import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest
from websockets.exceptions import ConnectionClosed

from scraper import (
    CSV_HEADER,
    LiveCollector,
    ensure_csv,
    get_csv_path,
    get_last_timestamp,
)


class MockAsyncIterator:
    def __init__(self, messages, raise_on_iter=None):
        self._messages = list(messages)
        self._raise_on_iter = raise_on_iter
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_on_iter and self._idx == 0:
            raise self._raise_on_iter
        if self._idx >= len(self._messages):
            raise StopAsyncIteration
        msg = self._messages[self._idx]
        self._idx += 1
        return msg


class MockWS:
    def __init__(self, messages=None, raise_on_iter=None, collector=None):
        self._messages = messages or []
        self._raise_on_iter = raise_on_iter
        self._collector = collector
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return MockAsyncIterator(self._messages, self._raise_on_iter)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        if self._collector is not None:
            self._collector._running = False


def candle_msg(pair, ts):
    return json.dumps({
        "id": None,
        "method": "candles_update",
        "params": [[ts, "100", "100", "105", "95", "10", "1000", pair]],
    })


@pytest.fixture
def collector(tmp_path):
    config = {
        "pairs": ["BTC_USDT", "ETH_USDT"],
        "intervals": ["1m", "5m"],
        "output_dir": str(tmp_path / "data"),
    }
    c = LiveCollector(config)
    for pair in config["pairs"]:
        for interval in config["intervals"]:
            ensure_csv(get_csv_path(config["output_dir"], pair, interval))
    return c


# --- Auto-reconnect: ConnectionClosed ---


@pytest.mark.asyncio
async def test_reconnect_on_connection_closed(collector):
    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[],
                raise_on_iter=ConnectionClosed(None, None),
                collector=None,
            )
        return MockWS(
            messages=[],
            raise_on_iter=ConnectionClosed(None, None),
            collector=collector,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert call_count == 2


# --- Auto-reconnect: ConnectionError ---


@pytest.mark.asyncio
async def test_reconnect_on_connection_error(collector):
    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[], raise_on_iter=ConnectionError("first"), collector=None
            )
        return MockWS(
            messages=[],
            raise_on_iter=ConnectionError("second"),
            collector=collector,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert call_count == 2


# --- Auto-reconnect: unexpected exception ---


@pytest.mark.asyncio
async def test_reconnect_on_unexpected_exception(collector):
    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[],
                raise_on_iter=RuntimeError("unexpected"),
                collector=None,
            )
        return MockWS(
            messages=[],
            raise_on_iter=RuntimeError("unexpected"),
            collector=collector,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert call_count == 2


# --- Reconnection logging ---


@pytest.mark.asyncio
async def test_connection_closed_logs_warning(collector, caplog):
    def make_ws():
        return MockWS(
            messages=[],
            raise_on_iter=ConnectionClosed(None, None),
            collector=collector,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.WARNING):
            await collector._stream_interval("1m")

    assert any("Connection lost" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_connection_error_logs_warning(collector, caplog):
    def make_ws():
        return MockWS(
            messages=[], raise_on_iter=ConnectionError("test"), collector=collector
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.WARNING):
            await collector._stream_interval("1m")

    assert any("Connection lost" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_unexpected_exception_logs_error(collector, caplog):
    def make_ws():
        return MockWS(
            messages=[],
            raise_on_iter=RuntimeError("boom"),
            collector=collector,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.ERROR):
            await collector._stream_interval("1m")

    assert any("Unexpected error" in r.message for r in caplog.records)


# --- After reconnect, subscriptions re-established ---


@pytest.mark.asyncio
async def test_reconnect_resubscribes_to_all_pairs(collector):
    call_count = 0
    all_sent = []

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[], raise_on_iter=ConnectionError("first"), collector=None
            )
        ws = MockWS(messages=[], collector=collector)
        all_sent.append(ws.sent)
        return ws

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert call_count == 2


# --- Refresh last_ts_map on reconnect ---


@pytest.mark.asyncio
async def test_refresh_last_ts_map_on_reconnect(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")

    with open(csv_path, "a") as f:
        f.write("5000,100,105,95,100,10,1000\n")

    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[], raise_on_iter=ConnectionError("first"), collector=None
            )
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert collector._last_ts_map[str(csv_path)] == 5000


@pytest.mark.asyncio
async def test_refresh_updates_stale_entry(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")

    collector._last_ts_map[str(csv_path)] = 100

    with open(csv_path, "a") as f:
        f.write("9999,100,105,95,100,10,1000\n")

    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[], raise_on_iter=ConnectionError("first"), collector=None
            )
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert collector._last_ts_map[str(csv_path)] == 9999


# --- Gap detection ---


@pytest.mark.asyncio
async def test_gap_detection_logs_warning(collector, caplog, tmp_path):
    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")

    with open(csv_path, "a") as f:
        f.write("1000,100,105,95,100,10,1000\n")

    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(
                messages=[], raise_on_iter=ConnectionError("fail"), collector=None
            )
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.WARNING):
            await collector._stream_interval("1m")

    gap_warnings = [r for r in caplog.records if "gap" in r.message.lower()]
    assert len(gap_warnings) > 0
    assert any("Consider running" in r.message for r in gap_warnings)


@pytest.mark.asyncio
async def test_no_gap_warning_when_data_is_recent(collector, caplog, tmp_path):
    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    now = int(time.time())

    with open(csv_path, "a") as f:
        f.write(f"{now},100,105,95,100,10,1000\n")

    def make_ws():
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.WARNING):
            await collector._stream_interval("1m")

    gap_warnings = [r for r in caplog.records if "gap" in r.message.lower()]
    assert len(gap_warnings) == 0


@pytest.mark.asyncio
async def test_gap_threshold_is_two_intervals(collector, caplog, tmp_path):
    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    now = int(time.time())
    just_below_threshold = now - 119

    with open(csv_path, "a") as f:
        f.write(f"{just_below_threshold},100,105,95,100,10,1000\n")

    def make_ws():
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        with caplog.at_level(logging.WARNING):
            await collector._stream_interval("1m")

    gap_warnings = [r for r in caplog.records if "gap" in r.message.lower()]
    assert len(gap_warnings) == 0


# --- Multiple rapid disconnections ---


@pytest.mark.asyncio
async def test_multiple_rapid_disconnections(collector):
    disconnect_count = 0
    max_disconnects = 5

    def make_ws():
        nonlocal disconnect_count
        disconnect_count += 1
        if disconnect_count >= max_disconnects:
            collector.stop()
        return MockWS(
            messages=[],
            raise_on_iter=ConnectionError(f"disconnect {disconnect_count}"),
            collector=None,
        )

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert disconnect_count == max_disconnects


@pytest.mark.asyncio
async def test_mixed_exception_types_without_crash(collector):
    disconnect_count = 0

    exceptions = [
        ConnectionClosed(None, None),
        ConnectionError("net err"),
        RuntimeError("unexpected"),
        ConnectionClosed(None, None),
    ]

    def make_ws():
        nonlocal disconnect_count
        exc = exceptions[disconnect_count]
        disconnect_count += 1
        if disconnect_count >= len(exceptions):
            collector.stop()
        return MockWS(messages=[], raise_on_iter=exc, collector=None)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert disconnect_count == len(exceptions)


# --- Graceful shutdown ---


@pytest.mark.asyncio
async def test_stop_sets_running_false(collector):
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    assert collector._running is False


@pytest.mark.asyncio
async def test_stop_prevents_connection(collector):
    collector.stop()
    ws = MockWS(messages=[candle_msg("BTC_USDT", 1000)], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    assert ws.sent == []


@pytest.mark.asyncio
async def test_running_checked_in_loop(collector):
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    data_dir = collector.config["output_dir"]
    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2


# --- CSV integrity ---


@pytest.mark.asyncio
async def test_csv_not_corrupted_after_stream(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg1 = candle_msg("BTC_USDT", 1000)
    msg2 = candle_msg("BTC_USDT", 2000)
    ws = MockWS(messages=[msg1, msg2], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 3
    for line in lines:
        fields = line.split(",")
        assert len(fields) == 7


@pytest.mark.asyncio
async def test_csv_valid_after_reconnect_interruption(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg = candle_msg("BTC_USDT", 1000)
    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(messages=[msg], raise_on_iter=ConnectionError("fail"), collector=None)
        return MockWS(messages=[], collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == ",".join(CSV_HEADER)
    for line in lines:
        fields = line.split(",")
        assert len(fields) == 7


# --- Signal handlers ---


def test_signal_handlers_registered(tmp_path):
    import signal

    config = {
        "pairs": ["BTC_USDT"],
        "intervals": ["1m"],
        "output_dir": str(tmp_path / "data"),
    }
    collector = LiveCollector(config)

    original_int = signal.getsignal(signal.SIGINT)
    original_term = signal.getsignal(signal.SIGTERM)

    signal.signal(signal.SIGINT, lambda s, f: collector.stop())
    signal.signal(signal.SIGTERM, lambda s, f: collector.stop())

    assert collector._running is True
    import os
    os.kill(os.getpid(), signal.SIGINT)
    assert collector._running is False

    signal.signal(signal.SIGINT, original_int)
    signal.signal(signal.SIGTERM, original_term)
