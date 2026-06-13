import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from scraper import (
    CSV_HEADER,
    LiveCollector,
    ensure_csv,
    get_csv_path,
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
    return json.dumps(
        {
            "id": None,
            "method": "candles_update",
            "params": [[ts, "100", "100", "105", "95", "10", "1000", pair]],
        }
    )


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


# --- run() ---


@pytest.mark.asyncio
async def test_run_initializes_last_ts_map(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    f = get_csv_path(data_dir, "BTC_USDT", "1m")
    ensure_csv(f)
    with open(f, "a") as fh:
        fh.write("1000,100,105,95,100,10,1000\n")

    ws = MockWS(messages=[], collector=collector)
    with patch("scraper.websockets.connect", return_value=ws):
        await collector.run()

    assert collector._last_ts_map[str(f)] == 1000


@pytest.mark.asyncio
async def test_run_creates_tasks_for_each_interval(collector):
    tasks = []

    async def fake_gather(*args, **kwargs):
        tasks.extend(args)
        return []

    ws = MockWS(messages=[], collector=collector)
    with patch("scraper.websockets.connect", return_value=ws):
        with patch("scraper.asyncio.gather", side_effect=fake_gather):
            await collector.run()

    assert len(tasks) == len(collector.config["intervals"])


# --- _stream_interval() ---


@pytest.mark.asyncio
async def test_stream_subscribes_to_all_pairs(collector):
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    subs = [m for m in ws.sent if m.get("method") == "candles_subscribe"]
    assert len(subs) == len(collector.config["pairs"])
    pairs_seen = {s["params"][0] for s in subs}
    assert pairs_seen == set(collector.config["pairs"])


@pytest.mark.asyncio
async def test_stream_sends_interval_seconds(collector):
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("5m")

    subs = [m for m in ws.sent if m.get("method") == "candles_subscribe"]
    for s in subs:
        assert s["params"][1] == 300


@pytest.mark.asyncio
async def test_stream_writes_candle_to_csv(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == ",".join(CSV_HEADER)
    assert len(lines) == 2
    assert lines[1] == "1000,100,105,95,100,10,1000"


@pytest.mark.asyncio
async def test_stream_candle_reordered_to_ohlcv(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg = json.dumps(
        {
            "id": None,
            "method": "candles_update",
            "params": [[2000, "open", "close", "high", "low", "vol", "deal", "BTC_USDT"]],
        }
    )
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert lines[1] == "2000,open,high,low,close,vol,deal"


@pytest.mark.asyncio
async def test_stream_updates_last_ts_map(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[msg], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    assert collector._last_ts_map[str(csv_path)] == 1000


@pytest.mark.asyncio
async def test_stream_overwrites_same_timestamp(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg1 = candle_msg("BTC_USDT", 1000)
    msg2 = json.dumps(
        {
            "id": None,
            "method": "candles_update",
            "params": [[1000, "200", "200", "205", "195", "20", "2000", "BTC_USDT"]],
        }
    )
    ws = MockWS(messages=[msg1, msg2], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[1] == "1000,200,205,195,200,20,2000"


@pytest.mark.asyncio
async def test_stream_appends_new_timestamp(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg1 = candle_msg("BTC_USDT", 1000)
    msg2 = candle_msg("BTC_USDT", 2000)
    ws = MockWS(messages=[msg1, msg2], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3


@pytest.mark.asyncio
async def test_stream_routes_to_correct_csv_files(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    msg1 = candle_msg("BTC_USDT", 1000)
    msg2 = candle_msg("ETH_USDT", 1000)
    ws = MockWS(messages=[msg1, msg2], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    btc_lines = (get_csv_path(data_dir, "BTC_USDT", "1m")).read_text().strip().split("\n")
    eth_lines = (get_csv_path(data_dir, "ETH_USDT", "1m")).read_text().strip().split("\n")
    assert len(btc_lines) == 2
    assert len(eth_lines) == 2


@pytest.mark.asyncio
async def test_stream_skips_non_candles_update(collector, tmp_path):
    data_dir = collector.config["output_dir"]
    non_update = json.dumps({"id": 1, "result": {"status": "success"}, "error": None})
    candle = candle_msg("BTC_USDT", 1000)
    ws = MockWS(messages=[non_update, candle], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    csv_path = get_csv_path(data_dir, "BTC_USDT", "1m")
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_stream_reconnects_on_error(collector):
    call_count = 0

    def make_ws():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockWS(messages=[], raise_on_iter=ConnectionError("first"), collector=None)
        return MockWS(messages=[], raise_on_iter=ConnectionError("second"), collector=collector)

    with (
        patch("scraper.websockets.connect", side_effect=lambda *a, **k: make_ws()),
        patch("scraper.RECONNECT_DELAY", 0),
    ):
        await collector._stream_interval("1m")

    assert call_count == 2


@pytest.mark.asyncio
async def test_stream_stops_when_not_running(collector):
    collector.stop()
    ws = MockWS(messages=[candle_msg("BTC_USDT", 1000)], collector=collector)

    with patch("scraper.websockets.connect", return_value=ws):
        await collector._stream_interval("1m")

    assert ws.sent == []


# --- _keepalive() ---


@pytest.mark.asyncio
async def test_keepalive_sends_pings():
    sent = []

    class FakeWS:
        async def send(self, data):
            sent.append(json.loads(data))

    with patch("scraper.PING_INTERVAL", 0):
        task = asyncio.create_task(LiveCollector._keepalive(LiveCollector({}), FakeWS()))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(sent) >= 1
    assert sent[0]["method"] == "ping"
    assert sent[0]["params"] == []


# --- _subscribe() ---


@pytest.mark.asyncio
async def test_subscribe_sends_message():
    sent = []

    class FakeWS:
        async def send(self, data):
            sent.append(json.loads(data))

    with patch("scraper.asyncio.sleep", new_callable=AsyncMock):
        await LiveCollector._subscribe(LiveCollector({}), FakeWS(), "BTC_USDT", 60)

    assert len(sent) == 1
    assert sent[0]["method"] == "candles_subscribe"
    assert sent[0]["params"] == ["BTC_USDT", 60]


@pytest.mark.asyncio
async def test_subscribe_includes_delay():
    sleep_called = False

    async def fake_sleep(_):
        nonlocal sleep_called
        sleep_called = True

    class FakeWS:
        async def send(self, data):
            pass

    with patch("scraper.asyncio.sleep", side_effect=fake_sleep):
        await LiveCollector._subscribe(LiveCollector({}), FakeWS(), "BTC_USDT", 60)

    assert sleep_called


# --- stop() ---


def test_stop_sets_running_false(collector):
    assert collector._running is True
    collector.stop()
    assert collector._running is False
