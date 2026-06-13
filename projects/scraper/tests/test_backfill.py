import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scraper import (
    CSV_HEADER,
    Backfiller,
    append_candles,
    ensure_csv,
    get_csv_path,
)


def _make_config(tmp_path, pairs=None, intervals=None, start_date=None, chunk_size=500, delay=0.0):
    return {
        "pairs": pairs or ["BTC_USDT"],
        "intervals": intervals or ["1m"],
        "output_dir": str(tmp_path / "data"),
        "start_date": start_date,
        "backfill_chunk_size": chunk_size,
        "backfill_delay": delay,
    }


def _whitebit_candle(ts, pair="BTC_USDT"):
    return [ts, "100.0", "100.0", "105.0", "95.0", "10.0", "1000.0", pair]


def _ws_response(req_id, candles):
    return json.dumps({"id": req_id, "result": candles, "error": None})


def _ws_error_response(req_id, error_msg):
    return json.dumps({"id": req_id, "result": None, "error": error_msg})


class FakeWS:
    """Returns pre-loaded responses in order. Used for precise control in unit tests."""

    def __init__(self, recv_queue=None):
        self._recv_queue = list(recv_queue or [])
        self._recv_idx = 0
        self.sent = []

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        if self._recv_idx < len(self._recv_queue):
            resp = self._recv_queue[self._recv_idx]
            self._recv_idx += 1
            return resp
        await asyncio.sleep(3600)
        raise asyncio.CancelledError()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class SmartFakeWS:
    """Auto-generates valid responses for candles_request.
    Used for integration-style tests where the exact number of chunks isn't known."""

    def __init__(self, result_factory=None):
        self.sent = []
        self._result_factory = result_factory or (lambda req: [_whitebit_candle(req["params"][1])])

    async def send(self, msg):
        self.sent.append(json.loads(msg))

    async def recv(self):
        for req in reversed(self.sent):
            if req.get("method") == "candles_request":
                result = self._result_factory(req)
                return _ws_response(req["id"], result)
        await asyncio.sleep(3600)
        raise asyncio.CancelledError()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeConnect:
    """Wraps a WS object to mimic websockets.connect() async context manager."""

    def __init__(self, ws_obj):
        self._ws = ws_obj

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *args):
        pass


def _seed_csv(csv_path, ts):
    ensure_csv(csv_path)
    append_candles(csv_path, [[ts, "1", "2", "3", "4", "5", "6", "X"]])


# --- _determine_start ---


class TestDetermineStart:
    def test_from_last_timestamp(self, tmp_path):
        csv_path = tmp_path / "BTC_USDT" / "1m.csv"
        ensure_csv(csv_path)
        append_candles(csv_path, [[1000, "1", "1", "1", "1", "1", "1", "X"]])
        config = _make_config(tmp_path)
        b = Backfiller(config)
        assert b._determine_start(csv_path, 60) == 1060

    def test_from_start_date(self, tmp_path):
        csv_path = tmp_path / "BTC_USDT" / "1m.csv"
        ensure_csv(csv_path)
        config = _make_config(tmp_path, start_date="2026-01-01")
        b = Backfiller(config)
        expected = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
        assert b._determine_start(csv_path, 60) == expected

    def test_default_30_days(self, tmp_path):
        csv_path = tmp_path / "BTC_USDT" / "1m.csv"
        ensure_csv(csv_path)
        config = _make_config(tmp_path)
        b = Backfiller(config)
        before = int((datetime.now(UTC) - timedelta(days=30)).timestamp())
        result = b._determine_start(csv_path, 60)
        after = int((datetime.now(UTC) - timedelta(days=30)).timestamp())
        assert before <= result <= after

    def test_csv_takes_priority_over_start_date(self, tmp_path):
        csv_path = tmp_path / "BTC_USDT" / "1m.csv"
        _seed_csv(csv_path, 5000)
        config = _make_config(tmp_path, start_date="2026-01-01")
        b = Backfiller(config)
        assert b._determine_start(csv_path, 60) == 5060


# --- _fetch_candles ---


class TestFetchCandles:
    @pytest.mark.asyncio
    async def test_matches_response_by_id(self):
        candle = _whitebit_candle(1000)
        target_id = 1
        ws = FakeWS(recv_queue=[_ws_response(target_id, [candle])])
        b = Backfiller(_make_config(Path("/tmp")))
        b._next_id = lambda: target_id
        result = await b._fetch_candles(ws, "BTC_USDT", 1000, 2000, 60)
        assert result == [candle]

    @pytest.mark.asyncio
    async def test_skips_unmatched_ids(self):
        candle = _whitebit_candle(1000)
        target_id = 5
        ws = FakeWS(
            recv_queue=[
                _ws_response(999, []),
                _ws_response(target_id, [candle]),
            ]
        )
        b = Backfiller(_make_config(Path("/tmp")))
        b._next_id = lambda: target_id
        result = await b._fetch_candles(ws, "BTC_USDT", 1000, 2000, 60)
        assert result == [candle]

    @pytest.mark.asyncio
    async def test_error_in_response(self):
        target_id = 3
        ws = FakeWS(recv_queue=[_ws_error_response(target_id, "bad request")])
        b = Backfiller(_make_config(Path("/tmp")))
        b._next_id = lambda: target_id
        with pytest.raises(RuntimeError, match="candles_request error"):
            await b._fetch_candles(ws, "BTC_USDT", 1000, 2000, 60)

    @pytest.mark.asyncio
    async def test_sends_correct_request_format(self):
        candle = _whitebit_candle(1000)
        target_id = 7
        ws = FakeWS(recv_queue=[_ws_response(target_id, [candle])])
        b = Backfiller(_make_config(Path("/tmp")))
        b._next_id = lambda: target_id
        await b._fetch_candles(ws, "BTC_USDT", 1000, 2000, 60)
        sent = ws.sent[0]
        assert sent["id"] == target_id
        assert sent["method"] == "candles_request"
        assert sent["params"] == ["BTC_USDT", 1000, 2000, 60]


# --- _keepalive ---


class TestKeepalive:
    @pytest.mark.asyncio
    async def test_sends_ping(self):
        ws = MagicMock()
        ws.send = AsyncMock()
        b = Backfiller(_make_config(Path("/tmp")))

        call_count = 0

        async def fast_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("scraper.asyncio.sleep", side_effect=fast_sleep):
            task = asyncio.ensure_future(b._keepalive(ws))
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert ws.send.call_count >= 1
        sent = json.loads(ws.send.call_args_list[0][0][0])
        assert sent == {"id": 0, "method": "ping", "params": []}


# --- run() ---


class TestRun:
    @pytest.mark.asyncio
    async def test_creates_csv_with_ohlcv_data(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 120

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        new_ts = existing_ts + 60
        candle = _whitebit_candle(new_ts)
        ws_obj = FakeWS(recv_queue=[_ws_response(1, [candle])])

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert csv_path.exists()
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == ",".join(CSV_HEADER)
        assert str(new_ts) in lines[2]

    @pytest.mark.asyncio
    async def test_csv_ohlcv_column_order(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 120

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        new_ts = existing_ts + 60
        candle = [new_ts, "100.0", "100.0", "105.0", "95.0", "10.0", "1000.0", "BTC_USDT"]
        ws_obj = FakeWS(recv_queue=[_ws_response(1, [candle])])

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        row = csv_path.read_text().strip().split("\n")[2].split(",")
        assert row == [f"{new_ts}", "100.0", "105.0", "95.0", "100.0", "10.0", "1000.0"]

    @pytest.mark.asyncio
    async def test_resume_from_last_timestamp(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 7200

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        new_ts = existing_ts + 60
        candle = _whitebit_candle(new_ts)
        requests_seen = []
        ws_obj = FakeWS(recv_queue=[_ws_response(1, [candle])])
        original_send = ws_obj.send

        async def tracking_send(msg):
            await original_send(msg)
            data = json.loads(msg)
            if data.get("method") == "candles_request":
                requests_seen.append(data["params"])

        ws_obj.send = tracking_send

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert len(requests_seen) == 1
        assert requests_seen[0][1] == new_ts

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_skip_when_up_to_date(self, tmp_path):
        now = int(time.time())
        config = _make_config(tmp_path)

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, now)

        requests_seen = []
        ws_obj = SmartFakeWS()
        original_send = ws_obj.send

        async def tracking_send(msg):
            await original_send(msg)
            data = json.loads(msg)
            if data.get("method") == "candles_request":
                requests_seen.append(data)

        ws_obj.send = tracking_send

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert len(requests_seen) == 0

    @pytest.mark.asyncio
    async def test_start_date_override(self, tmp_path):
        config = _make_config(
            tmp_path,
            pairs=["BTC_USDT"],
            intervals=["1m"],
            start_date="2026-01-01",
        )
        expected_start = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())

        requests_seen = []
        ws_obj = SmartFakeWS()
        original_send = ws_obj.send

        async def tracking_send(msg):
            await original_send(msg)
            data = json.loads(msg)
            if data.get("method") == "candles_request":
                requests_seen.append(data["params"][1])

        ws_obj.send = tracking_send

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert len(requests_seen) >= 1
        assert requests_seen[0] == expected_start

    @pytest.mark.asyncio
    async def test_multiple_pairs_and_intervals(self, tmp_path):
        config = _make_config(
            tmp_path,
            pairs=["BTC_USDT", "ETH_USDT"],
            intervals=["1m", "5m"],
        )
        now = int(time.time())

        for pair in ["BTC_USDT", "ETH_USDT"]:
            for interval in ["1m", "5m"]:
                csv_path = get_csv_path(str(tmp_path / "data"), pair, interval)
                _seed_csv(csv_path, now - 120)

        candle = _whitebit_candle(now - 60)
        responses = [_ws_response(i + 1, [candle]) for i in range(4)]
        ws_obj = FakeWS(recv_queue=responses)

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        for pair in ["BTC_USDT", "ETH_USDT"]:
            for interval in ["1m", "5m"]:
                csv_path = get_csv_path(str(tmp_path / "data"), pair, interval)
                assert csv_path.exists()

    @pytest.mark.asyncio
    async def test_chunking_multiple_chunks(self, tmp_path):
        config = _make_config(
            tmp_path,
            pairs=["BTC_USDT"],
            intervals=["1m"],
            chunk_size=2,
        )
        now = int(time.time())
        existing_ts = now - 300

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        requests_seen = []
        ws_obj = SmartFakeWS()
        original_send = ws_obj.send

        async def tracking_send(msg):
            await original_send(msg)
            data = json.loads(msg)
            if data.get("method") == "candles_request":
                requests_seen.append((data["params"][1], data["params"][2]))

        ws_obj.send = tracking_send

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert len(requests_seen) >= 2

    @pytest.mark.asyncio
    async def test_keepalive_task_cancelled(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 120

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        candle = _whitebit_candle(existing_ts + 60)
        ws_obj = FakeWS(recv_queue=[_ws_response(1, [candle])])

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 0.01):
                await b.run()

    @pytest.mark.asyncio
    async def test_empty_candle_response(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 120

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        ws_obj = SmartFakeWS(result_factory=lambda req: [])

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_request_ids_increment(self, tmp_path):
        config = _make_config(tmp_path)
        now = int(time.time())
        existing_ts = now - 300

        csv_path = get_csv_path(str(tmp_path / "data"), "BTC_USDT", "1m")
        _seed_csv(csv_path, existing_ts)

        request_ids = []
        ws_obj = SmartFakeWS()
        original_send = ws_obj.send

        async def tracking_send(msg):
            await original_send(msg)
            data = json.loads(msg)
            if data.get("method") == "candles_request":
                request_ids.append(data["id"])

        ws_obj.send = tracking_send

        b = Backfiller(config)
        with patch("scraper.websockets.connect", return_value=FakeConnect(ws_obj)):
            with patch("scraper.PING_INTERVAL", 999999):
                await b.run()

        assert len(request_ids) >= 1
        for i in range(1, len(request_ids)):
            assert request_ids[i] > request_ids[i - 1]
