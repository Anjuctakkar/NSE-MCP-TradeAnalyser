"""
NSE MCP Dashboard - local FastAPI app over two NSE MCP servers.

Changes in this version:
  * CM server URL aligned with the .bat/watcher (mcp.nseindia.in); both URLs
    can be overridden with NSE_BHAV_URL / NSE_CM_URL env vars.
  * One MCP session per server is cached and reused (re-handshake on expiry).
  * Stocks are analysed concurrently (bounded).
  * Splits/bonuses inside the history window are applied to earlier closes
    via get_corporate_actions, so SMA/RSI/MACD and signals aren't distorted.
  * /api/recommend is wired to recommend.py.

Run:  pip install -r requirements.txt && python server.py   (Python 3.11+)
"""

import asyncio
import datetime as dt
import itertools
import json
import os
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import backtest
import recommend
import technical
import tracker

app = FastAPI(title="NSE MCP Dashboard")

SERVERS: dict[str, str] = {
    "bhavcopy": os.getenv("NSE_BHAV_URL", "https://mcp.nseindia.in/bhavcopy/cm/mcp"),
    "cm": os.getenv("NSE_CM_URL", "https://mcp.nseindia.in/cmmkt/mcp"),
}

_ID_COUNTER = itertools.count(1)
_INIT_PARAMS = {
    "protocolVersion": "2025-03-26",  # same as the working watcher/.bat
    "capabilities": {},
    "clientInfo": {"name": "nse-mcp-dashboard", "version": "0.2.0"},
}

ANALYSIS_SERVER = "bhavcopy"
ANALYSIS_HISTORY_MONTHS = 6
ANALYSIS_CONCURRENCY = 3


class CallToolRequest(BaseModel):
    server: str
    tool: str
    arguments: dict[str, Any] = {}


class StockAllocation(BaseModel):
    symbol: str
    allocation: float = 0


class AnalyzeRequest(BaseModel):
    stocks: list[StockAllocation]


class RecommendRequest(BaseModel):
    budget: float
    existing: list[dict[str, Any]] = []


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _server_url(server_key: str) -> str:
    if server_key not in SERVERS:
        raise HTTPException(400, f"Unknown server '{server_key}'. Known: {list(SERVERS)}")
    return SERVERS[server_key]


def _describe_error(exc: BaseException) -> str:
    parts: list[str] = []

    def walk(e: BaseException) -> None:
        if isinstance(e, BaseExceptionGroup):
            for sub in e.exceptions:
                walk(sub)
        else:
            parts.append(f"{type(e).__name__}: {e}")

    walk(exc)
    return " | ".join(parts) if parts else str(exc)


def _looks_empty(value: Any) -> bool:
    if value in (None, {}, []):
        return True
    return isinstance(value, dict) and set(value.keys()) <= {"type"}


# ---------------------------------------------------------------------------
# MCP transport with a cached session per server URL
# ---------------------------------------------------------------------------
_client: Optional[httpx.AsyncClient] = None
_sessions: dict[str, Optional[str]] = {}
_session_locks: dict[str, asyncio.Lock] = {}


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def _post(url: str, method: str, params: dict[str, Any], session_id: Optional[str],
                expect_response: bool = True) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if expect_response:
        payload["id"] = next(_ID_COUNTER)

    resp = await _get_client().post(url, json=payload, headers=headers)
    resp.raise_for_status()
    new_sid = resp.headers.get("mcp-session-id", session_id)
    if not expect_response:
        return None, new_sid

    data: Optional[dict[str, Any]] = None
    if "text/event-stream" in resp.headers.get("content-type", ""):
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:") and line[5:].strip():
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
        if data is None:
            raise RuntimeError(f"Could not parse SSE response for '{method}'")
    else:
        data = resp.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
    return data.get("result"), new_sid


async def _ensure_session(url: str, force: bool = False) -> Optional[str]:
    lock = _session_locks.setdefault(url, asyncio.Lock())
    async with lock:
        if url in _sessions and not force:
            return _sessions[url]
        _, sid = await _post(url, "initialize", _INIT_PARAMS, None)
        await _post(url, "notifications/initialized", {}, sid, expect_response=False)
        _sessions[url] = sid
        return sid


async def _rpc(url: str, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Send a request on the cached session; re-handshake once if it looks expired."""
    sid = await _ensure_session(url)
    try:
        result, _ = await _post(url, method, params, sid)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (400, 401, 404):
            raise
        sid = await _ensure_session(url, force=True)
        result, _ = await _post(url, method, params, sid)
    return result or {}


async def _list_tools(url: str) -> list[dict[str, Any]]:
    try:
        result = await _rpc(url, "tools/list", {})
        return [{"name": t.get("name", ""), "description": t.get("description") or "",
                 "input_schema": t.get("inputSchema") or {}} for t in result.get("tools", [])]
    except BaseException as exc:  # noqa: BLE001
        raise RuntimeError(_describe_error(exc)) from exc


async def _call_tool(url: str, tool_name: str, arguments: dict[str, Any]) -> Any:
    try:
        result = await _rpc(url, "tools/call", {"name": tool_name, "arguments": arguments})
        is_error = bool(result.get("isError", False))

        structured = result.get("structuredContent")
        if not _looks_empty(structured):
            return {"is_error": is_error, "content": structured}

        blocks: list[Any] = []
        for block in result.get("content", []) or []:
            if not isinstance(block, dict):
                blocks.append(block)
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if text is None:
                    text = block.get("data") or block.get("value") or block.get("content")
                if text is None:
                    blocks.append(block)
                    continue
                try:
                    blocks.append(json.loads(text))
                except (json.JSONDecodeError, TypeError):
                    blocks.append(text)
            else:
                blocks.append(block)

        friendly = blocks if len(blocks) != 1 else blocks[0]
        if _looks_empty(friendly):
            return {"is_error": is_error, "content": result,
                    "note": "No populated 'content'/'structuredContent' - showing the raw MCP result."}
        return {"is_error": is_error, "content": friendly}
    except BaseException as exc:  # noqa: BLE001
        raise RuntimeError(_describe_error(exc)) from exc


@app.on_event("shutdown")
async def _close_client() -> None:
    if _client is not None:
        await _client.aclose()


@app.get("/api/servers")
def get_servers():
    return dict(SERVERS)


@app.get("/api/tools")
async def get_tools(server: str):
    url = _server_url(server)
    try:
        tools = await _list_tools(url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Could not reach {url}: {exc}")
    return {"server": server, "url": url, "tools": tools}


@app.post("/api/call")
async def call_tool(req: CallToolRequest):
    url = _server_url(req.server)
    try:
        return await _call_tool(url, req.tool, req.arguments)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Call failed: {exc}")


# ---------------------------------------------------------------------------
# History + corporate actions
# ---------------------------------------------------------------------------
async def _fetch_history(url: str, symbol: str, total_months: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    end_date = "today"
    months_left = total_months
    while months_left > 0:
        chunk = min(3, months_left)
        raw = await _call_tool(url, "get_stock_history", {"symbol": symbol, "months": chunk, "endDate": end_date})
        if raw.get("is_error"):
            raise RuntimeError(f"get_stock_history failed: {raw.get('note') or raw.get('content')}")
        content = raw.get("content")
        data = content.get("data") if isinstance(content, dict) else None
        if not data:
            break
        records = data + records
        months_left -= chunk
        next_end = content.get("next_end_date")
        if not next_end:
            break
        end_date = next_end
    return records


async def _corporate_actions(url: str, symbol: str, first_date: str, last_date: str) -> dict[str, Any]:
    """All corporate actions with an ex-date inside the history window.

    Returns {"status": "ok"|"unavailable", "notes": [...], "adjustments": [...]}.
    Splits/bonuses with a usable factor go into `adjustments` (applied to prices);
    everything else (dividends, other events) is note-only. A failed lookup is
    reported as "unavailable" so the UI can say prices are unchecked."""
    try:
        raw = await _call_tool(url, "get_corporate_actions",
                               {"symbol": symbol, "fromDate": first_date[:10], "toDate": last_date[:10]})
        if raw.get("is_error"):
            return {"status": "unavailable", "notes": [], "adjustments": []}
        events = recommend.find_records(raw.get("content"))
    except Exception:  # noqa: BLE001
        return {"status": "unavailable", "notes": [], "adjustments": []}
    notes, adjustments = [], []
    for ev in events:
        kind = str(ev.get("actionType", "OTHER")).upper()
        factor = _to_float(ev.get("adjustmentFactor"))
        ex = str(ev.get("exDate", ""))[:10]
        if not (first_date[:10] <= ex <= last_date[:10]):
            continue
        adjusted = kind in ("SPLIT", "BONUS") and bool(factor) and 0 < factor < 1
        notes.append({"exDate": ex, "type": kind, "purpose": ev.get("purpose"),
                      "factor": factor, "adjusted": adjusted})
        if adjusted:
            adjustments.append({"exDate": ex, "type": kind, "factor": factor, "purpose": ev.get("purpose")})
    notes.sort(key=lambda n: n["exDate"])
    return {"status": "ok", "notes": notes, "adjustments": adjustments}


def _apply_adjustments(dates: list[str], closes: list[Optional[float]],
                       adjustments: list[dict[str, Any]]) -> list[Optional[float]]:
    """Scale closes before each ex-date by that event's factor (factors compound)."""
    out = list(closes)
    for i, d in enumerate(dates):
        if out[i] is None:
            continue
        for adj in adjustments:
            if d[:10] < adj["exDate"]:
                out[i] *= adj["factor"]
    return out


async def _search_symbols(url: str, query: str) -> list[dict[str, Any]]:
    raw = await _call_tool(url, "search_symbols", {"query": query})
    if raw.get("is_error"):
        raise RuntimeError(f"search_symbols failed: {raw.get('note') or raw.get('content')}")
    content = raw.get("content")
    return (content.get("results") if isinstance(content, dict) else None) or []


@app.get("/api/search")
async def search_symbol(query: str):
    url = _server_url(ANALYSIS_SERVER)
    q = query.strip()
    if not q:
        return {"query": q, "tried": [], "results": []}

    words: list[str] = []
    for w in q.replace("-", " ").split():
        if len(w) >= 3 and w.lower() not in (x.lower() for x in words):
            words.append(w)
    attempts = [q] + [w for w in words if w.lower() != q.lower()]

    tried: list[str] = []
    found: dict[str, dict[str, Any]] = {}
    try:
        for attempt in attempts:
            tried.append(attempt)
            for r in await _search_symbols(url, attempt):
                found.setdefault(r["symbol"], r)
            if found:
                break
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Symbol search failed: {exc}")

    cutoff = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    active = [r for r in found.values() if str(r.get("last_date", "")) >= cutoff]
    q_upper = q.upper().replace(" ", "")
    active.sort(key=lambda r: (r["symbol"] != q_upper, len(r["symbol"]), r["symbol"]))
    return {"query": q, "tried": tried, "hidden_inactive": len(found) - len(active), "results": active[:10]}


async def _analyze_one(url: str, stock: StockAllocation, today: str) -> dict[str, Any]:
    symbol = stock.symbol.strip().upper()
    entry: dict[str, Any] = {"symbol": symbol, "allocation": stock.allocation}
    try:
        records = await _fetch_history(url, symbol, ANALYSIS_HISTORY_MONTHS)
        if not records:
            raise ValueError("get_stock_history returned no data for this symbol.")

        dates = [str(r.get("date", "")) for r in records]
        raw_closes: list[Optional[float]] = [_to_float(r.get("close")) for r in records]
        if all(c is None for c in raw_closes):
            raise ValueError("get_stock_history returned records with no usable 'close' values.")

        ca = await _corporate_actions(url, symbol, dates[0], dates[-1])
        adjustments = ca["adjustments"]
        closes = _apply_adjustments(dates, raw_closes, adjustments)

        ltp_date = next((d for d in reversed(dates) if d), today)
        ltp_raw = await _call_tool(url, "get_ltp_by_date", {"symbol": symbol, "date": ltp_date})
        if ltp_raw.get("is_error"):
            raise RuntimeError(f"get_ltp_by_date failed: {ltp_raw.get('note') or ltp_raw.get('content')}")
        latest_price = _to_float(ltp_raw.get("content")) or technical.latest_value(raw_closes)

        ma20_raw, ma50_raw = await asyncio.gather(
            _call_tool(url, "moving_average", {"symbol": symbol, "days": 20}),
            _call_tool(url, "moving_average", {"symbol": symbol, "days": 50}),
        )
        ma20_tool = None if ma20_raw.get("is_error") else _to_float(ma20_raw.get("content"))
        ma50_tool = None if ma50_raw.get("is_error") else _to_float(ma50_raw.get("content"))

        macd_line, macd_signal = technical.macd(closes)
        sma20_series = technical.sma(closes, 20)
        sma50_series = technical.sma(closes, 50)
        rsi_series = technical.rsi(closes, 14)
        trade_signal = technical.generate_trade_signal(
            closes, sma20_series, sma50_series, rsi_series, macd_line, macd_signal)

        # The signal's entry/stop/target come from adjusted closes; the latest
        # close is after any ex-date, so they're on the same scale as the LTP.
        try:
            ltp_age_days = (dt.date.fromisoformat(today) - dt.date.fromisoformat(ltp_date[:10])).days
        except ValueError:
            ltp_age_days = None

        qty = int(stock.allocation // latest_price) if latest_price else None
        invested = round(qty * latest_price, 2) if qty else 0.0
        entry.update({
            "dates": dates, "close": closes, "sma20": sma20_series, "sma50": sma50_series,
            "rsi": rsi_series, "macd_line": macd_line, "macd_signal": macd_signal,
            "moving_average_20": ma20_tool, "moving_average_50": ma50_tool,
            "latest_price": latest_price, "ltp_date": ltp_date, "ltp_age_days": ltp_age_days,
            "quantity": qty, "invested": invested,
            "leftover": round(stock.allocation - invested, 2),
            "trade_signal": trade_signal, "corporate_adjustments": adjustments,
            "ca_status": ca["status"], "ca_notes": ca["notes"],
        })
    except Exception as exc:  # noqa: BLE001
        entry["error"] = _describe_error(exc)
    return entry


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    url = _server_url(ANALYSIS_SERVER)
    today = dt.date.today().isoformat()
    sem = asyncio.Semaphore(ANALYSIS_CONCURRENCY)

    async def run(stock: StockAllocation) -> dict[str, Any]:
        async with sem:
            return await _analyze_one(url, stock, today)

    return {"results": await asyncio.gather(*(run(s) for s in req.stocks))}


async def _fetch_history_adjusted(url: str, symbol: str, total_months: int) -> list[dict[str, Any]]:
    """Like _fetch_history, but split/bonus-adjusts open/high/low/close so screening
    and backtests aren't distorted by ex-date price drops."""
    records = await _fetch_history(url, symbol, total_months)
    if not records:
        return records
    dates = [str(r.get("date", "")) for r in records]
    ca = await _corporate_actions(url, symbol, dates[0], dates[-1])
    if not ca["adjustments"]:
        return records
    keys = {k for r in records for k in r if str(k).lower() in ("open", "high", "low", "close")}
    cols = {k: _apply_adjustments(dates, [_to_float(r.get(k)) for r in records], ca["adjustments"]) for k in keys}
    return [{**r, **{k: cols[k][i] for k in keys if k in r}} for i, r in enumerate(records)]


@app.post("/api/recommend")
async def recommend_stocks(req: RecommendRequest):
    url = _server_url(ANALYSIS_SERVER)
    try:
        result = await recommend.recommend(url, req.existing, req.budget, _fetch_history_adjusted, _call_tool)
        result["logged"] = await tracker.add_picks(result.get("picks", []))
        return result
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Recommendation failed: {_describe_error(exc)}")


# ---------------------------------------------------------------------------
# Evaluation: backtest + forward track record of logged suggestions
# ---------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    symbols: list[str]
    months: int = 12
    max_hold: int = 20
    cost_pct: float = 0.2


class TakenRequest(BaseModel):
    taken: bool


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    url = _server_url(ANALYSIS_SERVER)
    symbols = list(dict.fromkeys(s.strip().upper() for s in req.symbols if s.strip()))[:10]
    if not symbols:
        raise HTTPException(400, "Give at least one symbol.")
    months = max(6, min(req.months, 24))
    max_hold = max(2, min(req.max_hold, 120))
    cost = max(0.0, min(req.cost_pct, 5.0))
    sem = asyncio.Semaphore(ANALYSIS_CONCURRENCY)

    async def one(sym: str) -> dict[str, Any]:
        async with sem:
            try:
                records = await _fetch_history_adjusted(url, sym, months)
                if not records:
                    raise ValueError("No price history returned.")
                return {"symbol": sym, **backtest.run(records, max_hold, cost)}
            except Exception as exc:  # noqa: BLE001
                return {"symbol": sym, "error": _describe_error(exc)}

    results = await asyncio.gather(*(one(s) for s in symbols))
    ok = [r for r in results if "error" not in r]
    return {"params": {"months": months, "max_hold": max_hold, "cost_pct": cost},
            "results": results, "pooled": backtest.pool(ok) if ok else None}


@app.get("/api/track")
async def track(max_hold: int = 20, cost_pct: float = 0.2):
    """Re-evaluate every logged suggestion against price data since it was made."""
    url = _server_url(ANALYSIS_SERVER)
    entries = await tracker.list_entries()
    if not entries:
        return {"entries": [], "summary": backtest.summarize_forward([]), "taken_summary": None}

    today = dt.date.today()
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_symbol.setdefault(e["symbol"], []).append(e)
    sem = asyncio.Semaphore(ANALYSIS_CONCURRENCY)

    async def history(sym: str, group: list[dict[str, Any]]):
        oldest = min(dt.date.fromisoformat(e["as_of"]) for e in group)
        months = max(1, min(6, (today - oldest).days // 30 + 2))
        async with sem:
            try:
                return sym, await _fetch_history_adjusted(url, sym, months), None
            except Exception as exc:  # noqa: BLE001
                return sym, None, _describe_error(exc)

    hist = {sym: (recs, err) for sym, recs, err in await asyncio.gather(
        *(history(s, g) for s, g in by_symbol.items()))}

    rows = []
    for e in entries:
        recs, err = hist[e["symbol"]]
        if not recs:
            ev = {"status": "no_data", "error": err}
        else:
            ev = backtest.evaluate_forward(recs, e["as_of"], e["entry"], e["stop_loss"], e["target"],
                                           max(2, min(max_hold, 120)), cost_pct)
        rows.append({**e, **ev})
    rows.sort(key=lambda r: (r["as_of"], r["symbol"]), reverse=True)
    taken = [r for r in rows if r.get("taken")]
    return {"entries": rows, "summary": backtest.summarize_forward(rows),
            "taken_summary": backtest.summarize_forward(taken) if taken else None}


@app.post("/api/track/{entry_id}/taken")
async def track_taken(entry_id: str, req: TakenRequest):
    if not await tracker.set_taken(entry_id, req.taken):
        raise HTTPException(404, "Suggestion not found.")
    return {"ok": True}


@app.delete("/api/track/{entry_id}")
async def track_delete(entry_id: str):
    if not await tracker.delete(entry_id):
        raise HTTPException(404, "Suggestion not found.")
    return {"ok": True}


STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
