"""Tiny JSON-file log of suggestions, so their real outcomes can be tracked."""
import asyncio
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

LOG_PATH = Path(__file__).resolve().parent / "suggestions_log.json"
_lock = asyncio.Lock()


def _load() -> list[dict[str, Any]]:
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(rows: list[dict[str, Any]]) -> None:
    tmp = LOG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tmp.replace(LOG_PATH)


async def add_picks(picks: list[dict[str, Any]]) -> int:
    """Log picks; the same symbol for the same as_of date is only logged once."""
    async with _lock:
        rows = _load()
        seen = {(r["symbol"], r["as_of"]) for r in rows}
        added = 0
        for p in picks:
            if (p["symbol"], p["as_of"]) in seen:
                continue
            rows.append({
                "id": uuid.uuid4().hex[:10], "symbol": p["symbol"], "as_of": p["as_of"],
                "entry": p["entry"], "stop_loss": p["stop_loss"], "target": p["target"],
                "score": p.get("score"), "logged_at": dt.datetime.now().isoformat(timespec="seconds"),
                "taken": False,
            })
            added += 1
        if added:
            _save(rows)
        return added


async def list_entries() -> list[dict[str, Any]]:
    async with _lock:
        return _load()


async def set_taken(entry_id: str, taken: bool) -> bool:
    async with _lock:
        rows = _load()
        for r in rows:
            if r["id"] == entry_id:
                r["taken"] = taken
                _save(rows)
                return True
        return False


async def delete(entry_id: str) -> bool:
    async with _lock:
        rows = _load()
        kept = [r for r in rows if r["id"] != entry_id]
        if len(kept) == len(rows):
            return False
        _save(kept)
        return True
