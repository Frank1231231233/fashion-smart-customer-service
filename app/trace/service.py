import json
from pathlib import Path
from threading import Lock

TRACE_FILE = Path(__file__).resolve().parent / "trace.jsonl"
_TRACE_LOCK = Lock()


def append_trace_event(event: dict[str, object]) -> None:
    with _TRACE_LOCK, TRACE_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def query_trace_events(trace_id: str) -> list[dict[str, object]]:
    with _TRACE_LOCK:
        if not TRACE_FILE.exists():
            return []

        events: list[dict[str, object]] = []
        with TRACE_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue
                event = json.loads(line)
                if event.get("trace_id") != trace_id:
                    continue
                events.append(event)
            return events
