import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB_PATH = Path("data/checkpoints.db")


@lru_cache
def get_checkpointer() -> SqliteSaver:
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        CHECKPOINT_DB_PATH,
        check_same_thread=False,
    )
    return SqliteSaver(connection)
