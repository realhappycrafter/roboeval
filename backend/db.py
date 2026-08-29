# -*- coding: utf-8 -*-
"""RoboEval 数据库层：SQLite + 血缘表结构。

血缘链路：datasets -> episodes -> experiments -> checkpoints -> rollouts -> annotations
"""
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "roboeval.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'local',
    robot_type TEXT,
    fps INTEGER,
    total_episodes INTEGER,
    total_frames INTEGER,
    cameras TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    UNIQUE(source, name)
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    episode_index INTEGER NOT NULL,
    length INTEGER,
    task TEXT,
    UNIQUE(dataset_id, episode_index)
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model TEXT,
    dataset_names TEXT,
    hyperparams TEXT,
    log_path TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    step INTEGER,
    path TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS rollouts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id INTEGER REFERENCES checkpoints(id) ON DELETE CASCADE,
    dataset_id INTEGER REFERENCES datasets(id),
    video_path TEXT,
    result TEXT NOT NULL DEFAULT 'unknown' CHECK (result IN ('success','failure','unknown')),
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rollout_id INTEGER NOT NULL REFERENCES rollouts(id) ON DELETE CASCADE,
    stage TEXT,
    cause TEXT,
    confidence REAL,
    source TEXT NOT NULL DEFAULT 'ai' CHECK (source IN ('ai','human')),
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
