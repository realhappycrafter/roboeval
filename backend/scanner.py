# -*- coding: utf-8 -*-
"""LeRobot 格式数据集扫描器：扫描 info.json，解析元信息与 episodes 写入数据库。"""
import json
from pathlib import Path

import pandas as pd

from db import get_conn

# 数据集根目录（相对 roboeval/ 上一级 AIC2026/assets）
ASSETS = Path(__file__).resolve().parent.parent.parent / "assets" / "datasets"
SCAN_ROOTS = [
    ("local-zip1", ASSETS / "raw1" / "数据集"),
    ("local-zip2", ASSETS / "raw2" / "test_merged"),
]


def _read_episodes_meta(dataset_dir: Path) -> list[dict]:
    """读取 meta/episodes 下的 parquet，返回 [{episode_index, length, task}]。"""
    out = []
    for pq in sorted(dataset_dir.glob("meta/episodes/**/*.parquet")):
        try:
            df = pd.read_parquet(pq)
        except Exception as e:  # noqa: BLE001
            print(f"[scanner] 读取失败 {pq}: {e}")
            continue
        for _, row in df.iterrows():
            task = row.get("tasks")
            if isinstance(task, (list, tuple)):
                task = "; ".join(str(t) for t in task)
            out.append(
                {
                    "episode_index": int(row.get("episode_index", len(out))),
                    "length": int(row["length"]) if "length" in row and pd.notna(row["length"]) else None,
                    "task": str(task) if task is not None else None,
                }
            )
    return out


def scan_dataset(dataset_dir: Path, source: str) -> dict | None:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    cameras = [k for k in features if "images" in k]
    eps = _read_episodes_meta(dataset_dir)
    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "source": source,
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes") or len(eps),
        "total_frames": info.get("total_frames"),
        "cameras": json.dumps(cameras, ensure_ascii=False),
        "episodes": eps,
    }


def scan_directory(root: Path, source: str) -> list[dict]:
    """扫描任意目录：root 本身是数据集则收它，否则遍历其直接子目录找数据集。"""
    if not root.exists() or not root.is_dir():
        raise ValueError(f"目录不存在或不是目录: {root}")
    recs = []
    if (root / "meta" / "info.json").exists():
        rec = scan_dataset(root, source)
        if rec:
            recs.append(rec)
    else:
        for child in sorted(root.iterdir()):
            if child.is_dir():
                rec = scan_dataset(child, source)
                if rec:
                    recs.append(rec)
    return recs


def ingest(recs: list[dict]) -> tuple[int, int]:
    """把扫描结果写入数据库，返回 (数据集数, episode 数)。"""
    conn = get_conn()
    n_ds = n_ep = 0
    for rec in recs:
        conn.execute(
            "INSERT OR REPLACE INTO datasets (name, path, source, robot_type, fps,"
            " total_episodes, total_frames, cameras) VALUES (?,?,?,?,?,?,?,?)",
            (rec["name"], rec["path"], rec["source"], rec["robot_type"], rec["fps"],
             rec["total_episodes"], rec["total_frames"], rec["cameras"]),
        )
        row = conn.execute("SELECT id FROM datasets WHERE name=? AND source=?",
                           (rec["name"], rec["source"])).fetchone()
        if row is None:
            continue
        ds_id = row["id"]
        conn.execute("DELETE FROM episodes WHERE dataset_id=?", (ds_id,))
        for ep in rec["episodes"]:
            conn.execute(
                "INSERT OR IGNORE INTO episodes (dataset_id, episode_index, length, task)"
                " VALUES (?,?,?,?)",
                (ds_id, ep["episode_index"], ep["length"], ep["task"]),
            )
            n_ep += 1
        n_ds += 1
    conn.commit()
    conn.close()
    return n_ds, n_ep


def run_scan(reset: bool = False) -> dict:
    """全量扫描内置根目录并写库。reset=True 时清空 datasets/episodes 重建。"""
    from db import init_db

    init_db()
    if reset:
        conn = get_conn()
        conn.execute("DELETE FROM episodes")
        conn.execute("DELETE FROM datasets")
        conn.commit()
        conn.close()

    all_recs: list[dict] = []
    for source, root in SCAN_ROOTS:
        if not root.exists():
            print(f"[scanner] 跳过不存在的根目录: {root}")
            continue
        recs = scan_directory(root, source)
        print(f"[scanner] {root} -> 发现 {len(recs)} 个数据集")
        all_recs.extend(recs)
    n_ds, n_ep = ingest(all_recs)

    conn = get_conn()
    total_ds = conn.execute("SELECT COUNT(*) c FROM datasets").fetchone()["c"]
    total_ep = conn.execute("SELECT COUNT(*) c FROM episodes").fetchone()["c"]
    conn.close()
    print(f"[scanner] 本次入库 {n_ds} 集 / {n_ep} 条；库内总计 {total_ds} 集 / {total_ep} 条")
    return {"datasets": total_ds, "episodes": total_ep, "imported": n_ds}


if __name__ == "__main__":
    run_scan(reset=True)
