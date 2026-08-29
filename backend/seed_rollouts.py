# -*- coding: utf-8 -*-
"""评测 rollout 播种：把已有评测视频登记到对应检查点下。幂等（按 video_path 去重）。"""
from pathlib import Path

from db import get_conn, init_db

SEED = [
    # (实验名, 检查点 step, 素材相对路径, 初始结果, 备注)
    ("act_so101_test", "last", "rollouts/eval_act_so101_test_20260803.mp4", "unknown",
     "训练后真机评测录像（微信传输），内容待标注/AI归因"),
]


def main() -> None:
    init_db()
    conn = get_conn()
    for exp_name, ckpt_step, rel, result, notes in SEED:
        exp = conn.execute("SELECT id FROM experiments WHERE name=?", (exp_name,)).fetchone()
        if exp is None:
            print(f"[seed] 实验不存在，跳过: {exp_name}")
            continue
        ck = conn.execute(
            "SELECT id FROM checkpoints WHERE experiment_id=? AND (step=? OR step IS NULL)",
            (exp["id"], int(ckpt_step) if ckpt_step.isdigit() else -1),
        ).fetchone()
        if ck is None:
            print(f"[seed] 检查点不存在，跳过: {exp_name}@{ckpt_step}")
            continue
        dup = conn.execute("SELECT id FROM rollouts WHERE video_path=?", (rel,)).fetchone()
        if dup:
            print(f"[seed] 已存在，跳过: {rel}")
            continue
        cur = conn.execute(
            "INSERT INTO rollouts (checkpoint_id, video_path, result, notes) VALUES (?,?,?,?)",
            (ck["id"], rel, result, notes),
        )
        print(f"[seed] rollout #{cur.lastrowid} -> {exp_name}@{ckpt_step}: {rel}")
    conn.commit()
    n = conn.execute("SELECT COUNT(*) c FROM rollouts").fetchone()["c"]
    conn.close()
    print(f"[seed] 完成：rollouts={n}")


if __name__ == "__main__":
    main()
