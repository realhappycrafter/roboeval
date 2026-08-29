# -*- coding: utf-8 -*-
"""血缘播种：从 WSL 训练产物目录读取 train_config.json，登记真实实验与检查点。

幂等：每次运行先清空 experiments/checkpoints/annotations 关联表再写入。
用法：python lineage_seeder.py
"""
import json
from pathlib import Path

from db import get_conn, init_db

WSL_ROOT = Path(
    "//wsl.localhost/Ubuntu/home/happy_711890/.cache/huggingface/lerobot/seeedstudio123"
)
OUTPUTS = WSL_ROOT / "outputs" / "train"

# 实验登记清单：只登记磁盘上真实存在的产物
EXPERIMENTS = [
    {
        "name": "act_so101_test",
        "model": "act",
        "config": OUTPUTS / "act_so101_test" / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
        "ckpts": [
            ("last", OUTPUTS / "act_so101_test" / "checkpoints" / "last" / "pretrained_model"),
        ],
        "notes": "首次 ACT 训练（1000 步）。训练日志结尾出现 dataloader pin_memory 线程异常，训练提前中断。",
    },
    {
        "name": "act_so101_verify",
        "model": "act",
        "config": OUTPUTS / "act_so101_verify" / "checkpoints" / "000010" / "pretrained_model" / "train_config.json",
        "ckpts": [
            ("000010", OUTPUTS / "act_so101_verify" / "checkpoints" / "000010" / "pretrained_model"),
        ],
        "notes": "训练链路验证：10 步短训，确认 LeRobot 训练管线可跑通。",
    },
]


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[seeder] 配置解析失败 {path}: {e}")
        return {}


def extract(cfg: dict) -> dict:
    opt = cfg.get("optimizer", {}) or {}
    pol = cfg.get("policy", {}) or {}
    ds = cfg.get("dataset", {}) or {}
    return {
        "steps": cfg.get("steps"),
        "batch_size": cfg.get("batch_size"),
        "seed": cfg.get("seed"),
        "lr": opt.get("lr"),
        "optimizer": opt.get("type"),
        "weight_decay": opt.get("weight_decay"),
        "grad_clip_norm": opt.get("grad_clip_norm"),
        "policy_type": pol.get("type"),
        "chunk_size": pol.get("chunk_size"),
        "dim_model": pol.get("dim_model"),
        "dim_feedforward": pol.get("dim_feedforward"),
        "kl_weight": pol.get("kl_weight"),
        "dataset_repo": ds.get("repo_id"),
        "log_freq": cfg.get("log_freq"),
        "save_freq": cfg.get("save_freq"),
    }


def main() -> None:
    init_db()
    conn = get_conn()
    conn.execute("DELETE FROM annotations")
    conn.execute("DELETE FROM rollouts")
    conn.execute("DELETE FROM checkpoints")
    conn.execute("DELETE FROM experiments")

    for exp in EXPERIMENTS:
        cfg = load_config(exp["config"])
        hp = extract(cfg)
        if not hp["steps"]:
            print(f"[seeder] 跳过 {exp['name']}：配置不存在")
            continue
        cur = conn.execute(
            "INSERT INTO experiments (name, model, dataset_names, hyperparams, log_path, notes)"
            " VALUES (?,?,?,?,?,?)",
            (
                exp["name"],
                exp["model"],
                hp["dataset_repo"],
                json.dumps(hp, ensure_ascii=False),
                str(OUTPUTS.parent.parent / "train_act_so101_test.log") if "test" in exp["name"] else None,
                exp["notes"],
            ),
        )
        exp_id = cur.lastrowid
        n_ckpt = 0
        for step_label, ckpt_dir in exp["ckpts"]:
            if not ckpt_dir.exists():
                print(f"[seeder] 检查点不存在，跳过: {ckpt_dir}")
                continue
            step = int(step_label) if step_label.isdigit() else None
            conn.execute(
                "INSERT INTO checkpoints (experiment_id, step, path) VALUES (?,?,?)",
                (exp_id, step, str(ckpt_dir)),
            )
            n_ckpt += 1
        print(
            f"[seeder] {exp['name']}: model={hp['policy_type']}, steps={hp['steps']}, "
            f"lr={hp['lr']}, batch={hp['batch_size']}, dataset={hp['dataset_repo']}, 检查点 {n_ckpt} 个"
        )

    conn.commit()
    n = conn.execute("SELECT COUNT(*) c FROM experiments").fetchone()["c"]
    nck = conn.execute("SELECT COUNT(*) c FROM checkpoints").fetchone()["c"]
    conn.close()
    print(f"[seeder] 完成：experiments={n}, checkpoints={nck}")


if __name__ == "__main__":
    main()
