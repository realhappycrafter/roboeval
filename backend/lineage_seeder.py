# -*- coding: utf-8 -*-
"""血缘播种：自动发现 WSL 中的 LeRobot 训练产物并登记进数据库。

扫描规则：在每个 OUTPUT_ROOTS 下的 <job>/checkpoints/**/pretrained_model/train_config.json
视为一个检查点，同一 job 目录归为一个训练实验。

幂等：每次运行先清空 experiments/checkpoints 及下游关联表再写入。
用法：python lineage_seeder.py
"""
import json
import re
from pathlib import Path

from db import get_conn, init_db

WSL_HOME = Path("//wsl.localhost/Ubuntu/home/happy_711890")

OUTPUT_ROOTS = [
    ("wsl-home", WSL_HOME / "outputs" / "train"),
    ("lerobot-repo", WSL_HOME / "lerobot" / "outputs" / "train"),
    ("hf-cache", WSL_HOME / ".cache" / "huggingface" / "lerobot" / "seeedstudio123" / "outputs" / "train"),
]

NOTES = {
    "act_so101_best": "完整训练：10 万步 ACT，每 1 万步保存检查点，数据集 seeedstudio123/test_merged。",
    "act_so101_test": "训练链路测试（短步数），用于验证训练/检查点保存流程。",
    "act_so101_verify": "训练管线验证（短步数），确认 LeRobot 训练可跑通。",
}


def load_config(path: Path) -> dict:
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


def _step_from_dir(path: Path) -> int | None:
    """从检查点目录名（如 010000、last）推断步数。"""
    for part in reversed(path.parts):
        if re.fullmatch(r"\d{5,}", part):
            return int(part)
    return None


def discover(roots: list[tuple[str, Path]] | None = None) -> list[dict]:
    """返回 [{name, source, job_dir, hyperparams, ckpts:[(step, path)]}]

    roots: [(来源标签, 目录)]，默认用内置 OUTPUT_ROOTS。
    每个 root 可以是 train 根目录（含多个 job 子目录），或单个 job 目录（自身含 checkpoints）。
    """
    jobs: dict[tuple[str, str], dict] = {}
    for source, root in (roots if roots is not None else OUTPUT_ROOTS):
        if not root.exists():
            print(f"[seeder] 根目录不存在，跳过: {root}")
            continue
        job_dirs = [root] if (root / "checkpoints").is_dir() else [
            p for p in sorted(root.iterdir()) if p.is_dir()
        ]
        for job_dir in job_dirs:
            configs = sorted(job_dir.glob("checkpoints/**/pretrained_model/train_config.json"))
            if not configs:
                continue
            key = (source, job_dir.name)
            jobs.setdefault(key, {"name": job_dir.name, "source": source,
                                  "job_dir": job_dir, "configs": []})
            jobs[key]["configs"].extend(configs)
    out = []
    for (source, job_name), data in jobs.items():
        # 用 steps 最大的配置作为实验超参代表（通常是最终配置）
        best_cfg, best_steps = {}, -1
        ckpts = []
        for cfg_path in data["configs"]:
            cfg = load_config(cfg_path)
            if not cfg:
                continue
            hp = extract(cfg)
            if (hp.get("steps") or 0) > best_steps:
                best_cfg, best_steps = hp, hp.get("steps") or 0
            step = _step_from_dir(cfg_path.parent.parent) or hp.get("steps")
            ckpts.append((step, str(cfg_path.parent)))
        out.append({
            "name": f"{job_name} [{source}]",
            "source": source,
            "job_dir": data["job_dir"],
            "hyperparams": best_cfg,
            "ckpts": sorted(ckpts, key=lambda x: (x[0] is None, x[0])),
        })
    return out


def register(experiments: list[dict], replace_existing: bool = True) -> tuple[int, int]:
    """把实验列表写入数据库，返回 (实验数, 检查点数)。

    replace_existing=True 时覆盖同名同来源的实验（用于刷新）。
    """
    init_db()
    conn = get_conn()
    n_exp = n_ck = 0
    for exp in experiments:
        hp = exp["hyperparams"]
        if replace_existing:
            old = conn.execute("SELECT id FROM experiments WHERE name=?", (exp["name"],)).fetchone()
            if old:
                conn.execute("DELETE FROM checkpoints WHERE experiment_id=?", (old["id"],))
                conn.execute("DELETE FROM experiments WHERE id=?", (old["id"],))
        cur = conn.execute(
            "INSERT INTO experiments (name, model, dataset_names, hyperparams, log_path, notes)"
            " VALUES (?,?,?,?,?,?)",
            (
                exp["name"],
                hp.get("policy_type") or "act",
                hp.get("dataset_repo"),
                json.dumps(hp, ensure_ascii=False),
                str(exp["job_dir"]),
                exp.get("notes") or NOTES.get(exp["name"].split(" [")[0], ""),
            ),
        )
        exp_id = cur.lastrowid
        for step, ckpt_path in exp["ckpts"]:
            conn.execute(
                "INSERT INTO checkpoints (experiment_id, step, path) VALUES (?,?,?)",
                (exp_id, step, ckpt_path),
            )
            n_ck += 1
        n_exp += 1
        print(
            f"[seeder] {exp['name']}: model={hp.get('policy_type')}, steps={hp.get('steps')}, "
            f"lr={hp.get('lr')}, batch={hp.get('batch_size')}, dataset={hp.get('dataset_repo')}, "
            f"检查点 {len(exp['ckpts'])} 个"
        )
    conn.commit()
    total_exp = conn.execute("SELECT COUNT(*) c FROM experiments").fetchone()["c"]
    total_ck = conn.execute("SELECT COUNT(*) c FROM checkpoints").fetchone()["c"]
    conn.close()
    return n_exp, n_ck


def main() -> None:
    # 全量重建：先清空下游关联表
    conn = get_conn()
    conn.execute("DELETE FROM annotations")
    conn.execute("DELETE FROM rollouts")
    conn.execute("DELETE FROM checkpoints")
    conn.execute("DELETE FROM experiments")
    conn.commit()
    conn.close()

    n_exp, n_ck = register(discover())
    print(f"[seeder] 完成：experiments={n_exp}, checkpoints={n_ck}")


if __name__ == "__main__":
    main()
