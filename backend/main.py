# -*- coding: utf-8 -*-
"""RoboEval 后端入口：FastAPI + SQLite。

启动（项目根 AIC2026/roboeval/ 下）：
    ../venv python -m uvicorn backend.main:app --port 8000
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db import get_conn, init_db
from scanner import run_scan

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

load_dotenv(Path(__file__).resolve().parent / ".env")  # DASHSCOPE_API_KEY 等

app = FastAPI(title="RoboEval", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def startup() -> None:
    init_db()
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) c FROM datasets").fetchone()["c"]
    conn.close()
    if n == 0:
        run_scan(reset=True)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "RoboEval"}


@app.post("/api/scan")
def scan():
    return run_scan(reset=True)


@app.get("/api/datasets")
def list_datasets():
    import json as _json

    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM datasets ORDER BY name")]
    conn.close()
    for r in rows:
        try:
            r["cameras"] = _json.loads(r["cameras"]) if r["cameras"] else []
        except Exception:  # noqa: BLE001
            r["cameras"] = []
    return rows


@app.get("/api/datasets/{dataset_id}/episodes")
def list_episodes(dataset_id: int):
    conn = get_conn()
    ds = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if ds is None:
        raise HTTPException(404, "dataset not found")
    eps = [dict(r) for r in conn.execute(
        "SELECT id, episode_index, length, task FROM episodes WHERE dataset_id=? ORDER BY episode_index",
        (dataset_id,),
    )]
    conn.close()
    return {"dataset": dict(ds), "episodes": eps}


@app.get("/api/experiments")
def list_experiments():
    import json as _json

    conn = get_conn()
    rows = []
    for r in conn.execute(
        """SELECT e.id, e.name, e.model, e.dataset_names, e.hyperparams, e.notes, e.created_at,
                  COUNT(c.id) AS ckpt_count
           FROM experiments e
           LEFT JOIN checkpoints c ON c.experiment_id = e.id
           GROUP BY e.id ORDER BY e.id"""
    ):
        d = dict(r)
        try:
            d["hyperparams"] = _json.loads(d["hyperparams"]) if d["hyperparams"] else {}
        except Exception:  # noqa: BLE001
            d["hyperparams"] = {}
        rows.append(d)
    conn.close()
    return rows


@app.get("/api/rollouts")
def list_rollouts():
    conn = get_conn()
    rows = []
    for r in conn.execute(
        """SELECT ro.id, ro.result, ro.notes, ro.video_path, ro.created_at,
                  COALESCE(ck.step, 'last') AS ckpt_step, ck.path AS ckpt_path,
                  e.name AS experiment_name, e.model,
                  (SELECT COUNT(*) FROM annotations a WHERE a.rollout_id = ro.id) AS annotation_count
           FROM rollouts ro
           LEFT JOIN checkpoints ck ON ck.id = ro.checkpoint_id
           LEFT JOIN experiments e ON e.id = ck.experiment_id
           ORDER BY ro.id DESC"""
    ):
        rows.append(dict(r))
    conn.close()
    return rows


@app.post("/api/rollouts")
def create_rollout(payload: dict):
    checkpoint_id = payload.get("checkpoint_id")
    video_path = payload.get("video_path")
    if not checkpoint_id or not video_path:
        raise HTTPException(422, "checkpoint_id 与 video_path 必填")
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO rollouts (checkpoint_id, video_path, result, notes) VALUES (?,?,?,?)",
        (checkpoint_id, video_path, payload.get("result", "unknown"), payload.get("notes")),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return {"id": rid}


@app.post("/api/rollouts/{rollout_id}/result")
def set_result(rollout_id: int, payload: dict):
    result = payload.get("result")
    if result not in ("success", "failure", "unknown"):
        raise HTTPException(422, "result 必须是 success/failure/unknown")
    conn = get_conn()
    cur = conn.execute("UPDATE rollouts SET result=? WHERE id=?", (result, rollout_id))
    conn.commit()
    n = cur.rowcount
    conn.close()
    if n == 0:
        raise HTTPException(404, "rollout not found")
    return {"id": rollout_id, "result": result}


@app.get("/api/rollouts/{rollout_id}/annotations")
def list_annotations(rollout_id: int):
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM annotations WHERE rollout_id=? ORDER BY id DESC", (rollout_id,)
    )]
    conn.close()
    return rows


@app.post("/api/rollouts/{rollout_id}/annotations")
def add_annotation(rollout_id: int, payload: dict):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO annotations (rollout_id, stage, cause, confidence, source) VALUES (?,?,?,?,?)",
        (
            rollout_id,
            payload.get("stage"),
            payload.get("cause"),
            payload.get("confidence"),
            payload.get("source", "human"),
        ),
    )
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return {"id": aid}


@app.get("/api/media/{rel_path:path}")
def media(rel_path: str, request: Request):
    """视频流：支持 HTTP Range，保证浏览器进度条拖动可用。"""
    import os
    import re

    from starlette.responses import StreamingResponse

    file_path = (BASE_DIR.parent / "assets" / rel_path).resolve()
    assets_root = (BASE_DIR.parent / "assets").resolve()
    if not str(file_path).startswith(str(assets_root)) or not file_path.exists():
        raise HTTPException(404, "media not found")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    chunk = 1024 * 1024

    def stream(start: int, end: int):
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    if range_header:
        m = re.search(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else min(start + chunk - 1, file_size - 1)
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        }
        return StreamingResponse(stream(start, end), status_code=206, headers=headers,
                                 media_type="video/mp4")
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    return StreamingResponse(stream(0, file_size - 1), headers=headers, media_type="video/mp4")


@app.post("/api/rollouts/{rollout_id}/attribute")
def run_attribution(rollout_id: int, payload: dict | None = None):
    """AI 归因：抽帧 -> qwen-vl-max -> 结构化归因写库。重跑会覆盖旧 AI 归因。"""
    import attribution as attr

    conn = get_conn()
    ro = conn.execute("SELECT * FROM rollouts WHERE id=?", (rollout_id,)).fetchone()
    if ro is None:
        conn.close()
        raise HTTPException(404, "rollout not found")
    video_rel = ro["video_path"]
    conn.close()

    video_path = (BASE_DIR.parent / "assets" / video_rel).resolve()
    if not video_path.exists():
        raise HTTPException(404, f"video not found: {video_rel}")

    task = (payload or {}).get("task", "将桌上的积木块抓起并放入透明收纳盒中")
    try:
        data = attr.attribute(video_path, task)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"归因失败: {e}")

    conn = get_conn()
    conn.execute("DELETE FROM annotations WHERE rollout_id=? AND source='ai'", (rollout_id,))
    for st in data.get("stages", []):
        conn.execute(
            "INSERT INTO annotations (rollout_id, stage, cause, confidence, source)"
            " VALUES (?,?,?,?, 'ai')",
            (rollout_id, st.get("stage"), st.get("cause"), st.get("confidence")),
        )
    if data.get("summary"):
        conn.execute(
            "INSERT INTO annotations (rollout_id, stage, cause, confidence, source)"
            " VALUES (?,?,?,?, 'ai')",
            (rollout_id, "总结", data["summary"], None),
        )
    if data.get("advice"):
        conn.execute(
            "INSERT INTO annotations (rollout_id, stage, cause, confidence, source)"
            " VALUES (?,?,?,?, 'ai')",
            (rollout_id, "建议", data["advice"], None),
        )
    # 结果仍是 unknown 时，用 AI 判断自动补标（可追溯：notes 记录）
    result_updated = False
    if ro["result"] == "unknown" and data.get("result_guess") in ("success", "failure"):
        conn.execute(
            "UPDATE rollouts SET result=?, notes=COALESCE(notes,'')||' | AI预标注' WHERE id=?",
            (data["result_guess"], rollout_id),
        )
        result_updated = True
    conn.commit()
    conn.close()
    return {
        "rollout_id": rollout_id,
        "result_guess": data.get("result_guess"),
        "summary": data.get("summary"),
        "advice": data.get("advice"),
        "stages": data.get("stages", []),
        "model": data.get("_model"),
        "result_updated": result_updated,
    }


@app.get("/api/report")
def report():
    """数据飞轮报告：成功率聚合 + 失败阶段×原因热力 + AI 采集建议。"""
    conn = get_conn()
    # 各实验的 rollout 结果统计
    per_exp = []
    for r in conn.execute(
        """SELECT e.id, e.name, e.model,
                  COALESCE(SUM(ro.result='success'),0) AS success,
                  COALESCE(SUM(ro.result='failure'),0) AS failure,
                  COALESCE(SUM(ro.result='unknown'),0) AS unknown,
                  COUNT(ro.id) AS total
           FROM experiments e
           LEFT JOIN checkpoints ck ON ck.experiment_id = e.id
           LEFT JOIN rollouts ro ON ro.checkpoint_id = ck.id
           GROUP BY e.id ORDER BY e.id"""
    ):
        d = dict(r)
        judged = d["success"] + d["failure"]
        d["success_rate"] = round(d["success"] / judged, 3) if judged else None
        per_exp.append(d)

    # 失败归因聚合：阶段 -> [(原因, 频次, 平均置信度)]
    stage_rows = conn.execute(
        """SELECT a.stage, a.cause, COUNT(*) AS cnt, AVG(a.confidence) AS avg_conf
           FROM annotations a
           JOIN rollouts ro ON ro.id = a.rollout_id AND ro.result = 'failure'
           WHERE a.source = 'ai' AND a.stage NOT IN ('建议','总结') AND a.cause IS NOT NULL
           GROUP BY a.stage, a.cause ORDER BY cnt DESC"""
    ).fetchall()
    stages: dict[str, list] = {}
    for r in stage_rows:
        stages.setdefault(r["stage"], []).append(
            {"cause": r["cause"], "count": r["cnt"], "avg_conf": round(r["avg_conf"], 3) if r["avg_conf"] is not None else None}
        )

    # AI 采集建议
    advice = [dict(r) for r in conn.execute(
        """SELECT a.cause, a.rollout_id, a.created_at FROM annotations a
           WHERE a.stage='建议' AND a.cause IS NOT NULL
           ORDER BY a.id DESC LIMIT 20"""
    )]
    conn.close()
    return {"per_experiment": per_exp, "failure_stages": stages, "advice": advice}


@app.get("/api/stats")
def stats():
    conn = get_conn()
    q = lambda sql: conn.execute(sql).fetchone()["c"]  # noqa: E731
    out = {
        "datasets": q("SELECT COUNT(*) c FROM datasets"),
        "episodes": q("SELECT COUNT(*) c FROM episodes"),
        "experiments": q("SELECT COUNT(*) c FROM experiments"),
        "checkpoints": q("SELECT COUNT(*) c FROM checkpoints"),
        "rollouts": q("SELECT COUNT(*) c FROM rollouts"),
        "annotations": q("SELECT COUNT(*) c FROM annotations"),
        "total_frames": conn.execute("SELECT COALESCE(SUM(total_frames),0) s FROM datasets").fetchone()["s"],
    }
    conn.close()
    return out


# 前端静态页（占位 dashboard）
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
