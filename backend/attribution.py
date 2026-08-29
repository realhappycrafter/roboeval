# -*- coding: utf-8 -*-
"""AI 归因引擎：视频抽帧 → qwen-vl-max 多模态推理 → 结构化归因 JSON。

流程：
  1. ffmpeg 均匀抽帧（默认每 3 秒一帧，最多 8 帧，缩放到 512px 宽）
  2. 帧图 base64 后与任务上下文一起发给 qwen-vl-max
  3. 解析模型输出的 JSON，得到 result_guess + stages[(stage, cause, confidence)]
"""
import base64
import json
import os
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
API_KEY = os.environ.get("DASHSCOPE_API_KEY")
MODEL = os.environ.get("DASHSCOPE_VL_MODEL", "qwen-vl-max")

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

FRAME_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "frames"

PROMPT_TEMPLATE = """你是一位具身智能（机器人学习）评测专家。下面是同一个机械臂评测 rollout 抽取的连续关键帧（按时间顺序）。

背景信息：
- 硬件：SO-101 机械臂（6 自由度桌面级），front/side 双相机
- 任务：{task}
- 训练策略：ACT（Action Chunking Transformer）

请仔细逐帧分析机械臂的行为过程，输出严格的 JSON（不要输出任何其他文字、不要用 markdown 代码块）：
{{
  "result_guess": "success 或 failure 或 uncertain",
  "summary": "整个过程的一段话中文描述（100字以内）",
  "stages": [
    {{
      "stage": "问题所属阶段，从这些值中选：接近目标/抓取/搬运/下放置入/全程",
      "cause": "该阶段的具体问题或表现，中文（40字以内）",
      "confidence": 0.0到1.0之间的小数
    }}
  ],
  "advice": "给下一轮数据采集或训练的建议，中文（60字以内）"
}}

注意：
- 只有当你在帧中明确看到任务失败证据（如未抓到物体、物体掉落、最终未放入容器）才判 failure
- 若帧序列不足以判断最终结果，用 uncertain
- stages 数组最多 4 项，只列有实际依据的，没有问题就列表现正常的阶段"""


def extract_frames(video_path: Path, max_frames: int = 8, interval: float = 3.0) -> list[Path]:
    """每 interval 秒抽一帧，返回帧图路径列表。"""
    out_dir = FRAME_DIR / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    pattern = str(out_dir / "frame_%03d.jpg")
    vf = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
        f"scale=512:-2"
    )
    cmd = [
        FFMPEG, "-y", "-i", str(video_path),
        "-vf", vf, "-frames:v", str(max_frames), "-vsync", "vfr", "-q:v", "5", pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"抽帧失败: {proc.stderr[-400:]}")
    return frames


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def attribute(video_path: Path, task: str = "将桌上的积木块抓起并放入透明收纳盒中") -> dict:
    frames = extract_frames(video_path)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    content = [{"type": "text", "text": PROMPT_TEMPLATE.format(task=task)}]
    for f in frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(f)}"},
        })
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}], temperature=0.2
    )
    text = resp.choices[0].message.content.strip()
    # 容错：剥掉可能的 ```json 包裹
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"模型未返回 JSON: {text[:200]}")
    data = json.loads(text[start:end + 1])
    data["_frames"] = [str(f) for f in frames]
    data["_model"] = MODEL
    return data


if __name__ == "__main__":
    import sys

    v = Path(sys.argv[1])
    result = attribute(v)
    print(json.dumps(result, ensure_ascii=False, indent=2))
