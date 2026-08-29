# RoboEval · 具身智能策略评测与数据闭环平台

> 面向 LeRobot 生态的**策略评测 + AI 失败归因 + 数据飞轮**平台
> 把机器人学习从"训完靠手感"变成"失败可归因、下一轮采集有依据"。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

---

## 为什么做这个

做具身智能的人都有同样的痛：

1. 训完 ACT / SmolVLA，**只知道 loss 降了，不知道成功率多少、为什么失败**。
2. 失败 rollout 的原因（抓空？放置偏移？目标超工作空间？）**靠人脑回放视频归类**。
3. 数据集版本、超参、checkpoint、评测结果**散落在文件夹里**，两个月后说不清"那个模型是用哪版数据训的"。
4. 业界共识：具身智能 80% 时间耗在数据环节，但**"补什么数据"全凭直觉**。

**现有工具止步于"采集 → 训练 → 看数据"，评测与归因是空白，而它正是数据闭环唯一缺的环。**

## 与现有工具的关系（互补，非替代）

| 工具 | 覆盖 | 未覆盖 |
|---|---|---|
| `huggingface/lerobot-dataset-visualizer` | 数据可视化、统计、规则级异常 episode 过滤 | 训练实验管理、策略评测、语义归因 |
| `lerobot-gui` 等 | 配置/标定/遥操作/采集/训练 | 评测环节（W&B 是通用工具，不懂 episode 语义） |
| **RoboEval** | **评测管理 + AI 失败归因 + 数据飞轮建议** | — |

> 官方 visualizer 看**数据**，RoboEval 看**策略**。

## 功能

| 模块 | 说明 |
|---|---|
| **数据集血缘** | 扫描 LeRobot 格式数据集（v2/v3），自动解析 `info.json` 与 episodes parquet，记录相机、帧数、任务 |
| **训练实验登记** | 从检查点目录的 `train_config.json` 自动提取超参（步数/学习率/batch/chunk_size…），串联数据→实验→检查点 |
| **评测 Rollout** | 导入评测视频（本地录像或云训练机导出），关联检查点，成功/失败一键标注，HTTP Range 流式播放 |
| **AI 归因引擎** ⭐ | ffmpeg 抽关键帧 → 多模态大模型（qwen-vl-max，可切换国产模型本地部署）→ 结构化归因：`阶段 × 原因 × 置信度` |
| **数据飞轮报告** | 成功率聚合 + 失败阶段×原因热力图 + AI 生成的下一轮采集建议 |
| **离线优先** | 本地部署、数据不出本机（机器人数据隐私） |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置多模态大模型（用于 AI 归因）
#    在 backend/.env 中填入：
#    DASHSCOPE_API_KEY=sk-xxx
#    DASHSCOPE_VL_MODEL=qwen-vl-max

# 3. 启动（Windows 可直接双击 start.bat）
cd backend && python -m uvicorn main:app --port 8000

# 4. 浏览器打开 http://127.0.0.1:8000
```

### 接入你自己的数据

把 LeRobot 格式的数据集目录放到 `assets/datasets/` 下，然后修改 `backend/scanner.py` 的 `SCAN_ROOTS`，或调用：

```bash
curl -X POST http://127.0.0.1:8000/api/scan
```

把训练产物（`train_config.json` 所在目录）登记进血缘：编辑 `backend/lineage_seeder.py` 的 `EXPERIMENTS`，然后：

```bash
python backend/lineage_seeder.py
```

把评测视频放进 `assets/rollouts/`，编辑 `backend/seed_rollouts.py` 登记，然后在网页点击 **🤖 运行 AI 归因**。

## 项目结构

```
roboeval/
├── backend/
│   ├── main.py            # FastAPI 入口与全部 API
│   ├── db.py              # SQLite 血缘表结构
│   ├── scanner.py         # LeRobot 数据集扫描器
│   ├── lineage_seeder.py  # 训练实验/检查点播种
│   ├── seed_rollouts.py   # 评测视频登记
│   ├── attribution.py     # AI 归因引擎（抽帧 + 多模态推理）
│   └── .env               # API Key（不会被提交）
├── frontend/index.html    # 单页仪表盘（零构建依赖）
├── assets/                # 数据集 / 评测视频 / 抽帧（不入仓库）
└── start.bat              # 一键启动
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/stats` | 全局统计 |
| POST | `/api/scan` | 重新扫描数据集 |
| GET | `/api/datasets` | 数据集列表 |
| GET | `/api/datasets/{id}/episodes` | 某数据集的 episodes |
| GET | `/api/experiments` | 训练实验 + 超参 + 检查点数 |
| GET/POST | `/api/rollouts` | 评测登记与列表 |
| POST | `/api/rollouts/{id}/result` | 标注成功/失败/待定 |
| POST | `/api/rollouts/{id}/attribute` | **运行 AI 归因** |
| GET | `/api/rollouts/{id}/annotations` | 归因记录 |
| GET | `/api/media/{path}` | 视频流（支持 Range） |
| GET | `/api/report` | 数据飞轮报告 |

## 归因输出示例

```json
{
  "result_guess": "failure",
  "summary": "机械臂尝试抓取红色积木，但未成功抓取，积木始终在桌面上，未放入收纳盒。",
  "stages": [
    {"stage": "抓取",     "cause": "夹爪未闭合或位置偏差导致未能抓取积木", "confidence": 0.9},
    {"stage": "搬运",     "cause": "无有效抓取，无法进行搬运动作",         "confidence": 0.8},
    {"stage": "下放置入", "cause": "未执行放置动作，因未抓取物体",         "confidence": 0.7}
  ],
  "advice": "增加抓取阶段的视觉反馈校准，优化夹爪开合时机与位置精度"
}
```

## 路线图

- [x] 数据集血缘与 episodes 解析
- [x] 训练实验与检查点自动登记
- [x] 评测 Rollout 管理与视频流播放
- [x] AI 失败归因（qwen-vl-max）
- [x] 数据飞轮报告（成功率 / 热力图 / 采集建议）
- [ ] AI 数据质检（episode 级异常检测）
- [ ] 真机直连评测执行器
- [ ] loss 曲线面板

## 许可

Apache-2.0。See [LICENSE](./LICENSE).
