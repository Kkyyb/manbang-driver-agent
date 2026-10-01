# 满帮货运司机决策 Agent

用于货运仿真评测的 Python 决策模块。大模型将司机的自然语言偏好转换成结构化约束，算法根据订单收益、时间、位置和偏好要求，在接单、等待、空驶三个动作中做选择。

本仓库发布的是 `提交/demo/agent` 的源码快照，不包含仿真服务、货源数据、司机数据、模型密钥或原项目的运行记录。

## 比赛背景与成绩

本项目参加 **满帮集团 Agent 算法大赛：基于 Agentic AI 的卡车司机连续找货决策**，于 **2026 年 5 月**取得 **95/963** 的排名。

比赛要求 Agent 在一个自然月的仿真过程中，连续为卡车司机选择接单、等待或空驶动作。比赛评分为：

```text
比赛评分 = 月度净收益 − 偏好罚分
```

每位司机都有独立且动态变化的自然语言偏好，包括禁运类目、夜间禁动和临时家事剧本等。核心挑战是在满足司机偏好的前提下最大化月度净收益，同时考虑当前订单对后续位置、可用时间和履约能力的影响。

项目采用“大模型解析偏好＋算法决策”的方案，将自然语言要求转换成订单过滤、收益评分和行程调度约束，持续权衡接单收益与偏好违约代价。

## 目录

```text
agent/
├── __init__.py
├── model_decision_service.py       # 对外决策入口
├── baseline_policy.py             # 订单评分、约束执行、休息及行程安排
├── general_preference_parser.py   # 通用偏好解析及约束编译
├── preference_taxonomy_schema.json # 偏好分类与字段定义
└── eval_taxonomy_parse.py          # 解析评估及运行时复用的提示词工具
```

## 工作流程

1. 从仿真接口读取司机位置、时间和当前可见偏好。
2. 偏好变化时调用模型解析，并转换成策略可执行的约束。
3. 优先处理预约行程、回家、禁动时段和整天休息等安排。
4. 查询候选货源，并在查询后重新读取仿真时间。
5. 检查订单可行性，按预估收益和偏好罚金评分。
6. 返回接单、等待或空驶动作，由仿真引擎执行。

约束执行分为三种角色：`GATE` 排除不符合条件的候选，`SCORE` 将偏好代价计入评分，`SCHEDULE` 安排必须优先执行的动作。

基础订单评分：

```text
预估净收益 = 运费 - 每公里成本 ×（赴装空驶公里数 + 装卸货地点间公里数）
订单评分 =（预估净收益 - 预估偏好罚金）/ 占用分钟数
```

## 接入方式

建议使用 Python 3.10 或以上版本。源码主要使用 Python 标准库；运行入口还需要评测环境提供 `simkit.ports.SimulationApiPort`。

将仓库根目录加入 Python 模块搜索路径，并确保评测环境中的 `simkit` 可导入：

```python
from agent.model_decision_service import ModelDecisionService

# api 由仿真评测环境创建并注入。
service = ModelDecisionService(api)
action = service.decide(driver_id)
# 仿真环境接收 action 后执行相应动作。
```

注入接口应提供以下方法：

- `get_driver_status(driver_id)`：读取司机状态。
- `query_cargo(driver_id, latitude, longitude)`：查询候选货源。
- `model_chat_completion(payload)`：调用模型；返回兼容 Chat Completions 的响应结构。
- `query_decision_history(driver_id, step)`：评测协议定义的历史查询方法。

返回动作示例：

```json
{"action": "take_order", "params": {"cargo_id": "123"}}
```

其他动作包括 `wait`（参数 `duration_minutes`）和 `reposition`（参数 `latitude`、`longitude`）。

## 常用配置

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MANBANG_SIM_DAYS` | `31` | 仿真总天数 |
| `MANBANG_COST_PER_KM` | `3.0` | 每公里预估成本，单位元/km |
| `MANBANG_REPO_SPEED` | `60.0` | 行驶速度估计，单位 km/h |
| `MANBANG_USE_GENERAL_PREF_PARSE` | `1` | 优先使用通用偏好解析器 |
| `MANBANG_USE_LLM_PARSE` | `1` | 启用旧版模型解析路径 |
| `MANBANG_USE_PLANNER` | `1` | 启用接单与休息权衡 |
| `MANBANG_USE_PLANNER_V2` | `0` | 实验性 V2 规划器；本快照缺少对应实现文件 |
| `MANBANG_USE_DENSITY` | `0` | 启用在线货源密度图 |
| `MANBANG_TAXONOMY_CACHE` | `1` | 缓存通用解析结果 |
| `MANBANG_TAXONOMY_CACHE_DIR` | 根据源码路径计算 | 通用解析缓存目录，可显式指定可写位置 |

如果需要使用纯规则解析，应同时将 `MANBANG_USE_GENERAL_PREF_PARSE` 和 `MANBANG_USE_LLM_PARSE` 设置为 `0`。默认配置会调用模型，源码中部分“零 Token”或“默认关闭”的旧注释与当前默认值不一致，以代码为准。

## 当前限制

- 本仓库是 Agent 源码模块，不能独立运行完整货运仿真。
- 日期换算以 `2026-03-01 00:00:00` 为基准。
- 规则分类已定义接单节奏和区域内行驶等偏好，但部分分类尚未编译成可执行约束。
- 通用解析器收到无效模型 JSON 时会返回空约束，当前调用方未针对这种返回值触发解析兜底。
- 部分运行统计在发出动作时提前更新，尚未根据动作执行结果统一校正。
- `planner_core.py`、`planner_beam.py` 不在此快照中，请保持 V2 规划器关闭。
- 独立解析评估需要额外的标准答案、数据集和模型网关；脚本中的默认路径沿用原项目布局。

## 验证范围

发布前完成 Python 源码语法检查、JSON 格式检查，以及模拟接口下的有货接单和无货等待检查。未运行真实模型调用或完整月度收益评测。
