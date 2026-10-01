"""决策服务：纯算法 baseline（不调用大模型，Token 消耗为 0）。

保留类名 ``ModelDecisionService`` 与构造签名 ``(__init__(self, api))``、入口 ``decide(self, driver_id)``，
以便评测编排（``bench.embedded_agent``）按原方式注入与调用。

决策逻辑见 `agent.baseline_policy.BaselinePolicy`：
读取司机状态 → 在 接单 / 休息 / 空驶 三类动作中按"净收益/时间 + 偏好规避 + 月末保护"择一。
"""

from __future__ import annotations

import logging
from typing import Any

from agent.baseline_policy import BaselinePolicy
from simkit.ports import SimulationApiPort


class ModelDecisionService:
    """单步决策：感知司机状态与候选货源，返回结构化动作。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._policy = BaselinePolicy()
        self._logger = logging.getLogger("agent.decision_service")

    def decide(self, driver_id: str) -> dict[str, Any]:
        try:
            status = self._api.get_driver_status(driver_id)
            action = self._policy.decide(self._api, driver_id, status)
            if not isinstance(action, dict) or "action" not in action:
                raise ValueError("policy 返回非法动作")
            self._logger.info(
                "decision driver_id=%s time_min=%s action=%s params=%s",
                driver_id,
                status.get("simulation_progress_minutes"),
                action.get("action"),
                action.get("params"),
            )
            return action
        except Exception:  # noqa: BLE001 — 决策异常不得中断该司机仿真，兜底为安全等待
            self._logger.exception("decision failed driver_id=%s，回退 wait", driver_id)
            return {"action": "wait", "params": {"duration_minutes": 30}}
