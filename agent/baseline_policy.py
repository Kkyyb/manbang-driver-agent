"""纯算法 baseline 决策核心（不依赖大模型，零 token）。

设计要点（与 `simkit` 口径对齐）：
- 仅通过注入的环境接口（``get_driver_status`` / ``query_cargo``）感知，不读原始数据文件。
- 动作物理由引擎计算，本模块只负责"选哪一个动作"，并用与引擎一致的公式做可行性与收益预估。
- 速度 60km/h 下「分钟 ≈ 公里」：``reposition``/赴装耗时 = ceil(距离km/速度*60)。
- 收益：``net = price - cost_per_km*(赴装空驶km + 干线直线km)``；其中 ``cost_per_km`` 对 Agent 不可见，
  用可调估计 ``MANBANG_COST_PER_KM`` 代替，仅用于候选排序（同一司机内为常数，不影响相对排序的方向）。
- horizon：评测总时长 = ``simulation_duration_days*1440`` 分钟，但 Agent 取不到，用 ``MANBANG_SIM_DAYS`` 注入。

第 2 步「偏好规避」覆盖高 ROI、低风险的通用项（不按 driver_id 硬编码，从偏好文本解析）：
类目禁运、夜间禁动窗、每日连续休息、整月歇业天数、赴装/干线距离上限、月度空驶上限、禁入圆区、城市范围。
更复杂的"回家点/到访点/家事"等留待后续（baseline 会承担这部分罚分）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_FMT = "%Y-%m-%d %H:%M:%S"

# LLM 解析结果磁盘缓存目录（仅 A/B 用，门控 MANBANG_LLM_CACHE=1 启用）：
# 把每司机的 LLM 基础约束解析按「偏好文本 + system 提示」哈希落盘，使 A/B 多组
# 复用同一份解析，消除 LLM 非确定性对基线的扰动。默认关 → 行为不变。
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LLM_CACHE_DIR = os.environ.get("MANBANG_LLM_CACHE_DIR") or os.path.join(_PROJ_ROOT, ".ab_tmp", "llm_cache")

# 密度选点：空驶成本摊销的参考接单时长（分钟）。把一次 reposition 的总空驶成本
# 摊到这么长的一次接单上，得到与 rate 同量级（元/分钟）的惩罚项。约一次中等单的 busy。
_REPO_AMORTIZE_MIN = 600.0


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else default
    except ValueError:
        return default


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """与引擎一致的大圆距离（km）。"""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * r * math.asin(math.sqrt(h))


def _drive_minutes(distance_km: float, speed_kmh: float) -> int:
    """赴装/空驶耗时；与引擎 ``distance_to_minutes`` 一致（零距离按 0 处理用于赴装预估）。"""
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil((distance_km / speed_kmh) * 60.0))


def _wall_to_minutes(text: str) -> int | None:
    try:
        dt = datetime.strptime(str(text).strip(), _WALL_FMT)
    except (ValueError, TypeError):
        return None
    return int((dt - _EPOCH).total_seconds() // 60)


def _parse_load_window(cargo: dict[str, Any]) -> tuple[int, int] | None:
    raw = cargo.get("load_time")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    a = _wall_to_minutes(raw[0])
    b = _wall_to_minutes(raw[1])
    if a is None or b is None or b < a:
        return None
    return (a, b)


# --------------------------------------------------------------------------- #
# 偏好 → 结构化约束
# --------------------------------------------------------------------------- #


@dataclass
class Constraints:
    night_windows: list[tuple[int, int]] = field(default_factory=list)  # 每日 [start,end] 分钟（end 可>1440 表跨夜）
    daily_rest_min: int = 0           # 每日需连续休息（分钟）
    days_off: int = 0                 # 整月需完全歇业的天数
    max_pickup_km: float | None = None
    max_haul_km: float | None = None
    max_month_deadhead_km: float | None = None
    banned_categories: set[str] = field(default_factory=set)
    forbidden_zones: list[tuple[float, float, float]] = field(default_factory=list)  # (lat,lng,radius_km)
    region_bbox: tuple[float, float, float, float] | None = None


def _pref_texts(preferences: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(preferences, list):
        return out
    for p in preferences:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            t = p.get("content") or p.get("text")
            if t:
                out.append(str(t))
    return out


def _parse_time_ranges(text: str) -> list[tuple[int, int]]:
    """从"不接单/不空驶/不出车"类句子里抽时间窗，返回每日分钟区间（跨夜则 end>1440）。"""
    if not any(k in text for k in ("不接单", "不空驶", "不出车", "不跑", "不接", "禁动")):
        return []
    ranges: list[tuple[int, int]] = []
    # 匹配 "23:00-04:00" / "23-6" / "2点到5点" / "23点至次日4点"
    pat = re.compile(r"(\d{1,2})\s*(?:[:：]\s*(\d{1,2}))?\s*(?:点|时)?\s*(?:-|—|~|到|至)\s*(?:次日)?\s*(\d{1,2})\s*(?:[:：]\s*(\d{1,2}))?\s*(?:点|时)?")
    for m in pat.finditer(text):
        h1 = int(m.group(1)); m1 = int(m.group(2) or 0)
        h2 = int(m.group(3)); m2 = int(m.group(4) or 0)
        if not (0 <= h1 <= 24 and 0 <= h2 <= 24):
            continue
        start = h1 * 60 + m1
        end = h2 * 60 + m2
        if "次日" in text or end <= start:
            end += 1440
        if end > start:
            ranges.append((start, end))
    return ranges


def _parse_constraints(preferences: Any) -> Constraints:
    c = Constraints()
    for text in _pref_texts(preferences):
        # 夜间/时段禁动
        c.night_windows.extend(_parse_time_ranges(text))

        # 每日连续休息（取最大要求）
        for m in re.finditer(r"(?:连续|休息|歇脚|停车|熄火)[^0-9]{0,8}(\d+)\s*小时", text):
            if any(k in text for k in ("休息", "歇", "停车", "熄火")):
                c.daily_rest_min = max(c.daily_rest_min, int(m.group(1)) * 60)

        # 整月歇业天数
        if any(k in text for k in ("不出车", "不接单", "歇", "放空", "完全")):
            mm = re.search(r"(?:至少|每月)?\s*(\d+)\s*(?:个)?(?:整|自然)?天", text)
            if mm and any(k in text for k in ("不出车", "歇", "放空", "完全不", "不接单")):
                c.days_off = max(c.days_off, int(mm.group(1)))

        # 距离上限
        for m in re.finditer(r"(空驶|赴装|去装|装货点|单笔|装卸|干线|运距|拉)[^0-9]{0,8}(?:≤|<=|不超过|不大于|小于|不高于)?\s*(\d+)\s*(?:km|公里|千米)", text):
            kind, km = m.group(1), float(m.group(2))
            if kind in ("空驶", "赴装", "去装", "装货点"):
                if "月" in text:
                    c.max_month_deadhead_km = km if c.max_month_deadhead_km is None else min(c.max_month_deadhead_km, km)
                else:
                    c.max_pickup_km = km if c.max_pickup_km is None else min(c.max_pickup_km, km)
            else:
                c.max_haul_km = km if c.max_haul_km is None else min(c.max_haul_km, km)

        # 类目禁运：仅从标准禁运句式「(不接|不拉|尽量不拉…)货源品类为「X」」中抽取，
        # 锚定"品类为"+引号，避免把"指定熟货（品类「X」）…不接则损失"这类【必接】语境
        # 误判为禁运（D009 熟货即此情形）。不依赖任何预置枚举。
        if any(k in text for k in ("不接", "不拉", "禁运", "尽量不", "避免", "不要")):
            # 从"品类为"起，连同其后以"或/、/和/及"并列的多个「X」一并抽取
            for seg in re.finditer(r"品类为\s*((?:[「『《][^」』》]{2,12}[」』》]\s*[或、和及]?\s*)+)", text):
                for m in re.finditer(r"[「『《]([^」』》]{2,12})[」』》]", seg.group(1)):
                    cat = m.group(1).strip()
                    if cat and not any(ch.isdigit() for ch in cat):
                        c.banned_categories.add(cat)

        # 禁入圆区：坐标 + 半径
        if any(k in text for k in ("禁入", "不进入", "不得进入", "禁区", "勿入")):
            coord = re.search(r"[\(（]\s*(\d+\.\d+)\s*[,，]\s*(\d+\.\d+)\s*[\)）]", text)
            rad = re.search(r"半径\s*(\d+)\s*(?:km|公里|千米)?", text)
            if coord:
                radius = float(rad.group(1)) if rad else 10.0
                c.forbidden_zones.append((float(coord.group(1)), float(coord.group(2)), radius))

        # 城市范围：优先从文本解析经纬度边界（如「北纬22.42至22.89，东经113.74至114.66」），
        # 不预置任何城市坐标表；复赛换城市/换数值时随文本自动跟上。
        bbox = _parse_region_bbox(text)
        if bbox is not None:
            c.region_bbox = bbox
    return c


def _parse_region_bbox(text: str) -> tuple[float, float, float, float] | None:
    """从「市内行驶」类文本中抽取经纬度范围，返回 (lat_min, lat_max, lng_min, lng_max)。"""
    if not any(k in text for k in ("范围", "市内", "境内", "不出市", "内行驶")):
        return None
    lat = re.search(r"北纬\s*(\d+\.?\d*)\s*(?:[-—~]|至|到)\s*(\d+\.?\d*)", text)
    lng = re.search(r"东经\s*(\d+\.?\d*)\s*(?:[-—~]|至|到)\s*(\d+\.?\d*)", text)
    if lat and lng:
        lat_a, lat_b = float(lat.group(1)), float(lat.group(2))
        lng_a, lng_b = float(lng.group(1)), float(lng.group(2))
        return (min(lat_a, lat_b), max(lat_a, lat_b), min(lng_a, lng_b), max(lng_a, lng_b))
    return None


_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_coord(text: str) -> tuple[float, float] | None:
    """抽取「（lat，lng）」首个经纬度对（中英文括号/逗号均可）。"""
    m = re.search(r"[（(]\s*(\d+\.\d+)\s*[，,]\s*(\d+\.\d+)\s*[）)]", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _parse_km_value(text: str) -> float | None:
    """抽取「X 公里/千米/km」的半径数值（含「一公里」等中文数词）。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|千米|km)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([一二两三四五六七八九十])\s*(?:公里|千米)", text)
    if m:
        return float(_CN_NUM.get(m.group(1), 1))
    return None


def _parse_presence(preferences: Any, use_home: bool, use_visit: bool) -> list[Constraint]:
    """从偏好文本抽取 presence 基元（S1，纯文本派生，不依赖预置坐标表）。

    daily_presence：「每天X点前…自家/回家（lat，lng）一公里内」→ 每日 deadline 前须在家半径内。
    visit_quota：「至少N个不同自然日到过（lat，lng）一公里内」→ 月内 N 个不同日到访。
    子门控关闭则不产出对应基元，保证 OFF 逐字节复现 S0。
    """
    out: list[Constraint] = []
    if not (use_home or use_visit):
        return out
    for t in _pref_texts(preferences):
        if use_home and "点前" in t and any(k in t for k in ("自家", "回家", "到家", "回到", "在家")):
            coord = _parse_coord(t)
            hm = re.search(r"(\d{1,2})\s*点前", t)
            if coord and hm:
                hh = int(hm.group(1))
                if 0 <= hh <= 24:
                    # 同一回家偏好里若含「X点至Y点不接单」静默窗，抽出来让 home_daily 自行守夜，
                    # 使其不依赖独立的 forbid_action（LLM 路径把禁动窗解析为 SCORE 时仍能守 D009 静默）。
                    qw = _parse_time_ranges(t)
                    out.append(Constraint(
                        kind="daily_presence",
                        params={
                            "point": coord,
                            "radius_km": _parse_km_value(t) or 1.0,
                            "deadline_min_of_day": hh * 60,
                            # 守卫边际：吸收 query_scan_cost（每次接单引擎前置 10min，Agent 的 finish
                            # 预估未含）+ 决策粒度，确保实际回家落在 deadline 前。
                            "guard_min": _env_int("MANBANG_PRESENCE_HOME_GUARD", 15),
                            "quiet_window": qw[0] if qw else None,
                        },
                        roles=("GATE", "SCHEDULE"),
                    ))
        if use_visit and "不同" in t and ("自然日" in t or "天" in t) and any(k in t for k in ("到过", "到访", "去过", "到达")):
            coord = _parse_coord(t)
            nm = re.search(r"至少\s*(\d+)\s*个?\s*不同", t)
            if coord and nm:
                out.append(Constraint(
                    kind="require_presence",
                    params={
                        "mode": "visit_quota",
                        "point": coord,
                        "radius_km": _parse_km_value(t) or 1.0,
                        "min_days": int(nm.group(1)),
                    },
                    roles=("SCHEDULE",),
                ))
    return out


def _cn_times_to_minutes(text: str) -> list[int]:
    """抽取中文日期时间「YYYY年M月D日 HH:MM」→ 自 _EPOCH 起的分钟，按出现顺序返回。"""
    out: list[int] = []
    for m in re.finditer(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[:：](\d{1,2})", text):
        y, mo, d, h, mi = (int(g) for g in m.groups())
        try:
            dt = datetime(y, mo, d, h, mi)
        except ValueError:
            continue
        out.append(int((dt - _EPOCH).total_seconds() // 60))
    return out


def _parse_event_constraints(preferences: Any, use_require_accept: bool,
                             use_home_event: bool) -> list[Constraint]:
    """时空预约型硬约束的确定性兜底解析。

    主路径优先让 LLM 输出 timed_route；这里保留货号/坐标/时间戳等结构化文本的兜底。
    兜底也统一产出 timed_route phase，避免继续扩散 home_event/require_accept 这类专用类型。
    """
    out: list[Constraint] = []
    if not isinstance(preferences, list):
        return out
    locs = _extract_location_table(preferences)
    for p in preferences:
        if not isinstance(p, dict):
            continue
        t = str(p.get("content") or p.get("text") or "")
        # —— 熟货必接：「指定熟货源编号NNN…上架时间：T…不接则…损失」 ——
        if use_require_accept and "熟货" in t and any(k in t for k in ("不接", "必接", "信任", "损失")):
            mid = re.search(r"编号\s*(\d+)", t)
            mup = re.search(r"上架时间[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", t)
            s = _wall_to_minutes(mup.group(1)) if mup else None
            e = _wall_to_minutes(p.get("end_time"))
            # 窗口下界=上架时刻（货上架前不可接）；上界=偏好失效时刻（end_time）
            if mid and s is not None and e is not None and e > s:
                out.append(Constraint(
                    "timed_route",
                    {"route_id": hashlib.sha1(t.encode("utf-8")).hexdigest()[:12],
                     "phases": [{"bind_cargo": True, "cargo_id": mid.group(1),
                                  "enter_after": s, "enter_by": e, "dwell_min": 0,
                                  "task": "熟货必接"}],
                     "penalty": float(p.get("penalty_amount", p.get("penalty")) or 0.0),
                     "guard_min": 15},
                    # 提前 12h 激活 GATE，使窗前就拒绝「会跨过上架时刻」的长单，确保窗开时空闲。
                    ("GATE", "SCHEDULE"), active_window=(s - 12 * 60, e), source="regex"))
        # —— 家事剧本：「先到(pickup)接配偶…返回老家(home)…待到T事情解决方可出车」 ——
        if use_home_event and ("家事" in t or ("配偶" in t and "接" in t)):
            coords = re.findall(r"[（(]\s*(\d+\.\d+)\s*[，,]\s*(\d+\.\d+)\s*[）)]", t)
            times = _cn_times_to_minutes(t)
            mhold = re.search(r"(?:停留|静止|待)\s*(?:不少于|至少)?\s*(\d+)\s*分钟", t)
            if len(coords) >= 2 and len(times) >= 2:
                ws = min(times)  # 事件起点（最早时刻）
                le = max(times)  # 「事情解决方可出车」截止（最晚时刻）
                hold = (int(mhold.group(1)) + 1) if mhold else 11  # +1 越过 calc 的「≥10min」临界
                out.append(Constraint(
                    "timed_route",
                    {"route_id": hashlib.sha1(t.encode("utf-8")).hexdigest()[:12],
                     "phases": [
                         {"loc": (float(coords[0][0]), float(coords[0][1])),
                          "enter_after": ws, "enter_by": ws, "dwell_min": hold, "task": "接配偶"},
                         {"loc": (float(coords[1][0]), float(coords[1][1])),
                          "immediate_after_prev": True, "hold_until": le, "dwell_min": 0, "task": "返家静止"},
                     ],
                     "penalty": float(p.get("penalty_amount", p.get("penalty")) or 0.0),
                     "radius_km": 1.0, "guard_min": 15},
                    # 仅 SCHEDULE：家事偏好窗前不可见（见 driver_state_manager 可见性过滤），
                    # 无法、也不应在窗前用 GATE 拦长单（那需不可见的未来信息）。
                    ("SCHEDULE",), active_window=(ws - 24 * 60, le), source="regex"))
        # —— 通用时序路点：某日到某地等待/办事，或按顺序经过多个地点。使用 shared location table
        # 把“增城老档口”这类跨偏好引用解析成坐标；handler 统一消费 timed_route phases。
        if use_home_event:
            timed = _parse_timed_route_event(t, p, locs)
            if timed is not None:
                out.append(timed)
    return out


def _extract_location_table(preferences: Any) -> dict[str, tuple[float, float]]:
    """从全量偏好中抽共享地点表：region/ref -> coord。

    这是 §4.3 的轻量落地：LLM 负责语义约束，程序用确定性坐标抽取校验精确数值。
    当前先覆盖 city/区县名 + 坐标的常见表达，避免为“增城/四会”写 driver 特例。
    """
    locs: dict[str, tuple[float, float]] = {}
    for t in _pref_texts(preferences):
        coord_pat = r"[（(]\s*(\d+\.\d+)\s*[，,]\s*(\d+\.\d+)\s*[）)]"
        for m in re.finditer(r"([\u4e00-\u9fff]{1,12})" + coord_pat, t):
            raw = m.group(1)
            coord = (float(m.group(2)), float(m.group(3)))
            _add_location_aliases(locs, raw, coord)
        # 坐标在句首、地名在句尾的表达也纳入同一地点表，如“到过（23.15,113.67）一公里内的增城区”。
        for m in re.finditer(coord_pat + r".{0,30}?([\u4e00-\u9fff]{2,6})(?:区|市|县|镇)", t):
            coord = (float(m.group(1)), float(m.group(2)))
            raw = m.group(3)
            _add_location_aliases(locs, raw, coord)
    return locs


def _add_location_aliases(locs: dict[str, tuple[float, float]], text: str, coord: tuple[float, float]) -> None:
    """把坐标附近的行政区/地点称呼归一成可复用别名。

    设计上这是“实体消解”的确定性兜底：同一坐标附近出现的“增城区”“增城老档口”
    都指向同一 loc，后续 timed_route 只消费坐标。
    """
    raw = str(text or "")
    if not raw:
        return
    locs.setdefault(raw, coord)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,12}[区市县镇]", raw):
        token = m.group(0)
        locs.setdefault(token, coord)
        locs.setdefault(token.rstrip("区市县镇"), coord)
        stem = token[:-1]
        for n in range(2, min(6, len(stem)) + 1):
            locs.setdefault(stem[-n:], coord)
            locs.setdefault(stem[-n:] + token[-1], coord)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}(?:档口|老家|县城|仓|厂|店)", raw):
        token = m.group(0)
        locs.setdefault(token, coord)
        for n in range(2, min(6, len(token)) + 1):
            locs.setdefault(token[-n:], coord)


def _month_day_to_min(day_of_month: int, minute_of_day: int = 0) -> int:
    return (day_of_month - 1) * 1440 + minute_of_day


def _parse_month_day(text: str) -> int | None:
    m = re.search(r"三月\s*(\d{1,2}|[一二两三四五六七八九十]{1,3})\s*[号日]", text)
    if not m:
        return None
    raw = m.group(1)
    return int(raw) if raw.isdigit() else _cn_num_to_int(raw)


def _find_location_ref(text: str, locs: dict[str, tuple[float, float]]) -> tuple[str, tuple[float, float]] | None:
    best: tuple[str, tuple[float, float]] | None = None
    for key, coord in locs.items():
        if key and key in text and (best is None or len(key) > len(best[0])):
            best = (key, coord)
    return best


def _parse_timed_route_event(text: str, pref: dict[str, Any], locs: dict[str, tuple[float, float]]) -> Constraint | None:
    per = pref.get("penalty_amount", pref.get("penalty"))
    penalty = float(per) if isinstance(per, (int, float)) else 0.0
    if penalty <= 0:
        return None
    route_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    # 单点等待型：如“某日到某地停一趟，花两小时”。
    if any(k in text for k in ("盘库", "清库存", "把数目对清楚", "停一趟")):
        day = _parse_month_day(text)
        loc = _find_location_ref(text, locs)
        mh = re.search(r"花\s*([一二两三四五六七八九十\d]+)\s*小时", text)
        if day and loc:
            dwell_h = _cn_num_to_int(mh.group(1)) if mh else 2
            dwell = max(1, int(dwell_h) * 60)
            day_start = _month_day_to_min(day)
            day_end = day_start + 1440
            return Constraint(
                "timed_route",
                {"route_id": route_id,
                 "phases": [{"loc": loc[1], "enter_by": day_end - dwell, "dwell_min": dwell}],
                 "penalty": penalty},
                ("GATE", "SCHEDULE"), active_window=(day_start - 24 * 60, day_end), source="regex")

    # 多点顺序型：如“先过A，12点前到B，赴宴到14点”。
    if any(k in text for k in ("赴宴", "寿宴", "先过", "赶到")):
        day = _parse_month_day(text)
        if not day:
            return None
        noon = _month_day_to_min(day, 12 * 60)
        phases: list[dict[str, Any]] = []
        # 按文本出现顺序取地点；同一坐标去重。
        seen: set[tuple[float, float]] = set()
        for key, coord in sorted(((k, v) for k, v in locs.items() if k in text), key=lambda kv: text.find(kv[0])):
            rounded = (round(coord[0], 4), round(coord[1], 4))
            if rounded in seen:
                continue
            seen.add(rounded)
            phases.append({"loc": coord, "enter_by": noon, "dwell_min": 0})
        if not phases:
            return None
        if "两点" in text or "2点" in text or "14:00" in text:
            phases[-1]["dwell_min"] = 120
        return Constraint(
            "timed_route",
            {"route_id": route_id, "phases": phases, "penalty": penalty},
            ("GATE", "SCHEDULE"), active_window=(noon - 48 * 60, noon + 14 * 60), source="regex")
    return None


def _cn_num_to_int(text: str) -> int:
    text = str(text).strip()
    if text.isdigit():
        return int(text)
    table = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if text in table:
        return table[text]
    if text.startswith("十"):
        return 10 + table.get(text[1:], 0)
    if "十" in text:
        left, right = text.split("十", 1)
        return table.get(left, 0) * 10 + table.get(right, 0)
    return 0


def _parse_soft_cargo(preferences: Any) -> list[Constraint]:
    """从「尽量不接/避免…品类为「X」」抽取软类目，连同 penalty_amount/cap（status 偏好 dict 自带）。

    与硬禁「不接/不拉/禁运」区分：软约束不硬拒，转 SCORE 期望罚分。返回每条 forbid_cargo_soft。
    """
    out: list[Constraint] = []
    if not isinstance(preferences, list):
        return out
    for p in preferences:
        if not isinstance(p, dict):
            continue
        t = str(p.get("content") or p.get("text") or "")
        if not any(k in t for k in ("尽量不", "避免")) or "品类为" not in t:
            continue
        names: set[str] = set()
        for seg in re.finditer(r"品类为\s*((?:[「『《][^」』》]{2,12}[」』》]\s*[或、和及]?\s*)+)", t):
            for m in re.finditer(r"[「『《]([^」』》]{2,12})[」』》]", seg.group(1)):
                cat = m.group(1).strip()
                if cat and not any(ch.isdigit() for ch in cat):
                    names.add(cat)
        if not names:
            continue
        per = p.get("penalty_amount")
        cap = p.get("penalty_cap")
        out.append(Constraint(
            kind="forbid_cargo_soft",
            params={
                "names": names,
                "per_order": float(per) if isinstance(per, (int, float)) else 0.0,
                "cap": float(cap) if isinstance(cap, (int, float)) else None,
            },
            roles=("SCORE",),
        ))
    return out


_SOFT_HAUL_RE = re.compile(
    r"卸货[^0-9]{0,12}(?:不得超过|不超过|≤|<=|小于|不大于|不高于)?\s*(\d+)\s*(?:公里|千米|km)")
_SOFT_PICK_RE = re.compile(
    r"(?:赴装货点|赴装|去装货点)[^0-9]{0,12}(?:不得超过|不超过|≤|<=|小于|不大于|不高于)?\s*(\d+)\s*(?:公里|千米|km)")


def _parse_soft_distance(preferences: Any) -> list[Constraint]:
    """从「单笔货装货点至卸货点的距离≤Nkm」「赴装货点空驶≤Nkm」抽每单距离上限，连同 penalty_amount/cap。

    这类是**每单软罚分**（calc 用 haversine(start,end) 与 Agent haul_km 同口径），转 SCORE：
    距离超限按机会成本折算，利润够高仍接、付罚分；不硬拒（硬拒会丢高价长途单）。
    """
    out: list[Constraint] = []
    if not isinstance(preferences, list):
        return out
    for p in preferences:
        if not isinstance(p, dict):
            continue
        t = str(p.get("content") or p.get("text") or "")
        per = p.get("penalty_amount")
        cap = p.get("penalty_cap")
        per_f = float(per) if isinstance(per, (int, float)) else 0.0
        cap_f = float(cap) if isinstance(cap, (int, float)) else None
        mh = _SOFT_HAUL_RE.search(t)
        mp = _SOFT_PICK_RE.search(t)
        # 装卸（haul）：仅在**无上限**时软化——无 cap 的罚分不封顶，必须管理（D005 净 +4777）。
        # 有 cap 的装卸罚分有界，A/B 显示「照接长途、付封顶罚分」优于软化（D006/D007 软化净降），
        # 故有 cap 则不解析（回退 baseline：不 GATE 不 SCORE，照接付封顶罚）。
        if mh and cap_f is None:
            out.append(Constraint(kind="limit_distance_soft",
                                  params={"metric": "haul", "max_km": float(mh.group(1)),
                                          "per_order": per_f, "cap": None},
                                  roles=("SCORE",)))
        if mp:
            out.append(Constraint(kind="limit_distance_soft",
                                  params={"metric": "pickup", "max_km": float(mp.group(1)),
                                          "per_order": per_f, "cap": cap_f},
                                  roles=("SCORE",)))
    return out


_LLM_PARSE_SYSTEM = (
    "你是货运司机偏好解析器。把偏好文本解析成结构化约束列表 JSON。只输出 JSON 对象，不要解释，不要 markdown。\n"
    "输入是若干条偏好，每条形如：[文本] (penalty_amount=A, penalty_cap=C)。A=每次违规罚金，C=该规则封顶(null=不封顶)。\n"
    "输出 {\"locations\":{地点key:{\"lat\":LA,\"lng\":LO,\"region\":\"地名\",\"aliases\":[...] }},\"constraints\":[...]}。"
    "locations 是共享地点表：只记录文本明确给出数字经纬度的地点；同一地点的不同叫法用 aliases 合并。"
    "约束优先用 location_ref 引用 locations 的 key；没有可引用地点时才直接写 lat/lng。每项约束是下列之一（数值从文本抽取，罚分额用给定 A/C；只抽文本明确写出的）：\n"
    "- {\"kind\":\"ban_cargo\",\"names\":[品类...]}                              // 明确「不接/不拉/禁运」某品类\n"
    "- {\"kind\":\"avoid_cargo\",\"names\":[...],\"per_order\":A,\"cap\":C}        // 「尽量不/避免」某品类\n"
    "- {\"kind\":\"limit_pickup_km\",\"max_km\":N,\"per_order\":A,\"cap\":C}       // 赴装/去装货点空驶上限\n"
    "- {\"kind\":\"limit_haul_km\",\"max_km\":N,\"per_order\":A,\"cap\":C}         // 装货点到卸货点/干线/运距 单笔上限\n"
    "- {\"kind\":\"limit_month_deadhead_km\",\"max_km\":N,\"per_km\":A,\"cap\":C}  // 一个月空驶里程总和上限，超出部分按公里罚\n"
    "- {\"kind\":\"avoid_window\",\"start_min\":S,\"end_min\":E,\"per_day\":A,\"cap\":C} // 每日某时段不接单/不空驶；分钟(00:00=0,13:00=780)，跨夜 E>1440\n"
    "- {\"kind\":\"daily_rest\",\"minutes\":M,\"per_day\":A,\"cap\":C}              // 每日需连续休息/停车/熄火 M 分钟（per_day=每天没休满的罚金，cap=封顶）\n"
    "- {\"kind\":\"days_off\",\"days\":D,\"penalty\":A}                            // 整月需完全歇业 D 天；penalty=没凑够的一次性罚金 A\n"
    "- {\"kind\":\"daily_presence\",\"location_ref\":\"地点key\",\"lat\":LA,\"lng\":LO,\"deadline_min_of_day\":M,\"radius_km\":R,\"quiet_start_min_of_day\":S,\"quiet_end_min_of_day\":E} // 每天在 M 点前必须回到/到达某坐标半径内（自家、回家、到家、回到家、在家）；若同一句还要求「X点至Y点不接单/不出车/在家」则输出 quiet_*，用于到家后守夜；只在文本明确数字经纬度时输出，优先用 location_ref\n"
    "- {\"kind\":\"visit_region\",\"regions\":[地名...],\"min_days\":N,\"penalty\":A} // 「装货地/卸货地在某地的货，接够 N 个不同自然日」——按货源 city 地名字符串匹配（如「增城」），不要坐标；penalty=没凑够的一次性罚金 A\n"
    "- {\"kind\":\"cargo_region_count\",\"regions\":[地名...],\"per_match\":A,\"cap\":C,\"cap_count\":N} // 计次型月配额：每接一次某地名货抵一次罚分，影子价格 s=A；cap 或 cap_count 到顶后停止奖励\n"
    "- {\"kind\":\"visit_quota\",\"location_ref\":\"地点key\",\"lat\":LA,\"lng\":LO,\"min_days\":N,\"radius_km\":R} // 月内车辆需到访某坐标点 ≥N 个不同自然日（仅当明确数字经纬度、且语义是「车到某点」而非「接某地货」），优先用 location_ref\n"
    "- {\"kind\":\"ban_region\",\"regions\":[地名...],\"location_refs\":[地点key...],\"per_order\":A,\"cap\":C}      // 「装货地/卸货地在某地的货一律不接/尽量不接」——按货源出发/到达城市名匹配的地名字符串（如「惠州」），可用 location_ref 的 region 字段补地名；不要用坐标匹配货源\n"
    "- {\"kind\":\"temporal_region\",\"regions\":[地名...],\"location_refs\":[地点key...],\"scopes\":[\"cargo_start\"|\"cargo_end\"],\"days_of_month\":[D...],\"active_start_min\":S,\"active_end_min\":E,\"per_order\":A,\"cap\":C} // 带生效时间窗的地名货源约束。日级表述（如「三月四号五号」「某几天」「限行日」）优先输出 days_of_month，不要自己换算绝对分钟；小时级/分钟级窗口才输出 active_start_min/active_end_min（从 2026-03-01 00:00 起算，[S,E)）。用于「X月Y号/某几天/某时段不接某地装货或卸货、不往某地派货、限行日避开某城市货」；按货源 city 字符串匹配，不要坐标\n"
    "- {\"kind\":\"timed_route\",\"penalty\":A,\"radius_km\":R,\"phases\":[{\"location_ref\":\"地点key\",\"lat\":LA,\"lng\":LO,\"day_of_month\":D,\"enter_by_min_of_day\":M,\"enter_after_min_of_day\":M,\"dwell_min\":N,\"hold_until_min_of_day\":M,\"immediate_after_prev\":false,\"bind_cargo\":false,\"cargo_id\":\"货号\",\"task\":\"说明\"}]} // 一次性时序剧本：某日到某坐标/地点办事、停留、赴宴、取物、盘库、接人后返家静止，或指定熟货在上架窗内必接。phase 按执行顺序输出；坐标可来自 locations 中其它偏好里同一地点的明确经纬度。熟货必接 phase 输出 bind_cargo=true+cargo_id，可没有 location_ref/lat/lng；家事/接人/返家静止也输出 timed_route，不要输出 home_event。日级时间优先输出 day_of_month + *_min_of_day，不要自己换算绝对分钟；无法确定坐标且不是 bind_cargo 则不要输出该 phase。enter_by_min_of_day 表示最晚到达，hold_until_min_of_day 表示到达后必须原地停留到该时刻；immediate_after_prev 表示上一 phase 完成后立刻去本 phase（如接人后马上返家/返仓）。若只是「X点前赶到B」，只给 B phase 的 enter_by，不要给前一个 phase 设置 hold_until\n"
    "- {\"kind\":\"forbidden_zone\",\"location_ref\":\"地点key\",\"lat\":LA,\"lng\":LO,\"radius_km\":R}          // 车辆禁入的圆区，优先用 location_ref\n"
    "- {\"kind\":\"region\",\"bbox\":[纬下,纬上,经下,经上]}                        // 限定行驶范围\n"
    "时刻换算成分钟（下午1点=780，次日/翌日 +1440）。\n"
    "**地名约束的两种处理（务必区分）：**\n"
    "① 针对**货源地理**的约束（「装货地/卸货地在某地的货」）→ 用**地名字符串**（货源自带 city、字符串匹配即可，"
    "**绝不要坐标**，即使偏好文本里同时写了经纬度也只取地名）：「不接/尽量不接某地货」→ ban_region；"
    "「接某地货凑够 N 个不同自然日」→ visit_region；带日期/时段限定（如"
    "「X月Y号不接某地货」「某几天不往某地跑/派货」）→ temporal_region。"
    "temporal_region 的 scopes：只说装货地则 cargo_start，只说卸货地则 cargo_end，说装/卸、货源、派去、往某地跑则同时给 cargo_start 和 cargo_end。"
    "日级日期必须输出 days_of_month（如「三月四号五号」→ [4,5]），不要输出 active_start_min/active_end_min。\n"
    "② 针对**车辆位置**的 forbidden_zone（禁入圆区）/ region（行驶范围）/ visit_quota（车到坐标点）→ "
    "**只在文本写出明确数字经纬度或 location_ref 指向明确数字经纬度时才输出**；只有地名没有数字经纬度则整条忽略，绝不臆造坐标"
    "（凭错误坐标圈区会误伤大片货源，损失远超照付罚分）。\n"
    "普通一次性到点办事/停留/赴宴/取物/盘库、家事接送返家静止、指定熟货必接都输出 timed_route；"
    "每天回家/到家/在家这类每日到点义务输出 daily_presence，不要输出 visit_quota；"
    "但「X点至Y点不接单/不空驶」这类禁动时段即使写在回家偏好句里，也要作为 avoid_window 输出。\n"
    "**只输出严格匹配上述类型的约束。凡不属于上述任何类型的偏好（如「首单到达不晚于某点」「同日接单数≤N」）"
    "一律直接忽略，绝不勉强映射成 avoid_window 或其它类型。**\n"
    "avoid_window 仅指明确的「整段时间不接单/不空驶」；「在某点前到达/早于/晚于某点」这类到达时限不是禁动窗，忽略。\n"
    "timed_route 时间归属：句式「先过A取物/捎东西，X点前赶到B赴宴/开会到Y点」→ A phase 只表示经过/取物，不设置 hold_until；"
    "X点前归属于 B phase 的 enter_by_min_of_day，Y点归属于 B phase 的 hold_until_min_of_day。只有文本明确说在某 phase 停留/等待到某时刻，才设置该 phase 的 hold_until_min_of_day。\n"
    "家事接送归属：句式「T点后到A接配偶/人，停留N分钟，返回B，待到U点方可出车」→ phase1=A enter_after=T enter_by=T dwell_min=N；"
    "phase2=B immediate_after_prev=true hold_until=U，radius_km 可按文本半径或默认1。\n"
    "休息类区分：『X点到Y点睡觉/休息/停车熄火/车停着不动』指定了起止时刻 → 用 avoid_window(start_min=X,end_min=Y)；"
    "『每天连续休息/停车 N 小时』未指定具体时段 → 用 daily_rest(minutes=N*60)。"
)


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型回复里取出 JSON 对象（容忍 ```json 围栏与前后说明）。"""
    s = str(text).strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except (ValueError, TypeError):
                return None
    return None


def _fnum(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def _fcap(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def _coord_from_obj(obj: dict[str, Any]) -> tuple[float, float] | None:
    lat, lng = obj.get("lat"), obj.get("lng")
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        coord = obj.get("coord") or obj.get("coords") or obj.get("point")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            lat, lng = coord[0], coord[1]
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return None
    lat_f, lng_f = float(lat), float(lng)
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0):
        return None
    return (round(lat_f, 2), round(lng_f, 2))


def _llm_location_table(obj: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """解析 P3 顶层 locations，并把 key/aliases 都映射到同一实体。

    LLM 负责实体消解；这里负责数值校验和别名索引。只有明确数字坐标会进入 coord。
    """
    raw = obj.get("locations") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, dict):
            continue
        k = key.strip()
        coord = _coord_from_obj(value)
        region = str(value.get("region") or "").strip()
        aliases = [a.strip() for a in value.get("aliases", []) if isinstance(a, str) and a.strip()]
        entry = {"key": k, "coord": coord, "region": region, "aliases": tuple(aliases)}
        for name in (k, *aliases):
            out.setdefault(name, entry)
    return out


def _location_ref(it: dict[str, Any]) -> str:
    for key in ("location_ref", "loc_ref", "ref"):
        value = it.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coord_from_constraint(it: dict[str, Any], locs: dict[str, dict[str, Any]]) -> tuple[float, float] | None:
    ref = _location_ref(it)
    if ref and isinstance(locs.get(ref), dict):
        coord = locs[ref].get("coord")
        if isinstance(coord, tuple) and len(coord) == 2:
            return coord
    return _coord_from_obj(it)


def _location_refs(it: dict[str, Any]) -> list[str]:
    out: list[str] = []
    single = _location_ref(it)
    if single:
        out.append(single)
    for key in ("location_refs", "loc_refs"):
        values = it.get(key)
        if isinstance(values, list):
            out.extend(v.strip() for v in values if isinstance(v, str) and v.strip())
    return out


def _regions_from_constraint(it: dict[str, Any], locs: dict[str, dict[str, Any]]) -> set[str]:
    regions = {str(r).strip() for r in (it.get("regions") or [])
               if isinstance(r, str) and r.strip()}
    for ref in _location_refs(it):
        entry = locs.get(ref)
        if not isinstance(entry, dict):
            continue
        region = str(entry.get("region") or "").strip()
        if region:
            regions.add(region)
        else:
            key = str(entry.get("key") or ref).strip()
            if key:
                regions.add(key)
    return regions


def _llm_phase_abs_min(ph: dict[str, Any], abs_key: str, day_key: str, minute_key: str) -> int | None:
    if isinstance(ph.get(abs_key), (int, float)):
        return int(ph[abs_key])
    day = ph.get(day_key)
    minute = ph.get(minute_key)
    if isinstance(day, (int, float)) and isinstance(minute, (int, float)):
        d, m = int(day), int(minute)
        if 1 <= d <= 31 and 0 <= m <= 2880:
            return (d - 1) * 1440 + m
    return None


def _llm_timed_route_to_constraint(it: dict[str, Any], locs: dict[str, dict[str, Any]] | None = None) -> Constraint | None:
    locs = locs or {}
    phases_in = it.get("phases")
    if not isinstance(phases_in, list):
        return None
    phases: list[dict[str, Any]] = []
    for ph in phases_in:
        if not isinstance(ph, dict):
            continue
        bind_cargo = bool(ph.get("bind_cargo"))
        cid = str(ph.get("cargo_id") or "").strip()
        loc = _coord_from_constraint(ph, locs)
        out: dict[str, Any] = {}
        if bind_cargo:
            if not cid:
                continue
            out["bind_cargo"] = True
            out["cargo_id"] = cid
        else:
            if loc is None:
                continue
            out["loc"] = loc
            ref = _location_ref(ph)
            if ref:
                out["location_ref"] = ref
        enter_by = _llm_phase_abs_min(ph, "enter_by_min", "day_of_month", "enter_by_min_of_day")
        enter_after = _llm_phase_abs_min(ph, "enter_after_min", "day_of_month", "enter_after_min_of_day")
        hold_until = _llm_phase_abs_min(ph, "hold_until_min", "day_of_month", "hold_until_min_of_day")
        if enter_by is not None:
            out["enter_by"] = enter_by
        if enter_after is not None:
            out["enter_after"] = enter_after
        dwell = int(ph.get("dwell_min") or 0)
        if hold_until is not None and (enter_by is None or hold_until > enter_by):
            out["hold_until"] = hold_until
            if enter_by is not None:
                dwell = max(dwell, hold_until - enter_by)
        if bool(ph.get("immediate_after_prev") or ph.get("asap_after_prev") or ph.get("immediate")):
            out["immediate_after_prev"] = True
        out["dwell_min"] = max(0, dwell)
        task = ph.get("task")
        if isinstance(task, str) and task.strip():
            out["task"] = task.strip()
        phases.append(out)
    if not phases:
        return None
    deadlines = [int(p[k]) for p in phases for k in ("enter_by", "hold_until") if isinstance(p.get(k), int)]
    if not deadlines:
        return None
    active_start = int(it.get("active_start_min", min(deadlines) - 48 * 60))
    active_end = int(it.get("active_end_min", max(deadlines) + 24 * 60))
    penalty = _fnum(it.get("penalty", it.get("per_order")))
    route_id_src = json.dumps(it, ensure_ascii=False, sort_keys=True)
    return Constraint(
        "timed_route",
        {"route_id": hashlib.sha1(route_id_src.encode("utf-8")).hexdigest()[:12],
         "phases": phases, "penalty": penalty,
         "radius_km": float(it.get("radius_km") or 2.0),
         "guard_min": int(it.get("guard_min") or 15)},
        ("GATE", "SCHEDULE"),
        active_window=(max(0, active_start), max(active_start + 1, active_end)),
        source="llm",
    )


def _timed_route_bound_cargo_ids(items: list[Constraint]) -> set[str]:
    ids: set[str] = set()
    for c in items:
        if c.kind != "timed_route":
            continue
        for ph in c.params.get("phases") or []:
            if isinstance(ph, dict) and ph.get("bind_cargo"):
                cid = str(ph.get("cargo_id") or "").strip()
                if cid:
                    ids.add(cid)
    return ids


def _llm_items_to_constraints(obj: dict[str, Any]) -> list[Constraint] | None:
    """LLM 富 JSON → Constraint 列表：软约束→SCORE、硬约束→GATE、调度→SCHEDULE。
    顶层缺 constraints 列表则 None（回退正则）；单项非法则跳过。"""
    cons = obj.get("constraints")
    if not isinstance(cons, list):
        return None
    locs = _llm_location_table(obj)

    def _names(it: dict) -> set[str]:
        return {str(n).strip() for n in (it.get("names") or [])
                if isinstance(n, str) and n.strip() and not any(ch.isdigit() for ch in n)}

    items: list[Constraint] = []
    for it in cons:
        if not isinstance(it, dict):
            continue
        k = it.get("kind")
        try:
            if k == "ban_cargo":
                nm = _names(it)
                if nm:
                    items.append(Constraint("forbid_cargo", {"names": nm}, ("GATE",), source="llm"))
            elif k == "avoid_cargo":
                nm = _names(it)
                if nm:
                    items.append(Constraint("forbid_cargo_soft",
                                            {"names": nm, "per_order": _fnum(it.get("per_order")), "cap": _fcap(it.get("cap"))},
                                            ("SCORE",), source="llm"))
            elif k in ("limit_pickup_km", "limit_haul_km"):
                mx = it.get("max_km")
                cap = _fcap(it.get("cap"))
                # A/B 结论（S2.b）：装卸（haul）仅在**无 cap**时软化；有 cap 的装卸软化净降，
                # 照接长途付封顶罚更优 → 不解析（忽略，回退 baseline）。赴装（pickup）总软化。
                if isinstance(mx, (int, float)) and mx > 0:
                    metric = "pickup" if k == "limit_pickup_km" else "haul"
                    if metric == "haul" and cap is not None:
                        pass  # 带 cap 的装卸：忽略
                    else:
                        items.append(Constraint("limit_distance_soft",
                                                {"metric": metric, "max_km": float(mx), "per_order": _fnum(it.get("per_order")), "cap": cap},
                                                ("SCORE",), source="llm"))
            elif k == "limit_month_deadhead_km":
                mx = it.get("max_km")
                if isinstance(mx, (int, float)) and mx > 0:
                    items.append(Constraint("limit_month_deadhead_soft",
                                            {"max_km": float(mx), "per_km": _fnum(it.get("per_km")), "cap": _fcap(it.get("cap"))},
                                            ("SCORE",), source="llm"))
            elif k == "avoid_window":
                s, e = int(it["start_min"]), int(it["end_min"])
                # 低价时段窗（跨夜 / 22点后起 / 上午9点前结束）→ 硬避让划算（且兼作休息块）；
                # 日间窗（如午休）覆盖高价时段，强制反亏（A/B：D004 午窗硬enf −9790）→ 忽略（照付罚）。
                low_value = (e > 1440) or (s >= 22 * 60) or (e <= 9 * 60 and s < 6 * 60)
                if 0 <= s and e > s and low_value:
                    # rest_per_day/rest_cap：仅供 planner V2 把「固定时段禁动」当休息义务前瞻用
                    # （§4.1 固定窗休息）；旧 SCHEDULE/GATE handler 不读这俩字段，行为不变。
                    items.append(Constraint("forbid_action",
                                            {"windows": [(s, e)],
                                             "rest_per_day": _fnum(it.get("per_day")),
                                             "rest_cap": _fcap(it.get("cap"))},
                                            ("SCHEDULE", "GATE"), source="llm"))
            elif k == "daily_rest":
                m = int(it.get("minutes") or 0)
                # 任意连续休息 N 分钟。产出两条（缓存存全，A/B 由门控过滤）：
                # ① forbid_action(0,m, source_daily_rest=True)：旧固定块方案（贪心架构用）——同日连续满足
                #    calc _eval_daily_rest，占低价夜间。GATE 拦会侵入休息块的接单，SCHEDULE 窗内 wait。
                # ② daily_rest_duty(minutes, per_day, cap)：planner 专用休息义务（§4.1 高初始紧迫度，
                #    高价单可抢占），roles=("REST",) 无旧 handler、非 planner 模式自动忽略。
                if 0 < m <= 1440:
                    items.append(Constraint("forbid_action",
                                            {"windows": [(0, m)], "source_daily_rest": True},
                                            ("SCHEDULE", "GATE"), source="llm"))
                    items.append(Constraint("daily_rest_duty",
                                            {"minutes": m, "per_day": _fnum(it.get("per_day")), "cap": _fcap(it.get("cap"))},
                                            ("REST", "SCHEDULE"), source="llm"))
            elif k == "days_off":
                d = int(it.get("days") or 0)
                if d > 0:
                    items.append(Constraint(
                        "require_idle_days",
                        {"days": d, "penalty": _fnum(it.get("penalty"))},
                        ("SCHEDULE",),
                        source="llm",
                    ))
            elif k == "daily_presence":
                point = _coord_from_constraint(it, locs)
                deadline = int(it.get("deadline_min_of_day") or 0)
                if point is not None and 0 < deadline <= 1440:
                    qs, qe = it.get("quiet_start_min_of_day"), it.get("quiet_end_min_of_day")
                    quiet = None
                    if isinstance(qs, (int, float)) and isinstance(qe, (int, float)) and int(qe) > int(qs):
                        quiet = (int(qs), int(qe))
                    params: dict[str, Any] = {
                        "point": point,
                        "radius_km": float(it.get("radius_km") or 1.0),
                        "deadline_min_of_day": deadline,
                        "guard_min": _env_int("MANBANG_PRESENCE_HOME_GUARD", 15),
                        "quiet_window": quiet,
                    }
                    ref = _location_ref(it)
                    if ref:
                        params["location_ref"] = ref
                    items.append(Constraint(
                        "daily_presence",
                        params,
                        ("GATE", "SCHEDULE"), source="llm"))
            elif k == "ban_region":
                # 货源地名禁运（按 city 字符串匹配，对齐 calc _cargo_touches_region）。SCORE 软罚：
                # 无 cap（惠州 800/次）则始终计罚、逼出「净收益>罚才接」；有 cap 达顶后边际 0。
                regions = _regions_from_constraint(it, locs)
                if regions:
                    items.append(Constraint("ban_region",
                                            {"regions": regions, "per_order": _fnum(it.get("per_order")), "cap": _fcap(it.get("cap"))},
                                            ("SCORE",), source="llm"))
            elif k == "temporal_region":
                # 带绝对生效时间窗的地名货源约束。抽象覆盖：
                # 「某几天不接某地货」「某时间段不往某地派货」「限行日避开某城市装/卸货」。
                regions = _regions_from_constraint(it, locs)
                raw_scopes = it.get("scopes") or []
                scopes = {str(s).strip() for s in raw_scopes
                          if str(s).strip() in {"cargo_start", "cargo_end"}}
                days = sorted({int(d) for d in (it.get("days_of_month") or [])
                               if isinstance(d, (int, float)) and 1 <= int(d) <= 31})
                if days:
                    a, b = (days[0] - 1) * 1440, days[-1] * 1440
                else:
                    a, b = int(it["active_start_min"]), int(it["active_end_min"])
                if regions and scopes and 0 <= a < b:
                    items.append(Constraint(
                        "temporal_region",
                        {"regions": regions, "scopes": scopes, "active_window": (a, b),
                         "per_order": _fnum(it.get("per_order")), "cap": _fcap(it.get("cap"))},
                        ("SCORE",), source="llm"))
            elif k == "timed_route":
                route = _llm_timed_route_to_constraint(it, locs)
                if route is not None:
                    items.append(route)
            elif k == "forbidden_zone":
                point = _coord_from_constraint(it, locs)
                if point is not None:
                    items.append(Constraint("forbid_location",
                                            {"zones": [(point[0], point[1], float(it.get("radius_km", 10)))]},
                                            ("GATE",), source="llm"))
            elif k == "region":
                bb = it.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    la, lA, lo, lO = (float(x) for x in bb)
                    items.append(Constraint("confine_location",
                                            {"bbox": (min(la, lA), max(la, lA), min(lo, lO), max(lo, lO))},
                                            ("GATE",), source="llm"))
            elif k == "visit_region":
                # 货源地名到访配额（按 city 字符串匹配，对齐 calc _eval_required_region_cargo_days）。
                # SCORE 奖励型：接该地货凑够 min_days 个不同日，影子价格激励（见 _score_visit_region）。
                regions = _regions_from_constraint(it, locs)
                md = int(it.get("min_days") or 0)
                if regions and md > 0:
                    items.append(Constraint("visit_region",
                                            {"regions": regions, "min_days": md, "penalty": _fnum(it.get("penalty"))},
                                            ("SCORE",), source="llm"))
            elif k == "cargo_region_count":
                regions = _regions_from_constraint(it, locs)
                per = _fnum(it.get("per_match", it.get("reward_per_match")))
                cap_count = it.get("cap_count")
                if not isinstance(cap_count, (int, float)):
                    cap = _fcap(it.get("cap"))
                    cap_count = int(cap // per) if cap is not None and per > 0 else None
                if regions and per > 0:
                    items.append(Constraint(
                        "cargo_region_count",
                        {"regions": regions, "per_match": per,
                         "cap_count": int(cap_count) if cap_count is not None else None},
                        ("SCORE",),
                        source="llm",
                    ))
            elif k == "visit_quota":
                point = _coord_from_constraint(it, locs)
                md = int(it.get("min_days") or 0)
                # 坐标铁律：仅接受明确数字经纬度（提示词已要求地名无坐标整条忽略）。
                if point is not None and md > 0:
                    params = {"mode": "visit_quota", "point": point,
                              "radius_km": float(it.get("radius_km") or 1.0), "min_days": md}
                    ref = _location_ref(it)
                    if ref:
                        params["location_ref"] = ref
                    items.append(Constraint("require_presence", params, ("SCHEDULE",), source="llm"))
            # 时空预约型统一走 timed_route；require_accept/home_event 只作为旧 IR/兜底兼容概念保留。
        except (KeyError, ValueError, TypeError):
            continue
    return items


def _pref_records(preferences: Any) -> list[str]:
    """偏好 → 「文本 (penalty_amount=A, penalty_cap=C, start_time=S, end_time=E)」喂给 LLM。

    附 penalty_amount/cap 使其能填罚分额；附 start_time/end_time 作为时段类约束
    （avoid_window 等）的时间上下文。时序剧本（熟货必接 / 家事 / 赴宴 / 盘库）主路径由 LLM
    解析为 timed_route；确定性解析只做兜底。"""
    out: list[str] = []
    if not isinstance(preferences, list):
        return out
    for p in preferences:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            t = p.get("content") or p.get("text")
            if t:
                out.append(
                    f"{t} (penalty_amount={p.get('penalty_amount')}, penalty_cap={p.get('penalty_cap')}, "
                    f"start_time={p.get('start_time')}, end_time={p.get('end_time')})"
                )
    return out


def _llm_parse_constraints(api: Any, preferences: Any) -> list[Constraint] | None:
    """调用模型把偏好解析成 Constraint 列表（软→SCORE）；失败/非法返回 None（由调用方回退正则）。

    O(1)/司机：仅在 `_refresh_constraints` 的 repr 脏检查命中（偏好变化）时调用一次。
    """
    recs = _pref_records(preferences)
    if not recs:
        return []  # 空偏好 → 空列表（无需调模型）
    # A/B 缓存（门控 MANBANG_LLM_CACHE）：命中则跳过模型调用，保证多组解析一致。
    use_cache = os.environ.get("MANBANG_LLM_CACHE", "0") == "1"
    cache_path: str | None = None
    if use_cache:
        key = hashlib.sha1(("\n".join(recs) + "|" + _LLM_PARSE_SYSTEM).encode("utf-8")).hexdigest()
        cache_path = os.path.join(_LLM_CACHE_DIR, key + ".pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:  # noqa: BLE001 — 缓存损坏则照常调模型
                pass
    payload = {
        "messages": [
            {"role": "system", "content": _LLM_PARSE_SYSTEM},
            {"role": "user", "content": "\n".join(recs)},
        ],
        "temperature": 0,
        # qwen reasoning 链是解析超时主因（默认 ~150s 贴死 60s 超时→回退裸正则→罚分爆炸）。
        # 关闭 thinking 后 150s→1.1s、token 降 90%，解析结果一致。payload 由网关透传，agent 侧可控。
        "enable_thinking": False,
    }
    try:
        data = api.model_chat_completion(payload)
        content = data["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 — 调用/取值失败一律回退正则
        return None
    obj = _extract_json(content)
    if obj is None:
        return None
    result = _llm_items_to_constraints(obj)
    if use_cache and cache_path is not None and result is not None:
        try:
            os.makedirs(_LLM_CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:  # noqa: BLE001 — 落盘失败不影响本次解析
            pass
    return result


def _build_constraints(preferences: Any, use_presence_home: bool = False,
                       use_presence_visit: bool = False,
                       use_soft_cargo: bool = False,
                       use_soft_distance: bool = False,
                       api: Any = None, use_llm: bool = False,
                       use_require_accept: bool = True,
                       use_home_event: bool = True,
                       use_ban_region: bool = False,
                       use_planner: bool = False,
                       use_visit_region: bool = False,
                       use_idle_gate: bool = False,
                       use_general_parser: bool = False,
                       driver_id: str = "") -> ConstraintSet:
    """偏好 → ConstraintSet（IR）。

    LLM 路径（门控开 + 注入 api）：LLM 产出软/硬分类的 Constraint 列表（软→SCORE），叠加正则
    require_presence（S1）；任一失败回退下方正则路径。正则路径（默认/兜底）保持不变。
    """
    if use_general_parser:
        try:
            try:
                from general_preference_parser import parse_preferences as _parse_general_preferences
            except Exception:
                from agent.general_preference_parser import parse_preferences as _parse_general_preferences
            parsed = _parse_general_preferences(preferences, api=api, driver_id=driver_id)
            items = [
                Constraint(
                    kind=spec.kind,
                    params=dict(spec.params),
                    roles=tuple(spec.roles),
                    active_window=spec.active_window,
                    source=spec.source,
                )
                for spec in parsed.constraints
            ]
            if not use_ban_region:
                items = [it for it in items if it.kind not in ("ban_region", "temporal_region")]
            if not use_visit_region:
                items = [it for it in items if it.kind != "visit_region"]
            if use_idle_gate:
                for it in items:
                    if it.kind == "require_idle_days" and "GATE" not in it.roles:
                        it.roles = it.roles + ("GATE",)
            if use_planner:
                items = [
                    it for it in items
                    if not (it.kind == "forbid_action" and it.params.get("source_daily_rest"))
                ]
                for it in items:
                    if it.kind == "daily_rest_duty" and "SCHEDULE" not in it.roles:
                        it.roles = it.roles + ("SCHEDULE",)
            else:
                items = [it for it in items if it.kind != "daily_rest_duty"]
            cset = ConstraintSet(items=items)
            cset.rebuild_index()
            return cset
        except Exception:
            pass

    if use_llm and api is not None:
        llm_items = _llm_parse_constraints(api, preferences)
        if llm_items is not None:
            # S5 门控：关则剔除地名禁货软约束（全程 ban_region + 带时窗 temporal_region），
            # 回退「不处理地名货=照接付罚」。LLM 缓存按提示词哈希存完整解析，A/B 共用同一份，
            # 仅此处过滤 → 干净归因。
            if not use_ban_region:
                llm_items = [it for it in llm_items if it.kind not in ("ban_region", "temporal_region")]
            # S7 门控：关则剔除 visit_region（地名到访配额），回退「无有效约束=凑不够付 failed」。
            if not use_visit_region:
                llm_items = [it for it in llm_items if it.kind != "visit_region"]
            # S8 门控：开则给 require_idle_days 加 GATE role（拒 finish 跨入歇业日的单，防污染 off-day）。
            if use_idle_gate:
                for it in llm_items:
                    if it.kind == "require_idle_days" and "GATE" not in it.roles:
                        it.roles = it.roles + ("GATE",)
            # 休息约束二选一（缓存存全两条，门控过滤、A/B 干净归因）：
            # planner 模式 → 删固定块 forbid_action(source_daily_rest)，由 daily_rest_duty 交 planner 编排；
            # 非 planner 模式 → 删 daily_rest_duty，保留固定块（旧行为，逐字节复现）。
            if use_planner:
                llm_items = [it for it in llm_items
                             if not (it.kind == "forbid_action" and it.params.get("source_daily_rest"))]
                # 兼容旧 LLM 缓存：缓存里的 daily_rest_duty 可能只有 REST role。planner 模式下给它补
                # SCHEDULE role，启用 pre-query 休息兜底，避免 query scan 吃掉最后的连续休息余量。
                for it in llm_items:
                    if it.kind == "daily_rest_duty" and "SCHEDULE" not in it.roles:
                        it.roles = it.roles + ("SCHEDULE",)
            else:
                llm_items = [it for it in llm_items if it.kind != "daily_rest_duty"]
            # 时空预约型：timed_route 优先使用 LLM 解析出的 phase；若 LLM 未产出 timed_route，
            # 再用确定性解析兜底。兜底也尽量产出 timed_route，减少专用 kind 扩散。
            event_items = _parse_event_constraints(preferences, use_require_accept, use_home_event)
            if any(it.kind == "timed_route" for it in llm_items):
                event_items = [it for it in event_items if it.kind != "timed_route"]
            bound_cargo_ids = _timed_route_bound_cargo_ids(llm_items)
            if bound_cargo_ids:
                event_items = [
                    it for it in event_items
                    if not (it.kind == "require_accept" and str(it.params.get("cargo_id") or "") in bound_cargo_ids)
                ]
            llm_items.extend(event_items)
            presence_items = _parse_presence(preferences, use_presence_home, use_presence_visit)
            if any(it.kind == "daily_presence" for it in llm_items):
                presence_items = [it for it in presence_items if it.kind != "daily_presence"]
            llm_items.extend(presence_items)
            cset = ConstraintSet(items=llm_items)
            cset.rebuild_index()
            return cset

    legacy = _parse_constraints(preferences)
    items: list[Constraint] = []

    if legacy.night_windows:
        items.append(Constraint(
            kind="forbid_action",
            params={"windows": list(legacy.night_windows)},
            roles=("SCHEDULE", "GATE"),
        ))
    if legacy.daily_rest_min > 0:
        items.append(Constraint(
            kind="require_rest",
            params={"minutes": legacy.daily_rest_min},
            roles=("SCHEDULE",),
        ))
    if legacy.days_off > 0:
        items.append(Constraint(
            kind="require_idle_days",
            params={"days": legacy.days_off},
            roles=("SCHEDULE", "GATE") if use_idle_gate else ("SCHEDULE",),
        ))
    # S2.b：每单距离上限（赴装/装卸）默认硬 GATE；软门控开时转 SCORE（见下），此处对软覆盖的 metric 跳过。
    soft_dist_items = _parse_soft_distance(preferences) if use_soft_distance else []
    soft_metrics = {c.params.get("metric") for c in soft_dist_items}
    if legacy.max_pickup_km is not None and "pickup" not in soft_metrics:
        items.append(Constraint(
            kind="limit_distance",
            params={"metric": "pickup", "max_km": legacy.max_pickup_km},
            roles=("GATE",),
        ))
    if legacy.max_haul_km is not None and "haul" not in soft_metrics:
        items.append(Constraint(
            kind="limit_distance",
            params={"metric": "haul", "max_km": legacy.max_haul_km},
            roles=("GATE",),
        ))
    if legacy.max_month_deadhead_km is not None:
        items.append(Constraint(
            kind="limit_distance",
            params={"metric": "month_deadhead", "max_km": legacy.max_month_deadhead_km},
            roles=("GATE",),
        ))
    # S2：软类目（「尽量不/避免」）从硬禁集剥离，转 SCORE；门控关时仍按硬禁（S0/S1 口径）。
    soft_items = _parse_soft_cargo(preferences) if use_soft_cargo else []
    soft_names = {n for c in soft_items for n in c.params.get("names", set())}
    hard_banned = set(legacy.banned_categories) - soft_names
    if hard_banned:
        items.append(Constraint(
            kind="forbid_cargo",
            params={"names": hard_banned},
            roles=("GATE",),
        ))
    if legacy.forbidden_zones:
        items.append(Constraint(
            kind="forbid_location",
            params={"zones": list(legacy.forbidden_zones)},
            roles=("GATE",),
        ))
    if legacy.region_bbox is not None:
        items.append(Constraint(
            kind="confine_location",
            params={"bbox": legacy.region_bbox},
            roles=("GATE",),
        ))

    items.extend(_parse_presence(preferences, use_presence_home, use_presence_visit))
    items.extend(soft_items)
    items.extend(soft_dist_items)

    cset = ConstraintSet(items=items)
    cset.rebuild_index()
    return cset



def _in_bbox(lat: float, lng: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= lat <= bbox[1] and bbox[2] <= lng <= bbox[3]


def _in_any_zone(lat: float, lng: float, zones: list[tuple[float, float, float]]) -> bool:
    return any(haversine_km(lat, lng, zlat, zlng) <= zr for zlat, zlng, zr in zones)


# --------------------------------------------------------------------------- #
# 约束 IR（中间表示）+ Handler 注册表
# --------------------------------------------------------------------------- #
# 把"可执行模型"与"解析"解耦：解析层只产出 Constraint 列表，决策引擎只消费它们。
# 加一种约束类型 = 往注册表加一个 handler，而非改固定字段。IR 为唯一决策路径。


@dataclass
class Constraint:
    kind: str                          # 基元类型（见 _GATE_HANDLERS / _SCHEDULE_HANDLERS）
    params: dict[str, Any]             # kind 专属载荷
    roles: tuple[str, ...]             # ("GATE",) / ("SCHEDULE",) / ("GATE","SCHEDULE")
    active_window: tuple[int, int] | None = None  # [start,end] 仿真分钟；None=全程
    source: str = "regex"              # provenance，便于日志


@dataclass
class ConstraintSet:
    items: list[Constraint] = field(default_factory=list)
    by_role: dict[str, list[Constraint]] = field(default_factory=dict)

    def rebuild_index(self) -> None:
        idx: dict[str, list[Constraint]] = {"GATE": [], "SCHEDULE": []}
        for c in self.items:
            for r in c.roles:
                idx.setdefault(r, []).append(c)
        self.by_role = idx

    def daily_rest_minutes(self) -> int:
        """跨约束依赖：夜间窗等待时需知道休息要求，以便顺带标记当日已休。"""
        m = 0
        for c in self.items:
            if c.kind == "require_rest":
                m = max(m, int(c.params.get("minutes", 0)))
        return m


@dataclass
class CandidateCtx:
    """_best_candidate 内每候选构建一次；reposition 复用（start=end=目标点，is_reposition=True）。"""
    cargo: dict[str, Any]
    deadhead_km: float
    haul_km: float
    arrival: int
    ready: int
    finish: int
    slat: float
    slng: float
    elat: float
    elng: float
    net: float
    busy: int
    st: "DriverRunState"
    now: int
    lat: float
    lng: float
    cost_per_km: float
    horizon_min: int
    speed: float
    is_reposition: bool = False


@dataclass
class ScheduleCtx:
    """decide() 候选循环前构建一次。"""
    st: "DriverRunState"
    now: int
    day: int
    mins_into_day: int
    mins_to_midnight: int
    lat: float
    lng: float
    cost_per_km: float
    horizon_min: int
    speed: float
    horizon_guard: int

# 判断一个约束现在是否生效
def _active(c: Constraint, now: int) -> bool:
    if c.active_window is None:
        return True
    return c.active_window[0] <= now < c.active_window[1]


def _timed_route_days(cset: ConstraintSet) -> set[int]:
    days: set[int] = set()
    for c in cset.items:
        if c.kind != "timed_route":
            continue
        for ph in c.params.get("phases") or []:
            enter_by = ph.get("enter_by")
            if isinstance(enter_by, int):
                days.add(enter_by // 1440)
    return days


def _idle_days(st: "DriverRunState", days: int) -> set[int]:
    """选择整天歇业日：优先月末，但避开 timed_route 占用日。"""
    remaining = max(0, days - len(st.idle_days_done))
    if remaining <= 0:
        return set()
    blocked = _timed_route_days(st.constraints)
    out: list[int] = []
    for d in range(st.total_days - 1, -1, -1):
        if d in blocked or d in st.idle_days_done:
            continue
        out.append(d)
        if len(out) >= remaining:
            break
    return set(out)


def _current_route_phase(st: "DriverRunState", c: Constraint) -> dict[str, Any] | None:
    rid = str(c.params.get("route_id") or "")
    phases = c.params.get("phases") or []
    idx = st.timed_route_progress.get(rid, 0)
    if idx < 0 or idx >= len(phases):
        return None
    ph = phases[idx]
    return ph if isinstance(ph, dict) else None


def _route_phase_latest_arrive(c: Constraint, idx: int, speed: float, guard_default: int) -> int:
    """从当前 phase 倒推最晚到达时间。"""
    phases = c.params.get("phases") or []
    if idx < 0 or idx >= len(phases) or not isinstance(phases[idx], dict):
        return 0
    cur = phases[idx]
    latest = int(cur.get("enter_by", 10**9) or 10**9)
    loc = cur.get("loc")
    if not loc:
        return latest
    travel = int(cur.get("dwell_min", 0) or 0)
    hold = cur.get("hold_until")
    if isinstance(hold, int) and latest < 10**9:
        travel = max(travel, hold - latest)
    prev_loc = loc
    for j in range(idx + 1, len(phases)):
        ph = phases[j]
        if not isinstance(ph, dict):
            continue
        nxt = ph.get("loc")
        if not nxt:
            continue
        guard = int(ph.get("guard_min", guard_default) or guard_default)
        travel += _drive_minutes(haversine_km(prev_loc[0], prev_loc[1], nxt[0], nxt[1]), speed) + guard
        latest = min(latest, int(ph.get("enter_by", 10**9) or 10**9) - travel)
        ph_dwell = int(ph.get("dwell_min", 0) or 0)
        ph_hold = ph.get("hold_until")
        if isinstance(ph_hold, int) and isinstance(ph.get("enter_by"), int):
            ph_dwell = max(ph_dwell, ph_hold - int(ph.get("enter_by")))
        travel += ph_dwell
        prev_loc = nxt
    return latest


# -- GATE handlers：返回 True 表示候选/目标点仍允许 ----------------------------- #
# 候选场景：ctx.is_reposition=False，cargo 真实，slat/slng=起点、elat/elng=终点。
# reposition 场景：ctx.is_reposition=True，slat/slng=elat/elng=目标点，deadhead_km=step_km。


def _gate_forbid_cargo(c: Constraint, ctx: CandidateCtx) -> bool:
    if ctx.is_reposition:
        return True  # reposition 不涉及货源类目
    name = str(ctx.cargo.get("cargo_name", "") or "")
    names = c.params.get("names") or set()
    if name and name in names:
        return False
    return True


def _gate_limit_distance(c: Constraint, ctx: CandidateCtx) -> bool:
    metric = c.params.get("metric")
    max_km = c.params.get("max_km")
    if max_km is None:
        return True
    if metric == "pickup":
        if ctx.is_reposition:
            return True  # 现行 _step_toward 不校验赴装上限
        return ctx.deadhead_km <= max_km
    if metric == "haul":
        if ctx.is_reposition:
            return True
        return ctx.haul_km <= max_km
    if metric == "month_deadhead":
        # 候选：累计 + 本单赴装；reposition：累计 + 本步 step_km（deadhead_km 即 step_km）
        return ctx.st.month_deadhead_km + ctx.deadhead_km <= max_km
    return True


def _gate_confine_location(c: Constraint, ctx: CandidateCtx) -> bool:
    bbox = c.params.get("bbox")
    if bbox is None:
        return True
    if ctx.is_reposition:
        return _in_bbox(ctx.elat, ctx.elng, bbox)  # 仅目标点
    return _in_bbox(ctx.slat, ctx.slng, bbox) and _in_bbox(ctx.elat, ctx.elng, bbox)


def _gate_forbid_location(c: Constraint, ctx: CandidateCtx) -> bool:
    zones = c.params.get("zones") or []
    if not zones:
        return True
    if ctx.is_reposition:
        return not _in_any_zone(ctx.elat, ctx.elng, zones)  # 仅目标点
    if _in_any_zone(ctx.slat, ctx.slng, zones) or _in_any_zone(ctx.elat, ctx.elng, zones):
        return False
    return True


def _gate_forbid_action(c: Constraint, ctx: CandidateCtx) -> bool:
    windows = c.params.get("windows") or []
    if BaselinePolicy._hits_night_window(windows, ctx.now, ctx.finish):
        return False
    return True


def _is_daily_presence(c: Constraint) -> bool:
    return c.kind == "daily_presence" or (c.kind == "require_presence" and c.params.get("mode") == "home_daily")


def _gate_require_presence(c: Constraint, ctx: CandidateCtx) -> bool:
    """daily_presence：拒绝"完单后无法在当日 deadline 前驶回家半径内"的单（依赖 finish/elat）。

    visit_quota 无 GATE（只在 SCHEDULE 倒推）；reposition 不受此闸门约束。
    """
    if not _is_daily_presence(c) or ctx.is_reposition:
        return True
    point = c.params.get("point")
    if not point:
        return True
    hlat, hlng = point
    guard = int(c.params.get("guard_min", 15))
    deadline_abs = (ctx.now // 1440) * 1440 + int(c.params.get("deadline_min_of_day", 23 * 60))
    if ctx.finish <= deadline_abs:
        drive_home = _drive_minutes(haversine_km(ctx.elat, ctx.elng, hlat, hlng), ctx.speed)
        if ctx.finish + drive_home > deadline_abs - guard:
            return False
    return True


def _gate_require_accept(c: Constraint, ctx: CandidateCtx) -> bool:
    """指定熟货必接：窗口开启前，拒绝「完单会越过 win_start」的单——避免被长单拖住、
    错过在线窗内 take_order 该熟货的时机。窗口内/已接则不限（SCHEDULE 抢占去接）。"""
    if ctx.is_reposition:
        return True
    cid = str(c.params.get("cargo_id") or "")
    if cid in ctx.st.require_accept_done:
        return True
    win = c.params.get("win")
    if not win:
        return True
    win_start = win[0]
    guard = int(c.params.get("guard_min", 15))
    # 仅在窗前生效：本单若会跑过 win_start（减 guard 余量），则拒，保持窗口期空闲可接熟货
    if ctx.now < win_start and ctx.finish > win_start - guard:
        return False
    return True


def _gate_require_idle_days(c: Constraint, ctx: CandidateCtx) -> bool:
    """整月歇业（off_days，落地路径 2b）：拒绝「finish 跨入月末歇业日」的单。

    calc _eval_off_days 要求 off-day 当天 active_minutes==0（含跨午夜单在该日的重叠分钟）。
    月末歇业日由 SCHEDULE 整天 wait，但**前一天若接了跨午夜的长单、其卸货会落进歇业日凌晨**→
    污染该 off-day（D001 day27 单跨进 day28 → off_days 3→2 → failed 10000）。本 GATE 在歇业日
    起点(idle_start)前夜直接拦掉会跨入的单，从根保证歇业日完全无活动。reposition 即时不跨午夜、
    且歇业日当天 SCHEDULE 已 wait，不拦。"""
    if ctx.is_reposition:
        return True
    days = int(c.params.get("days", 0))
    if days <= 0:
        return True
    for d in _idle_days(ctx.st, days):
        if _interval_overlap(ctx.now, ctx.finish, d * 1440, (d + 1) * 1440):
            return False
    return True


def _gate_timed_route(c: Constraint, ctx: CandidateCtx) -> bool:
    """deadline 前保护 timed_route：拒绝会越过当前 phase 最晚出发点的单。"""
    if ctx.is_reposition:
        return True
    phase = _current_route_phase(ctx.st, c)
    if phase is None:
        return True
    if phase.get("bind_cargo"):
        enter_after = int(phase.get("enter_after", phase.get("enter_by", 0)) or 0)
        enter_by = int(phase.get("enter_by", 0) or 0)
        guard = int(phase.get("guard_min", c.params.get("guard_min", 15)) or 15)
        if enter_after and ctx.now < enter_after and ctx.finish > enter_after - guard:
            return False
        if enter_by and ctx.now < enter_by <= ctx.finish + guard:
            return False
        return True
    loc = phase.get("loc")
    enter_by = phase.get("enter_by")
    if not loc or enter_by is None:
        return True
    guard = int(phase.get("guard_min", 15) or 15)
    rid = str(c.params.get("route_id") or "")
    idx = ctx.st.timed_route_progress.get(rid, 0)
    latest_arrive = _route_phase_latest_arrive(c, idx, ctx.speed, guard)
    drive = _drive_minutes(haversine_km(ctx.elat, ctx.elng, loc[0], loc[1]), ctx.speed)
    return ctx.finish + drive + guard <= latest_arrive


_GATE_HANDLERS: dict[str, Any] = {
    "forbid_cargo": _gate_forbid_cargo,
    "limit_distance": _gate_limit_distance,
    "confine_location": _gate_confine_location,
    "forbid_location": _gate_forbid_location,
    "forbid_action": _gate_forbid_action,
    "daily_presence": _gate_require_presence,
    "require_presence": _gate_require_presence,
    "require_accept": _gate_require_accept,
    "require_idle_days": _gate_require_idle_days,  # 2b：防跨午夜单污染月末歇业日
    "timed_route": _gate_timed_route,
    # home_event 仅作旧 IR 兼容；新解析不再生成该 kind。
}


# -- SCORE handlers：返回候选的期望罚分（元），并入 rate=(net-expected_penalty)/busy -- #
# 软约束（「尽量不/避免」）不硬拒，而是按罚分折算机会成本——利润够高就接、付罚分。


def _score_forbid_cargo(c: Constraint, ctx: CandidateCtx) -> float:
    """软类目：候选属软禁类目时返回边际罚分。
    边际口径 = 该规则累计未达上限则 per_order，否则 0（上限后再接不增罚，calc 同口径）。"""
    if ctx.is_reposition:
        return 0.0
    name = str(ctx.cargo.get("cargo_name", "") or "")
    names = c.params.get("names") or set()
    if not name or name not in names:
        return 0.0
    per = float(c.params.get("per_order", 0.0))
    cap = c.params.get("cap")
    taken = sum(ctx.st.soft_cargo_taken.get(n, 0) for n in names)
    if cap is not None and taken * per >= float(cap):
        return 0.0
    return per


def _score_limit_distance(c: Constraint, ctx: CandidateCtx) -> float:
    """软距离上限：候选 metric 超限则返回边际罚分（累计未达 cap 则 per_order，否则 0）。"""
    if ctx.is_reposition:
        return 0.0  # 每单距离针对接单的赴装/装卸，不约束 reposition
    metric = c.params.get("metric")
    max_km = c.params.get("max_km")
    if max_km is None:
        return 0.0
    val = ctx.deadhead_km if metric == "pickup" else ctx.haul_km
    if val <= float(max_km):
        return 0.0
    per = float(c.params.get("per_order", 0.0))
    cap = c.params.get("cap")
    taken = ctx.st.soft_distance_taken.get(str(metric), 0)
    if cap is not None and taken * per >= float(cap):
        return 0.0
    return per


def _score_limit_month_deadhead(c: Constraint, ctx: CandidateCtx) -> float:
    """软月度空驶上限（「仅对超出部分按公里计罚」）：本单把累计空驶推过上限的**增量超额** × per_km。
    与 calc 的 excess-based 同口径；达 cap 后边际 0。"""
    if ctx.is_reposition:
        return 0.0
    max_km = c.params.get("max_km")
    per = float(c.params.get("per_km", 0.0))
    if max_km is None or per <= 0:
        return 0.0
    before = ctx.st.month_deadhead_km
    over_before = max(0.0, before - float(max_km))
    over_after = max(0.0, before + ctx.deadhead_km - float(max_km))
    marginal = (over_after - over_before) * per
    cap = c.params.get("cap")
    if cap is not None:
        spent = min(float(cap), over_before * per)
        marginal = max(0.0, min(marginal, float(cap) - spent))
    return marginal


def _cargo_in_region(cargo: dict[str, Any], regions: set[str]) -> bool:
    """货源 start/end 城市名是否含任一 region（与 calc 的 _cargo_touches_region 同口径）。

    query 接口返回的 cargo 自带嵌套 `start.city`/`end.city`（如「广东省惠州市惠城区…」），
    calc 也按城市名字符串包含判定 → agent 直接字符串匹配即可精确对齐，无需坐标。"""
    spec = CargoMatchSpec(regions=frozenset(str(r).strip() for r in regions if str(r).strip()))
    return _cargo_match_hit(cargo, spec)


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


_CARGO_BOTH_SCOPES = frozenset({"cargo_start", "cargo_end"})


@dataclass(frozen=True)
class CargoMatchSpec:
    """货源地理匹配谓词。

    这是货源类约束的公共抽象：ban_region / temporal_region / visit_region
    本质都只是在 cargo.start.city / cargo.end.city 上做 region 字符串匹配。
    """
    regions: frozenset[str]
    scopes: frozenset[str] = _CARGO_BOTH_SCOPES
    active_window: tuple[int, int] | None = None


def _cargo_match_from_constraint(
    c: Constraint,
    default_scopes: frozenset[str] = _CARGO_BOTH_SCOPES,
) -> CargoMatchSpec:
    regions = frozenset(str(r).strip() for r in (c.params.get("regions") or set()) if str(r).strip())
    scopes_raw = c.params.get("scopes")
    scopes = frozenset(str(s).strip() for s in scopes_raw if str(s).strip()) if scopes_raw else default_scopes
    scopes = frozenset(s for s in scopes if s in _CARGO_BOTH_SCOPES)
    return CargoMatchSpec(regions=regions, scopes=scopes, active_window=c.params.get("active_window"))


def _cargo_matched_regions(cargo: dict[str, Any], spec: CargoMatchSpec) -> list[str]:
    start = cargo.get("start") or {}
    end = cargo.get("end") or {}
    cities = {
        "cargo_start": str(start.get("city", "") or ""),
        "cargo_end": str(end.get("city", "") or ""),
    }
    out: list[str] = []
    for r in sorted(spec.regions):
        if not r:
            continue
        for scope in spec.scopes:
            if r in cities.get(scope, ""):
                out.append(r)
                break
    return out


def _cargo_match_hit(
    cargo: dict[str, Any],
    spec: CargoMatchSpec,
    start_min: int | None = None,
    end_min: int | None = None,
) -> bool:
    if not spec.regions or not spec.scopes:
        return False
    if spec.active_window is not None:
        if start_min is None or end_min is None:
            return False
        win_start, win_end = spec.active_window
        if not _interval_overlap(start_min, end_min, int(win_start), int(win_end)):
            return False
    return bool(_cargo_matched_regions(cargo, spec))


def _cargo_region_scope_hit(cargo: dict[str, Any], regions: set[str], scopes: set[str]) -> bool:
    spec = CargoMatchSpec(
        regions=frozenset(str(r).strip() for r in regions if str(r).strip()),
        scopes=frozenset(str(s).strip() for s in scopes if str(s).strip()),
    )
    return _cargo_match_hit(cargo, spec)


def _score_ban_region(c: Constraint, ctx: CandidateCtx) -> float:
    """软禁地名货：候选货源装/卸城市命中禁地名时返回边际罚分。
    边际口径与 calc 一致——每接一单触及 region 罚 per_order；有 cap 且累计达 cap 后边际 0
    （超出免费可放弃）；无 cap（如 D001 惠州 800/次）则始终计罚，逼出「净收益>罚分才接」。"""
    if ctx.is_reposition:
        return 0.0
    spec = _cargo_match_from_constraint(c)
    if not _cargo_match_hit(ctx.cargo, spec):
        return 0.0
    per = float(c.params.get("per_order", 0.0))
    cap = c.params.get("cap")
    taken = sum(ctx.st.ban_region_taken.get(r, 0) for r in spec.regions)
    if cap is not None and taken * per >= float(cap):
        return 0.0
    return per


def _temporal_region_key(c: Constraint) -> str:
    regions = c.params.get("regions") or set()
    scopes = c.params.get("scopes") or set()
    win = c.params.get("active_window") or (0, 0)
    return f"{'|'.join(sorted(regions))}@{win[0]}-{win[1]}@{'|'.join(sorted(scopes))}"


def _temporal_region_hit(c: Constraint, cargo: dict[str, Any], start_min: int, end_min: int) -> bool:
    return _cargo_match_hit(cargo, _cargo_match_from_constraint(c), start_min, end_min)


def _score_temporal_region(c: Constraint, ctx: CandidateCtx) -> float:
    """带生效时间窗的地名货源软禁。

    例：三月四五号不往深圳跑/不派深圳货。动作执行区间与 active_window 交叠，且装/卸城市
    命中 scopes 指定地名时，返回边际罚分；cap 达顶后边际为 0。
    """
    if ctx.is_reposition or not _temporal_region_hit(c, ctx.cargo, ctx.now, ctx.finish):
        return 0.0
    per = float(c.params.get("per_order", 0.0))
    cap = c.params.get("cap")
    key = _temporal_region_key(c)
    taken = ctx.st.temporal_region_taken.get(key, 0)
    if cap is not None and taken * per >= float(cap):
        return 0.0
    return per


def _region_key(regions: set[str]) -> str:
    """visit_region 进度跟踪用的稳定 key（region 集合 → 排序拼接）。"""
    return "|".join(sorted(regions))


def _score_visit_region(c: Constraint, ctx: CandidateCtx) -> float:
    """货源地名到访配额的**影子价格奖励**（§5 凑够型 s=P×r/d）。返回负值（=奖励，抬高 rate）。

    仅当候选货命中 region、今天尚未记到访、且仍需更多天时给 s：
      s = penalty × 剩余需求天数 r / 剩余天数 d（含今日）。月初 r/d 小→顺路才接；月末 d 小→s 飙升必接。
    r≤0（已凑够）或 r>d（注定完不成，止损）→ 0。今天已记到访 → 0（calc 同日只算一天，不重复激励）。"""
    if ctx.is_reposition:
        return 0.0
    spec = _cargo_match_from_constraint(c)
    if not _cargo_match_hit(ctx.cargo, spec):
        return 0.0
    min_days = int(c.params.get("min_days", 0))
    penalty = float(c.params.get("penalty", 0.0))
    if min_days <= 0 or penalty <= 0:
        return 0.0
    day = ctx.now // 1440
    done = ctx.st.visit_region_days.get(_region_key(set(spec.regions)), set())
    if day in done:
        return 0.0
    r = min_days - len(done)
    if r <= 0:
        return 0.0
    d = max(1, ctx.st.total_days - day)  # 剩余天数（含今日）
    if r > d:
        return 0.0  # 注定完不成，止损不空驶
    return -(penalty * r / d)  # 负 = 奖励


def _score_cargo_region_count(c: Constraint, ctx: CandidateCtx) -> float:
    """计次型地名货月配额：每命中一次给固定影子价格 s=c，到 cap_count 后停止。

    返回负数是因为旧 SCORE handler 使用 expected_penalty 语义；负 expected_penalty 等价于奖励。
    """
    if ctx.is_reposition:
        return 0.0
    spec = _cargo_match_from_constraint(c)
    if not _cargo_match_hit(ctx.cargo, spec):
        return 0.0
    per = float(c.params.get("per_match", 0.0))
    if per <= 0:
        return 0.0
    key = _region_key(set(spec.regions))
    cap_count = c.params.get("cap_count")
    taken = ctx.st.cargo_region_count_done.get(key, 0)
    if cap_count is not None and taken >= int(cap_count):
        return 0.0
    return -per


_SCORE_HANDLERS: dict[str, Any] = {
    "forbid_cargo_soft": _score_forbid_cargo,
    "limit_distance_soft": _score_limit_distance,
    "limit_month_deadhead_soft": _score_limit_month_deadhead,
    "ban_region": _score_ban_region,
    "temporal_region": _score_temporal_region,
    "visit_region": _score_visit_region,
    "cargo_region_count": _score_cargo_region_count,
}


# -- SCHEDULE handlers：返回动作 dict 表示抢占，None 表示不触发 ------------------- #
# 紧迫度优先级：值小者先跑（复现今日 decide 闸门顺序）。


_SCHEDULE_PRIORITY: dict[str, int] = {
    "home_event": 5,           # 家事剧本（最高，时空预约不可错过）
    "timed_route": 6,          # 通用时序路点：先于整天休息，避免 off-day 抢占预约日
    "require_accept": 8,       # 指定熟货必接（次高）
    "require_idle_days": 10,   # 整天 wait
    "daily_presence": 15,      # 每日到点义务：deadline 前回到指定点
    "forbid_action": 20,       # 夜间窗内等待（顺带补休）
    "require_rest": 30,        # 傍晚补足连续休息
    "daily_rest_duty": 30,     # planner 休息义务：临近午夜时先休息，避免 query scan 吃掉余量
}


def _sched_require_idle_days(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    days = int(c.params.get("days", 0))
    if days > 0 and ctx.day in _idle_days(ctx.st, days):
        ctx.st.idle_days_done.add(ctx.day)
        return _wait(max(1, ctx.mins_to_midnight))
    return None


def _sched_forbid_action(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    windows = c.params.get("windows") or []
    win_end = BaselinePolicy._in_night_window_until(windows, ctx.now)
    if win_end is None:
        return None
    wait_min = max(1, win_end - ctx.now)
    rest_min = ctx.st.constraints.daily_rest_minutes()
    if wait_min >= rest_min > 0 and (ctx.now + wait_min) // 1440 == ctx.day:
        ctx.st.rested_days.add(ctx.day)
    return _wait(wait_min)


def _sched_require_rest(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    rest_min = int(c.params.get("minutes", 0))
    if rest_min <= 0 or ctx.day in ctx.st.rested_days:
        return None
    if ctx.mins_into_day >= 1440 - rest_min:
        wait_min = max(1, min(rest_min, ctx.mins_to_midnight))
        if wait_min >= rest_min:
            ctx.st.rested_days.add(ctx.day)
        return _wait(wait_min)
    return None


def _sched_daily_rest_duty(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    """planner 模式下的每日连续休息兜底。

    V2 会在 query 后用 beam 权衡接单/休息；但 query 本身会消耗仿真时间。
    当今天只剩 rest_min 加少量 guard 时，如果仍先 query，真实 wait 很容易不足 rest_min。
    这里在 query 前抢占，保证“最后一段可满足休息窗口”不会被感知耗时吃掉。
    """
    rest_min = int(c.params.get("minutes", 0))
    if rest_min <= 0 or ctx.day in ctx.st.rested_days:
        return None
    query_guard = max(0, _env_int("MANBANG_QUERY_GUARD_MIN", 15))
    if rest_min <= ctx.mins_to_midnight <= rest_min + query_guard:
        ctx.st.rested_days.add(ctx.day)
        return _wait(rest_min)
    return None


def _reposition_action(ctx: ScheduleCtx, plat: float, plng: float) -> dict[str, Any] | None:
    """直驶指定点（不限步长），坐标 round(.,2) 守住精度不变量；返回 reposition 动作。
    目标点本身已是 2 位小数，故落点精确等于该点（haversine=0 即满足 calc 半径）。"""
    tlat, tlng = round(plat, 2), round(plng, 2)
    dist = haversine_km(ctx.lat, ctx.lng, tlat, tlng)
    if dist <= 1e-6:
        return None
    ctx.st.month_deadhead_km += dist
    return {"action": "reposition", "params": {"latitude": tlat, "longitude": tlng}}


def _sched_require_presence(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    point = c.params.get("point")
    if not point:
        return None
    plat, plng = point
    radius = float(c.params.get("radius_km", 1.0))
    mode = "home_daily" if c.kind == "daily_presence" else c.params.get("mode")

    if mode == "home_daily":
        guard = int(c.params.get("guard_min", 15))
        deadline_abs = ctx.day * 1440 + int(c.params.get("deadline_min_of_day", 23 * 60))
        # 静默窗守夜：now 落在「23点至次日X点不接单」窗内且已到家 → 原地静止到窗尾（自行守夜，
        # 不依赖 forbid_action）。若不在家则由下方 must_leave 逻辑先驶回家。
        quiet = c.params.get("quiet_window")
        if quiet:
            qend = BaselinePolicy._in_night_window_until([quiet], ctx.now)
            if qend is not None and haversine_km(ctx.lat, ctx.lng, plat, plng) <= radius:
                return _wait(max(1, qend - ctx.now))
        if ctx.now >= deadline_abs:
            return None  # 过 deadline 且不在静默窗：由其它闸门接管
        home_dist = haversine_km(ctx.lat, ctx.lng, plat, plng)
        drive_home = _drive_minutes(home_dist, ctx.speed)
        if ctx.now + drive_home + guard < deadline_abs:
            return None  # 仍有富余，继续经营（GATE 保证每单可返家）
        # must_leave 已到：到家则 hold 到 deadline，否则直驶回家
        if home_dist <= radius:
            ctx.st.home_satisfied_days.add(ctx.day)
            return _wait(max(1, deadline_abs - ctx.now))
        return _reposition_action(ctx, plat, plng)

    if mode == "visit_quota":
        min_days = int(c.params.get("min_days", 0))
        if min_days <= 0 or ctx.day in ctx.st.visit_days_done:
            return None
        need = min_days - len(ctx.st.visit_days_done)
        if need <= 0:
            return None
        remaining_days = ctx.st.total_days - ctx.day  # 含今日
        if need < remaining_days:
            return None  # 配额仍宽裕，推迟到月末（到访点接近休息/歇业日，可廉价合并）
        # slack 收紧：今日必须到访
        if haversine_km(ctx.lat, ctx.lng, plat, plng) <= radius:
            return None  # 已在点内，由 decide 顶端标记当日
        return _reposition_action(ctx, plat, plng)

    return None


def _sched_require_accept(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    """指定熟货必接：在线窗 [win_start, win_end) 内且未接 → 直接发 take_order（引擎自动空驶赴装）。
    货离线/未上架则引擎返回 accepted=False（仅耗 1min），下个 step 重试，直至接到或窗口结束。"""
    cid = str(c.params.get("cargo_id") or "")
    if not cid or cid in ctx.st.require_accept_done:
        return None
    win = c.params.get("win")
    if not win:
        return None
    win_start, win_end = win
    if win_start <= ctx.now < win_end:
        # 不预标记 done：货未上架时引擎返回 accepted=False（仅耗 1min），下个 step 在窗内重试；
        # 一旦接到，引擎自动空驶赴装 + 跑 haul（数百分钟），下次 decide 已过 win_end → 不再触发。
        return {"action": "take_order", "params": {"cargo_id": cid}}
    return None


def _sched_home_event(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    """家事剧本：接配偶(pickup 停≥hold) → 返老家(home) → 静止到 leave_end。分阶段确定性编排。"""
    pickup = c.params.get("pickup")
    home = c.params.get("home")
    if not pickup or not home:
        return None
    win_start = int(c.params.get("win_start", 0))
    leave_end = int(c.params.get("leave_end", 0))
    if ctx.now >= leave_end:
        return None  # 事件结束，恢复经营
    radius = float(c.params.get("radius_km", 1.0))
    guard = int(c.params.get("guard_min", 15))
    hold = int(c.params.get("pickup_hold_min", 11))
    st = ctx.st
    at_pickup = haversine_km(ctx.lat, ctx.lng, pickup[0], pickup[1]) <= radius
    at_home = haversine_km(ctx.lat, ctx.lng, home[0], home[1]) <= radius

    # 阶段 0：窗前预定位——临近 win_start 才驶向 pickup（避免过早空驶）。
    # 注：本偏好窗前不可见，此分支对 D010 实际不触发；保留以兼容「偏好提前可见」的数据变体。
    if ctx.now < win_start:
        drive = _drive_minutes(haversine_km(ctx.lat, ctx.lng, pickup[0], pickup[1]), ctx.speed)
        if ctx.now + drive + guard < win_start:
            return None  # 仍宽裕，继续经营
        if at_pickup:
            return _wait(max(1, win_start - ctx.now))  # 已到接驳点，hold 到窗口开启
        return _reposition_action(ctx, pickup[0], pickup[1])

    # 阶段 1：接配偶——在 pickup 累计停留满 hold 分钟
    if st.family_pickup_arrival_min is None:
        if at_pickup:
            st.family_pickup_arrival_min = ctx.now
            return _wait(hold)  # 一次性停满（calc 要求 pickup_run>=10）
        return _reposition_action(ctx, pickup[0], pickup[1])
    if not st.family_home_arrived and ctx.now - st.family_pickup_arrival_min < hold and at_pickup:
        return _wait(max(1, hold - (ctx.now - st.family_pickup_arrival_min)))

    # 阶段 2：返老家
    if not at_home:
        return _reposition_action(ctx, home[0], home[1])
    st.family_home_arrived = True
    # 阶段 3：静止到 leave_end（避免 early_leave + 最小化 minutes_not_home）
    return _wait(max(1, leave_end - ctx.now))


def _sched_timed_route(c: Constraint, ctx: ScheduleCtx) -> dict[str, Any] | None:
    """通用时序路点：按 phase 顺序到达/等待。

    phase = {loc, enter_by, dwell_min, enter_after?}。handler 只执行当前 phase 的第一步；
    后续由滚动重规划继续推进。
    """
    rid = str(c.params.get("route_id") or "")
    if not rid:
        return None
    radius = float(c.params.get("radius_km", 2.0) or 2.0)
    guard_default = int(c.params.get("guard_min", 15) or 15)
    while True:
        phase = _current_route_phase(ctx.st, c)
        if phase is None:
            return None
        if phase.get("bind_cargo"):
            cid = str(phase.get("cargo_id") or "")
            if not cid:
                return None
            enter_after = int(phase.get("enter_after", phase.get("enter_by", 0)) or 0)
            enter_by = int(phase.get("enter_by", ctx.horizon_min) or ctx.horizon_min)
            if ctx.now < enter_after:
                return None
            if ctx.now < enter_by:
                return {"action": "take_order", "params": {"cargo_id": cid}}
            ctx.st.timed_route_progress[rid] = ctx.st.timed_route_progress.get(rid, 0) + 1
            continue
        loc = phase.get("loc")
        if not loc:
            return None
        enter_after = int(phase.get("enter_after", 0) or 0)
        enter_by = int(phase.get("enter_by", ctx.horizon_min) or ctx.horizon_min)
        dwell = int(phase.get("dwell_min", 0) or 0)
        hold_until = int(phase.get("hold_until", 0) or 0)
        guard = int(phase.get("guard_min", guard_default) or guard_default)
        at_loc = haversine_km(ctx.lat, ctx.lng, loc[0], loc[1]) <= radius
        idx = ctx.st.timed_route_progress.get(rid, 0)
        latest_arrive = _route_phase_latest_arrive(c, idx, ctx.speed, guard)

        if ctx.now < enter_after:
            drive = _drive_minutes(haversine_km(ctx.lat, ctx.lng, loc[0], loc[1]), ctx.speed)
            if ctx.now + drive + guard < enter_after:
                return None
            if at_loc:
                return _wait(max(1, enter_after - ctx.now))
            return _reposition_action(ctx, loc[0], loc[1])

        if not at_loc:
            drive = _drive_minutes(haversine_km(ctx.lat, ctx.lng, loc[0], loc[1]), ctx.speed)
            if phase.get("immediate_after_prev"):
                return _reposition_action(ctx, loc[0], loc[1])
            if ctx.now + drive + guard < latest_arrive:
                return None
            return _reposition_action(ctx, loc[0], loc[1])

        key = (rid, ctx.st.timed_route_progress.get(rid, 0))
        if key not in ctx.st.timed_route_arrivals:
            ctx.st.timed_route_arrivals[key] = ctx.now
        arrived = ctx.st.timed_route_arrivals[key]
        dwell_remaining = max(0, dwell - (ctx.now - arrived))
        hold_remaining = max(0, hold_until - ctx.now)
        wait_remaining = max(dwell_remaining, hold_remaining)
        if wait_remaining > 0:
            return _wait(max(1, wait_remaining))
        ctx.st.timed_route_progress[rid] = ctx.st.timed_route_progress.get(rid, 0) + 1


_SCHEDULE_HANDLERS: dict[str, Any] = {
    "require_idle_days": _sched_require_idle_days,
    "forbid_action": _sched_forbid_action,
    "require_rest": _sched_require_rest,
    "daily_rest_duty": _sched_daily_rest_duty,
    "daily_presence": _sched_require_presence,
    "require_presence": _sched_require_presence,
    "require_accept": _sched_require_accept,
    "home_event": _sched_home_event,
    "timed_route": _sched_timed_route,
}


def _schedule_priority(c: Constraint) -> int:
    """紧迫度：值小者先跑。daily_presence/旧 home_daily 早于夜间窗等待；到访最低。"""
    if c.kind == "require_presence":
        return 15 if c.params.get("mode") == "home_daily" else 45
    return _SCHEDULE_PRIORITY.get(c.kind, 100)


# --------------------------------------------------------------------------- #
# 在线货源密度图（第 4 步，§11）
# --------------------------------------------------------------------------- #




class DensityMap:
    """每司机一份的在线时空货源图（纯内存、零 token）。

    只靠司机沿途多次 ``query_cargo`` 增量累积（接口只返回最近 k 条，看不到全图）。
    累积量用「位置无关的 intrinsic_rate」= ``(price - cost_per_km*haul_km)/max(cost_time,1)``，
    归属到货源 **start 所在 cell·hour**；避免 deadhead 随司机位置变化而污染图。

    粒度：0.05°≈5.5km 网格 × hour-of-day(24)；另存全时段聚合做 fallback。
    """

    def __init__(self, grid_deg: float, alpha: float, ucb_c: float) -> None:
        self._grid = max(1e-4, grid_deg)
        self._alpha = min(1.0, max(0.01, alpha))      # EWMA 学习率
        self._ucb_c = max(0.0, ucb_c)
        # key=(ci,cj,hour) / (ci,cj) → [ewma_rate, ewma_count, n_obs]
        self._th: dict[tuple[int, int, int], list[float]] = {}
        self._agg: dict[tuple[int, int], list[float]] = {}
        self._global_rate = 0.0   # 全局 intrinsic_rate 的 EWMA（λ_global 代理）
        self._total_obs = 0

    def _cell(self, lat: float, lng: float) -> tuple[int, int]:
        return (int(math.floor(lat / self._grid)), int(math.floor(lng / self._grid)))

    def _center(self, ci: int, cj: int) -> tuple[float, float]:
        return ((ci + 0.5) * self._grid, (cj + 0.5) * self._grid)

    @staticmethod
    def _bump(d: dict, key, rate: float, count: float, alpha: float) -> None:
        cur = d.get(key)
        if cur is None:
            d[key] = [rate, count, 1.0]
        else:
            cur[0] += alpha * (rate - cur[0])
            cur[1] += alpha * (count - cur[1])
            cur[2] += 1.0

    def observe(self, hour: int, start_lat: float, start_lng: float,
                intrinsic_rate: float, cell_supply: float) -> None:
        ci, cj = self._cell(start_lat, start_lng)
        self._bump(self._th, (ci, cj, hour % 24), intrinsic_rate, cell_supply, self._alpha)
        self._bump(self._agg, (ci, cj), intrinsic_rate, cell_supply, self._alpha)
        if self._total_obs == 0:
            self._global_rate = intrinsic_rate
        else:
            self._global_rate += self._alpha * (intrinsic_rate - self._global_rate)
        self._total_obs += 1

    @property
    def total_obs(self) -> int:
        return self._total_obs

    @property
    def global_rate(self) -> float:
        return self._global_rate

    def local_rate(self, lat: float, lng: float, hour: int, n_min: int) -> tuple[float, int]:
        """返回 (期望 intrinsic_rate, 样本数)；当前 cell·hour 不足则退全时段，再退全局。"""
        ci, cj = self._cell(lat, lng)
        rec = self._th.get((ci, cj, hour % 24))
        if rec is not None and rec[2] >= n_min:
            return (rec[0], int(rec[2]))
        rec = self._agg.get((ci, cj))
        if rec is not None and rec[2] >= n_min:
            return (rec[0], int(rec[2]))
        return (self._global_rate, 0)

    def opportunity_rate(self, lat: float, lng: float, hour: int, n_min: int,
                         cost_per_km: float, ref_supply: float,
                         max_pickup_km: float) -> tuple[float, int]:
        """估计从当前位置出发，下一单可行装货起点的机会率。

        专业名称是 opportunity value approximation（机会价值近似）：不要求车辆正好停在货源
        start cell，而是在可接受赴装半径内扫描已观测 start cell，并扣掉去装货点的空驶成本。
        这样 terminal reward 更接近“下一单是否真能接”，而不是只看落点局部密度。
        """
        if self._total_obs <= 0 or max_pickup_km <= 0:
            return (0.0, 0)

        def _best(records: Iterable[tuple[tuple[int, int], list[float]]]) -> tuple[float, int]:
            best_rate = 0.0
            best_n = 0
            for (ci, cj), rec in records:
                rate, supply, n = rec[0], rec[1], rec[2]
                if n < n_min:
                    continue
                clat, clng = self._center(ci, cj)
                dist = haversine_km(lat, lng, clat, clng)
                if dist > max_pickup_km:
                    continue
                supply_factor = min(1.0, supply / max(ref_supply, 1.0))
                pickup_cost_rate = cost_per_km * dist / _REPO_AMORTIZE_MIN
                score = rate * supply_factor - pickup_cost_rate
                if score > best_rate:
                    best_rate = score
                    best_n = int(n)
            return (best_rate, best_n)

        hh = hour % 24
        exact = [((ci, cj), rec) for (ci, cj, h), rec in self._th.items() if h == hh]
        best = _best(exact)
        if best[1] >= n_min:
            return best
        return _best(self._agg.items())

    def best_cell(self, lat: float, lng: float, hour: int, n_min: int,
                  cost_per_km: float, ref_supply: float) -> tuple[float, float, float] | None:
        """在已观测 cell 中选 reposition 目标，返回 (中心lat, 中心lng, score)。

        score = ewma_rate × 充裕度 − 空驶折算 + UCB 探索加成；遍历全时段聚合表。
        约束（bbox/禁区/月度上限）由调用方 ``_reposition_target`` 复核，这里只给候选。
        """
        if self._total_obs <= 0:
            return None
        best: tuple[float, float, float] | None = None
        ln_total = math.log(self._total_obs + 1.0)
        for (ci, cj), rec in self._agg.items():
            rate, supply, n = rec[0], rec[1], rec[2]
            if n < n_min:
                continue
            clat, clng = self._center(ci, cj)
            dist = haversine_km(lat, lng, clat, clng)
            if dist <= 1e-6:
                continue  # 已在本 cell
            supply_factor = min(1.0, supply / max(ref_supply, 1.0))
            # 空驶成本折算到「元/分钟」与 rate 同量级：总成本 cost_per_km*dist 摊到一次接单的
            # 参考 busy 分钟（_REPO_AMORTIZE_MIN）上，故 penalty 随距离单调增——远 cell 自然劣后。
            penalty = cost_per_km * dist / _REPO_AMORTIZE_MIN
            ucb = self._ucb_c * math.sqrt(ln_total / n) if n > 0 else 0.0
            score = rate * supply_factor - penalty + ucb
            if best is None or score > best[2]:
                best = (clat, clng, score)
        return best


# --------------------------------------------------------------------------- #
# 每司机运行态
# --------------------------------------------------------------------------- #


@dataclass
class DriverRunState:
    rested_days: set[int] = field(default_factory=set)     # 已满足"每日连续休息"的日序
    month_deadhead_km: float = 0.0                          # 累计空驶（赴装 + reposition）
    empty_streak: int = 0                                   # 连续"无可接货"次数
    hotspot: tuple[float, float] | None = None             # 近期高价货源中心（用于逃离空区）
    last_pref_repr: str = ""
    constraints: "ConstraintSet" = field(default_factory=ConstraintSet)
    total_days: int = 31
    density: DensityMap | None = None                       # 在线密度图（门控开启时注入，§11）
    home_satisfied_days: set[int] = field(default_factory=set)  # 已在 deadline 前驶回家的日序（S1 home_daily）
    visit_days_done: set[int] = field(default_factory=set)      # 已到访指定点的自然日（S1 visit_quota）
    idle_days_done: set[int] = field(default_factory=set)       # 已完成整天歇业的自然日（P2 days_off）
    soft_cargo_taken: dict[str, int] = field(default_factory=dict)  # 已接软禁类目计数（S2 SCORE，按 cargo_name）
    soft_distance_taken: dict[str, int] = field(default_factory=dict)  # 已接超软距离上限计数（S2.b SCORE，按 metric）
    ban_region_taken: dict[str, int] = field(default_factory=dict)  # 已接禁地名货计数（S5 SCORE，按 region 字符串）
    temporal_region_taken: dict[str, int] = field(default_factory=dict)  # 已接带时窗禁地名货计数（按 region+window+scope）
    visit_region_days: dict[str, set[int]] = field(default_factory=dict)  # 已接某地名货的不同自然日（S7 visit_region 配额）
    cargo_region_count_done: dict[str, int] = field(default_factory=dict)  # 计次型地名货月配额完成次数（P2）
    require_accept_done: set[str] = field(default_factory=set)  # 已接的「指定熟货必接」cargo_id（S4 require_accept）
    family_pickup_arrival_min: int | None = None  # 家事剧本：首次抵达接驳点的仿真分钟（S4 home_event）
    family_home_arrived: bool = False             # 家事剧本：接配偶后是否已返抵老家（S4 home_event）
    timed_route_progress: dict[str, int] = field(default_factory=dict)  # timed_route：route_id -> 当前 phase 下标
    timed_route_arrivals: dict[tuple[str, int], int] = field(default_factory=dict)  # timed_route：phase 首次到达分钟


class BaselinePolicy:
    """纯算法决策：感知 → 候选评估 → 在 接单/休息/空驶 间择一。"""

    def __init__(self) -> None:
        self._speed = _env_float("MANBANG_REPO_SPEED", 60.0)
        self._cost_per_km = _env_float("MANBANG_COST_PER_KM", 3.0)
        self._horizon_min = _env_int("MANBANG_SIM_DAYS", 31) * 1440
        self._min_rate = _env_float("MANBANG_MIN_RATE", 0.0)          # 接单的最低 元/分钟
        self._refresh_wait = _env_int("MANBANG_REFRESH_WAIT", 90)     # 无货时等待刷新分钟
        self._escape_after = _env_int("MANBANG_ESCAPE_AFTER", 3)      # 连续空区后空驶逃离
        self._reposition_step_km = _env_float("MANBANG_REPO_STEP_KM", 60.0)
        self._horizon_guard_min = _env_int("MANBANG_HORIZON_GUARD", 30)  # 月末保护边际
        # 密度图（§11）：默认关，行为与 baseline 逐字节一致；A/B 达标后再设默认开。
        # MANBANG_USE_DENSITY 仅控制「观测累积 + 估值（local_rate）」；「reposition 选点（best_cell）」
        # 拆出独立门控 MANBANG_DENSITY_REPO（默认跟随 USE_DENSITY，显式置 0 可只开估值不改 reposition）。
        # A/B 结论：density 观测/估值零副作用，但 reposition 选点有害——会破坏时空硬约束
        # （D010 家事窗口被带离家），故 reposition 选点默认随 density 关。
        self._use_density = _env_int("MANBANG_USE_DENSITY", 0) == 1
        self._use_density_repo = _env_int("MANBANG_DENSITY_REPO", 1 if self._use_density else 0) == 1
        # S1 require_presence 子门控：A/B 达标（D009 回家 27000→0、D010 到访 3000→0，他人零回归，
        # failed=0），默认开；置 0 可回退到 S0（不解析对应基元，逐字节复现）。
        self._use_presence_home = _env_int("MANBANG_PRESENCE_HOME", 1) == 1
        self._use_presence_visit = _env_int("MANBANG_PRESENCE_VISIT", 1) == 1
        # S2 软类目 SCORE 子门控：A/B 达标（+271 net、failed=0、他人零回归），默认开；
        # 置 0 则「尽量不/避免」回退硬禁（逐字节复现 S0/S1）。
        self._use_soft_cargo = _env_int("MANBANG_SOFT_CARGO", 1) == 1
        # S2.b 软距离上限 SCORE 子门控：A/B 达标（+5886 net，D005/D008 受益、他人零回归、failed=0），默认开；
        # 置 0 回退（赴装上限硬 GATE、装卸上限不解析）。
        self._use_soft_distance = _env_int("MANBANG_SOFT_DISTANCE", 1) == 1
        # S3 LLM 解析基础约束子门控：**默认开**（LLM 主、正则兜底）——评测用默认门控值，须开才生效；
        # ban_region/daily_rest_duty 等均出自 LLM 路径。LLM 失败自动回退正则（优雅降级）。A/B 对照显式置 0。
        self._use_llm_parse = _env_int("MANBANG_USE_LLM_PARSE", 1) == 1
        # 泛化偏好解析架构灰度门控：宽 schema envelope + 确定性 compiler。默认关，避免影响当前
        # 已调过的 LLM/regex 解析路径；打开后优先走 general_preference_parser。
        self._use_general_pref_parse = _env_int("MANBANG_USE_GENERAL_PREF_PARSE", 1) == 1
        # S4 时空预约硬约束子门控（仅 LLM 路径下解析时生效）：默认开，置 0 则不产出对应基元。
        self._use_require_accept = _env_int("MANBANG_REQUIRE_ACCEPT", 1) == 1
        self._use_home_event = _env_int("MANBANG_HOME_EVENT", 1) == 1
        # S5 地名禁货（ban_region）：按货源 city 字符串匹配的软罚约束（对齐 calc _cargo_touches_region）。
        # **默认开**（A/B 验证 +2629、惠州完全回收），显式置 0 回退对照。
        self._use_ban_region = _env_int("MANBANG_BAN_REGION", 1) == 1
        # S6 全局规划器骨架（PLANNER 落地路径 1b）：滚动时域 + beam search + 统一目标函数。**默认开**
        # （A/B 验证 +8295、failed=0、D002 零回归）。接管「接单 vs 休息」联合决策：daily_rest 改用
        # 「高初始紧迫度休息义务」（§4.1）取代固定块，beam 前瞻当天能否补休、让高价单可抢占。
        # 歇业/夜间窗/家事等仍走 SCHEDULE 抢占（月配额留第 2 步）。显式置 0 回退贪心对照。
        self._use_planner = _env_int("MANBANG_USE_PLANNER", 1) == 1
        self._planner_beam_k = _env_int("MANBANG_PLANNER_BEAM_K", 6)  # beam 束宽
        self._planner_horizon_h = _env_int("MANBANG_PLANNER_HORIZON_H", 0)  # 前瞻小时；0=到当天午夜
        # P1 第二小步：新 beam 规划器 V2（planner_core/planner_beam，浮动 horizon + 保守终值）。
        # 灰度开关，**默认关**——A/B 对照 ≥37955 前不接管现网；import/异常一律回退现网逻辑。
        self._use_planner_v2 = _env_int("MANBANG_USE_PLANNER_V2", 0) == 1
        # V2 目标函数灰度：默认用真实净收益；置 1 后用 rate×参考时长排序，缓解短单密集区少接单。
        # 专业名称是 time-normalized objective（时间归一化目标函数）。默认关，避免影响当前稳定分。
        self._planner_rate_objective = _env_int("MANBANG_PLANNER_RATE_OBJECTIVE", 0) == 1
        self._planner_rate_objective_ref_min = max(
            1.0, _env_float("MANBANG_PLANNER_RATE_OBJECTIVE_REF_MIN", 600.0)
        )
        # V2 固定时段休息义务（如 D002 0–6 点停车熄火）：把 avoid_window→forbid_action 当休息义务
        # 接进 beam，让前瞻避开「接单跨进禁动窗」。仅 V2 路径生效，默认开；A/B 对照显式置 0。
        self._planner_fixed_rest = _env_int("MANBANG_PLANNER_FIXED_REST", 1) == 1
        # V2 诊断日志：默认关。开启后只写 trace，不参与决策；用于分析保守终值为何放弃高收益单。
        self._planner_trace = _env_int("MANBANG_PLANNER_TRACE", 0) == 1
        self._planner_trace_top_n = max(1, _env_int("MANBANG_PLANNER_TRACE_TOP_N", 5))
        # P4 最小产能终值：默认关。只使用运行时 query 写入的 density，不读原始数据。
        self._planner_terminal_density = _env_int("MANBANG_PLANNER_TERMINAL_DENSITY", 0) == 1
        self._planner_terminal_discount = max(0.0, min(1.0, _env_float("MANBANG_PLANNER_TERMINAL_DISCOUNT", 0.25)))
        self._planner_terminal_cap = max(0.0, _env_float("MANBANG_PLANNER_TERMINAL_CAP", 1200.0))
        self._planner_terminal_minutes_cap = max(0, _env_int("MANBANG_PLANNER_TERMINAL_MINUTES_CAP", 480))
        self._planner_terminal_order_net_ratio = max(
            0.0, _env_float("MANBANG_PLANNER_TERMINAL_ORDER_NET_RATIO", 0.5)
        )
        self._planner_terminal_order_rate_ratio = max(
            0.0, _env_float("MANBANG_PLANNER_TERMINAL_ORDER_RATE_RATIO", 0.5)
        )
        self._planner_terminal_pickup_radius_km = max(
            0.0, _env_float("MANBANG_PLANNER_TERMINAL_PICKUP_RADIUS_KM", 60.0)
        )
        # S7 地名到访配额（visit_region，落地路径第 2 步）：按货源 city 接货天数计（对齐 calc
        # _eval_required_region_cargo_days），影子价格 s=P×r/d 激励接该地货凑够 N 天。**默认开**
        # （A/B 验证 +11138：D002 增城凑够、连带休息/赴装改善），显式置 0 回退对照。
        self._use_visit_region = _env_int("MANBANG_VISIT_REGION", 1) == 1
        # S8 歇业日防污染（idle_gate，落地路径 2b）：给 require_idle_days 加 GATE，拒绝 finish 跨入
        # 月末歇业日的单，从根防「前夜跨午夜单污染 off-day」（D001 off_days 3→2 失守 10000）。
        # **默认开**（A/B 验证 +8513：D001 歇业凑够 3 天、failed 清零），显式置 0 回退对照。
        self._use_idle_gate = _env_int("MANBANG_IDLE_GATE", 1) == 1
        # 动态 λ 子门控：A/B 显示其拖累高产司机，默认关；仅密度 reposition 选点保留。
        self._use_lambda = _env_int("MANBANG_USE_LAMBDA", 0) == 1
        self._density_grid = _env_float("MANBANG_DENSITY_GRID", 0.05)
        self._density_alpha = _env_float("MANBANG_DENSITY_ALPHA", 0.3)
        self._density_ucb = _env_float("MANBANG_DENSITY_UCB", 0.5)
        self._density_nmin = _env_int("MANBANG_DENSITY_NMIN", 3)        # cell 可用最低样本数
        self._density_warmup = _env_int("MANBANG_DENSITY_WARMUP", 20)   # 全局冷启动观测数
        self._density_ref_supply = _env_float("MANBANG_DENSITY_REFSUPPLY", 5.0)
        self._state: dict[str, DriverRunState] = {}
        self._logger = logging.getLogger("agent.baseline")

    # -- 公共入口 ---------------------------------------------------------- #

    def decide(self, api: Any, driver_id: str, status: dict[str, Any]) -> dict[str, Any]:
        now = int(status.get("simulation_progress_minutes", 0))
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        st = self._state_for(driver_id, now)  # 累计空驶里程、连续无货次数、已经完成休息的日期和密度图
        self._refresh_constraints(st, status.get("preferences"), api, driver_id=driver_id)
        c = st.constraints

        day = now // 1440
        mins_into_day = now % 1440
        mins_to_midnight = 1440 - mins_into_day

        # 观测当前位置：落入到访/回家点半径即标记当日（与 calc 按 after-pos 计数同口径）
        self._mark_presence(st, lat, lng, day)

        # 闸门 1：月末——只休息，避免接超 horizon 的单白干（仿真边界，非偏好）
        if now >= self._horizon_min - self._horizon_guard_min:
            return _wait(max(1, mins_to_midnight))

        # 闸门 2-4：SCHEDULE handler 按紧迫度跑（整天歇业→夜间窗等待→傍晚补休），命中即抢占
        sctx = ScheduleCtx(
            st=st, now=now, day=day, mins_into_day=mins_into_day,
            mins_to_midnight=mins_to_midnight, lat=lat, lng=lng,
            cost_per_km=self._cost_per_km, horizon_min=self._horizon_min,
            speed=self._speed, horizon_guard=self._horizon_guard_min,
        )
        forced = self._run_schedule_handlers(st.constraints, sctx)
        if forced is not None:
            return forced

        # 感知：仅在需要做经营决策时才 query（query 会耗仿真时间）
        try:
            resp = api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
            items = resp.get("items", []) or []
        except Exception:  # noqa: BLE001 — 感知失败兜底为短暂等待
            return _wait(self._refresh_wait)

        # query_cargo 会消耗仿真时间。候选动作必须用 query 后的时间做 GATE/finish 评估，
        # 否则会出现“预测未跨入固定窗，真实执行因 query scan 延后而跨窗”的时间口径错位。
        try:
            post = api.get_driver_status(driver_id)
            now = int(post.get("simulation_progress_minutes", now))
            lat = float(post.get("current_lat", lat))
            lng = float(post.get("current_lng", lng))
        except Exception:  # noqa: BLE001 — 状态回读失败时保守沿用 query 前状态
            pass
        day = now // 1440
        mins_into_day = now % 1440
        mins_to_midnight = 1440 - mins_into_day
        sctx = ScheduleCtx(
            st=st, now=now, day=day, mins_into_day=mins_into_day,
            mins_to_midnight=mins_to_midnight, lat=lat, lng=lng,
            cost_per_km=self._cost_per_km, horizon_min=self._horizon_min,
            speed=self._speed, horizon_guard=self._horizon_guard_min,
        )
        forced = self._run_schedule_handlers(st.constraints, sctx)
        if forced is not None:
            return forced

        self._update_hotspot(st, now, items)
        if self._use_planner:
            if self._use_planner_v2:
                # 新 beam 规划器 V2（默认关，灰度 A/B）：返回 None 则回退到下方兜底。
                act = self._planner_v2_decide(st, now, lat, lng, day, mins_to_midnight, items)
            else:
                # 全局规划器（落地路径 1b 骨架）：在接单/休息间用统一目标函数 + 当天补休前瞻择优。
                # 返回 take_order/wait 则直接执行；返回 None（既不接单也不休息）落到下方空驶/等待兜底。
                act = self._planner_decide(st, now, lat, lng, day, mins_to_midnight, items)
            if act is not None:
                return act
        else:
            best = self._best_candidate(st, now, lat, lng, items)
            if best is not None:
                return self._commit_take(st, best)

        # 无可接货：累计空区计数 → 逃离 / 等待刷新
        st.empty_streak += 1
        if st.empty_streak >= self._escape_after:  # 当连续无货次数达到阈值后，尝试换位置。默认阈值为 3。
            move = self._reposition_target(st, now, lat, lng, c)
            # 逃离选点不得危及当日回家：若驶向该点后赶不回家，放弃逃离，改为受 must_leave 限制的等待
            if move is not None and self._escape_ok_for_home(st, now, move["lat"], move["lng"], move["dist_km"]):
                st.month_deadhead_km += move["dist_km"]
                st.empty_streak = 0
                return {"action": "reposition", "params": {"latitude": move["lat"], "longitude": move["lng"]}}
        # 等待刷新，但不跨过午夜（让休息/夜间闸门下一步接管）；
        # 若有回家约束，亦不睡过「must_leave」——确保 deadline 前能醒来驶回家。
        cap = min(self._refresh_wait, mins_to_midnight)
        ml = self._home_must_leave(st, now, lat, lng)
        if ml is not None and ml > now:
            cap = min(cap, ml - now)
        # 时空预约（熟货必接 / 家事剧本）：不睡过其唤醒点，确保 decide 在窗口/预定位时刻被触发。
        ew = self._event_must_wake(st, now, lat, lng)
        if ew is not None and ew > now:
            cap = min(cap, ew - now)
        return _wait(max(1, cap))

    # -- 候选评估 ---------------------------------------------------------- #

    def _eval_candidate(
        self, st: DriverRunState, now: int, lat: float, lng: float, item: dict[str, Any],
        gates: list[Constraint], scores: list[Constraint], min_rate: float,
        skip_score_kinds: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """单个候选货的可行性 + 经济性评分。可行返回完整 dict（含 rate/net/finish/终点/期望罚），
        否则 None。被 _feasible_candidates（planner 复用）与 _best_candidate（贪心）共享，确保口径一致。"""
        cargo = item.get("cargo") or {}
        cid = str(cargo.get("cargo_id", "")).strip()
        if not cid:
            return None
        try:
            price = float(cargo.get("price", 0.0))
            cost_time = int(cargo.get("cost_time_minutes", 0))
            start = cargo.get("start") or {}
            end = cargo.get("end") or {}
            slat, slng = float(start["lat"]), float(start["lng"])
            elat, elng = float(end["lat"]), float(end["lng"])
        except (KeyError, TypeError, ValueError):
            return None
        if cost_time < 0:
            return None

        deadhead_km = haversine_km(lat, lng, slat, slng)
        haul_km = haversine_km(slat, slng, elat, elng)

        # 时间可行性（与引擎一致）——先算 finish，供 forbid_action 的 GATE 使用
        dead_min = _drive_minutes(deadhead_km, self._speed)
        arrival = now + dead_min
        window = _parse_load_window(cargo)
        if window is not None:
            ls, le = window
            if arrival > le:
                return None  # 错过装货窗，必失败
            ready = ls if arrival < ls else arrival
        else:
            ready = arrival
        finish = ready + cost_time

        net = price - self._cost_per_km * (deadhead_km + haul_km)
        busy = (ready - now) + cost_time
        ctx = CandidateCtx(
            cargo=cargo, deadhead_km=deadhead_km, haul_km=haul_km,
            arrival=arrival, ready=ready, finish=finish,
            slat=slat, slng=slng, elat=elat, elng=elng,
            net=net, busy=busy, st=st, now=now, lat=lat, lng=lng,
            cost_per_km=self._cost_per_km, horizon_min=self._horizon_min,
            speed=self._speed, is_reposition=False,
        )

        # GATE：类目/距离/区域/禁区（这些不依赖 finish）
        _FINISH_DEP = ("forbid_action", "daily_presence", "require_presence", "require_idle_days")
        if any(not _GATE_HANDLERS[c.kind](c, ctx) for c in gates if c.kind not in _FINISH_DEP):
            return None

        # 月末 horizon（引擎边界守卫，留在循环内与内联路径同序）
        if finish > self._horizon_min - self._horizon_guard_min:
            return None

        # 依赖 finish 的 GATE（夜间窗 / 回家可达）放在 horizon 之后
        if any(not _GATE_HANDLERS[c.kind](c, ctx) for c in gates if c.kind in _FINISH_DEP):
            return None

        if net <= 0:
            return None
        # SCORE：把软约束的期望罚分并入 rate（空 SCORE → 0，与 S0/S1 逐字节兼容）
        expected_penalty = 0.0
        skip_score_kinds = skip_score_kinds or set()
        for c in scores:
            if c.kind in skip_score_kinds:
                continue
            expected_penalty += _SCORE_HANDLERS[c.kind](c, ctx)
        rate = (net - expected_penalty) / max(busy, 1)
        if rate < min_rate:
            return None
        return {"cargo_id": cid, "rate": rate, "net": net, "deadhead_km": deadhead_km,
                "haul_km": haul_km, "cargo_name": str(cargo.get("cargo_name", "") or ""),
                "cargo": cargo, "expected_penalty": expected_penalty, "finish": finish,
                "ready": ready, "busy": busy, "now": now, "elat": elat, "elng": elng,
                "now_day": now // 1440}

    def _feasible_candidates(
        self, st: DriverRunState, now: int, lat: float, lng: float, items: list[dict[str, Any]],
        skip_score_kinds: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """所有可行候选货的评分列表（保持 items 顺序）。供 planner 在「接单/休息」间统一权衡。"""
        cset = st.constraints
        assert isinstance(cset, ConstraintSet)
        gates = [c for c in cset.by_role.get("GATE", []) if _active(c, now)]
        scores = [c for c in cset.by_role.get("SCORE", []) if _active(c, now)]
        min_rate = self._lambda(st, now, lat, lng)
        out: list[dict[str, Any]] = []
        for item in items:
            r = self._eval_candidate(st, now, lat, lng, item, gates, scores, min_rate, skip_score_kinds)
            if r is not None:
                out.append(r)
        return out

    def _best_candidate(
        self, st: DriverRunState, now: int, lat: float, lng: float, items: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """候选 gate 走 GATE handler 注册表，按 rate（元/分钟）取最优（贪心路径，逐字节复现旧行为）。"""
        best: dict[str, Any] | None = None
        for cand in self._feasible_candidates(st, now, lat, lng, items):
            if best is None or cand["rate"] > best["rate"]:
                best = cand
        return best

    def _commit_take(self, st: DriverRunState, best: dict[str, Any]) -> dict[str, Any]:
        """接单成功后推进运行态（月度空驶 / 软约束 / 禁地名计数）并返回 take_order 动作。
        被贪心与 planner 共用，确保口径一致。"""
        st.month_deadhead_km += best["deadhead_km"]
        st.empty_streak = 0
        cname = best.get("cargo_name")
        if cname:  # S2：软禁类目接单数
            st.soft_cargo_taken[cname] = st.soft_cargo_taken.get(cname, 0) + 1
        for con in st.constraints.by_role.get("SCORE", []):
            if con.kind == "limit_distance_soft":  # S2.b：超软距离上限接单数（按 metric）
                m = con.params.get("metric"); mx = con.params.get("max_km")
                v = best["deadhead_km"] if m == "pickup" else best["haul_km"]
                if mx is not None and v > float(mx):
                    st.soft_distance_taken[str(m)] = st.soft_distance_taken.get(str(m), 0) + 1
            elif con.kind == "ban_region":  # S5：禁地名接单数（按 region，一单记一次命中）
                bc = best.get("cargo") or {}
                for rr in _cargo_matched_regions(bc, _cargo_match_from_constraint(con)):
                    st.ban_region_taken[rr] = st.ban_region_taken.get(rr, 0) + 1
                    break
            elif con.kind == "temporal_region":  # S5：带时窗地名禁货，按 region+window+scope 计数
                bc = best.get("cargo") or {}
                if _temporal_region_hit(con, bc, int(best.get("now", 0)), int(best.get("finish", 0))):
                    key = _temporal_region_key(con)
                    st.temporal_region_taken[key] = st.temporal_region_taken.get(key, 0) + 1
            elif con.kind == "visit_region":  # S7：记到访该地名货的自然日（凑配额）
                bc = best.get("cargo") or {}
                spec = _cargo_match_from_constraint(con)
                if _cargo_match_hit(bc, spec):
                    key = _region_key(set(spec.regions))
                    st.visit_region_days.setdefault(key, set()).add(best.get("now_day", 0))
            elif con.kind == "cargo_region_count":  # P2：计次型地名货配额
                bc = best.get("cargo") or {}
                spec = _cargo_match_from_constraint(con)
                if _cargo_match_hit(bc, spec):
                    key = _region_key(set(spec.regions))
                    cap_count = con.params.get("cap_count")
                    taken = st.cargo_region_count_done.get(key, 0)
                    if cap_count is None or taken < int(cap_count):
                        st.cargo_region_count_done[key] = taken + 1
        return {"action": "take_order", "params": {"cargo_id": best["cargo_id"]}}

    @staticmethod
    def _rest_duty(cset: ConstraintSet) -> tuple[int, float, float | None] | None:
        """取每日休息义务 (rest_min, per_day, cap)；无则 None。第 1 步仅支持单条 daily_rest。"""
        for c in cset.items:
            if c.kind == "daily_rest_duty":
                return (int(c.params.get("minutes", 0)),
                        float(c.params.get("per_day", 0.0)),
                        c.params.get("cap"))
        return None

    @staticmethod
    def _fixed_rest_windows(cset: ConstraintSet) -> list[tuple[int, int, float, float | None]]:
        """取固定时段休息(=停车熄火/夜间禁动)窗：[(win_start, win_end, per_day, cap), ...]。

        来自 avoid_window→forbid_action（解析时存了 rest_per_day/rest_cap）；无 rest_per_day 的
        forbid_action（如 source_daily_rest 固定块）不算固定窗休息，跳过。供 V2 beam 当义务前瞻。
        """
        out: list[tuple[int, int, float, float | None]] = []
        for c in cset.items:
            if c.kind != "forbid_action":
                continue
            per_day = c.params.get("rest_per_day")
            if not per_day:                       # 无罚分信息 → 非休息类禁动，不纳入
                continue
            for win in c.params.get("windows", []):
                ws, we = int(win[0]), int(win[1])
                out.append((ws, we, float(per_day), c.params.get("rest_cap")))
        return out

    def _planner_decide(self, st: DriverRunState, now: int, lat: float, lng: float,
                        day: int, mins_to_midnight: int,
                        items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """落地路径 1b 骨架：接单 vs 休息的统一目标函数决策 + 当天补休前瞻（§4.1）。

        beam 思想的浅层实例（前瞻 H=当天）：把「现在接最优单」与「现在休息」折成同尺度 Δ 比较——
        - 接单 Δ = 净收益 − 单笔罚 − (若接单挤掉今日休息 → 休息违约边际 per_day)
        - 休息 Δ = 0（占时但避违约）；今天来不及凑连续 rest_min 则横竖违约
        选 Δ 最大者。无世界模型，接单后只前瞻确定性补休、不前瞻未知落点货（continuation=0），
        规避 lookahead 估值覆辙。返回 take_order/wait；既不接单也不宜现在休息 → None（交兜底）。

        休息何时发生由统一目标自然决定：白天有正收益单 → 接单；凌晨无好单 → 休息（落低价时段）；
        临近 deadline 接单会挤掉休息时 → 仅高价单（net>per_day）抢占，否则转休息。
        """
        cset = st.constraints
        cands = self._feasible_candidates(st, now, lat, lng, items)
        best: dict[str, Any] | None = None
        for c in cands:
            if best is None or c["rate"] > best["rate"]:
                best = c
        duty = self._rest_duty(cset)

        # 无休息义务 / 今日已休 → 纯接单（与贪心同口径）
        if duty is None or day in st.rested_days:
            return self._commit_take(st, best) if best is not None else None

        rest_min, per_day, cap = duty
        # 休息违约边际成本（达 cap 后 0，与 calc 一致）：粗估已违约天数 = 已过天数 − 已休天数
        rest_cost = per_day
        if cap is not None and per_day > 0:
            approx_violated = max(0, day - len(st.rested_days))
            if approx_violated * per_day >= float(cap):
                rest_cost = 0.0

        can_rest_today = mins_to_midnight >= rest_min  # 今天剩余仍够一次连续 rest_min

        take_score: float | None = None
        if best is not None:
            net_after = best["net"] - best.get("expected_penalty", 0.0)
            slack_after = mins_to_midnight - best["busy"]  # 接单占用后当天剩余
            preempts_rest = can_rest_today and slack_after < rest_min  # 接单会挤掉今日休息
            take_score = net_after - (rest_cost if preempts_rest else 0.0)

        rest_score = 0.0 if can_rest_today else -rest_cost  # 来不及休 → 横竖违约

        # 接单更优且为正 → 接单
        if take_score is not None and take_score > rest_score and take_score > 0:
            return self._commit_take(st, best)
        # 否则今天还能休 → 一次性 wait rest_min（同日连续），标记当日已休
        if can_rest_today:
            st.rested_days.add(day)
            return _wait(rest_min)
        # 今天来不及补休：有正收益单仍接（已无休息可挤），否则交兜底
        if take_score is not None and take_score > 0:
            return self._commit_take(st, best)
        return None

    def _planner_v2_decide(self, st: DriverRunState, now: int, lat: float, lng: float,
                           day: int, mins_to_midnight: int,
                           items: list[dict[str, Any]]) -> dict[str, Any] | None:
        """P1 第二小步：beam（planner_core/planner_beam）+ 保守终值做「接单 vs 休息」决策。

        复用 _feasible_candidates 的候选（net/busy/终点/单笔罚已算好），仅替换决策内核：
        浮动 horizon（第二个义务 deadline，48h 封顶）+ beam top-K + 保守终值（营收记0、休息违约才扣）。
        import 失败 / 任何异常一律回退 None（不影响现网；本方法仅 MANBANG_USE_PLANNER_V2=1 才被调用）。
        """
        try:
            from planner_core import (PlannerState, Action, DailyRestConstraint,
                                       FixedWindowRestConstraint,
                                       VisitRegionQuotaConstraint,
                                       MonthDeadheadOverageConstraint,
                                       RequiredIdleDaysConstraint,
                                       CargoRegionCountRewardConstraint,
                                       TimedRouteConstraint)
            from planner_beam import beam_search, compute_horizon, explain_plan, SimProvider
        except Exception:
            try:
                from agent.planner_core import (PlannerState, Action, DailyRestConstraint,
                                                FixedWindowRestConstraint,
                                                VisitRegionQuotaConstraint,
                                                MonthDeadheadOverageConstraint,
                                                RequiredIdleDaysConstraint,
                                                CargoRegionCountRewardConstraint,
                                                TimedRouteConstraint)
                from agent.planner_beam import beam_search, compute_horizon, explain_plan, SimProvider
            except Exception:
                return None

        try:
            # 这些 P2 SCORE 已迁到 planner_core 约束；V2 候选基准分先跳过，避免双重计分。
            p2_core_score_kinds = {"visit_region", "limit_month_deadhead_soft", "cargo_region_count"}
            cands = self._feasible_candidates(st, now, lat, lng, items, p2_core_score_kinds)
            duty = self._rest_duty(st.constraints)

            orders: list[Any] = []
            cand_by_id: dict[str, dict[str, Any]] = {}
            for c in cands:
                net_after_penalty = float(c["net"]) - float(c.get("expected_penalty", 0.0))
                planner_base_net = net_after_penalty
                if self._planner_rate_objective:
                    planner_base_net = float(c.get("rate", 0.0) or 0.0) * self._planner_rate_objective_ref_min
                orders.append(Action(
                    "take_order", start_min=now, duration_min=int(c["busy"]),
                    dest=(float(c["elat"]), float(c["elng"])),
                    base_net=planner_base_net,
                    deadhead_km=float(c["deadhead_km"]), cargo_id=c["cargo_id"],
                    meta={"cargo": c.get("cargo") or {},
                          "raw_base_net": net_after_penalty,
                          "expected_penalty": float(c.get("expected_penalty", 0.0) or 0.0),
                          "rate": float(c.get("rate", 0.0) or 0.0),
                          "busy": int(c.get("busy", 0) or 0)},
                ))
                cand_by_id[c["cargo_id"]] = c

            constraints: list[Any] = []
            obligations: list[Any] = []
            root_resources: dict[str, Any] = {"rested_days": frozenset(st.rested_days)}

            for con in st.constraints.by_role.get("SCORE", []):
                if con.kind == "visit_region":
                    regions = set(con.params.get("regions") or set())
                    if regions:
                        key = _region_key(regions)
                        res_key = f"visit_region_days:{key}"
                        constraints.append(VisitRegionQuotaConstraint(
                            regions=regions,
                            min_days=int(con.params.get("min_days", 0) or 0),
                            penalty=float(con.params.get("penalty", 0.0) or 0.0),
                            total_days=st.total_days,
                            resource_key=res_key,
                        ))
                        root_resources[res_key] = frozenset(st.visit_region_days.get(key, set()))
                elif con.kind == "cargo_region_count":
                    regions = set(con.params.get("regions") or set())
                    if regions:
                        key = _region_key(regions)
                        res_key = f"cargo_region_count:{key}"
                        constraints.append(CargoRegionCountRewardConstraint(
                            regions=regions,
                            reward_per_match=float(con.params.get("per_match", 0.0) or 0.0),
                            cap_count=con.params.get("cap_count"),
                            resource_key=res_key,
                        ))
                        root_resources[res_key] = int(st.cargo_region_count_done.get(key, 0))
                elif con.kind == "limit_month_deadhead_soft":
                    max_km = con.params.get("max_km")
                    if isinstance(max_km, (int, float)):
                        constraints.append(MonthDeadheadOverageConstraint(
                            max_km=float(max_km),
                            per_km=float(con.params.get("per_km", 0.0) or 0.0),
                            cap=con.params.get("cap"),
                        ))
                        root_resources["month_deadhead_km"] = float(st.month_deadhead_km)

            full_day_rest = False
            for con in st.constraints.items:
                if con.kind != "require_idle_days":
                    continue
                idle_days = int(con.params.get("days", 0) or 0)
                idle_penalty = float(con.params.get("penalty", 0.0) or 0.0)
                if idle_days <= 0 or idle_penalty <= 0:
                    continue
                constraints.append(RequiredIdleDaysConstraint(
                    min_days=idle_days,
                    penalty=idle_penalty,
                    total_days=st.total_days,
                ))
                root_resources["idle_days_done"] = frozenset(st.idle_days_done)
                full_day_rest = True

            rest_min = 0
            if duty is not None:
                rest_min, per_day, cap = duty
                rc = DailyRestConstraint(min_minutes=int(rest_min),
                                         per_day_penalty=float(per_day), cap=cap)
                constraints.append(rc)
                obligations.append(rc)
            # 固定时段休息（如 0–6 点停车熄火）：当义务接进 beam，让前瞻能避开「接单跨进禁动窗」。
            fixed_wins = self._fixed_rest_windows(st.constraints) if self._planner_fixed_rest else []
            for (ws, we, pd, cp) in fixed_wins:
                fc = FixedWindowRestConstraint(win_start=ws, win_end=we,
                                               per_day_penalty=pd, cap=cp)
                constraints.append(fc)
                obligations.append(fc)

            # P5 timed_route：先把非 bind_cargo 的 waypoint/hold phase 交给 beam 统一比较。
            # 老 SCHEDULE/GATE 仍保留；V2 默认关，且真实 decide() 当前仍先跑 handler，避免突然切换调度权。
            route_constraints: list[Any] = []
            for con in st.constraints.items:
                if con.kind != "timed_route":
                    continue
                rid = str(con.params.get("route_id") or "")
                phases = con.params.get("phases") or []
                if not rid or not isinstance(phases, list):
                    continue
                tr = TimedRouteConstraint(
                    route_id=rid,
                    phases=phases,
                    penalty=float(con.params.get("penalty", 0.0) or 0.0),
                    radius_km=float(con.params.get("radius_km", 2.0) or 2.0),
                )
                progress = int(st.timed_route_progress.get(rid, 0) or 0)
                if progress >= len(tr.phases):
                    continue
                route_constraints.append(tr)
                constraints.append(tr)
                obligations.append(tr)
                root_resources[str(tr.progress_key)] = progress
                for (arid, idx), arr in st.timed_route_arrivals.items():
                    if arid == rid:
                        root_resources[f"{tr.arrival_prefix}{idx}"] = int(arr)

            if not orders and not (full_day_rest and now % 1440 == 0) and not route_constraints:
                # 无可行货：beam 无可编排 → 回退现网内核（保留其找货/睡觉兜底），避免 wait(90) 空耗
                return self._planner_decide(st, now, lat, lng, day, mins_to_midnight, items)

            root = PlannerState(time_min=now, lat=lat, lng=lng, resources=root_resources)
            provider = SimProvider(root_min=now, first_step_orders=orders,
                                   rest_minutes=int(rest_min),
                                   fixed_windows=tuple((w[0], w[1]) for w in fixed_wins),
                                   full_day_rest=full_day_rest,
                                   route_constraints=tuple(route_constraints),
                                   route_cost_per_km=self._cost_per_km,
                                   route_speed_kmh=self._speed)
            horizon_min = compute_horizon(root, obligations)
            terminal_fn = self._planner_v2_terminal_fn(st, root.time_min + horizon_min)
            plan, score = beam_search(root, provider, constraints, obligations,
                                      horizon_min=horizon_min, beam_k=self._planner_beam_k,
                                      terminal_fn=terminal_fn)
            if not plan:
                return None
            if self._planner_trace:
                self._log_planner_v2_trace(
                    now=now, lat=lat, lng=lng, root=root, provider=provider,
                    constraints=constraints, obligations=obligations,
                    plan=plan, score=score, explain_plan=explain_plan,
                    cand_by_id=cand_by_id, terminal_fn=terminal_fn,
                )
            a0 = plan[0]
            if a0.kind == "take_order":
                best = cand_by_id.get(a0.cargo_id or "")
                return self._commit_take(st, best) if best is not None else None
            if a0.kind == "rest":
                dur = max(1, int(a0.duration_min))
                if a0.meta.get("timed_route") is not None:
                    rid = str(a0.meta.get("timed_route") or "")
                    phase_idx = a0.meta.get("phase")
                    if isinstance(phase_idx, int):
                        st.timed_route_arrivals.setdefault((rid, phase_idx), now)
                    return _wait(dur)
                if a0.meta.get("idle_day") is not None:
                    st.idle_days_done.add(int(a0.meta["idle_day"]))
                    return _wait(dur)
                if rest_min > 0 and mins_to_midnight >= rest_min:   # 今天确实凑得满 → 标记当日已休
                    st.rested_days.add(day)
                    return _wait(dur)
                if fixed_wins:   # 固定窗义务：beam 选停驶即执行（停到窗结束，避开跨窗违约）；不动 rested_days
                    return _wait(dur)
                return None  # 仅有「任意连续休息」但今天来不及凑 → 交兜底（不误标记）
            if a0.kind == "reposition" and a0.dest is not None:
                st.month_deadhead_km += float(a0.deadhead_km)
                return {"action": "reposition",
                        "params": {"latitude": round(float(a0.dest[0]), 2),
                                   "longitude": round(float(a0.dest[1]), 2)}}
            return None
        except Exception:  # noqa: BLE001 — V2 任何异常都回退现网，绝不影响分数
            return None

    def _log_planner_v2_trace(self, *, now: int, lat: float, lng: float,
                              root: Any, provider: Any, constraints: list[Any],
                              obligations: list[Any], plan: list[Any], score: float,
                              explain_plan: Any, cand_by_id: dict[str, dict[str, Any]],
                              terminal_fn: Any = None) -> None:
        """写 V2 规划诊断日志。默认关闭；诊断异常只记录 debug，不影响决策。

        这是“可观测性/observability”代码：目的是解释 planner，不参与 planner。
        """
        try:
            root_actions = provider.candidates(root)
            previews: list[dict[str, Any]] = []
            for action in root_actions:
                one = explain_plan(root, [action], constraints, obligations, terminal_fn=terminal_fn)
                item: dict[str, Any] = {
                    "kind": action.kind,
                    "cargo_id": action.cargo_id,
                    "duration_min": action.duration_min,
                    "base_net": round(float(action.base_net), 2),
                    "one_step_total": one.get("total_score"),
                    "one_step_terminal": one.get("terminal_value"),
                }
                if action.cargo_id and action.cargo_id in cand_by_id:
                    cand = cand_by_id[action.cargo_id]
                    item.update({
                        "rate": round(float(cand.get("rate", 0.0)), 4),
                        "raw_net": round(float(cand.get("net", 0.0)), 2),
                        "expected_penalty": round(float(cand.get("expected_penalty", 0.0)), 2),
                    })
                if action.meta:
                    item["meta"] = dict(action.meta)
                previews.append(item)
            previews.sort(key=lambda x: float(x.get("one_step_total") or 0.0), reverse=True)
            payload = {
                "now": now,
                "lat": round(float(lat), 4),
                "lng": round(float(lng), 4),
                "beam_score": round(float(score), 2),
                "first_action": {
                    "kind": plan[0].kind if plan else None,
                    "cargo_id": plan[0].cargo_id if plan else None,
                    "duration_min": plan[0].duration_min if plan else None,
                    "meta": dict(plan[0].meta) if plan and plan[0].meta else {},
                },
                "chosen_plan": explain_plan(root, plan, constraints, obligations, terminal_fn=terminal_fn),
                "root_candidate_preview": previews[: self._planner_trace_top_n],
            }
            self._logger.info("planner_v2_trace %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("planner_v2_trace_failed error=%s", exc)

    def _planner_v2_terminal_fn(self, st: DriverRunState, horizon_end: int) -> Any:
        """P4 最小产能终值闭包；门控关闭或没有足够运行时观测时返回 None。

        专业名称是 terminal value shaping（终值塑形）：在 beam 截断处给落点一个保守残值，
        缓解“未来营收记 0”导致的过度保守。这里只使用在线 query 形成的 DensityMap。
        """
        if not self._planner_terminal_density:
            return None
        density = st.density
        if density is None or density.total_obs < self._density_warmup:
            return None
        pickup_radius_km = self._planner_terminal_pickup_radius(st)

        def _terminal(end_state: Any, root_min: int, obligations: list[Any],
                      last_action: Any = None) -> float:
            base = sum(ob.terminal_penalty(end_state, root_min) for ob in obligations)
            future = 0.0
            remain = min(max(0, horizon_end - int(end_state.time_min)), self._planner_terminal_minutes_cap)
            # 只给真实订单后的落点估未来机会。等待/空驶/timed_route 的落点缺少“下一单可接性”证据，
            # 高权重时会把车吸向局部高密度格子，反而增加空驶和丢单。
            action_kind = getattr(last_action, "kind", None)
            meta = getattr(last_action, "meta", {}) if last_action is not None else {}
            known_penalty = float(meta.get("expected_penalty", 0.0) or 0.0) if isinstance(meta, dict) else 0.0
            if remain > 0 and (last_action is None or action_kind == "take_order") and known_penalty <= 0:
                hour = (int(end_state.time_min) // 60) % 24
                rate, n = density.opportunity_rate(
                    float(end_state.lat), float(end_state.lng), hour, self._density_nmin,
                    self._cost_per_km, self._density_ref_supply, pickup_radius_km,
                )
                if n >= self._density_nmin and rate > 0:
                    future = min(self._planner_terminal_cap, rate * remain * self._planner_terminal_discount)
                    if last_action is not None:
                        action_rate = float(meta.get("rate", 0.0) or 0.0) if isinstance(meta, dict) else 0.0
                        if action_rate > 0 and self._planner_terminal_order_rate_ratio > 0:
                            future = min(future, action_rate * remain * self._planner_terminal_order_rate_ratio)
                        elif self._planner_terminal_order_net_ratio > 0:
                            proven = max(0.0, float(getattr(last_action, "base_net", 0.0) or 0.0))
                            if proven > 0:
                                future = min(future, proven * self._planner_terminal_order_net_ratio)
            return future - base

        return _terminal

    def _planner_terminal_pickup_radius(self, st: DriverRunState) -> float:
        """terminal 的下一单赴装半径；优先遵守司机偏好里的软上限。"""
        radius = self._planner_terminal_pickup_radius_km
        for con in st.constraints.by_role.get("SCORE", []):
            if con.kind != "limit_distance_soft" or con.params.get("metric") != "pickup":
                continue
            max_km = con.params.get("max_km")
            if isinstance(max_km, (int, float)):
                radius = min(radius, float(max_km))
        return radius

    def _run_schedule_handlers(self, cset: Any, sctx: ScheduleCtx) -> dict[str, Any] | None:
        """按紧迫度优先级跑 SCHEDULE handler，命中即返回抢占动作。"""
        if not isinstance(cset, ConstraintSet):
            return None
        scheds = [c for c in cset.by_role.get("SCHEDULE", []) if _active(c, sctx.now)]
        scheds.sort(key=_schedule_priority)
        for c in scheds:
            fn = _SCHEDULE_HANDLERS.get(c.kind)
            if fn is None:
                continue
            act = fn(c, sctx)
            if act is not None:
                return act
        return None

    @staticmethod
    def _mark_presence(st: DriverRunState, lat: float, lng: float, day: int) -> None:
        """当前位置落入 presence 点半径内则记当日（home/visit 各自集合）。"""
        cset = st.constraints
        if not isinstance(cset, ConstraintSet):
            return
        for c in cset.items:
            if c.kind not in ("daily_presence", "require_presence"):
                continue
            point = c.params.get("point")
            if not point:
                continue
            if haversine_km(lat, lng, point[0], point[1]) <= float(c.params.get("radius_km", 1.0)):
                if c.params.get("mode") == "visit_quota":
                    st.visit_days_done.add(day)
                elif c.kind == "daily_presence" or c.params.get("mode") == "home_daily":
                    st.home_satisfied_days.add(day)

    def _home_must_leave(self, st: DriverRunState, now: int, lat: float, lng: float) -> int | None:
        """回家约束的最晚出发绝对分钟 = deadline − 当前回家耗时（取各约束最早者）。"""
        cset = st.constraints
        if not isinstance(cset, ConstraintSet):
            return None
        earliest: int | None = None
        for c in cset.items:
            if not _is_daily_presence(c):
                continue
            point = c.params.get("point")
            if not point:
                continue
            deadline_abs = (now // 1440) * 1440 + int(c.params.get("deadline_min_of_day", 23 * 60))
            if now >= deadline_abs:
                continue
            guard = int(c.params.get("guard_min", 15))
            drive_home = _drive_minutes(haversine_km(lat, lng, point[0], point[1]), self._speed)
            ml = deadline_abs - drive_home - guard
            earliest = ml if earliest is None else min(earliest, ml)
        return earliest

    def _escape_ok_for_home(self, st: DriverRunState, now: int,
                            tlat: float, tlng: float, step_km: float) -> bool:
        """逃离 reposition 不得使「当日 deadline 前回不了家」。驶向 (tlat,tlng) 后须仍能回家。"""
        cset = st.constraints
        if not isinstance(cset, ConstraintSet):
            return True
        arrive = now + _drive_minutes(step_km, self._speed)
        for c in cset.items:
            if not _is_daily_presence(c):
                continue
            point = c.params.get("point")
            if not point:
                continue
            deadline_abs = (now // 1440) * 1440 + int(c.params.get("deadline_min_of_day", 23 * 60))
            if now >= deadline_abs:
                continue
            guard = int(c.params.get("guard_min", 15))
            drive_home = _drive_minutes(haversine_km(tlat, tlng, point[0], point[1]), self._speed)
            if arrive + drive_home + guard > deadline_abs:
                return False
        return True

    def _event_must_wake(self, st: DriverRunState, now: int, lat: float, lng: float) -> int | None:
        """时空预约唤醒点（取最早）：
        - require_accept：旧熟货 IR 的在线窗 win_start（须在窗开启时醒来去 take_order）。
        - home_event：旧家事 IR 的预定位出发时刻。
        - timed_route：当前 phase 的最晚出发点。"""
        cset = st.constraints
        if not isinstance(cset, ConstraintSet):
            return None
        earliest: int | None = None

        def _upd(t: int) -> None:
            nonlocal earliest
            if t > now:
                earliest = t if earliest is None else min(earliest, t)

        for c in cset.items:
            if c.kind == "require_accept":
                if str(c.params.get("cargo_id") or "") in st.require_accept_done:
                    continue
                win = c.params.get("win")
                if win:
                    _upd(int(win[0]))
            elif c.kind == "home_event":
                leave_end = int(c.params.get("leave_end", 0))
                if now >= leave_end:
                    continue
                win_start = int(c.params.get("win_start", 0))
                pickup = c.params.get("pickup")
                if not pickup:
                    continue
                guard = int(c.params.get("guard_min", 15))
                if now < win_start:
                    drive = _drive_minutes(haversine_km(lat, lng, pickup[0], pickup[1]), self._speed)
                    _upd(win_start - drive - guard)
                else:
                    _upd(now + 1)  # 事件进行中：每步都要醒来推进剧本
            elif c.kind == "timed_route":
                phase = _current_route_phase(st, c)
                if phase is None:
                    continue
                if phase.get("bind_cargo"):
                    enter_after = int(phase.get("enter_after", phase.get("enter_by", 0)) or 0)
                    enter_by = int(phase.get("enter_by", 0) or 0)
                    if now < enter_after:
                        _upd(enter_after)
                    elif enter_by and now < enter_by:
                        _upd(now + 1)
                    continue
                loc = phase.get("loc")
                if not loc:
                    continue
                guard = int(phase.get("guard_min", c.params.get("guard_min", 15)) or 15)
                rid = str(c.params.get("route_id") or "")
                idx = st.timed_route_progress.get(rid, 0)
                latest = _route_phase_latest_arrive(c, idx, self._speed, guard)
                drive = _drive_minutes(haversine_km(lat, lng, loc[0], loc[1]), self._speed)
                _upd(latest - drive - guard)
        return earliest

    def _lambda(self, st: DriverRunState, now: int, lat: float, lng: float) -> float:
        """动态接单阈值（元/分钟）。

        A/B 结论（§11.7）：动态 λ 会抬高高产司机（D003/D004/D007）的接单门槛，让其错过本该接的
        好单，净收益反降；而密度 reposition 选点（独立）对 D009 有效。故 λ 默认 **关**
        （独立子门控 ``MANBANG_USE_LAMBDA``），门控关时返回 ``_min_rate``，行为与 baseline 一致。
        """
        if not self._use_lambda or st.density is None or st.density.total_obs < self._density_warmup:
            return self._min_rate
        hour = (now // 60) % 24
        g = st.density.global_rate
        local, n = st.density.local_rate(lat, lng, hour, self._density_nmin)
        blend = 0.5 * g + 0.5 * local if n > 0 else g
        # 月末衰减：剩余时间越少阈值越低，避免月末挑食空等
        remain = max(0.0, self._horizon_min - now)
        decay = min(1.0, remain / max(1.0, self._horizon_min * 0.5))
        lam = 0.5 * blend * decay
        return max(self._min_rate, lam)

    # -- 辅助 -------------------------------------------------------------- #

    def _state_for(self, driver_id: str, now: int) -> DriverRunState:
        st = self._state.get(driver_id)
        if st is None:
            st = DriverRunState(total_days=max(1, self._horizon_min // 1440))
            if self._use_density or self._planner_terminal_density:
                st.density = DensityMap(self._density_grid, self._density_alpha, self._density_ucb)
            self._state[driver_id] = st
        return st

    def _refresh_constraints(self, st: DriverRunState, preferences: Any, api: Any = None, driver_id: str = "") -> None:
        if self._use_general_pref_parse:
            rep = json.dumps(_pref_records(preferences), ensure_ascii=False, sort_keys=True)
        else:
            rep = repr(preferences)
        if rep != st.last_pref_repr:
            st.constraints = _build_constraints(
                preferences, self._use_presence_home, self._use_presence_visit,
                self._use_soft_cargo, self._use_soft_distance,
                api=api, use_llm=self._use_llm_parse,
                use_require_accept=self._use_require_accept,
                use_home_event=self._use_home_event,
                use_ban_region=self._use_ban_region,
                use_planner=self._use_planner,
                use_visit_region=self._use_visit_region,
                use_idle_gate=self._use_idle_gate,
                use_general_parser=self._use_general_pref_parse,
                driver_id=driver_id,
            )
            st.last_pref_repr = rep

    def _update_hotspot(self, st: DriverRunState, now: int, items: list[dict[str, Any]]) -> None:
        """更新冷启动用的最高价热点；门控开启时同步喂密度图（按 start cell·hour 累积）。"""
        best_price = -1.0
        spot: tuple[float, float] | None = None
        hour = (now // 60) % 24
        # 统计每个 start-cell 的货源条数，作为充裕度（仅本次 query 的局部视图）
        supply: dict[tuple[float, float], int] = {}
        for item in items:
            cargo = item.get("cargo") or {}
            try:
                price = float(cargo.get("price", 0.0))
                start = cargo.get("start") or {}
                slat, slng = float(start["lat"]), float(start["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            if price > best_price:
                best_price, spot = price, (slat, slng)
            if st.density is not None:
                key = (round(slat, 1), round(slng, 1))
                supply[key] = supply.get(key, 0) + 1
        if spot is not None:
            st.hotspot = spot
        if st.density is None:
            return
        for item in items:
            cargo = item.get("cargo") or {}
            try:
                price = float(cargo.get("price", 0.0))
                cost_time = max(1, int(cargo.get("cost_time_minutes", 0)))
                start = cargo.get("start") or {}
                end = cargo.get("end") or {}
                slat, slng = float(start["lat"]), float(start["lng"])
                elat, elng = float(end["lat"]), float(end["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            haul_km = haversine_km(slat, slng, elat, elng)
            # 位置无关 intrinsic_rate：不含 deadhead（deadhead 随司机位置变，会污染图）
            intrinsic = (price - self._cost_per_km * haul_km) / cost_time
            cell_supply = float(supply.get((round(slat, 1), round(slng, 1)), 1))
            st.density.observe(hour, slat, slng, intrinsic, cell_supply)

    def _reposition_target(
        self, st: DriverRunState, now: int, lat: float, lng: float, c: Any
    ) -> dict[str, Any] | None:
        # 目标点：门控开启且密度图有候选 → 密度选点；否则回退到最高价热点（冷启动）。
        target = self._density_reposition_target(st, lat, lng)
        if target is None:
            if st.hotspot is None:
                return None
            target = st.hotspot
        return self._step_toward(st, now, lat, lng, target, c)

    def _density_reposition_target(
        self, st: DriverRunState, lat: float, lng: float
    ) -> tuple[float, float] | None:
        # reposition 选点独立门控：关时退回 hotspot 老逻辑（估值/observe 不受影响）。
        if not self._use_density_repo:
            return None
        if st.density is None or st.density.total_obs < self._density_warmup:
            return None
        cell = st.density.best_cell(
            lat, lng, 0, self._density_nmin, self._cost_per_km, self._density_ref_supply
        )
        if cell is None:
            return None
        return (cell[0], cell[1])

    def _step_toward(
        self, st: DriverRunState, now: int, lat: float, lng: float,
        target: tuple[float, float], c: Any
    ) -> dict[str, Any] | None:
        tlat, tlng = target
        dist = haversine_km(lat, lng, tlat, tlng)
        if dist <= 1e-6:
            return None
        # 按上限步长靠近目标
        frac = min(1.0, self._reposition_step_km / dist)
        # 坐标四舍五入到 2 位小数，与编排器日志 / 收益脚本的精度对齐：
        # 引擎内部用全精度坐标算 ceil 耗时，但日志只落 2 位小数，calc 据此重算。
        # 若发出全精度坐标，舍入会把 haversine 推过整分钟边界，导致 reposition/接单
        # 耗时校验差 1 分钟而失败（D003/D009 即此因）。统一精度后整条链一致。
        nlat = round(lat + (tlat - lat) * frac, 2)
        nlng = round(lng + (tlng - lng) * frac, 2)
        step_km = haversine_km(lat, lng, nlat, nlng)
        drive_min = _drive_minutes(step_km, self._speed)
        # 尊重区域 / 禁入 / 月度空驶上限：复用 GATE handler，构造 reposition 形态 ctx
        # （start=end=目标点，deadhead=step_km）。坐标舍入留在本函数，绝不进 handler。
        rctx = CandidateCtx(
            cargo={}, deadhead_km=step_km, haul_km=0.0,
            arrival=0, ready=0, finish=now + drive_min,
            slat=nlat, slng=nlng, elat=nlat, elng=nlng,
            net=0.0, busy=drive_min, st=st, now=now, lat=lat, lng=lng,
            cost_per_km=self._cost_per_km, horizon_min=self._horizon_min,
            speed=self._speed, is_reposition=True,
        )
        if isinstance(c, ConstraintSet):
            for cons in c.by_role.get("GATE", []):
                if not _GATE_HANDLERS[cons.kind](cons, rctx):
                    return None
        return {"lat": nlat, "lng": nlng, "dist_km": step_km}

    @staticmethod
    def _in_night_window_until(windows: list[tuple[int, int]], now: int) -> int | None:
        """若 ``now`` 落在某夜间禁动窗内，返回应等待到的绝对分钟（取最晚窗尾）。"""
        latest: int | None = None
        for d in (now // 1440 - 1, now // 1440):
            base = d * 1440
            for s, e in windows:
                ws, we = base + s, base + e
                if ws <= now < we:
                    latest = we if latest is None else max(latest, we)
        return latest

    @staticmethod
    def _hits_night_window(windows: list[tuple[int, int]], a: int, b: int) -> bool:
        if a >= b or not windows:
            return False
        for d in range(a // 1440 - 1, b // 1440 + 2):
            base = d * 1440
            for s, e in windows:
                ws, we = base + s, base + e
                if max(a, ws) < min(b, we):
                    return True
        return False


def _wait(minutes: int) -> dict[str, Any]:
    return {"action": "wait", "params": {"duration_minutes": max(1, int(minutes))}}
