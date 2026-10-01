# -*- coding: utf-8 -*-
"""Taxonomy envelope parser/compiler for executable preference constraints.

This module is intentionally small: the LLM produces the generalized taxonomy
envelope, and this compiler maps that semantic envelope into the Constraint IR
already consumed by ``BaselinePolicy.decide()``.
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[2]
_CACHE_DIR = Path(os.environ.get("MANBANG_TAXONOMY_CACHE_DIR") or (PROJECT_ROOT / ".ab_tmp" / "taxonomy_cache"))
_CACHE_VERSION = "taxonomy_itinerary_enrichment_v2"


@dataclass
class ConstraintSpec:
    kind: str
    params: dict[str, Any]
    roles: tuple[str, ...]
    active_window: tuple[int, int] | None = None
    source: str = "taxonomy_llm"


@dataclass
class ParseAudit:
    source_index: int | None
    category: str
    subcategory: str
    emitted: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    constraints: list[ConstraintSpec]
    audit: list[ParseAudit]
    raw: dict[str, Any] | None = None


def _load_eval_helpers() -> tuple[Any, Any, dict[str, Any], Any]:
    try:
        from eval_taxonomy_parse import ENVELOPE_SCHEMA, build_system_prompt, build_user_prompt, extract_json
    except Exception:
        from agent.eval_taxonomy_parse import ENVELOPE_SCHEMA, build_system_prompt, build_user_prompt, extract_json
    return build_system_prompt, build_user_prompt, ENVELOPE_SCHEMA, extract_json


def _taxonomy_runtime_schema(base_schema: dict[str, Any]) -> dict[str, Any]:
    """Extend eval schema with itinerary, needed by executable timed_route constraints."""
    schema = copy.deepcopy(base_schema)
    place_record_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "aliases": {"type": ["array", "null"], "items": {"type": "string"}},
            "kind": {"type": ["string", "null"]},
            "lat": {"type": ["number", "null"]},
            "lng": {"type": ["number", "null"]},
            "representative_lat": {"type": ["number", "null"]},
            "representative_lng": {"type": ["number", "null"]},
            "admin_radius_km": {"type": ["number", "null"]},
            "defined_in": {"type": ["string", "null"]},
            "referenced_in": {"type": ["array", "null"], "items": {"type": "string"}},
            "note": {"type": ["string", "null"]},
        },
        "required": [
            "key", "aliases", "kind", "lat", "lng",
            "representative_lat", "representative_lng", "admin_radius_km",
            "defined_in", "referenced_in", "note",
        ],
        "additionalProperties": False,
    }
    if isinstance(schema.get("properties"), dict):
        schema["properties"]["place_registry"] = {
            "type": ["array", "null"],
            "items": place_record_schema,
        }
        required = schema.get("required")
        if isinstance(required, list) and "place_registry" not in required:
            required.append("place_registry")
    constraint = (
        schema.get("properties", {})
        .get("preferences", {})
        .get("items", {})
        .get("properties", {})
        .get("constraint", {})
    )
    props = constraint.get("properties") if isinstance(constraint, dict) else None
    required = constraint.get("required") if isinstance(constraint, dict) else None
    if not isinstance(props, dict) or not isinstance(required, list):
        return schema
    step_schema = {
        "type": "object",
        "properties": {
            "seq": {"type": ["integer", "null"]},
            "action": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"]},
            "raw_time_expr": {"type": ["string", "null"]},
            "time_constraint": {
                "type": ["object", "null"],
                "properties": {
                    "type": {"type": ["string", "null"]},
                    "start": {"type": ["string", "null"]},
                    "end": {"type": ["string", "null"]},
                },
                "required": ["type", "start", "end"],
                "additionalProperties": False,
            },
            "location": {
                "type": ["object", "null"],
                "properties": {"lat": {"type": ["number", "null"]}, "lng": {"type": ["number", "null"]}},
                "required": ["lat", "lng"],
                "additionalProperties": False,
            },
            "named_location": {"type": ["string", "null"]},
            "place_role": {"type": ["string", "null"]},
            "arrive_before": {"type": ["string", "null"]},
            "depart_after": {"type": ["string", "null"]},
            "dwell_min_minutes": {"type": ["number", "null"]},
            "must_be_static_until": {"type": ["string", "null"]},
            "place_ref": {"type": ["string", "null"]},
            "immediate_after_prev": {"type": ["boolean", "null"]},
        },
        "required": [
            "seq", "action", "date", "raw_time_expr", "time_constraint",
            "location", "named_location", "place_role",
            "arrive_before", "depart_after", "dwell_min_minutes",
            "must_be_static_until", "place_ref", "immediate_after_prev",
        ],
        "additionalProperties": False,
    }
    props["itinerary"] = {
        "type": ["object", "null"],
        "properties": {"steps": {"type": "array", "items": step_schema}},
        "required": ["steps"],
        "additionalProperties": False,
    }
    if "itinerary" not in required:
        required.append("itinerary")
    return schema


def _norm_place_name(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text.strip() if not ch.isspace())


def _valid_lat_lng(lat: Any, lng: Any) -> tuple[float, float] | None:
    lat_num, lng_num = _as_num(lat), _as_num(lng)
    if lat_num is None or lng_num is None:
        return None
    if not (-90 <= lat_num <= 90 and -180 <= lng_num <= 180):
        return None
    return (round(lat_num, 6), round(lng_num, 6))


def _place_names(*values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if isinstance(value, list):
            candidates = value
        else:
            candidates = [value]
        for item in candidates:
            name = _norm_place_name(item)
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _put_place(
    registry: dict[str, dict[str, Any]],
    names: list[str],
    point: tuple[float, float] | None,
    kind: Any = None,
    admin_radius_km: Any = None,
) -> None:
    if point is None or not names:
        return
    record = {
        "lat": point[0],
        "lng": point[1],
        "kind": str(kind or ""),
        "admin_radius_km": _as_num(admin_radius_km),
        "names": set(names),
    }
    for name in names:
        existing = registry.get(name)
        if existing is None:
            registry[name] = record
            continue
        if existing.get("lat") is None or existing.get("lng") is None:
            existing.update(record)
        existing.setdefault("names", set()).update(names)


def _registry_entries(raw_registry: Any) -> list[dict[str, Any]]:
    if isinstance(raw_registry, list):
        return [item for item in raw_registry if isinstance(item, dict)]
    if isinstance(raw_registry, dict):
        places = raw_registry.get("places")
        source = places if isinstance(places, dict) else raw_registry
        entries = []
        for key, value in source.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("key", key)
                entries.append(item)
        return entries
    return []


def _build_place_registry(obj: dict[str, Any], prefs: list[Any]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for entry in _registry_entries(obj.get("place_registry") or obj.get("driver_place_registry") or obj.get("locations")):
        kind = entry.get("kind")
        point = _valid_lat_lng(entry.get("lat"), entry.get("lng"))
        if point is None:
            point = _valid_lat_lng(entry.get("representative_lat"), entry.get("representative_lng"))
        names = _place_names(
            entry.get("key"),
            entry.get("name"),
            entry.get("named_area"),
            entry.get("place_ref"),
            entry.get("aliases"),
        )
        _put_place(registry, names, point, kind, entry.get("admin_radius_km"))

    for item in prefs:
        c = _constraint(item) if isinstance(item, dict) else {}
        region = c.get("region")
        if isinstance(region, dict):
            point = _valid_lat_lng(region.get("center_lat"), region.get("center_lng"))
            names = _place_names(region.get("place_ref"), region.get("named_area"))
            _put_place(registry, names, point, region.get("place_kind"), region.get("admin_radius_km"))
        itin = c.get("itinerary")
        steps = itin.get("steps") if isinstance(itin, dict) else None
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                loc = step.get("location")
                point = None
                if isinstance(loc, dict):
                    point = _valid_lat_lng(loc.get("lat"), loc.get("lng"))
                if point is None:
                    point = _valid_lat_lng(step.get("lat"), step.get("lng"))
                names = _place_names(step.get("place_ref"), step.get("named_location"))
                _put_place(registry, names, point, step.get("place_kind"), step.get("admin_radius_km"))
    return registry


def _match_place(registry: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any] | None:
    for name in names:
        if name in registry:
            return registry[name]
    for name in names:
        for key, record in registry.items():
            aliases = record.get("names") if isinstance(record.get("names"), set) else set()
            all_names = {key, *aliases}
            if any(name in candidate or candidate in name for candidate in all_names):
                return record
    return None


def enrich_place_coordinates(obj: dict[str, Any]) -> dict[str, Any]:
    """Fill region/itinerary coordinates from the driver-level place registry."""
    prefs = obj.get("preferences") if isinstance(obj, dict) else None
    if not isinstance(prefs, list):
        return obj
    registry = _build_place_registry(obj, prefs)
    if not registry:
        return obj

    for item in prefs:
        if not isinstance(item, dict):
            continue
        c = _constraint(item)
        region = c.get("region")
        if isinstance(region, dict):
            point = _valid_lat_lng(region.get("center_lat"), region.get("center_lng"))
            record = _match_place(registry, _place_names(region.get("place_ref"), region.get("named_area")))
            if point is None and record is not None:
                region["center_lat"] = record["lat"]
                region["center_lng"] = record["lng"]
            if record is not None and region.get("admin_radius_km") is None and record.get("admin_radius_km") is not None:
                region["admin_radius_km"] = record["admin_radius_km"]

        itin = c.get("itinerary")
        steps = itin.get("steps") if isinstance(itin, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            loc = step.get("location")
            point = _valid_lat_lng(loc.get("lat"), loc.get("lng")) if isinstance(loc, dict) else None
            if point is not None:
                continue
            record = _match_place(registry, _place_names(step.get("place_ref"), step.get("named_location")))
            if record is not None:
                step["location"] = {"lat": record["lat"], "lng": record["lng"]}
    return obj


def _as_num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _as_int(v: Any) -> int | None:
    return int(v) if isinstance(v, (int, float)) else None


def _penalty(item: dict[str, Any]) -> tuple[float, float | None]:
    p = item.get("penalty") if isinstance(item.get("penalty"), dict) else {}
    amount = _as_num(p.get("raw_penalty_amount"))
    cap = _as_num(p.get("raw_penalty_cap"))
    return (float(amount or 0.0), cap if cap and cap > 0 else None)


def _constraint(item: dict[str, Any]) -> dict[str, Any]:
    c = item.get("constraint")
    return c if isinstance(c, dict) else {}


def _time_to_min(text: Any) -> int | None:
    if not isinstance(text, str) or ":" not in text:
        return None
    try:
        hh, mm = text.strip().split(":", 1)
        h, m = int(hh), int(mm)
    except ValueError:
        return None
    if not (0 <= h <= 24 and 0 <= m < 60):
        return None
    return h * 60 + m


def _daily_window(c: dict[str, Any]) -> tuple[int, int] | None:
    tw = c.get("time_window")
    if not isinstance(tw, dict):
        return None
    start = _time_to_min(tw.get("start"))
    end = _time_to_min(tw.get("end"))
    if start is None or end is None:
        return None
    if bool(tw.get("crosses_midnight")) or end <= start:
        end += 1440
    return (start, end) if end > start else None


def _region(c: dict[str, Any]) -> dict[str, Any]:
    r = c.get("region")
    return r if isinstance(r, dict) else {}


def _named_area(c: dict[str, Any]) -> str:
    r = _region(c)
    for key in ("named_area", "place_ref"):
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _region_scopes(c: dict[str, Any]) -> set[str]:
    role = str(_region(c).get("place_role") or "").strip()
    if role == "pickup":
        return {"cargo_start"}
    if role == "dropoff":
        return {"cargo_end"}
    return {"cargo_start", "cargo_end"}


def _point(c: dict[str, Any]) -> tuple[float, float] | None:
    r = _region(c)
    lat, lng = _as_num(r.get("center_lat")), _as_num(r.get("center_lng"))
    if lat is None or lng is None:
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return (round(lat, 2), round(lng, 2))


def _radius(c: dict[str, Any], default: float = 1.0) -> float:
    r = _region(c)
    for key in ("radius_km", "tolerance_km", "admin_radius_km"):
        v = _as_num(r.get(key))
        if v is not None and v > 0:
            return float(v)
    return default


def _value_set(c: dict[str, Any]) -> list[str]:
    values = c.get("value_set")
    if isinstance(values, list):
        return [str(v).strip() for v in values if str(v).strip()]
    cargo = c.get("cargo")
    if isinstance(cargo, dict):
        cats = cargo.get("categories")
        if isinstance(cats, list):
            return [str(v).strip() for v in cats if str(v).strip()]
        ids = cargo.get("specific_ids")
        if isinstance(ids, list):
            return [str(v).strip() for v in ids if str(v).strip()]
    return []


def _period_dates(c: dict[str, Any]) -> list[int]:
    period = c.get("period")
    raw = period.get("specific_dates") if isinstance(period, dict) else None
    out: list[int] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        text = str(item)
        try:
            out.append(int(text[-2:]))
        except ValueError:
            continue
    return sorted({d for d in out if 1 <= d <= 31})


def _date_text_to_day(text: Any) -> int | None:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if len(s) >= 10:
        try:
            return int(s[8:10])
        except ValueError:
            return None
    try:
        day = int(s[-2:])
    except ValueError:
        return None
    return day if 1 <= day <= 31 else None


def _active_window_for_dates(days: list[int]) -> tuple[int, int] | None:
    if not days:
        return None
    return ((days[0] - 1) * 1440, days[-1] * 1440)


def _compile_time_rest(item: dict[str, Any], c: dict[str, Any], audit: ParseAudit) -> list[ConstraintSpec]:
    amount, cap = _penalty(item)
    sub = item.get("subcategory")
    if sub == "daily_continuous_rest":
        fixed = _daily_window(c)
        if fixed is not None:
            audit.emitted.append("forbid_action")
            return [ConstraintSpec(
                "forbid_action",
                {"windows": [fixed], "rest_per_day": amount, "rest_cap": cap},
                ("SCHEDULE", "GATE"),
            )]
        hours = _as_num(c.get("scalar_value")) or 0.0
        minutes = int(hours * 60) if str(c.get("scalar_unit") or "") == "hour" else int(hours)
        if minutes <= 0:
            audit.warnings.append("daily_rest missing minutes")
            return []
        audit.emitted.extend(["forbid_action", "daily_rest_duty"])
        return [
            ConstraintSpec("forbid_action", {"windows": [(0, minutes)], "source_daily_rest": True}, ("SCHEDULE", "GATE")),
            ConstraintSpec("daily_rest_duty", {"minutes": minutes, "per_day": amount, "cap": cap}, ("REST", "SCHEDULE")),
        ]
    if sub == "daily_forbidden_window":
        win = _daily_window(c)
        if win is None:
            audit.warnings.append("forbidden_window missing time_window")
            return []
        audit.emitted.append("forbid_action")
        return [ConstraintSpec("forbid_action", {"windows": [win], "rest_per_day": amount, "rest_cap": cap}, ("SCHEDULE", "GATE"))]
    if sub == "monthly_full_rest_days":
        q = c.get("quota") if isinstance(c.get("quota"), dict) else {}
        days = _as_int(q.get("count")) or _as_int(c.get("scalar_value")) or 0
        if days <= 0:
            audit.warnings.append("monthly_full_rest_days missing days")
            return []
        audit.emitted.append("require_idle_days")
        return [ConstraintSpec("require_idle_days", {"days": days, "penalty": amount}, ("SCHEDULE",))]
    return []


def _compile_cargo(item: dict[str, Any], c: dict[str, Any], audit: ParseAudit) -> list[ConstraintSpec]:
    names = set(_value_set(c))
    if not names:
        audit.warnings.append("cargo constraint missing names")
        return []
    amount, cap = _penalty(item)
    if item.get("subcategory") == "cargo_forbid" and item.get("hardness") == "hard":
        audit.emitted.append("forbid_cargo_soft")
        return [ConstraintSpec("forbid_cargo_soft", {"names": names, "per_order": amount, "cap": cap}, ("SCORE",))]
    audit.emitted.append("forbid_cargo_soft")
    return [ConstraintSpec("forbid_cargo_soft", {"names": names, "per_order": amount, "cap": cap}, ("SCORE",))]


def _compile_geo(item: dict[str, Any], c: dict[str, Any], audit: ParseAudit) -> list[ConstraintSpec]:
    amount, cap = _penalty(item)
    sub = item.get("subcategory")
    if sub == "geo_named_place_forbid":
        area = _named_area(c)
        if not area:
            audit.warnings.append("geo_named_place_forbid missing named_area")
            return []
        days = _period_dates(c)
        params = {"regions": {area}, "scopes": _region_scopes(c), "per_order": amount, "cap": cap}
        win = _active_window_for_dates(days)
        if win is not None:
            params["active_window"] = win
            audit.emitted.append("temporal_region")
            return [ConstraintSpec("temporal_region", params, ("SCORE",))]
        audit.emitted.append("ban_region")
        return [ConstraintSpec("ban_region", params, ("SCORE",))]
    if sub in {"geo_must_outside", "geo_must_inside"}:
        pt = _point(c)
        if pt is None:
            audit.warnings.append(f"{sub} missing coordinate")
            return []
        if sub == "geo_must_outside":
            audit.emitted.append("forbid_location")
            return [ConstraintSpec("forbid_location", {"zones": [(pt[0], pt[1], _radius(c, 10.0))]}, ("GATE",))]
    if sub == "geo_homing":
        pt = _point(c)
        win = _daily_window(c)
        if pt is None or win is None:
            audit.warnings.append("geo_homing missing point/window")
            return []
        audit.emitted.append("daily_presence")
        return [ConstraintSpec("daily_presence", {
            "point": pt, "radius_km": _radius(c, 1.0),
            "deadline_min_of_day": win[0],
            "guard_min": 15,
            "quiet_window": win,
        }, ("GATE", "SCHEDULE"))]
    return []


def _compile_mileage(item: dict[str, Any], c: dict[str, Any], audit: ParseAudit) -> list[ConstraintSpec]:
    amount, cap = _penalty(item)
    max_km = _as_num(c.get("scalar_value"))
    if max_km is None or max_km <= 0:
        audit.warnings.append("mileage missing scalar_value")
        return []
    sub = item.get("subcategory")
    if sub == "mileage_deadhead_monthly_cap":
        audit.emitted.append("limit_month_deadhead_soft")
        return [ConstraintSpec("limit_month_deadhead_soft", {"max_km": max_km, "per_km": amount, "cap": cap}, ("SCORE",))]
    metric = "pickup" if sub == "mileage_deadhead_single_cap" else "haul"
    if metric == "haul" and cap is not None:
        audit.warnings.append("ignored capped haul soft constraint")
        return []
    audit.emitted.append("limit_distance_soft")
    return [ConstraintSpec("limit_distance_soft", {"metric": metric, "max_km": max_km, "per_order": amount, "cap": cap}, ("SCORE",))]


def _compile_point_visit(item: dict[str, Any], c: dict[str, Any], audit: ParseAudit) -> list[ConstraintSpec]:
    amount, cap = _penalty(item)
    sub = item.get("subcategory")
    if sub == "visit_frequency_quota":
        area = _named_area(c)
        q = c.get("quota") if isinstance(c.get("quota"), dict) else {}
        min_days = _as_int(q.get("count")) or _as_int(c.get("scalar_value")) or 0
        if area and min_days > 0:
            audit.emitted.append("visit_region")
            return [ConstraintSpec("visit_region", {"regions": {area}, "min_days": min_days, "penalty": amount}, ("SCORE",))]
        pt = _point(c)
        if pt is not None and min_days > 0:
            audit.emitted.append("require_presence")
            return [ConstraintSpec("require_presence", {
                "mode": "visit_quota", "point": pt, "radius_km": _radius(c, 1.0), "min_days": min_days,
            }, ("SCHEDULE",))]
        audit.warnings.append("visit_frequency_quota missing region/point or quota")
    if sub == "designated_order_must_take":
        ids = _value_set(c)
        if not ids:
            audit.warnings.append("designated_order missing cargo id")
            return []
        specs = []
        for cid in ids:
            specs.append(ConstraintSpec("timed_route", {
                "route_id": hashlib.sha1(cid.encode("utf-8")).hexdigest()[:12],
                "phases": [{"bind_cargo": True, "cargo_id": cid, "enter_after": 0, "enter_by": 31 * 1440}],
                "penalty": amount, "radius_km": 2.0, "guard_min": 15,
            }, ("GATE", "SCHEDULE")))
        audit.emitted.extend(["timed_route"] * len(specs))
        return specs
    if sub == "adhoc_complex_itinerary":
        route = _compile_itinerary(item, c, amount)
        if route is not None:
            audit.emitted.append("timed_route")
            skipped = _as_int(route.params.get("skipped_steps")) or 0
            if skipped > 0:
                audit.warnings.append(f"itinerary skipped {skipped} incomplete step(s)")
            phases = route.params.get("phases") if isinstance(route.params, dict) else None
            if isinstance(phases, list) and any(
                isinstance(ph, dict) and ph.get("enter_by") is None and ph.get("hold_until") is None
                for ph in phases
            ):
                audit.warnings.append("itinerary contains phase without deadline")
            return [route]
        audit.warnings.append("itinerary missing executable steps")
    return []


def _dt_to_min(text: Any) -> int | None:
    if not isinstance(text, str) or len(text) < 16:
        return None
    try:
        from datetime import datetime
        epoch = datetime(2026, 3, 1, 0, 0, 0)
        normalized = text[:19].replace("T", " ")
        dt = datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
        return int((dt - epoch).total_seconds() // 60)
    except Exception:
        return None


def _day_start_min(day: int) -> int:
    return (day - 1) * 1440


def _step_date_day(step: dict[str, Any], period_days: list[int]) -> int | None:
    day = _date_text_to_day(step.get("date"))
    if day is not None:
        return day
    return period_days[0] if len(period_days) == 1 else None


def _time_constraint_bounds(step: dict[str, Any], day: int | None) -> tuple[int | None, int | None]:
    tc = step.get("time_constraint")
    if not isinstance(tc, dict):
        return (None, None)
    start = _dt_to_min(tc.get("start"))
    end = _dt_to_min(tc.get("end"))
    if start is None and day is not None:
        start_min = _time_to_min(tc.get("start"))
        if start_min is not None:
            start = _day_start_min(day) + start_min
    if end is None and day is not None:
        end_min = _time_to_min(tc.get("end"))
        if end_min is not None:
            end = _day_start_min(day) + end_min
    typ = str(tc.get("type") or "").strip()
    if typ == "anytime_on_date" and day is not None:
        return (_day_start_min(day), _day_start_min(day) + 1440)
    if typ == "before":
        return (None, end)
    if typ == "after":
        return (start, None)
    if typ in {"between", "exact_hold_until"}:
        return (start, end)
    return (start, end)


def _compile_itinerary(item: dict[str, Any], c: dict[str, Any], amount: float) -> ConstraintSpec | None:
    itin = c.get("itinerary")
    steps = itin.get("steps") if isinstance(itin, dict) else None
    if not isinstance(steps, list):
        return None
    phases: list[dict[str, Any]] = []
    skipped_steps = 0
    period_days = _period_dates(c)
    for step in steps:
        if not isinstance(step, dict):
            continue
        loc = step.get("location")
        pt = None
        if isinstance(loc, dict):
            lat, lng = _as_num(loc.get("lat")), _as_num(loc.get("lng"))
            if lat is not None and lng is not None:
                pt = (round(lat, 2), round(lng, 2))
        if pt is None:
            lat, lng = _as_num(step.get("lat")), _as_num(step.get("lng"))
            if lat is not None and lng is not None:
                pt = (round(lat, 2), round(lng, 2))
        if pt is None:
            skipped_steps += 1
            continue
        ph: dict[str, Any] = {"loc": pt}
        for src, dst in (
            ("arrive_before", "enter_by"),
            ("depart_after", "enter_after"),
            ("must_be_static_until", "hold_until"),
        ):
            v = _dt_to_min(step.get(src))
            if v is not None:
                ph[dst] = v
        day = _step_date_day(step, period_days)
        tc_start, tc_end = _time_constraint_bounds(step, day)
        if "enter_after" not in ph and tc_start is not None:
            ph["enter_after"] = tc_start
        if "enter_by" not in ph and tc_end is not None:
            ph["enter_by"] = tc_end
        if "hold_until" not in ph:
            tc = step.get("time_constraint")
            if isinstance(tc, dict) and str(tc.get("type") or "").strip() == "exact_hold_until" and tc_end is not None:
                ph["hold_until"] = tc_end
        dwell = _as_int(step.get("dwell_min_minutes"))
        if dwell is not None:
            ph["dwell_min"] = max(0, dwell)
        if phases:
            ph["immediate_after_prev"] = bool(step.get("immediate_after_prev"))
        phases.append(ph)
    if not phases:
        return None
    for idx, ph in enumerate(phases[:-1]):
        if ph.get("enter_by") is not None:
            continue
        downstream = [
            int(p[k])
            for p in phases[idx + 1:]
            for k in ("enter_after", "enter_by", "hold_until")
            if isinstance(p.get(k), int)
        ]
        if downstream:
            ph["enter_by"] = min(downstream)
    deadlines = [int(p[k]) for p in phases for k in ("enter_by", "hold_until") if isinstance(p.get(k), int)]
    if not deadlines and period_days:
        day_end = _day_start_min(period_days[-1]) + 1440
        phases[-1]["enter_by"] = day_end
        deadlines.append(day_end)
    active_start = max(0, min(deadlines) - 48 * 60) if deadlines else 0
    active_end = max(deadlines) + 24 * 60 if deadlines else 31 * 1440
    src = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return ConstraintSpec("timed_route", {
        "route_id": hashlib.sha1(src.encode("utf-8")).hexdigest()[:12],
        "phases": phases,
        "penalty": amount,
        "radius_km": 2.0,
        "guard_min": 15,
        "skipped_steps": skipped_steps,
    }, ("GATE", "SCHEDULE"), active_window=(active_start, active_end))


def compile_envelope(obj: dict[str, Any]) -> ParseResult:
    prefs = obj.get("preferences") if isinstance(obj, dict) else None
    constraints: list[ConstraintSpec] = []
    audit: list[ParseAudit] = []
    if not isinstance(prefs, list):
        return ParseResult([], [ParseAudit(None, "", "", warnings=["missing preferences array"])], obj)
    obj = enrich_place_coordinates(obj)
    prefs = obj.get("preferences")
    for item in prefs:
        if not isinstance(item, dict):
            continue
        c = _constraint(item)
        a = ParseAudit(
            source_index=_as_int(item.get("source_index")),
            category=str(item.get("category") or ""),
            subcategory=str(item.get("subcategory") or ""),
        )
        cat = item.get("category")
        if cat == "TIME_REST":
            out = _compile_time_rest(item, c, a)
        elif cat == "CARGO_CATEGORY":
            out = _compile_cargo(item, c, a)
        elif cat == "GEO_SPATIAL":
            out = _compile_geo(item, c, a)
        elif cat == "MILEAGE_DISTANCE":
            out = _compile_mileage(item, c, a)
        elif cat == "POINT_VISIT_TASK":
            out = _compile_point_visit(item, c, a)
        else:
            out = []
            a.warnings.append(f"unsupported category {cat}")
        constraints.extend(out)
        audit.append(a)
    return ParseResult(constraints, audit, obj)


def parse_preferences(preferences: Any, api: Any = None, driver_id: str = "") -> ParseResult:
    if api is None:
        return ParseResult([], [ParseAudit(None, "", "", warnings=["missing api"])])
    if not isinstance(preferences, list) or not preferences:
        return ParseResult([], [])
    build_system_prompt, build_user_prompt, envelope_schema, extract_json = _load_eval_helpers()
    envelope_schema = _taxonomy_runtime_schema(envelope_schema)
    pref_key = json.dumps(preferences, ensure_ascii=False, sort_keys=True, default=str)
    system = (
        build_system_prompt()
        + "\n\n（务必输出合法 json 对象；envelope 的每个字段都要出现，用不到的填 null，不要省略 key。"
        "同时输出顶层 place_registry，先汇总该司机全部偏好里出现的地名、别名、坐标和行政区代表点；"
        "后续 constraint.region.place_ref / constraint.itinerary.steps[].place_ref 均应引用该注册表。"
        "一次性行程/办事/赴宴/取物/盘库必须输出 constraint.itinerary.steps；"
        "每个 step 尽量填 date、raw_time_expr、time_constraint。"
        "time_constraint.type 只能表达为 anytime_on_date、before、after、between、exact_hold_until 或 null；"
        "“当天/某日停一趟/办事”用 anytime_on_date，“上午/下午/某点前”用 before/between 归一化为 start/end；"
        "多步骤行程按执行顺序输出，前置取物/经过步骤没有明确停留到某时刻时不要填 must_be_static_until。）"
    )
    cache_key = hashlib.sha1((driver_id + "|" + pref_key + "|" + system + "|" + _CACHE_VERSION).encode("utf-8")).hexdigest()
    cache_path = _CACHE_DIR / f"{cache_key}.pkl"
    if os.environ.get("MANBANG_TAXONOMY_CACHE", "1") == "1" and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, ParseResult):
                return cached
        except Exception:
            pass
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_user_prompt(driver_id or "UNKNOWN", preferences)},
        ],
        "temperature": 0,
        "enable_thinking": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "preference_envelope",
                "strict": True,
                "schema": envelope_schema,
            },
        },
    }
    data = api.model_chat_completion(payload)
    content = data["choices"][0]["message"]["content"]
    obj = extract_json(content)
    if obj is None:
        return ParseResult([], [ParseAudit(None, "", "", warnings=["invalid model json"])])
    result = compile_envelope(obj)
    if os.environ.get("MANBANG_TAXONOMY_CACHE", "1") == "1":
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
    return result
