# -*- coding: utf-8 -*-
"""用真实模型(api.model_chat_completion)测试偏好分类 schema 的解析能力。

流程：
  1. system prompt = preference_taxonomy_schema.json 去掉样例(examples)后的全部定义 + 输出格式约定。
  2. 对每个司机，把它的【全部偏好】一次性喂给模型（带序号），让模型做指代消解，
     并把多谓词偏好拆成多条 envelope。
  3. 不信任模型 source_index，按 category/subcategory 语义对齐 golden，分层比对：
       覆盖率 / category 命中 / subcategory 命中 / 关键字段逐字段命中（主指标）。
  4. 新旧两个偏好集分别汇总 + 总计。

用法：
  python eval_taxonomy_parse.py            # 真跑 qwen（需 DASHSCOPE_API_KEY）
  python eval_taxonomy_parse.py --dry-run  # 不调模型，只检查 prompt/对齐/打分管线
  python eval_taxonomy_parse.py --verbose  # 打印每条 miss 的字段
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parents[2]  # .../manbang
RELEASE_SERVER = PROJECT_ROOT / "release_20260529" / "demo" / "server"
DEFAULT_CONFIG = PROJECT_ROOT / "demo" / "server" / "config" / "config.json"
SCHEMA_PATH = AGENT_DIR / "preference_taxonomy_schema.json"
GOLD_PATH = AGENT_DIR / "testdata" / "preference_taxonomy_gold_split.json"

if str(RELEASE_SERVER) not in sys.path:
    sys.path.insert(0, str(RELEASE_SERVER))


# ---------------------------------------------------------------- 模型接线
class LiveModelApi:
    """复用项目网关客户端，只取 url/key/name/timeout 四个字段，绕开 settings 全量校验。"""

    def __init__(self, config_path: Path) -> None:
        from bench.model_gateway_client import ModelGatewayClient  # type: ignore

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        url = cfg["model_api_url"]
        key = (
            os.environ.get("DASHSCOPE_API_KEY", "").strip()
            or os.environ.get("TIANCHI_MODEL_API_KEY", "").strip()
            or str(cfg.get("model_api_key", "")).strip()
        )
        if not key:
            raise RuntimeError("未配置模型密钥：请设置环境变量 DASHSCOPE_API_KEY。")
        name = cfg["model_name"]
        timeout = float(cfg.get("model_timeout_seconds", 60))
        self._client = ModelGatewayClient(url, key, name, timeout)
        self.model_name = name

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.chat_completion(payload)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("模型网关返回不是 JSON 对象")
        return data

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- prompt 构造
OUTPUT_SPEC = """
你是货运司机偏好解析器。请严格按上面的 Schema，把司机偏好解析成 envelope。

铁律：
1. 同一司机的【全部偏好放在一起】解析：地名若某条没给坐标，要从该司机其它偏好里把坐标补全（指代消解）。
2. 一条原文若含多个独立判定，【拆成多条 envelope】：每条只表达一个谓词（一个 subject/metric/operator），
   这些同源条目用【相同的 source_index】（来自的那条原文序号）标明同源。
3. 单谓词偏好输出一条；source_index 标明对应输入的第几条原文（从 1 开始）。同一条原文拆出的多条 source_index 必须相同。

字段判定细则（按此消除口径歧义）：
- period.type：每天重复→daily；整月持续生效或月度累计（品类禁接、地理围栏、月度配额/里程上限）→monthly；
  针对单笔订单（单笔运距、单次空驶）→once；特定日期的临时任务（某月某号、临时约定，哪怕跨几天）→once。不要用 date_range。
- penalty.model：每违反一次罚定额→per_violation_fixed；一次性大额（临时约定/错过即罚一笔）→one_time_fixed；
  月度配额没达标按缺口罚（至少 N 天/N 单未满足）→per_missing_quota；
  超出部分按单位累进（每超 1 公里/1 单）→per_unit_overflow；按时间累进（每迟到 1 分钟）→per_time_unit。
- 地名点/面区分：具体地点（XX 档口、XX 老家、给了坐标的点）是 named_point，region.type=point_radius；
  行政区（XX 区、XX 市、XX 县城）是 admin_area，region.type=named_admin_area。
  “装卸货在某区接够 N 个不同日子”属于 POINT_VISIT_TASK / visit_frequency_quota（到访频次），不是普通地理围栏。
- 指代消解回填坐标：地名若本条没给坐标，必须从该司机其它偏好找到坐标，填进 region.center_lat/center_lng，不要留 null。
- 地名保留完整：“XX市”不要截成“XX”，“XX市”要原样填入 region.named_area，不要留空。

只输出 JSON（不要解释、不要 markdown 代码块），格式：
{
  "preferences": [
    {
      "source_index": 1,
      "category": "TIME_REST|CARGO_CATEGORY|GEO_SPATIAL|MILEAGE_DISTANCE|ORDER_RHYTHM|POINT_VISIT_TASK",
      "subcategory": "<enums.subcategory 之一>",
      "intent_direction": "MUST|FORBID|CAP_MAX|QUOTA_MIN|SOFT_PREFER",
      "hardness": "hard|soft",
      "constraint": {
        "subject": "...", "metric": "...", "operator": "...",
        "scalar_value": null, "scalar_unit": null, "value_set": null,
        "period": {"type": "daily|monthly|once"},
        "region": null, "time_window": null, "cargo": null, "quota": null
      },
      "penalty": {"model": "...", "raw_penalty_amount": 0, "raw_penalty_cap": null}
    }
  ]
}
region/time_window/cargo/quota 用到时按 Schema 字段填，没用到填 null。
penalty.raw_penalty_amount / raw_penalty_cap 必须如实回填输入的 penalty_amount / penalty_cap。
同一条原文拆出的多条，source_index 必须相同。
"""


# ---------------------------------------------------------------- strict json schema
def _obj(props: dict, nullable: bool = False) -> dict:
    """strict 规则：所有 property 必须 required，additionalProperties=false。"""
    return {
        "type": ["object", "null"] if nullable else "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


_STR = {"type": ["string", "null"]}
_NUM = {"type": ["number", "null"]}
_BOOL = {"type": ["boolean", "null"]}
_ARR_STR = {"type": ["array", "null"], "items": {"type": "string"}}

ENVELOPE_SCHEMA = _obj({
    "preferences": {
        "type": "array",
        "items": _obj({
            "source_index": {"type": "integer"},
            "category": {"type": "string", "enum": [
                "TIME_REST", "CARGO_CATEGORY", "GEO_SPATIAL",
                "MILEAGE_DISTANCE", "ORDER_RHYTHM", "POINT_VISIT_TASK",
            ]},
            "subcategory": {"type": "string"},
            "intent_direction": {"type": "string", "enum": [
                "MUST", "FORBID", "CAP_MAX", "QUOTA_MIN", "SOFT_PREFER",
            ]},
            "hardness": {"type": "string", "enum": ["hard", "soft"]},
            "constraint": _obj({
                "subject": {"type": "string"},
                "metric": {"type": "string"},
                "operator": {"type": "string"},
                "scalar_value": _NUM,
                "scalar_unit": _STR,
                "value_set": _ARR_STR,
                "period": _obj({"type": {"type": "string"}, "specific_dates": _ARR_STR, "reset": _STR}),
                "region": _obj({
                    "type": _STR, "containment": _STR,
                    "center_lat": _NUM, "center_lng": _NUM, "radius_km": _NUM,
                    "named_area": _STR, "place_role": _STR, "tolerance_km": _NUM,
                    "place_ref": _STR, "place_kind": _STR, "admin_radius_km": _NUM,
                }, nullable=True),
                "time_window": _obj({
                    "start": _STR, "end": _STR, "crosses_midnight": _BOOL, "applies_to": _ARR_STR,
                }, nullable=True),
                "cargo": _obj({
                    "match_by": _STR, "categories": _ARR_STR, "specific_ids": _ARR_STR,
                }, nullable=True),
                "quota": _obj({
                    "count": _NUM, "count_unit": _STR, "comparator": _STR, "distinct_by": _STR,
                }, nullable=True),
            }),
            "penalty": _obj({
                "model": {"type": "string"},
                "raw_penalty_amount": {"type": "number"},
                "raw_penalty_cap": _NUM,
            }),
        }),
    },
})


def build_system_prompt() -> str:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema.pop("category_templates_with_examples", None)  # 样例属于 golden，不进 prompt
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return "### 偏好分类 Schema（定义部分）\n" + schema_text + "\n\n### 任务\n" + OUTPUT_SPEC


def build_user_prompt(driver_id: str, preferences: list[dict[str, Any]]) -> str:
    lines = [f"司机 {driver_id} 的全部偏好共 {len(preferences)} 条，请逐条解析并输出 JSON："]
    for i, p in enumerate(preferences, start=1):
        content = p.get("content") or p.get("text") or ""
        lines.append(
            f"[{i}] {content}"
            f"（penalty_amount={p.get('penalty_amount')}, penalty_cap={p.get('penalty_cap')}, "
            f"start_time={p.get('start_time')}, end_time={p.get('end_time')}）"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- 解析输出
def extract_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i : j + 1])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 比对
def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def as_set(v: Any) -> set:
    if v is None:
        return set()
    if isinstance(v, (list, tuple)):
        return {tuple(x) if isinstance(x, list) else x for x in v}
    return {v}


def field_equal(expected: Any, got: Any) -> bool:
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(float(expected) - float(got)) < 1e-6
    if isinstance(expected, list):
        return as_set(expected) == as_set(got)
    if isinstance(expected, str) and isinstance(got, str):
        return expected.strip() == got.strip()  # 容忍枚举值尾随空格（'>=' vs '>= '）
    return expected == got


# 放宽口径：以下字段的这些取值视为等价（口径两可，不算模型错）。
RELAX_EQUIV = {
    "period.type": [{"once", "date_range"}],            # 特定日期任务 once/date_range 两可
    "scalar_unit": [{"day", "distinct_days", "full_days"}],  # 都表示“天”
}


def field_equal_relaxed(path: str, expected: Any, got: Any) -> bool:
    if field_equal(expected, got):
        return True
    for suffix, groups in RELAX_EQUIV.items():
        if path.endswith(suffix):
            for g in groups:
                if expected in g and got in g:
                    return True
    if path.endswith("named_area") or path.endswith("place_ref"):  # 地名互为子串视为等价（惠州/惠州市）
        if isinstance(expected, str) and isinstance(got, str) and expected and got and (expected in got or got in expected):
            return True
    return False


class Acc:
    def __init__(self) -> None:
        self.cases = 0
        self.covered = 0
        self.cat_hit = 0
        self.sub_hit = 0
        self.field_checks = 0
        self.field_hits = 0

    def add(self, other: "Acc") -> None:
        self.cases += other.cases
        self.covered += other.covered
        self.cat_hit += other.cat_hit
        self.sub_hit += other.sub_hit
        self.field_checks += other.field_checks
        self.field_hits += other.field_hits


def eval_case(case: dict[str, Any], item: dict[str, Any] | None, verbose: bool, relaxed: bool = False) -> Acc:
    a = Acc()
    a.cases = 1
    exp = case["expected"]
    fields = exp.get("fields", {})
    a.field_checks = len(fields)
    if item is None:
        if verbose:
            print(f"    {case['preference_id']}: 未解析出（覆盖 miss），字段全记 0/{len(fields)}")
        return a
    a.covered = 1
    cat_ok = item.get("category") == exp["category"]
    sub_ok = item.get("subcategory") == exp["subcategory"]
    a.cat_hit = int(cat_ok)
    a.sub_hit = int(sub_ok)
    misses = []
    for path, ev in fields.items():
        gv = get_path(item, path)
        ok = field_equal_relaxed(path, ev, gv) if relaxed else field_equal(ev, gv)
        if ok:
            a.field_hits += 1
        else:
            misses.append((path, ev, gv))
    if verbose:
        flag = "" if (cat_ok and sub_ok and not misses) else "  <-- 有 miss"
        print(
            f"    {case['preference_id']}: cat={'Y' if cat_ok else 'N'} sub={'Y' if sub_ok else 'N'} "
            f"fields={a.field_hits}/{a.field_checks}{flag}"
        )
        for path, ev, gv in misses:
            print(f"        miss {path}: expected={ev!r} got={gv!r}")
    return a


def load_driver_prefs(dataset_rel: str, driver_id: str) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / dataset_rel
    data = json.loads(path.read_text(encoding="utf-8"))
    for d in data:
        if str(d.get("driver_id")) == driver_id:
            return d.get("preferences") or []
    raise KeyError(f"driver {driver_id} not found in {dataset_rel}")


def pct(num: int, den: int) -> str:
    return f"{(num / den * 100):.1f}%" if den else "n/a"


def report(title: str, acc: Acc) -> None:
    print(f"--- {title} 汇总 ---")
    print(f"  覆盖率        : {acc.covered}/{acc.cases}  ({pct(acc.covered, acc.cases)})")
    print(f"  category 命中 : {acc.cat_hit}/{acc.cases}  ({pct(acc.cat_hit, acc.cases)})")
    print(f"  子类 命中     : {acc.sub_hit}/{acc.cases}  ({pct(acc.sub_hit, acc.cases)})")
    print(f"  字段级命中(主): {acc.field_hits}/{acc.field_checks}  ({pct(acc.field_hits, acc.field_checks)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="不调模型，仅校验管线")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dump", default=None, help="把模型原始输出写到该 JSON 文件")
    ap.add_argument("--gold", default=str(GOLD_PATH), help="指定 golden 文件")
    ap.add_argument("--replay", default=None, help="从已 dump 的模型输出文件回放，不调模型(省 token)")
    ap.add_argument("--structured", action="store_true", help="结构化输出：payload 加 response_format=json_object")
    ap.add_argument("--json-schema", dest="json_schema", action="store_true",
                    help="strict 模式：payload 加 response_format=json_schema（强制每个字段都填）")
    ap.add_argument("--relaxed", action="store_true", help="放宽 golden 口径：period/scalar_unit/地名 等价归一")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    datasets = gold["datasets"]
    cases = gold["cases"]
    print(f"golden: {gold_path.name}  cases={len(cases)}  mode=split")

    # 按 (dataset, driver) 分组
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in cases:
        groups.setdefault((c["dataset"], c["driver_id"]), []).append(c)

    system_prompt = build_system_prompt()
    if args.structured or args.json_schema:
        system_prompt += "\n\n（务必输出合法 json 对象；envelope 的每个字段都要出现，用不到的填 null，不要省略 key。）"
    print(f"system prompt 长度: {len(system_prompt)} 字符  "
          f"structured={args.structured} json_schema={args.json_schema} relaxed={args.relaxed}")

    replay_data = json.loads(Path(args.replay).read_text(encoding="utf-8")) if args.replay else None
    api = None if (args.dry_run or args.replay) else LiveModelApi(Path(args.config))
    if api is not None:
        print(f"模型: {api.model_name}（真跑）\n")
    elif replay_data is not None:
        print(f"REPLAY：从 {args.replay} 回放模型输出（不调模型）\n")
    else:
        print("DRY-RUN：不调用模型\n")

    by_dataset: dict[str, Acc] = {k: Acc() for k in datasets}
    dumps: dict[str, Any] = {}

    for (ds, drv), drv_cases in sorted(groups.items()):
        prefs = load_driver_prefs(datasets[ds], drv)
        obj: dict[str, Any] | None = None
        if replay_data is not None:
            content = replay_data.get(f"{ds}:{drv}")
            obj = extract_json(content) if isinstance(content, str) else None
        elif api is not None:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": build_user_prompt(drv, prefs)},
                ],
                "temperature": 0,
                "enable_thinking": False,  # qwen 关闭思维链：避免超时、token 降九成
            }
            if args.json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "preference_envelope",
                        "strict": True,
                        "schema": ENVELOPE_SCHEMA,
                    },
                }
            elif args.structured:
                payload["response_format"] = {"type": "json_object"}
            try:
                data = api.model_chat_completion(payload)
                content = data["choices"][0]["message"]["content"]
                obj = extract_json(content)
                if args.dump is not None:
                    dumps[f"{ds}:{drv}"] = content
            except Exception as e:  # noqa: BLE001
                print(f"[{ds}:{drv}] 调用失败：{e}（该司机全部 case 记未覆盖）")

        print(f"=== {ds}:{drv}  偏好{len(prefs)}条  golden{len(drv_cases)}条 ===")
        # 不信任模型的 source_index：该司机全部输出条目放一个池子，按语义对齐 golden。
        pool = [m for m in (obj.get("preferences") if isinstance(obj, dict)
                and isinstance(obj.get("preferences"), list) else []) if isinstance(m, dict)]
        used: set[int] = set()
        matched: dict[int, dict[str, Any]] = {}
        for gc in drv_cases:  # pass1: category + subcategory 精确匹配
            exp = gc["expected"]
            for i, m in enumerate(pool):
                if i in used:
                    continue
                if m.get("category") == exp["category"] and m.get("subcategory") == exp["subcategory"]:
                    matched[id(gc)] = m
                    used.add(i)
                    break
        for gc in drv_cases:  # pass2: 剩余 case 降级为 category-only
            if id(gc) in matched:
                continue
            exp = gc["expected"]
            for i, m in enumerate(pool):
                if i in used:
                    continue
                if m.get("category") == exp["category"]:
                    matched[id(gc)] = m
                    used.add(i)
                    break
        for gc in drv_cases:
            a = eval_case(gc, matched.get(id(gc)), args.verbose, args.relaxed)
            by_dataset[ds].add(a)

    if api is not None:
        api.close()
    if args.dump is not None:
        Path(args.dump).write_text(json.dumps(dumps, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n模型原始输出已写入 {args.dump}")

    print("\n========== 结果 ==========")
    total = Acc()
    for ds in datasets:
        report(f"{ds} 偏好集", by_dataset[ds])
        total.add(by_dataset[ds])
        print()
    report("总计", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
