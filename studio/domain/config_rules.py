"""disable_when rule engine -- dual-sided enforcement of field-interaction declarations (cut 2 / R2 v2).

Design (docs/design/config-pipeline-refactor.md §6, D5): rather than build a
parallel rule table, the `disable_when` + `disable_value` + `disable_hint`
metadata already on a field IS the complete "pin value" rule declaration
(condition / pinned value / reason); `option_disable_when` is its option-level
sibling ("forbid value": when the condition holds, that enum value can't be
selected). This module elevates that single declaration into multiple facets:

1. Backend validation -- TrainingConfig._enforce_disable_rules calls
   disable_rule_violations and raises on any violation (replacing the old 8
   hand-written mutual-exclusion validators);
2. Tolerant fixing -- _tolerant_validate calls apply_disable_rule_fixes, with
   fix semantics matching the frontend's takeover (writes disable_value,
   rather than a blanket reset to the schema default);
3. Frontend UI -- SchemaForm consumes the same metadata for graying out +
   takeover (existing behavior);
4. R6 confirmation dialog -- the list of lossy value changes is derived by
   evaluating the same declaration.

Enforcement criterion: disable_when enforces hard rules where violating them
means runtime silently misbehaves or is semantically wrong. UI-guidance-only
soft pins (e.g. learning_rate is a real, effective scale factor for Prodigy;
pinning it to 1.0 is only a recommendation) go into ADVISORY_DISABLE_FIELDS,
staying UI-only and excluded from validation and fixing.

eval_show_when / _js_str moved down here from config_prune (this module is a
zero-dependency leaf; training.py's validator needs to import it, and
config_prune imports training, so keeping it there would create a cycle);
config_prune still re-exports them, so existing consumers are unaffected.
"""
from typing import Any, Mapping

from pydantic import BaseModel

_MISSING = object()

#: Fields where disable_when stays UI-only (soft pins). See the module
#: docstring for the criterion; before adding a field here, first confirm
#: that violating it does NOT actually make runtime silently misbehave.
ADVISORY_DISABLE_FIELDS: frozenset = frozenset({"learning_rate"})

#: The gate-first set for tolerant fixing: when a violated rule's `when`
#: expression references a switch in this set, turn that switch off first
#: (preserving the user's investment in the target field), otherwise fix the
#: target field by default. This generalizes the old InfoNoise-specific shim
#: in presets/io.py ("turn off InfoNoise first to preserve loss_weighting
#: etc"), which this mechanism has replaced.
TOLERANT_FIX_GATE_FIRST: frozenset = frozenset({"infonoise_enabled"})


def _js_str(value: Any) -> str:
    """Equivalent of JS `String(v)` -- show_when comparisons follow the frontend's stringification semantics.

    Differences: JS's true/false are lowercase; a float with an integer value
    has no decimal point (String(1.0) === "1", whereas Python's str(1.0) ==
    "1.0" -- training.py's `timestep_schedule_shift!=1` relies on this).
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if value is _MISSING:
        return "undefined"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def eval_show_when(expr: str | None, values: Mapping[str, Any]) -> bool:
    """Verbatim mirror of schema.ts's evalShowWhen: `||` branches are any(),
    `&&` is all(), `==`/`!=` are string comparisons; an empty expression or
    one that fails to parse both return True (failsafe)."""
    if not expr:
        return True
    branches = [p.strip() for p in expr.split("||") if p.strip()]
    if len(branches) > 1:
        return any(eval_show_when(b, values) for b in branches)
    ands = [p.strip() for p in expr.split("&&") if p.strip()]
    if len(ands) > 1:
        return all(eval_show_when(c, values) for c in ands)
    eq = expr.split("==")
    if len(eq) == 2:
        return _js_str(values.get(eq[0].strip(), _MISSING)) == eq[1].strip()
    ne = expr.split("!=")
    if len(ne) == 2:
        return _js_str(values.get(ne[0].strip(), _MISSING)) != ne[1].strip()
    return True


def _expr_fields(expr: str) -> list[str]:
    """Field names appearing in a when expression (left side of each comparison), de-duplicated in order of appearance."""
    out: list[str] = []
    for branch in expr.split("||"):
        for clause in branch.split("&&"):
            for op in ("==", "!="):
                parts = clause.split(op)
                if len(parts) == 2:
                    name = parts[0].strip()
                    if name and name not in out:
                        out.append(name)
                    break
    return out


def _schema_default(field) -> Any:
    return field.get_default(call_default_factory=True)


def iter_pin_rules(model_cls: type[BaseModel]):
    """Iterate over the mandatory pin-value rules: yield (field, when_expr, pin_value, hint).

    pin_value = disable_value when declared (so fix/validation semantics
    match the frontend's takeover), otherwise the schema default (the same
    fallback the frontend's takeover uses).
    """
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        expr = extra.get("disable_when")
        if not isinstance(expr, str) or not expr or name in ADVISORY_DISABLE_FIELDS:
            continue
        pin = extra.get("disable_value", _MISSING)
        if pin is _MISSING:
            pin = _schema_default(field)
        yield name, expr, pin, str(extra.get("disable_hint") or "")


def iter_forbid_rules(model_cls: type[BaseModel]):
    """Iterate over the forbidden-value rules: yield (field, forbidden_value_str, when_expr, hint).

    Source = the field metadata's option_disable_when: {enum value: when expression}.
    """
    for name, field in model_cls.model_fields.items():
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            continue
        gates = extra.get("option_disable_when")
        if not isinstance(gates, dict):
            continue
        for value, expr in gates.items():
            if isinstance(expr, str) and expr:
                yield name, str(value), expr, str(extra.get("disable_hint") or "")


def apply_pin_setdefaults(
    data: dict[str, Any], model_cls: type[BaseModel]
) -> dict[str, Any]:
    """构造期 setdefault ——「缺省跟随钉值,显式违反才报错」。

    pin 规则 when 为真且目标字段 **未显式提供** 时,落到钉值而非 schema 默认
    (与前端 takeover / FAMILY_CONFIG_DEFAULTS overlay 的 setdefault 同构):
    `TrainingConfig(navit_packing=True)` 的 attention_backend 应自动落 xformers,
    而不是拿着 schema 默认 flash_attn 去撞 after 校验。显式提供且违反的值
    不在此改写 —— 由 _enforce_disable_rules fail-fast,用户显式配置绝不静默改。

    when 求值视图 = data 缺键补 schema default(条件字段缺失按默认求值)。
    """
    view: dict[str, Any] | None = None
    out = data
    for name, expr, pin, _hint in iter_pin_rules(model_cls):
        if name in data:
            continue
        if view is None:  # lazy:多数构造没有缺失的 pin 字段
            view = {
                k: (data[k] if k in data else _schema_default(f))
                for k, f in model_cls.model_fields.items()
            }
        if eval_show_when(expr, view):
            if out is data:
                out = dict(data)
            out[name] = pin
            view[name] = pin
    return out


def disable_rule_violations(
    values: Mapping[str, Any], model_cls: type[BaseModel]
) -> list[dict[str, Any]]:
    """返回违反清单,每项 {field, expected, actual, hint, kind}。

    kind = "pin"(须等于 expected)| "forbid"(不可等于 actual,expected 为
    修复回退值 = schema default)。对完整 values 求值(与前端对完整表单 state
    求值一致)。
    """
    out: list[dict[str, Any]] = []
    for name, expr, pin, hint in iter_pin_rules(model_cls):
        if eval_show_when(expr, values):
            actual = values.get(name, _MISSING)
            if actual is not _MISSING and _js_str(actual) != _js_str(pin):
                out.append({
                    "field": name, "expected": pin, "actual": actual,
                    "hint": hint, "kind": "pin",
                })
    for name, forbidden, expr, hint in iter_forbid_rules(model_cls):
        actual = values.get(name, _MISSING)
        if actual is not _MISSING and _js_str(actual) == forbidden and eval_show_when(expr, values):
            out.append({
                "field": name,
                "expected": _schema_default(model_cls.model_fields[name]),
                "actual": actual, "hint": hint, "kind": "forbid",
            })
    return out


def apply_disable_rule_fixes(
    data: dict[str, Any], model_cls: type[BaseModel]
) -> tuple[dict[str, Any], list[str]]:
    """tolerant 修复:按规则把违反字段修到合法值,返回 (新 dict, 修过的字段名)。

    每轮只修一处然后重算 —— 双向对称锁下一个冲突会产生两条违反(A 的规则钉 A、
    B 的规则按 gate 关 A),同轮全修会把两边都改掉、过度修复(如 huber+InfoNoise
    冲突把 huber 也抹了);单步 + 重算后,关掉 gate 的那一步会让对侧违反自然消失。

    选步策略:任何违反的 when 表达式含 TOLERANT_FIX_GATE_FIRST 开关且该开关
    当前为真 → 先关开关(牺牲开关、保住用户在目标字段上的投入,历史 InfoNoise
    垫片语义的泛化);否则钉第一条违反的目标字段(pin 写 disable_value /
    forbid 写 schema default)。

    注意:对 **完整** dict 求值;缺键字段不参与判定(pydantic 构造时的族
    overlay / schema 默认不会造出违反 —— 默认组合恒合法,由测试锁)。
    """
    out = dict(data)
    fixed: list[str] = []
    for _ in range(20):  # 上限 > 规则总数;正常几步内收敛
        violations = disable_rule_violations(out, model_cls)
        if not violations:
            break
        target = value = None
        for v in violations:
            gates = [
                g for g in _expr_fields_of_violation(v, model_cls)
                if g in TOLERANT_FIX_GATE_FIRST and bool(out.get(g))
            ]
            if gates:
                target, value = gates[0], False
                break
        if target is None:
            v = violations[0]
            target, value = v["field"], v["expected"]
        out[target] = value
        if target not in fixed:
            fixed.append(target)
    return out, fixed


def _expr_fields_of_violation(violation: dict[str, Any], model_cls: type[BaseModel]) -> list[str]:
    """取该违反对应规则的 when 表达式字段(gate 候选)。"""
    name = violation["field"]
    extra = model_cls.model_fields[name].json_schema_extra
    if not isinstance(extra, dict):
        return []
    if violation["kind"] == "pin":
        expr = extra.get("disable_when") or ""
    else:
        expr = (extra.get("option_disable_when") or {}).get(_js_str(violation["actual"]), "")
    return _expr_fields(expr) if isinstance(expr, str) else []
