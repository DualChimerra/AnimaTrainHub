"""Reverse-compiles a pydantic v2 model into an `argparse.ArgumentParser`.

Design goals:
    - make schema.py's TrainingConfig the single source of truth for CLI / YAML / web form
    - preserve the existing anima_train.py CLI convention: a few fields use an alias (e.g.
      --lr <-> learning_rate) -- declared explicitly via json_schema_extra={"cli_alias": "--lr"}
    - doesn't try to replace argparse's semantics, just auto-translates field types/constraints/
      defaults into argument declarations

Supported field types:
    bool                  -> BooleanOptionalAction, --foo / --no-foo
    Literal["a", "b"]     -> choices=["a","b"]
    int / float / str     -> type=...
    list[T]               -> nargs="*", type=T
    Optional[T]           -> same as T, but defaults to None; both empty string / None fall
                             back to the default
"""
from __future__ import annotations

import argparse
import types
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


# ---------------------------------------------------------------------------
# Type analysis
# ---------------------------------------------------------------------------


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Optional[X] / X | None -> (X, True). Otherwise (annotation, False)."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


def _is_list(annotation: Any) -> bool:
    return get_origin(annotation) in (list,)


def _list_item_type(annotation: Any) -> Any:
    args = get_args(annotation)
    return args[0] if args else str


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


# ---------------------------------------------------------------------------
# Field -> argparse
# ---------------------------------------------------------------------------


def _default_value(field: FieldInfo) -> Any:
    """Extracts the Field default / default_factory (pydantic v2 uses PydanticUndefined for "no default")."""
    from pydantic_core import PydanticUndefined

    if field.default is not PydanticUndefined and field.default is not None:
        return field.default
    if field.default_factory is not None:
        try:
            return field.default_factory()  # type: ignore[call-arg]
        except TypeError:  # validator-based factory
            return None
    return None if field.default is None else field.default


def _flag_for(name: str, field: FieldInfo) -> str:
    extra = field.json_schema_extra or {}
    alias = extra.get("cli_alias") if isinstance(extra, dict) else None
    return alias or "--" + name.replace("_", "-")


def add_argument_for(
    parser: argparse.ArgumentParser,
    name: str,
    field: FieldInfo,
    *,
    suppress_default: bool = False,
) -> None:
    """Adds a single field to the parser. dest is always the field name (underscore form).

    When suppress_default=True, every argument gets default=argparse.SUPPRESS -- the resulting
    parse Namespace only contains keys the user explicitly passed (sparse), so CLI explicitness
    can be determined precisely from key presence, for namespace_from_config's CLI > YAML merge.
    """
    flag = _flag_for(name, field)
    annotation, is_optional = _unwrap_optional(field.annotation)
    default = _default_value(field)
    # argparse's format_help treats description as a printf template and expands `% params`;
    # a bare `%` in the description (e.g. "takes up 90%") makes --help raise ValueError outright.
    # Schema descriptions in this project are also used by the web UI / i18n and shouldn't be
    # polluted by argparse semantics, so every bare `%` gets escaped here at the bridge layer --
    # nobody in this project uses argparse named substitutions like %(default)s, so the escape
    # doesn't break any existing usage.
    help_text = (field.description or "").strip().replace("%", "%%")

    # bool ----------------------------------------------------------------
    if annotation is bool:
        # An Optional[bool] field's default stays None -- meaning "not specified";
        # a non-Optional bool falls back to False.
        if is_optional:
            actual_default = default  # keep None
        else:
            actual_default = bool(default) if default is not None else False
        if suppress_default:
            actual_default = argparse.SUPPRESS
        # Python 3.13+'s argparse rejects passing a --no-X style flag to BooleanOptionalAction
        # (it would auto-derive --no-no-X, colliding with the field name).
        # For field names starting with no_, fall back to a pair of mutually exclusive
        # store_true/store_false actions instead:
        #   --no-X    -> store_true  (no_X = True)
        #   --X       -> store_false (no_X = False)
        if name.startswith("no_") and len(name) > 3:
            positive = "--" + name[3:].replace("_", "-")
            parser.add_argument(
                flag,
                dest=name,
                action="store_true",
                default=actual_default,
                help=help_text or None,
            )
            # The default of the second action sharing this dest must match the first: argparse
            # setattrs the default in registration order at the start of parsing, and a non-
            # SUPPRESS None would inflate the sparse namespace with a fake "explicit" key.
            parser.add_argument(
                positive, dest=name, action="store_false",
                default=actual_default, help=None,
            )
        else:
            parser.add_argument(
                flag,
                dest=name,
                action=argparse.BooleanOptionalAction,
                default=actual_default,
                help=help_text or None,
            )
        return

    # Literal -------------------------------------------------------------
    if _is_literal(annotation):
        choices = list(get_args(annotation))
        # Literal elements share a type; if all str, type=str
        item_t = type(choices[0]) if choices else str
        parser.add_argument(
            flag,
            dest=name,
            choices=choices,
            type=item_t,
            default=argparse.SUPPRESS if suppress_default else default,
            help=help_text or None,
        )
        return

    # list[T] -------------------------------------------------------------
    if _is_list(annotation):
        item_t = _list_item_type(annotation)
        parser.add_argument(
            flag,
            dest=name,
            nargs="*",
            type=item_t,
            default=argparse.SUPPRESS if suppress_default
            else (default if default is not None else []),
            help=help_text or None,
        )
        return

    # int / float / str ---------------------------------------------------
    if annotation is int:
        parser.add_argument(
            flag, dest=name, type=int,
            default=argparse.SUPPRESS if suppress_default else default,
            help=help_text or None,
        )
        return
    if annotation is float:
        parser.add_argument(
            flag, dest=name, type=float,
            default=argparse.SUPPRESS if suppress_default else default,
            help=help_text or None,
        )
        return

    # Default to treating it as a string (covers str, Optional[str], and unknown types)
    parser.add_argument(
        flag,
        dest=name,
        type=str,
        default=argparse.SUPPRESS if suppress_default
        else (default if default is not None else ("" if not is_optional else None)),
        help=help_text or None,
    )


def build_parser(
    model_cls: type[BaseModel],
    *,
    prog: str | None = None,
    description: str | None = None,
    add_config_arg: bool = True,
    suppress_defaults: bool = False,
) -> argparse.ArgumentParser:
    """Generates a complete parser from a pydantic model.

    When add_config_arg=True, automatically adds `--config PATH` (pointing at the YAML config;
    not subject to suppress -- the caller needs to read it before merging).
    When suppress_defaults=True, every schema field gets default=argparse.SUPPRESS, so the
    parse result only contains keys the user explicitly passed; used together with
    namespace_from_config.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    if add_config_arg:
        parser.add_argument("--config", default="", help="Path to the YAML config file")
    for name, field in model_cls.model_fields.items():
        add_argument_for(parser, name, field, suppress_default=suppress_defaults)
    return parser


# ---------------------------------------------------------------------------
# YAML + explicit CLI values -> pydantic validation -> full Namespace
# ---------------------------------------------------------------------------


def namespace_from_config(
    args: argparse.Namespace,
    yaml_data: dict[str, Any],
    model_cls: type[BaseModel],
) -> argparse.Namespace:
    """Merges YAML and explicit CLI values, then expands into a Namespace after a full
    model_cls construction.

    Replaces the old merge_yaml_into_namespace (which approximated CLI explicitness by
    "value == default" and bypassed every validator -- field migration / the
    FAMILY_CONFIG_DEFAULTS per-family default overlay / mutual-exclusion and capability
    validation all silently stopped working on the trainer path; e.g. krea2 missing the
    shuffle_caption key would fall back to anima's semantic default of True and reject
    training). The caller's parser must be built with suppress_defaults=True so `args` only
    contains explicit keys, letting CLI > YAML precedence be determined precisely.

    Merge semantics:
    - yaml_data is handed to pydantic as-is -- old keys get migrated by a before-validator,
      unknown keys follow the model's extra policy (ignore, for TrainingConfig)
    - keys in args that belong to the schema override YAML (CLI explicit wins); the merged
      result goes through a single model construction, so any illegal config produced by
      combining sources fails fast right here
    - keys in args that don't belong to the schema (CLI-only switches like --interactive) are
      carried through unchanged

    Raises:
        pydantic.ValidationError: the merged result is invalid (mutual-exclusion conflict /
        family capability out of bounds / etc).
    """
    fields = model_cls.model_fields
    explicit = vars(args)
    merged = dict(yaml_data)
    merged.update({k: v for k, v in explicit.items() if k in fields})
    cfg = model_cls(**merged)
    out = argparse.Namespace(
        **{k: v for k, v in explicit.items() if k not in fields}
    )
    for key, value in cfg.model_dump(mode="python").items():
        setattr(out, key, value)
    return out
