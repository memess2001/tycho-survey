"""Set keys of a survey root's tycho.yaml in place, keeping its comments: ``python scripts/set_config.py [--root DIR]
section.key=value ...`` (for example ``survey.sky=configs/sky_v1.yaml scheduler.followups_per_day=10000``).

Only two-level keys (``section.key``) are supported, which covers every setting of tycho.yaml. A key's existing line
``  key: old`` inside its ``section:`` block is rewritten; a missing key is added at the end of the block, and a
missing section is appended. Values are written as given (YAML parses them). The result must load with
:func:`tycho.config.load_config` and, apart from the assigned keys, parse to exactly what the file held before (this
catches shapes the line editing cannot handle, such as a flow-style ``survey: {...}`` section); otherwise the file is
left unchanged.
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml


def set_key(text: str, section: str, key: str, value: str) -> str:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(rf"^{re.escape(section)}:\s*(#.*)?$", line)), None)
    if start is None:
        return text.rstrip("\n") + f"\n\n{section}:\n  {key}: {value}\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith((" ", "#"))),
               len(lines))
    pattern = re.compile(rf"^(  ){re.escape(key)}:(\s*)([^#]*?)(\s*#.*)?$")
    for i in range(start + 1, end):
        m = pattern.match(lines[i])
        if m:
            lines[i] = f"  {key}: {value}{m.group(4) or ''}"
            return "\n".join(lines) + "\n"
    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def _without(doc: object, assigned: list[tuple[str, str]]) -> object:
    """``doc`` with the assigned keys removed (and sections that become empty because of it)."""
    if not isinstance(doc, dict):
        return doc
    out = copy.deepcopy(doc)
    for section, key in assigned:
        sec = out.get(section)
        if isinstance(sec, dict):
            sec.pop(key, None)
            if not sec:
                out.pop(section)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="survey root (default: current directory)")
    ap.add_argument("assignments", nargs="+", help="section.key=value")
    args = ap.parse_args(argv)
    path = Path(args.root) / "tycho.yaml"
    if not path.exists():
        print(f"no {path}", file=sys.stderr)
        return 1
    original = path.read_text(encoding="utf-8")
    try:
        before = yaml.safe_load(original) or {}
    except yaml.YAMLError as exc:
        print(f"{path} does not parse ({exc}); fix it by hand first", file=sys.stderr)
        return 1
    if not isinstance(before, dict):
        print(f"{path} is not a mapping; fix it by hand first", file=sys.stderr)
        return 1
    text = original
    assigned: list[tuple[str, str]] = []
    for item in args.assignments:
        name, sep, value = item.partition("=")
        section, dot, key = name.partition(".")
        if not (sep and dot and section and key and "." not in key):
            print(f"expected section.key=value, got {item!r}", file=sys.stderr)
            return 2
        text = set_key(text, section, key, value)
        assigned.append((section, key))
    try:
        after = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        print(f"tycho.yaml would not parse ({exc}); left unchanged", file=sys.stderr)
        return 1
    if _without(before, assigned) != _without(after, assigned):
        print(f"editing {path} line by line would change other settings too (unusual layout, e.g. a flow-style "
              f"section); left unchanged, edit it by hand", file=sys.stderr)
        return 1
    path.write_text(text, encoding="utf-8")
    try:
        from tycho.config import load_config

        cfg = load_config(Path(args.root))
    except Exception as exc:  # noqa: BLE001 - restore the file on any failure
        path.write_text(original, encoding="utf-8")
        print(f"tycho.yaml would not load ({type(exc).__name__}: {exc}); left unchanged", file=sys.stderr)
        return 1
    for item in args.assignments:
        name = item.partition("=")[0]
        section, _, key = name.partition(".")
        print(f"{name} = {cfg[section][key]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
