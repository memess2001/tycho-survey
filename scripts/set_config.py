"""Set keys of a survey root's tycho.yaml in place, keeping its comments: ``python scripts/set_config.py [--root DIR]
section.key=value ...`` (for example ``survey.sky=configs/sky_v1.yaml scheduler.followups_per_day=10000``).

Only two-level keys (``section.key``) are supported, which covers every setting of tycho.yaml. A key's existing line
``  key: old`` inside its ``section:`` block is rewritten; a missing key is added at the end of the block, and a
missing section is appended. Values are written as given (YAML parses them). The result must load with
:func:`tycho.config.load_config`, otherwise the file is left unchanged.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


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
    text = original
    for item in args.assignments:
        name, sep, value = item.partition("=")
        section, dot, key = name.partition(".")
        if not (sep and dot and section and key and "." not in key):
            print(f"expected section.key=value, got {item!r}", file=sys.stderr)
            return 2
        text = set_key(text, section, key, value)
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
