#!/usr/bin/env python3
"""Build a single-page blind-review gallery over iteration_v1 review charts.

    python3 scripts/build_review_gallery.py

Output: reports/iteration_v1/review_gallery.html (relative <img> paths into
review_charts/). Decision views render immediately; each outcome view sits
behind a <details> fold so the audit stays blind until the reviewer chooses
to peek. No external resources; open via file://.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "reports" / "iteration_v1"
CHARTS = BASE / "review_charts"


def main() -> None:
    sections = []
    for arm in ("anchored", "legacy"):
        for split in ("validation", "test"):
            split_dir = CHARTS / arm / split
            if not split_dir.exists():
                continue
            groups = []
            for set_dir in sorted(split_dir.iterdir()):
                ddir = set_dir / "decision_view"
                odir = set_dir / "outcome_view"
                if not ddir.exists():
                    continue
                items = []
                for png in sorted(ddir.glob("*.png")):
                    rel_d = png.relative_to(BASE).as_posix()
                    rel_o = (odir / png.name).relative_to(BASE).as_posix()
                    items.append(
                        f'<div class="item"><img loading="lazy" src="{rel_d}">'
                        f'<details><summary>看 outcome(未来 6h)</summary>'
                        f'<img loading="lazy" src="{rel_o}"></details></div>'
                    )
                if items:
                    groups.append(f"<h3>{set_dir.name}({len(items)} 张)</h3>" + "\n".join(items))
            if groups:
                sections.append(f'<h2 id="{arm}-{split}">{arm} / {split}</h2>' + "\n".join(groups))

    nav = " | ".join(
        f'<a href="#{arm}-{split}">{arm}/{split}</a>'
        for arm in ("anchored", "legacy") for split in ("validation", "test")
    )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>iteration_v1 盲审图廊</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:20px;background:#fafafa;color:#222}}
img{{max-width:100%;border:1px solid #ddd;border-radius:4px;display:block}}
.item{{margin:14px 0;padding:10px;background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
details{{margin-top:6px}} summary{{cursor:pointer;color:#1565c0;font-size:14px}}
h2{{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:36px}}
nav{{position:sticky;top:0;background:#fafafa;padding:8px 0;font-size:15px;z-index:1}}
.tip{{color:#666;font-size:13px}}
</style></head><body>
<h1>iteration_v1 盲审图廊</h1>
<p class="tip">审核方式:先只看 decision 图(标题无任何未来信息),判断"这是不是我要的空头形态";
需要验证时再点开 outcome。anchored 优先。</p>
<nav>{nav}</nav>
{"".join(sections)}
</body></html>"""
    out = BASE / "review_gallery.html"
    out.write_text(html)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
