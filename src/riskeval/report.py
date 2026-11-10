"""Self-contained HTML report: no external links, light and dark theme, readable at 360 px.

Every string that comes from data goes through ``html.escape``.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone

_CSS = """
:root{--bg:#ffffff;--fg:#16181d;--muted:#5b6170;--line:#d9dce3;--accent:#2457c5;--warn:#9a5b00}
@media (prefers-color-scheme: dark){:root{--bg:#121417;--fg:#e8eaee;--muted:#a3a9b6;
--line:#2c3038;--accent:#8fb0ff;--warn:#f0b35a}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:900px;margin:0 auto}
h1{font-size:1.4rem;margin:.2rem 0 1rem}h2{font-size:1.1rem;margin:1.6rem 0 .5rem}
p.muted,small{color:var(--muted)}
.table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
.warn{color:var(--warn)}
svg{max-width:100%;height:auto}
"""


def _fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if math.isnan(x):
            return "n/a"
        return f"{x:.{digits}f}"
    if isinstance(x, int):
        return str(x)
    return html.escape(str(x))


def table(rows: Iterable[Mapping[str, object]], columns: list[str] | None = None) -> str:
    rows = list(rows)
    if not rows:
        return "<p class='muted'>No rows.</p>"
    cols = columns or list(rows[0].keys())
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def calibration_svg(rows: Sequence[Mapping[str, float]], size: int = 320) -> str:
    """Reliability diagram as inline SVG: predicted on x, observed on y."""
    pad = 36
    span = size - 2 * pad

    def xy(px: float, py: float) -> tuple[float, float]:
        return pad + px * span, size - pad - py * span

    pts = " ".join(
        f"{xy(r['mean_pred'], r['observed'])[0]:.1f},{xy(r['mean_pred'], r['observed'])[1]:.1f}"
        for r in rows
    )
    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    circles = "".join(
        f"<circle cx='{xy(r['mean_pred'], r['observed'])[0]:.1f}' cy='{xy(r['mean_pred'], r['observed'])[1]:.1f}' r='3' fill='currentColor'/>"
        for r in rows
    )
    return (
        f"<svg viewBox='0 0 {size} {size}' role='img' aria-label='Calibration plot' "
        f"style='color:var(--accent)'>"
        f"<rect x='{pad}' y='{pad}' width='{span}' height='{span}' fill='none' stroke='var(--line)'/>"
        f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y1}' stroke='var(--muted)' stroke-dasharray='4 4'/>"
        f"<polyline points='{pts}' fill='none' stroke='currentColor' stroke-width='2'/>{circles}"
        f"<text x='{size / 2}' y='{size - 6}' text-anchor='middle' font-size='12' fill='var(--muted)'>mean predicted</text>"
        f"<text x='12' y='{size / 2}' text-anchor='middle' font-size='12' fill='var(--muted)' "
        f"transform='rotate(-90 12 {size / 2})'>observed rate</text></svg>"
    )


def render(title: str, sections: list[tuple[str, str]], note: str = "") -> str:
    """Assemble the page. ``sections`` are (heading, already-safe HTML) pairs."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{html.escape(title)}</h1><p class='muted'>Generated {stamp} by riskeval. "
        f"{html.escape(note)}</p>",
    ]
    for heading, body in sections:
        parts.append(f"<section><h2>{html.escape(heading)}</h2>{body}</section>")
    parts.append("</main></body></html>")
    return "".join(parts)
