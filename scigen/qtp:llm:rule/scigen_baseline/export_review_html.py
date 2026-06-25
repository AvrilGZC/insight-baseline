from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .scigen_loader import load_scigen_records


def export_review_html(
    input_jsonl: str,
    output: str,
    mode: str,
    data_dir: str = "data/scigen",
) -> None:
    records = _load_jsonl(input_jsonl)
    source = {record.record_id: record for record in load_scigen_records(data_dir, split="test")}
    title = "SciGen Audit Baseline Review" if mode == "audit" else "SciGen Insight Discovery Review"
    body = [_header(title, mode)]
    for idx, row in enumerate(records, start=1):
        source_record = source.get(str(row.get("record_id")))
        body.append(_render_record(idx, row, mode, source_record))
    body.append("</main></body></html>")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(body), encoding="utf-8")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _header(title: str, mode: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #1f2933;
      --muted: #5d6673;
      --line: #d8ddd8;
      --panel: #ffffff;
      --soft: #eef3f0;
      --accent: #0f766e;
      --warn: #a16207;
      --bad: #b42318;
      --good: #177245;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 24px;
    }}
    h1 {{ margin: 0 0 4px; font-size: 20px; }}
    .sub {{ color: var(--muted); }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 18px 24px 48px; }}
    details.record {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 0 0 16px;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      padding: 14px 16px;
      background: var(--soft);
      font-weight: 650;
    }}
    .meta {{ color: var(--muted); font-weight: 500; margin-left: 8px; }}
    .section {{ padding: 14px 16px; border-top: 1px solid var(--line); }}
    .caption {{ font-weight: 650; margin-bottom: 8px; }}
    .description {{ white-space: pre-wrap; color: #2d3742; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat({3 if mode == "audit" else 2}, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }}
    .col {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .col h2 {{
      margin: 0;
      padding: 10px 12px;
      font-size: 15px;
      background: #f4f6f4;
      border-bottom: 1px solid var(--line);
    }}
    .counts {{ padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 6px; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      background: #edf2f7;
      color: #374151;
      font-size: 12px;
      font-weight: 650;
    }}
    .covered {{ background: #dff3e7; color: var(--good); }}
    .partially_covered {{ background: #fff3cf; color: var(--warn); }}
    .missing {{ background: #eceff3; color: #4b5563; }}
    .contradicted {{ background: #fde2df; color: var(--bad); }}
    .insight {{ padding: 10px 12px; border-top: 1px solid var(--line); }}
    .claim {{ font-weight: 600; }}
    .small {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .reason {{ margin-top: 6px; color: #334155; }}
    .evidence {{ margin-top: 6px; color: var(--muted); font-size: 12px; }}
    table.data {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
      table-layout: fixed;
    }}
    table.data th, table.data td {{
      border: 1px solid var(--line);
      padding: 5px 6px;
      vertical-align: top;
      word-break: break-word;
    }}
    table.data th {{ background: #f3f5f5; }}
    .questions {{ margin: 0; padding-left: 18px; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{_e(title)}</h1>
  <div class="sub">Open each record to inspect table, description, generated insights, evidence, and coverage labels.</div>
</header>
<main>"""


def _render_record(idx: int, row: Dict[str, Any], mode: str, source_record: Any) -> str:
    record_id = str(row.get("record_id", ""))
    domain = str(row.get("domain", ""))
    paper_id = str(row.get("paper_id", ""))
    caption = str(row.get("caption", ""))
    description = str(row.get("description", ""))
    table_html = _render_table(source_record) if source_record else "<p class='small'>Source table not found.</p>"
    if mode == "audit":
        comparison = _render_audit_grid(row)
    else:
        comparison = _render_discovery_grid(row)
    return f"""<details class="record">
  <summary>#{idx} {_e(record_id)} <span class="meta">{_e(domain)} · {_e(paper_id)}</span></summary>
  <section class="section">
    <div class="caption">{_e(caption)}</div>
    <div class="description">{_e(description)}</div>
  </section>
  <section class="section">
    {table_html}
  </section>
  <section class="section">
    {comparison}
  </section>
</details>"""


def _render_audit_grid(row: Dict[str, Any]) -> str:
    qtp_prefix = _render_questions(row.get("qtp_questions", []), "QTP Questions")
    columns = [
        ("Pure LLM w/ Description", row.get("pure_counts", {}), row.get("pure_top_insights", []), ""),
        ("QTP w/ Description", row.get("qtp_counts", {}), row.get("qtp_top_insights", []), qtp_prefix),
        ("Rule + LLM Verifier", row.get("rule_llm_counts", {}), row.get("rule_top_insights", []), ""),
    ]
    return "<div class='grid'>" + "\n".join(
        _render_column(name, counts, insights, prefix=prefix)
        for name, counts, insights, prefix in columns
    ) + "</div>"


def _render_discovery_grid(row: Dict[str, Any]) -> str:
    q_html = _render_questions(row.get("qtp_questions", []), "QTP Questions")
    qtp_col = _render_column("QTP Table-Only", {}, row.get("qtp_top_insights", []), prefix=q_html)
    rule_col = _render_column("Rule-Based", {}, row.get("rule_top_insights", []))
    return f"<div class='grid'>{qtp_col}{rule_col}</div>"


def _render_questions(questions: Any, title: str) -> str:
    if not isinstance(questions, list) or not questions:
        return ""
    items = []
    for q in questions[:5]:
        if not isinstance(q, dict):
            continue
        items.append(
            f"<li>{_e(q.get('question', ''))}<div class='small'>{_e(q.get('answer', ''))}</div></li>"
        )
    if not items:
        return ""
    return f"<div class='counts'><span class='pill'>{_e(title)}</span></div><ol class='questions'>{''.join(items)}</ol>"


def _render_column(name: str, counts: Dict[str, Any], insights: Iterable[Dict[str, Any]], prefix: str = "") -> str:
    counts_html = ""
    if counts:
        counts_html = "<div class='counts'>" + "".join(
            f"<span class='pill {key}'>{_e(key)}: {_e(value)}</span>"
            for key, value in counts.items()
        ) + "</div>"
    insight_html = "\n".join(_render_insight(item, idx) for idx, item in enumerate(insights, start=1))
    return f"<article class='col'><h2>{_e(name)}</h2>{counts_html}{prefix}{insight_html}</article>"


def _render_insight(item: Dict[str, Any], idx: int) -> str:
    status = str(item.get("status") or "")
    status_html = f"<span class='pill {status}'>{_e(status)}</span>" if status else ""
    subtype = str(item.get("subtype") or "")
    typ = str(item.get("type") or "")
    reason = str(item.get("rationale") or "")
    evidence = item.get("evidence") or []
    if isinstance(evidence, list):
        evidence_text = "; ".join(
            f"{cell.get('row_header') or cell.get('row')} | {cell.get('column_header') or cell.get('column')} = {cell.get('value')}"
            for cell in evidence[:6]
            if isinstance(cell, dict)
        )
    else:
        evidence_text = ""
    return f"""<div class="insight">
  <div class="claim">{idx}. {_e(str(item.get("claim", "")))} {status_html}</div>
  <div class="small">{_e(typ)} · {_e(subtype)}</div>
  {f'<div class="reason">{_e(reason)}</div>' if reason else ''}
  {f'<div class="evidence">Evidence: {_e(evidence_text)}</div>' if evidence_text else ''}
</div>"""


def _render_table(record: Any) -> str:
    headers = record.column_headers
    rows = record.rows
    head = "".join(f"<th>{_e(cell)}</th>" for cell in headers)
    body = []
    for row in rows[:80]:
        body.append("<tr>" + "".join(f"<td>{_e(cell)}</td>" for cell in row) + "</tr>")
    if len(rows) > 80:
        body.append(f"<tr><td colspan='{max(1, len(headers))}'>... {len(rows) - 80} more rows</td></tr>")
    return f"<table class='data'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SciGen manual review JSONL as static HTML.")
    parser.add_argument("input_jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["audit", "discovery"], required=True)
    parser.add_argument("--data-dir", default="data/scigen")
    args = parser.parse_args()
    export_review_html(args.input_jsonl, args.output, args.mode, args.data_dir)
    print(args.output)


if __name__ == "__main__":
    main()
