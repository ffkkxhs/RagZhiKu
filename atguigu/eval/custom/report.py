# atguigu/eval/custom/report.py
"""
把评估结果渲染成一个自包含的 HTML 报告。

为什么用纯 HTML/CSS：
- 不引入 Streamlit/Gradio/前端依赖，双击就能打开；
- 报告只有表格和 CSS 条形图，足够回答三个问题：
  1. 总体指标是多少？
  2. 哪类题最差？
  3. 具体哪几条失败，失败在哪？
- 等评估稳定、需要多人协作时，再换 Langfuse/Streamlit 这类工具。

报告结构：
- 顶部：本次运行的基本信息；
- 指标卡片：评估条数、Recall@5、MRR、faithfulness 等；
- 分题型条形图；
- 逐条 case 的明细，可展开看 gold、召回结果、答案和 judge 理由。
"""

import html
import json
from pathlib import Path


def _read_jsonl(path: Path):
    """读取 JSONL，每行一个 case 结果。"""
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def _read_json(path: Path):
    """读取 summary.json。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _escape(value) -> str:
    """
    转义 HTML 特殊字符，防止答案里的 <、>、& 把页面结构弄坏。
    同时把 None 显示成空字符串。
    """
    if value is None:
        return ""
    return html.escape(str(value))


def _percent(value) -> str:
    """
    把 0~1 的指标显示成百分比。
    None 显示成 "-"。
    """
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value * 100:.1f}%"


def _score(value) -> str:
    """把 1~5 的 judge 分数显示成一位小数；None 显示成 '-'。"""
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.1f}"


def _metric_card(label: str, value: str) -> str:
    """生成一张指标卡片。"""
    return f"""
    <div class="card">
      <div class="card-label">{_escape(label)}</div>
      <div class="card-value">{_escape(value)}</div>
    </div>
    """


def _bar_chart(title: str, data: dict, max_value: float) -> str:
    """
    生成一个纯 CSS 的水平条形图。

    参数：
    - title: 图表标题
    - data: {类别: 数值}
    - max_value: 数值上限（Recall 用 1，judge 分用 5）
    """
    if not data:
        return f"<h3>{_escape(title)}</h3><p class='muted'>暂无数据</p>"

    bars = []

    for label, value in data.items():
        if not isinstance(value, (int, float)):
            continue

        width = max(0.0, min(1.0, value / max_value)) * 100
        bars.append(f"""
        <div class="bar-row">
          <div class="bar-label">{_escape(label)}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{width:.1f}%"></div>
          </div>
          <div class="bar-value">{_score(value)}</div>
        </div>
        """)

    return f"<h3>{_escape(title)}</h3>{''.join(bars)}"


def _format_docs(docs) -> str:
    """
    把 gold 或 retrieved 的文档列表渲染成可读文本。

    每条是 {"file_title": ..., "title": ...}。
    """
    if not docs:
        return "<span class='muted'>（空）</span>"

    items = []
    for doc in docs:
        file_title = _escape(doc.get("file_title"))
        title = _escape(doc.get("title") or "")
        items.append(f"<li>{file_title} / {title}</li>")

    return f"<ul>{''.join(items)}</ul>"


def _case_row(row: dict) -> str:
    """
    渲染一条 case。

    - ok：显示指标，并展开 gold/召回/答案/judge 理由；
    - skipped/error：显示原因。
    """
    status = row.get("status")

    if status != "ok":
        return f"""
        <tr class="bad">
          <td>{_escape(row.get("id"))}</td>
          <td>{_escape(row.get("type"))}</td>
          <td>{_escape(status)}</td>
          <td>{_escape(row.get("question"))}</td>
          <td colspan="6">{_escape(row.get("reason"))}</td>
        </tr>
        """

    context_recall = row.get("context_recall@10")
    context_precision = row.get("context_precision@10")
    faithfulness = row.get("judge_faithfulness")
    answer_relevancy = row.get("judge_answer_relevancy")
    answer_correctness = row.get("judge_answer_correctness")

    # 五大指标里任意一个明显异常就标红；拒答题没有 gold，不按召回标红。
    is_bad = not row.get("expect_refusal") and (
        (isinstance(context_recall, (int, float)) and context_recall == 0)
        or (isinstance(faithfulness, (int, float)) and faithfulness < 3)
        or (isinstance(answer_correctness, (int, float)) and answer_correctness < 3)
    )
    row_class = "bad" if is_bad else ""

    detail = f"""
    <details>
      <summary>查看明细</summary>
      <div class="detail-block">
        <b>标准答案（gold）</b>
        {_format_docs(row.get("gold"))}
        <b>实际召回（reranked_docs）</b>
        {_format_docs(row.get("retrieved"))}
        <b>识别出的主体名</b>
        <p>{_escape(", ".join(row.get("item_names") or []))}</p>
        <b>辅助检索指标</b>
        <p>
          Recall@5: {_percent(row.get("recall@5"))} ·
          Recall@10: {_percent(row.get("recall@10"))} ·
          MRR: {_percent(row.get("mrr"))} ·
          nDCG@10: {_percent(row.get("ndcg@10"))} ·
          主体名命中: {_percent(row.get("item_name_hit"))} ·
          引用: {_score(row.get("judge_citation"))}
        </p>
        <b>最终回答</b>
        <pre>{_escape(row.get("answer"))}</pre>
        <b>judge 理由</b>
        <p>{_escape(row.get("judge_reason"))}</p>
      </div>
    </details>
    """

    return f"""
    <tr class="{row_class}">
      <td>{_escape(row.get("id"))}</td>
      <td>{_escape(row.get("type"))}</td>
      <td>{_escape(row.get("outcome"))}</td>
      <td>{_escape(row.get("question"))}</td>
      <td>{_percent(context_recall)}</td>
      <td>{_percent(context_precision)}</td>
      <td>{_score(faithfulness)}</td>
      <td>{_score(answer_relevancy)}</td>
      <td>{_score(answer_correctness)}</td>
      <td>{detail}</td>
    </tr>
    """


def build_report(results_path: Path, summary_path: Path, output_dir: Path) -> Path:
    """
    生成 report.html，并返回它的路径。

    参数：
    - results_path: 逐条结果 JSONL
    - summary_path: 汇总 JSON
    - output_dir: 输出目录

    返回：
    - report.html 的路径
    """
    rows = _read_jsonl(results_path)
    summary = _read_json(summary_path)
    overall = summary.get("overall", {})
    by_type = summary.get("by_type", {})
    outcomes = summary.get("outcomes", {})

    # ========================================================================
    # 【改造说明：报告优先展示“常说的五个指标”】
    # Context Recall / Context Precision / Faithfulness /
    # Answer Relevancy / Answer Correctness。
    # Recall@5、MRR、nDCG 仍然保留，但作为辅助检索指标放在后面。
    # ========================================================================
    def by_type_metric(metric_name):
        """按题型抽取某个指标，供条形图使用；只保留有数值的题型。"""
        return {
            case_type: values.get(metric_name, 0)
            for case_type, values in by_type.items()
            if isinstance(values.get(metric_name), (int, float))
        }

    context_recall_by_type = by_type_metric("context_recall@10")
    context_precision_by_type = by_type_metric("context_precision@10")
    faithfulness_by_type = by_type_metric("judge_faithfulness")
    answer_relevancy_by_type = by_type_metric("judge_answer_relevancy")
    answer_correctness_by_type = by_type_metric("judge_answer_correctness")
    recall_by_type = by_type_metric("recall@5")

    cards = "".join([
        _metric_card("总 case 数", summary.get("total", 0)),
        _metric_card("已评估", summary.get("evaluated", 0)),
        _metric_card("跳过（缺文档）", summary.get("skipped", 0)),
        _metric_card("执行失败", summary.get("errors", 0)),
        _metric_card("待选词短路", outcomes.get("clarification", 0)),
        _metric_card("无结果", outcomes.get("no_result", 0)),
        # 五大指标
        _metric_card("Context Recall@10", _percent(overall.get("context_recall@10"))),
        _metric_card("Context Precision@10", _percent(overall.get("context_precision@10"))),
        _metric_card("Faithfulness", _score(overall.get("judge_faithfulness"))),
        _metric_card("Answer Relevancy", _score(overall.get("judge_answer_relevancy"))),
        _metric_card("Answer Correctness", _score(overall.get("judge_answer_correctness"))),
        # 辅助检索指标
        _metric_card("Recall@5", _percent(overall.get("recall@5"))),
        _metric_card("MRR", _percent(overall.get("mrr"))),
    ])

    charts = ""
    charts += _bar_chart("分题型 Context Recall@10", context_recall_by_type, 1.0)
    charts += _bar_chart("分题型 Context Precision@10", context_precision_by_type, 1.0)
    if faithfulness_by_type:
        charts += _bar_chart("分题型 Faithfulness（1~5）", faithfulness_by_type, 5.0)
    if answer_relevancy_by_type:
        charts += _bar_chart("分题型 Answer Relevancy（1~5）", answer_relevancy_by_type, 5.0)
    if answer_correctness_by_type:
        charts += _bar_chart("分题型 Answer Correctness（1~5）", answer_correctness_by_type, 5.0)
    if recall_by_type:
        charts += _bar_chart("分题型 Recall@5（辅助）", recall_by_type, 1.0)

    case_rows = "".join(_case_row(row) for row in rows)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>RAG 评估报告</title>
  <style>
    body {{
      font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
      margin: 0;
      padding: 24px;
      background: #f5f7fb;
      color: #1f2937;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin-top: 28px; }}
    h3 {{ margin: 18px 0 10px; font-size: 15px; }}
    .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 20px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .card-label {{ font-size: 12px; color: #6b7280; }}
    .card-value {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
    .panel {{
      background: #fff;
      border-radius: 10px;
      padding: 18px;
      margin-top: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .bar-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
    .bar-label {{ width: 120px; font-size: 13px; color: #374151; }}
    .bar-track {{
      flex: 1;
      height: 14px;
      background: #e5e7eb;
      border-radius: 7px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #3b82f6, #60a5fa);
      border-radius: 7px;
    }}
    .bar-value {{ width: 48px; text-align: right; font-size: 12px; color: #374151; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
    th {{ text-align: left; background: #f9fafb; font-weight: 600; }}
    tr.bad td {{ background: #fff7f7; }}
    details summary {{ cursor: pointer; color: #2563eb; }}
    .detail-block {{ margin-top: 10px; line-height: 1.6; }}
    .detail-block pre {{
      white-space: pre-wrap;
      background: #f9fafb;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid #e5e7eb;
    }}
    .muted {{ color: #9ca3af; }}
  </style>
</head>
<body>
<div class="container">
  <h1>RAG 评估报告</h1>
  <div class="meta">
    生成时间：{_escape(summary.get("generated_at"))} ·
    模式：{_escape(summary.get("mode"))} ·
    测试集：{_escape(summary.get("dataset"))}
  </div>

  <div class="cards">{cards}</div>

  <div class="panel">
    <h2>分题型表现</h2>
    {charts}
  </div>

  <div class="panel">
    <h2>逐条结果</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>类型</th>
          <th>路径</th>
          <th>问题</th>
          <th>C-Recall</th>
          <th>C-Precision</th>
          <th>Faith</th>
          <th>Ans-Rel</th>
          <th>Ans-Corr</th>
          <th>明细</th>
        </tr>
      </thead>
      <tbody>
        {case_rows}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""

    report_path = output_dir / "report.html"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    return report_path
