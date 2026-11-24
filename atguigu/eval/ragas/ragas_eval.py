# atguigu/eval/ragas/ragas_eval.py
"""
RAGAS 评估方案（独立于现有的 run_eval.py）。

# ============================================================================
# 【为什么单独开一套】
#
# 现有 run_eval.py 是“自己实现”的评估：
# - 检索指标（Recall/MRR/nDCG/Context Precision/Recall）
# - 自己的中文 LLM judge（faithfulness / answer relevancy / correctness）
#
# 这套 ragas_eval.py 用 RAGAS 官方指标做对照：
# - Context Recall
# - Context Precision
# - Faithfulness
# - Answer Relevancy
# - Answer Correctness
#
# 两套结果可以并排看，互不覆盖：
# - run_eval.py 的结果在 results/<time>_retrieval/full
# - ragas_eval.py 的结果在 results/<time>_ragas
#
# 为什么不是直接替换：
# - RAGAS 的 prompt/指标实现会随版本变化；
# - RAGAS 默认偏英文，中文数据需要验证；
# - RAGAS 需要 ground truth（reference_answer）；
# - RAGAS 会额外产生不少 LLM 调用和成本。
#
# 依赖：
#   uv add "ragas>=0.2,<0.3"
#   或
#   uv pip install "ragas>=0.2,<0.3"
#
# 用法：
#   python -m atguigu.eval.ragas.ragas_eval --limit 5
#   python -m atguigu.eval.ragas.ragas_eval --dataset atguigu/eval/dataset/travel_eval.jsonl
#
# 注意：
# - 只跑有 reference_answer 的 case；没有参考答案的 case 会跳过；
# - 默认 max_workers=1，避免 RAGAS 并发调用把 LLM 打限流；
# - 不要和导入服务同时跑，BGE/LLM 会抢资源。
# ============================================================================
"""

import argparse
import html
import json
import math
import os
import sys
import types
from datetime import datetime
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from atguigu.config.config import LLMConfig
from atguigu.eval.custom.run_eval import (
    DEFAULT_DATASET,
    case_is_available,
    ensure_collections_loaded,
    get_available_file_titles,
    load_dataset,
)
from atguigu.query_process.main_graph import QueryMainGraphRunner
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.logger import logger
from atguigu.tool.mongo_tool import clear_history

# RAGAS 五个指标在 DataFrame 里的列名（不同版本可能略有差异，用包含匹配）
METRIC_KEYS = [
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]


class BGE3LangchainEmbeddings(Embeddings):
    """
    把项目的 BGE-M3 dense 向量包装成 LangChain Embeddings。

    RAGAS 的 Answer Relevancy / Answer Correctness 需要 embedding；
    这里复用项目已经在用的 BGE-M3，不额外引入 OpenAI embedding。
    """

    def embed_documents(self, texts):
        """批量编码文档文本，只取 dense 向量。"""
        if not texts:
            return []
        return get_bgem3_embeddings(list(texts))["dense"]

    def embed_query(self, text):
        """编码单条查询文本。"""
        return self.embed_documents([text])[0]


def _install_vertexai_compat_shim():
    """
    兼容 RAGAS 0.2.15 + langchain-community 0.4.x。

    背景：
    - RAGAS 0.2.15 的 ragas/llms/base.py 会无条件 import：
        from langchain_community.chat_models.vertexai import ChatVertexAI
    - langchain-community 0.4.2 已经移除了 chat_models.vertexai 模块；
    - 直接升级/降级 LangChain 会影响整个项目。

    做法：
    - 注册一个空的 vertexai 模块和占位 ChatVertexAI 类；
    - 我们只用 ChatOpenAI，不会实例化 ChatVertexAI；
    - 它只用于 RAGAS 内部的 isinstance 类型列表，占位类足够。
    """
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - 仅用于 RAGAS 的类型列表
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = module


def _load_ragas():
    """
    延迟导入 RAGAS，避免没安装时影响 --help。

    返回：
    - dict：包含 evaluate/EvaluationDataset/SingleTurnSample/包装器/指标/RunConfig
    """
    _install_vertexai_compat_shim()

    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
        )

        try:
            from ragas.run_config import RunConfig
        except ImportError:
            from ragas import RunConfig

        return {
            "evaluate": evaluate,
            "EvaluationDataset": EvaluationDataset,
            "SingleTurnSample": SingleTurnSample,
            "LangchainLLMWrapper": LangchainLLMWrapper,
            "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
            "RunConfig": RunConfig,
            "metrics": [
                Faithfulness(),
                AnswerRelevancy(),
                LLMContextRecall(),
                LLMContextPrecisionWithReference(),
                AnswerCorrectness(),
            ],
        }
    except Exception as error:
        raise RuntimeError(
            "RAGAS 未安装或版本不兼容。请先安装：\n"
            "  uv sync --extra ragas\n"
            "或\n"
            '  uv pip install "ragas>=0.2,<0.3"\n'
            f"原始错误: {error}"
        ) from error


def _build_judge_llm():
    """
    构建 RAGAS 使用的 LLM。

    推荐用 EVAL_JUDGE_MODEL 指定一个比生成模型更强的模型；
    没配置时退回项目默认模型。
    """
    model_name = os.getenv("EVAL_JUDGE_MODEL", LLMConfig.llm_default_model)

    return ChatOpenAI(
        model=model_name,
        api_key=LLMConfig.openai_api_key,
        base_url=LLMConfig.openai_api_base,
        temperature=0,
        timeout=120,
    )


def _collect_sample(case):
    """
    跑一条 case，构造成 RAGAS 样本。

    参数：
    - case: 测试集里的一行

    返回：
    - dict: RAGAS 样本 + 元信息
    - None: 没有参考答案，或没有生成答案
    """
    session_id = f"ragas-{case['id']}"

    try:
        clear_history(session_id)
    except Exception as error:
        logger.warning(f"清理历史失败，忽略: {error}")

    state = {
        "task_id": case["id"],
        "original_query": case["question"],
        "session_id": session_id,
    }

    result = QueryMainGraphRunner.create_and_run(state)

    answer = result.get("answer") or ""
    if not answer.strip():
        logger.warning(f"{case['id']} 没有生成答案，跳过")
        return None

    # 答案节点使用父块扩展后的文档；RAGAS 的 contexts 也用同一份
    docs = result.get("expanded_docs") or result.get("reranked_docs") or []
    contexts = [
        doc.get("content") or ""
        for doc in docs
        if (doc.get("content") or "").strip()
    ][:10]

    if not contexts:
        logger.warning(f"{case['id']} 没有检索上下文，跳过")
        return None

    reference = (case.get("reference_answer") or "").strip()
    if not reference:
        logger.warning(f"{case['id']} 没有 reference_answer，跳过")
        return None

    return {
        "id": case["id"],
        "type": case.get("type", "unknown"),
        "question": case["question"],
        "answer": answer,
        "reference": reference,
        "contexts": contexts,
    }


def _metric_column_map(df):
    """
    在 RAGAS 结果 DataFrame 里找到五个指标列。

    不同 RAGAS 版本的列名可能带前缀/后缀，
    这里用“包含匹配”做兼容，返回 {标准名: 实际列名}。
    """
    mapping = {}

    for key in METRIC_KEYS:
        for column in df.columns:
            if key in str(column).lower():
                mapping[key] = column
                break

    return mapping


def _to_float(value):
    """把 numpy/NaN/空值转成 JSON 可写的 float 或 None。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number):
        return None
    return round(number, 4)


def _write_report(output_dir, rows, summary):
    """
    生成一个轻量 HTML 报告。

    只展示 RAGAS 五个指标，和现有 report.html 分开，避免口径混在一起。
    """
    cards = "".join(
        f'<div class="card"><div class="label">{html.escape(key)}</div>'
        f'<div class="value">{summary.get(key, "-")}</div></div>'
        for key in METRIC_KEYS
    )

    rows_html = []

    for row in rows:
        cells = [
            html.escape(str(row.get("id", ""))),
            html.escape(str(row.get("type", ""))),
            html.escape(str(row.get("question", ""))),
        ]

        for key in METRIC_KEYS:
            cells.append(html.escape(str(row.get(key, "-"))))

        cells.append(html.escape(str(row.get("answer", ""))[:300]))
        rows_html.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    header = "".join(f"<th>{html.escape(key)}</th>" for key in METRIC_KEYS)

    content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>RAGAS 评估报告</title>
  <style>
    body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
            background:#f5f7fb; color:#1f2937; padding:24px; }}
    .container {{ max-width:1400px; margin:0 auto; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
              gap:12px; margin:20px 0; }}
    .card {{ background:#fff; padding:14px 16px; border-radius:10px;
             box-shadow:0 2px 8px rgba(0,0,0,.06); }}
    .label {{ font-size:12px; color:#6b7280; }}
    .value {{ font-size:24px; font-weight:700; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff;
             border-radius:10px; overflow:hidden; font-size:13px; }}
    th, td {{ padding:10px 8px; border-bottom:1px solid #e5e7eb; text-align:left;
              vertical-align:top; }}
    th {{ background:#f9fafb; }}
    .meta {{ color:#6b7280; font-size:13px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>RAGAS 评估报告</h1>
  <div class="meta">生成时间：{html.escape(summary.get("generated_at", ""))}</div>
  <div class="cards">{cards}</div>
  <table>
    <thead><tr><th>ID</th><th>类型</th><th>问题</th>{header}<th>答案摘要</th></tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
</body>
</html>
"""

    path = output_dir / "ragas_report.html"
    path.write_text(content, encoding="utf-8")
    return path


def run_ragas(samples, max_workers, max_retries):
    """
    调用 RAGAS 计算五个指标。

    参数：
    - samples: _collect_sample 返回的样本列表
    - max_workers: RAGAS 并发数，默认 1
    - max_retries: 单条样本失败重试次数

    返回：
    - (DataFrame, summary dict)
    """
    ragas = _load_ragas()

    dataset = ragas["EvaluationDataset"](
        samples=[
            ragas["SingleTurnSample"](
                user_input=sample["question"],
                response=sample["answer"],
                retrieved_contexts=sample["contexts"],
                reference=sample["reference"],
            )
            for sample in samples
        ]
    )

    llm = ragas["LangchainLLMWrapper"](_build_judge_llm())
    embeddings = ragas["LangchainEmbeddingsWrapper"](BGE3LangchainEmbeddings())

    try:
        run_config = ragas["RunConfig"](
            max_workers=max_workers,
            max_retries=max_retries,
            max_wait=60,
            timeout=180,
        )
    except TypeError:
        # 兼容字段名不同的 RAGAS 版本
        run_config = ragas["RunConfig"](
            max_workers=max_workers,
            max_retries=max_retries,
        )

    try:
        result = ragas["evaluate"](
            dataset=dataset,
            metrics=ragas["metrics"],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,
        )
    except TypeError:
        # 兼容少数 RAGAS 版本：evaluate 不接受 run_config/raise_exceptions
        result = ragas["evaluate"](
            dataset=dataset,
            metrics=ragas["metrics"],
            llm=llm,
            embeddings=embeddings,
        )

    df = result.to_pandas()
    column_map = _metric_column_map(df)

    # 把标准指标名映射到实际列；同时保留原始列
    normalized_rows = []

    for index, sample in enumerate(samples):
        row = {
            "id": sample["id"],
            "type": sample["type"],
            "question": sample["question"],
            "answer": sample["answer"],
            "reference": sample["reference"],
        }

        if index < len(df):
            for key in METRIC_KEYS:
                column = column_map.get(key)
                row[key] = _to_float(df.iloc[index][column]) if column else None

        normalized_rows.append(row)

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples": len(samples),
    }

    for key in METRIC_KEYS:
        values = [
            row[key]
            for row in normalized_rows
            if isinstance(row.get(key), (int, float))
        ]
        summary[key] = round(sum(values) / len(values), 4) if values else None

    return normalized_rows, summary


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="RAGAS 评估（独立方案）")

    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--max-workers", type=int, default=1, help="RAGAS 并发数，默认 1")
    parser.add_argument("--max-retries", type=int, default=5, help="RAGAS 单条重试次数")
    parser.add_argument("--enable-web", action="store_true", help="评估时启用 Web 搜索")

    return parser.parse_args()


def main():
    """评估主流程。"""
    args = parse_args()

    # 和 run_eval 保持一致：默认关闭 Web，必须生成答案
    os.environ["EVAL_DISABLE_WEB"] = "0" if args.enable_web else "1"
    os.environ["EVAL_SKIP_ANSWER"] = "0"

    try:
        _load_ragas()
    except RuntimeError as error:
        print(error)
        return 1

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    ensure_collections_loaded()
    available_titles = get_available_file_titles()

    if not available_titles:
        print("Milvus 里没有 kb_chunks，请先导入文档。")
        return 1

    samples = []
    skipped = []

    for index, case in enumerate(cases, start=1):
        logger.info(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")

        if not case_is_available(case, available_titles):
            skipped.append({"id": case["id"], "reason": "缺少文档"})
            continue

        try:
            sample = _collect_sample(case)
        except Exception as error:
            logger.exception(f"{case['id']} 执行失败")
            skipped.append({"id": case["id"], "reason": str(error)})
            continue

        if not sample:
            skipped.append({"id": case["id"], "reason": "无答案/无上下文/无参考答案"})
            continue

        samples.append(sample)

    if not samples:
        print("没有可评估样本（需要文档已导入，且有 reference_answer）。")
        return 1

    output_dir = (
        Path(__file__).parent
        / "results"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ragas"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = run_ragas(
        samples,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )

    summary["skipped"] = skipped
    summary["total_cases"] = len(cases)
    summary["evaluated"] = len(samples)

    # 保存样本、逐条结果、汇总和 HTML
    with open(output_dir / "ragas_samples.jsonl", "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(output_dir / "ragas_scores.jsonl", "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(output_dir / "ragas_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    report_path = _write_report(output_dir, rows, summary)

    print(f"RAGAS 评估完成，目录: {output_dir}")
    print(f"HTML 报告: {report_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
