# atguigu/eval/run_eval.py
"""
评估入口：读测试集 -> 跑查询图 -> 算指标 -> 写结果/汇总/HTML 报告。

用法示例：
    # 只测检索：不生成答案、不调用 Web，速度快、成本低
    python -m atguigu.eval.run_eval --mode retrieval

    # 端到端：生成答案，并让 LLM judge 打分
    python -m atguigu.eval.run_eval --mode full

    # 只跑前 5 条，方便调试
    python -m atguigu.eval.run_eval --mode retrieval --limit 5

    # 需要把 Web 搜索也纳入评估时
    python -m atguigu.eval.run_eval --mode full --enable-web

设计说明：
- 测试集里 gold_docs 只标到文件级时，同一文件的任意章节都算命中；
  以后改造出 title_path 后，可以在 gold_docs 里补 title_path，做章节级评估。
- 如果某条 case 需要的文档还没导入 Milvus，会标记为 skipped，
  不会把“没导入”误判成“检索效果差”。
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from atguigu.eval.judge import judge_answer
from atguigu.eval.metrics import (
    average,
    item_name_hit,
    mrr,
    ndcg_at_k,
    recall_at_k,
    refusal_correct,
    retrieval_key,
)
from atguigu.eval.report import build_report
from atguigu.query_process.main_graph import QueryMainGraphRunner
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_milvus_client
from atguigu.tool.mongo_tool import clear_history

# 默认测试集路径：和本文件同目录下的 dataset/travel_eval.jsonl
DEFAULT_DATASET = Path(__file__).parent / "dataset" / "travel_eval.jsonl"


def load_dataset(path: Path):
    """
    读取 JSONL 测试集。

    每一行是一个 JSON 对象，字段说明见 dataset/travel_eval.jsonl 顶部注释
    （JSONL 不支持注释，具体说明放在 README.md 里）。
    """
    cases = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    return cases


def get_available_file_titles():
    """
    从 Milvus 读出“已经导入的文档名”集合。

    为什么要先查这个：
    - 你可能正在一份一份导入文档；
    - 如果测试集里的文档还没导入，跑出来的低分不是检索差，而是没数据；
    - 提前跳过这些 case，结果才可信。
    """
    try:
        client = get_milvus_client()

        if not client.has_collection("kb_chunks"):
            return set()

        # 查询前先 load，避免服务重启后集合未加载导致 query 失败
        client.load_collection("kb_chunks")

        # 只取 file_title，避免把 content 和向量都拉回来
        rows = client.query(
            collection_name="kb_chunks",
            filter="",
            output_fields=["file_title"],
            limit=10000,
        )

        return {
            row.get("file_title")
            for row in rows
            if row.get("file_title")
        }

    except Exception as error:
        logger.warning(f"读取 kb_chunks 失败，无法做导入检查: {error}")
        return set()


def ensure_collections_loaded():
    """
    评估前确保 kb_chunks / kb_item_names 已 load。

    为什么需要：
    - Milvus 的 hybrid_search 要求集合处于 loaded 状态；
    - 服务重启后，集合可能没有自动 load；
    - 不 load 的话，查询会报 collection not loaded，看起来像“检索效果差”，
      实际只是没准备好。
    """
    try:
        client = get_milvus_client()
        for name in ("kb_chunks", "kb_item_names"):
            if client.has_collection(name):
                client.load_collection(name)
    except Exception as error:
        logger.warning(f"预加载集合失败，继续尝试评估: {error}")


def case_is_available(case, available_titles):
    """
    判断一条 case 需要的文档是否都已导入。

    - 拒答题（expect_refusal=true）不依赖具体文档，直接跑；
    - 其他题要求 gold_docs 里的所有 file_title 都在 Milvus 里。
    """
    if case.get("expect_refusal"):
        return True

    required = {
        doc["file_title"]
        for doc in case.get("gold_docs", [])
    }

    return required.issubset(available_titles)


def run_case(case, mode):
    """
    跑一条测试 case，并计算检索指标/答案指标。

    参数：
    - case: 测试集里的一行
    - mode: retrieval（只检索）或 full（检索 + 生成 + judge）

    返回：
    - dict：包含状态、召回结果、答案、各项指标
    """
    session_id = f"eval-{case['id']}"

    # 每个 case 用独立 session，避免上一题的历史影响下一题
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

    # reranked_docs 可能为空（比如商品名确认阶段就拒答了）
    reranked_docs = result.get("reranked_docs") or []

    # 统一成 (file_title, title) 二元组，方便和 gold 比较
    retrieved = [retrieval_key(doc) for doc in reranked_docs]

    gold = [
        (
            doc["file_title"],
            doc.get("title_path") or doc.get("title") or "",
        )
        for doc in case.get("gold_docs", [])
    ]

    answer = result.get("answer") or ""

    row = {
        "id": case["id"],
        "type": case.get("type", "unknown"),
        "question": case["question"],
        "status": "ok",
        # 记录是否本来就是拒答题，报告里不会把“gold 为空”误判成 bad case
        "expect_refusal": case.get("expect_refusal", False),
        "item_names": result.get("item_names") or [],
        "answer": answer,
        "retrieved": [
            {"file_title": file_title, "title": title}
            for file_title, title in retrieved
        ],
        "gold": [
            {"file_title": file_title, "title": title}
            for file_title, title in gold
        ],
        "recall@5": recall_at_k(retrieved, gold, 5),
        "recall@10": recall_at_k(retrieved, gold, 10),
        "mrr": mrr(retrieved, gold),
        "ndcg@10": ndcg_at_k(retrieved, gold, 10),
        "item_name_hit": item_name_hit(
            result.get("item_names") or [],
            case.get("gold_item_names", []),
        ),
        # retrieval 模式不生成答案，拒答指标没有意义，留空
        "refusal_correct": (
            refusal_correct(answer, case.get("expect_refusal", False))
            if mode == "full"
            else None
        ),
    }

    # 只有 full 模式才调用 judge，避免 retrieval 模式产生额外 LLM 成本
    if mode == "full":
        contexts = [
            doc.get("content") or ""
            for doc in reranked_docs
        ]
        judge_result = judge_answer(
            question=case["question"],
            answer=answer,
            contexts=contexts,
            gold_facts=case.get("gold_facts", []),
        )
        row.update({
            f"judge_{key}": value
            for key, value in judge_result.items()
        })

    return row


def summarize(rows, mode, dataset_path):
    """
    汇总总体指标和分题型指标。

    为什么要分题型：
    总体平均分会被题型比例影响；分题型看，才能知道是
    “交通类差”还是“对比类差”，从而决定下一轮改什么。
    """
    evaluated = [row for row in rows if row["status"] == "ok"]
    skipped = [row for row in rows if row["status"] == "skipped"]
    errors = [row for row in rows if row["status"] == "error"]

    metric_names = [
        "recall@5",
        "recall@10",
        "mrr",
        "ndcg@10",
        "item_name_hit",
        "refusal_correct",
        "judge_faithfulness",
        "judge_relevance",
        "judge_citation",
    ]

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "dataset": str(dataset_path),
        "total": len(rows),
        "evaluated": len(evaluated),
        "skipped": len(skipped),
        "errors": len(errors),
        "overall": {},
        "by_type": {},
    }

    # 总体平均：None 会被 average() 自动忽略
    for name in metric_names:
        summary["overall"][name] = average([row.get(name) for row in evaluated])

    # 分题型平均
    grouped = defaultdict(list)
    for row in evaluated:
        grouped[row["type"]].append(row)

    for case_type, items in grouped.items():
        summary["by_type"][case_type] = {
            "count": len(items),
            **{
                name: average([row.get(name) for row in items])
                for name in metric_names
            },
        }

    return summary


def parse_args():
    """
    解析命令行参数。

    参数说明：
    - --dataset: 测试集路径
    - --mode: retrieval（只测检索）或 full（端到端）
    - --limit: 只跑前 N 条，调试用
    - --enable-web: 是否启用 Web 搜索（默认关闭，降低成本/噪音）
    """
    parser = argparse.ArgumentParser(description="RAG 项目评估脚本")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="测试集 JSONL 路径",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "full"],
        default="retrieval",
        help="retrieval=只测检索，full=检索+生成+judge",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 条，调试用",
    )
    parser.add_argument(
        "--enable-web",
        action="store_true",
        help="评估时启用 Web 搜索（默认关闭）",
    )

    return parser.parse_args()


def main():
    """
    评估主流程。

    1. 解析参数、设置评估开关；
    2. 读测试集；
    3. 检查哪些文档已经导入；
    4. 逐条跑 case，跳过缺失文档；
    5. 写 results.jsonl、summary.json、report.html。
    """
    args = parse_args()

    # 评估默认关闭 Web：否则每条题都会多一次外部调用，噪音和成本都高
    os.environ["EVAL_DISABLE_WEB"] = "0" if args.enable_web else "1"

    # retrieval 模式跳过答案生成：不需要 LLM 生成，只要 reranked_docs
    os.environ["EVAL_SKIP_ANSWER"] = "1" if args.mode == "retrieval" else "0"

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    logger.info(f"测试集共 {len(cases)} 条，模式={args.mode}")

    # 先确保集合已 load，再做“哪些文档已导入”的检查，
    # 否则服务重启后 query 可能直接报 collection not loaded
    ensure_collections_loaded()

    available_titles = get_available_file_titles()
    logger.info(f"Milvus 中已导入文档数: {len(available_titles)}")

    if not available_titles:
        logger.error(
            "Milvus 里还没有 kb_chunks 数据。请先导入至少一份文档，再运行评估。"
        )
        return 1

    rows = []

    for index, case in enumerate(cases, start=1):
        logger.info(f"[{index}/{len(cases)}] {case['id']}: {case['question']}")

        if not case_is_available(case, available_titles):
            required = [doc["file_title"] for doc in case.get("gold_docs", [])]
            logger.warning(f"跳过 {case['id']}，缺少文档: {required}")
            rows.append({
                "id": case["id"],
                "type": case.get("type", "unknown"),
                "question": case["question"],
                "status": "skipped",
                "reason": f"缺少文档: {required}",
            })
            continue

        try:
            rows.append(run_case(case, args.mode))
        except Exception as error:
            logger.exception(f"{case['id']} 执行失败")
            rows.append({
                "id": case["id"],
                "type": case.get("type", "unknown"),
                "question": case["question"],
                "status": "error",
                "reason": str(error),
            })

    output_dir = (
        Path(__file__).parent
        / "results"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    with open(results_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, args.mode, args.dataset)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    report_path = build_report(results_path, summary_path, output_dir)

    logger.info(f"评估完成，结果目录: {output_dir}")
    logger.info(f"HTML 报告: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
