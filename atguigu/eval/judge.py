# atguigu/eval/judge.py
"""
LLM-as-judge：用大模型给最终答案打分。

为什么需要它：
- Recall/MRR 只能评估“资料找得准不准”，不能判断“答案写得好不好”；
- 人工逐条打分成本太高，先用模型做初筛，再人工抽检校准。

三个注意点：
1. judge 模型最好和生成答案的模型不是同一个，避免“自己夸自己”；
2. temperature 设 0，固定评分标准（rubric），减少随机性；
3. judge 输出必须是结构化 JSON；解析失败就返回 None，不能中断整个评估。
"""

import json
import os

from langchain.chat_models import init_chat_model

from atguigu.config.config import LLMConfig
from atguigu.tool.logger import logger

# 评分标准写在提示词里，所有 case 共用同一套标准，保证可比性
JUDGE_PROMPT = """
你是一名严格的 RAG 答案评审员。请根据【问题】【参考答案要点】【检索资料】【模型回答】打分。

评分维度（每项 1~5 分）：
- faithfulness：回答中的事实是否都能在【检索资料】里找到依据，有没有编造。
  5 = 所有关键事实都有依据；1 = 大量编造。
- relevance：回答是否切题，有没有答非所问。
  5 = 直接回答问题；1 = 完全跑题。
- citation：回答里的来源/章节是否和【检索资料】对得上。
  5 = 引用准确；1 = 引用错误或完全没有依据。

只返回一个 JSON 对象，不要输出 Markdown 代码块，不要输出额外解释：
{{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "citation": 1-5,
  "reason": "一句话说明扣分原因；如果满分就写‘无’"
}}

【问题】
{question}

【参考答案要点】
{gold_facts}

【检索资料】
{contexts}

【模型回答】
{answer}
"""


def _build_judge_model():
    """
    创建 judge 模型。

    优先读环境变量 EVAL_JUDGE_MODEL，方便你换一个和生成模型不同的模型；
    没配置时退回项目默认模型（demo 阶段可以接受，生产最好换掉）。
    """
    model_name = os.getenv("EVAL_JUDGE_MODEL", LLMConfig.llm_default_model)

    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=LLMConfig.openai_api_key,
        base_url=LLMConfig.openai_api_base,
        temperature=0,  # 评分任务要稳定，不要发挥
    )


def _format_contexts(contexts) -> str:
    """
    把召回片段拼成带编号的文本。

    编号的作用：
    - 让 judge 能对应“第 1 条资料、第 2 条资料”；
    - 如果答案里也有 [1]、[2] 引用，评分时更容易对照。
    """
    if not contexts:
        return "（无检索资料）"

    return "\n\n".join(
        f"[{index}] {text}"
        for index, text in enumerate(contexts, start=1)
    )


def _clamp_score(value):
    """
    把分数限制在 1~5 的整数；无法解析时返回 None。

    模型有时会返回 0、6、3.5 或者字符串，这里统一收敛，
    避免后续求平均值时出现异常值。
    """
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None

    return max(1, min(5, score))


def judge_answer(question: str, answer: str, contexts, gold_facts):
    """
    对一条答案做 LLM 判分。

    参数：
    - question: 用户问题
    - answer: 模型最终回答
    - contexts: 检索到的资料文本列表
    - gold_facts: 人工标注的关键事实列表（可以为空）

    返回：
    - dict: {"faithfulness": 1-5|None, "relevance": ..., "citation": ..., "reason": ...}
    - 任何一步失败都返回 None 分数的 dict，不抛异常
    """
    # 空答案直接判 0 分，不值得再花一次 LLM 调用
    if not (answer or "").strip():
        return {
            "faithfulness": None,
            "relevance": None,
            "citation": None,
            "reason": "答案为空",
        }

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_facts="\n".join(gold_facts) if gold_facts else "（未提供，按资料是否自洽判断）",
        contexts=_format_contexts(contexts),
        answer=answer,
    )

    try:
        model = _build_judge_model()
        response = model.invoke(prompt)
        text = (response.content or "").strip()

        # 兼容模型偶尔加上的 Markdown 代码块
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)

    except Exception as error:
        logger.warning(f"judge 调用或解析失败: {error}")
        return {
            "faithfulness": None,
            "relevance": None,
            "citation": None,
            "reason": f"judge 失败: {error}",
        }

    return {
        "faithfulness": _clamp_score(data.get("faithfulness")),
        "relevance": _clamp_score(data.get("relevance")),
        "citation": _clamp_score(data.get("citation")),
        "reason": str(data.get("reason", "")),
    }
