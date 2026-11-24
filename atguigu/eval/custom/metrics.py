# atguigu/eval/custom/metrics.py
"""
检索评估指标（纯函数，不连数据库、不调模型）。

先解释一个最基础的概念：
- “召回结果”是系统返回的片段列表，已经按相关性从高到低排好序；
- “标准答案”（gold）是人工标注的、本来应该被找到的片段列表。

评估检索，本质上就是比较这两个列表：
- 该找到的有没有找到？ -> Recall
- 找到的排得靠不靠前？ -> MRR / nDCG

为了让比较逻辑只写一次，这里先把每条结果和每条标准答案都统一成
(file_title, title) 这样的二元组，再写一个 is_match 判断它们算不算同一条。
"""

import math
from typing import Sequence, Tuple

# 类型别名：一条检索结果/标准答案用 (文件名, 标题) 表示
RetrievalKey = Tuple[str, str]


def retrieval_key(doc: dict) -> RetrievalKey:
    """
    把一条召回结果转成 (file_title, 标题) 的二元组。

    为什么需要：
    - 召回结果里可能同时有 title、title_path；
    - 标准答案可能只标到文件级，也可能标到章节级；
    - 统一成二元组后，后面的匹配、召回率、MRR 都只处理一种结构。

    参数：
    - doc: 一条 reranked_doc，至少包含 file_title

    返回：
    - (file_title, title)；优先用 title_path，没有就用 title，再没有就是空串
    """
    title = doc.get("title_path") or doc.get("title") or ""
    return (doc.get("file_title") or "", title)


def is_match(retrieved: RetrievalKey, gold: RetrievalKey) -> bool:
    """
    判断一条召回结果是否命中一条标准答案。

    匹配规则：
    1. 文件名必须相同；
    2. 如果标准答案只标到文件级（gold_title 为空），
       那么同一文件下的任意章节都算命中；
    3. 如果标准答案标到章节级，标题完全相同算命中；
       另外兼容 "## 核心景点推荐" 和 "核心景点推荐" 这种格式差异，
       只要一方是另一方的后缀或包含关系，也算命中。

    参数：
    - retrieved: 召回结果的 (file_title, title)
    - gold: 标准答案的 (file_title, title)

    返回：
    - True 表示命中，False 表示未命中
    """
    retrieved_file, retrieved_title = retrieved
    gold_file, gold_title = gold

    # 文件不同，肯定不是同一条
    if retrieved_file != gold_file:
        return False

    # 标准答案只标到文件级：同一文件的任何章节都算命中
    if not gold_title:
        return True

    # 标题完全一致
    if retrieved_title == gold_title:
        return True

    # 兼容 Markdown 标题前缀（## / ###）和 title_path 拼接差异
    return (
        retrieved_title.endswith(gold_title)
        or gold_title.endswith(retrieved_title)
        or gold_title in retrieved_title
    )


def _hit_count(retrieved: Sequence[RetrievalKey], gold: Sequence[RetrievalKey], k: int) -> int:
    """
    统计前 k 条结果覆盖了多少条标准答案。

    注意：一条标准答案只要被前 k 条里的任意一条命中，就算覆盖一次；
    不要求它在前 k 条里出现多次（去重后只看“有没有”）。

    参数：
    - retrieved: 按相关性排好序的召回结果
    - gold: 标准答案
    - k: 只看前 k 条

    返回：
    - 被覆盖的标准答案条数
    """
    top_k = retrieved[:k]
    hits = 0

    for gold_item in gold:
        if any(is_match(retrieved_item, gold_item) for retrieved_item in top_k):
            hits += 1

    return hits


def recall_at_k(retrieved: Sequence[RetrievalKey], gold: Sequence[RetrievalKey], k: int) -> float:
    """
    Recall@k（召回率）。

    用大白话解释：
    标准答案里一共有 3 个“应该被找到”的片段，
    系统返回的前 k 条里找到了 2 个，
    那么 Recall@k = 2 / 3 ≈ 0.67。

    它衡量的是“有没有漏”：
    - 1.0 表示该找到的全找到了；
    - 0.0 表示一个都没找到；
    - 它不关心命中的片段排在第几，只看在不在前 k 条里。

    参数：
    - retrieved: 召回结果（已按相关性排序）
    - gold: 标准答案
    - k: 只看前 k 条，常用 5 和 10

    返回：
    - 0~1 的浮点数；gold 为空时返回 0
    """
    if not gold:
        return 0.0

    return _hit_count(retrieved, gold, k) / len(gold)


def context_recall_at_k(
    retrieved: Sequence[RetrievalKey],
    gold: Sequence[RetrievalKey],
    k: int,
) -> float:
    """
    Context Recall@k（上下文召回）。

    用大白话：
    - 标准答案里有多少份文档/章节，被前 k 条召回结果覆盖了；
    - 它和 Recall@k 在“文档级 gold”下数值一样，但名字对齐 RAGAS 口径，
      方便和行业指标对话。

    和 RAGAS 的区别：
    - RAGAS 会用 LLM 把参考答案拆成 claim，再逐条检查有没有被上下文覆盖；
    - 这里先用 gold 文档/章节做确定性代理，成本低、可复现；
    - 如果以后要更细的 claim 级 recall，可以再走 RAGAS。

    参数：
    - retrieved: 召回结果（已排序）
    - gold: 标准答案文档/章节
    - k: 只看前 k 条

    返回：
    - 0~1 的浮点数；gold 为空时返回 0
    """
    if not gold:
        return 0.0

    top_k = retrieved[:k]
    covered = 0

    for gold_item in gold:
        if any(is_match(retrieved_item, gold_item) for retrieved_item in top_k):
            covered += 1

    return covered / len(gold)


def context_precision_at_k(
    retrieved: Sequence[RetrievalKey],
    gold: Sequence[RetrievalKey],
    k: int,
) -> float:
    """
    Context Precision@k（上下文精确率）。

    用大白话：
    - 召回结果里，相关片段是不是排在前面；
    - 如果相关片段都排在第 1、2 位，分数高；
    - 如果相关片段排在第 9、10 位，分数低；
    - 如果前 k 条里一个相关片段都没有，分数是 0。

    计算方式（RAGAS 口径）：
    - 对每个“相关”的位置 i，算前 i 条里的精确率（相关数 / i）；
    - 把这些精确率求平均；
    - 这样既考虑了“有没有相关”，也考虑了“相关得靠不靠前”。

    这是确定性指标，不需要 LLM，适合每次改动都跑。

    参数：
    - retrieved: 召回结果（已排序）
    - gold: 标准答案文档/章节
    - k: 只看前 k 条

    返回：
    - 0~1 的浮点数
    """
    top_k = retrieved[:k]
    if not top_k or not gold:
        return 0.0

    # 先标出每个位置是否相关
    relevant_flags = [
        any(is_match(retrieved_item, gold_item) for gold_item in gold)
        for retrieved_item in top_k
    ]

    total_relevant = sum(relevant_flags)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_so_far = 0

    for index, is_relevant in enumerate(relevant_flags, start=1):
        if not is_relevant:
            continue

        relevant_so_far += 1
        precision_sum += relevant_so_far / index

    return precision_sum / total_relevant


def mrr(retrieved: Sequence[RetrievalKey], gold: Sequence[RetrievalKey]) -> float:
    """
    MRR（Mean Reciprocal Rank，平均倒数排名）。

    用大白话解释：
    - 第一个正确结果排第 1，得 1 分；
    - 排第 2，得 1/2 分；
    - 排第 5，得 1/5 分；
    - 一条都没找到，得 0 分。

    它衡量的是“第一个正确答案出现得早不早”。
    在只有一个标准答案的场景里，MRR 比 Recall 更敏感。

    参数：
    - retrieved: 召回结果（已按相关性排序）
    - gold: 标准答案

    返回：
    - 0~1 的浮点数
    """
    for rank, retrieved_item in enumerate(retrieved, start=1):
        if any(is_match(retrieved_item, gold_item) for gold_item in gold):
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved: Sequence[RetrievalKey],
    gold: Sequence[RetrievalKey],
    k: int,
) -> float:
    """
    nDCG@k（归一化折损累计增益）。

    用大白话解释：
    - 命中的位置越靠前，贡献越大；
    - 把所有命中的贡献加起来，得到 DCG；
    - 再除以“理想排序下能拿到的最大 DCG”，得到 0~1 的分数。

    这里用最简单的一档相关性：
    命中标准答案 = 1，没命中 = 0。

    # ========================================================================
    # 【改造说明：每条 gold 只让第一次命中的结果计分】
    #
    # 旧实现的问题：
    # - 当前测试集是“文档级 gold”（gold 标题为空）；
    # - 同一文档的多个 chunk 都会被 is_match 判成相关；
    # - 但 IDCG 只按 len(gold)=1 算理想值；
    # - 结果 nDCG 会算到 3、4，超过 1，完全不可信。
    #
    # 新实现：
    # - 用 used_gold 记录哪些 gold 已经被命中；
    # - 每条 gold 只让第一次命中的结果得分，后面的同文档 chunk 不再重复计分；
    # - 这样 DCG 不会超过 IDCG，nDCG 稳定在 0~1。
    #
    # 这是通用修正：文档级 gold 和章节级 gold 都适用。
    # ========================================================================

    和 MRR 的区别：
    - MRR 只看第一个正确结果；
    - nDCG 会考虑多个正确结果的整体排序质量。

    参数：
    - retrieved: 召回结果（已按相关性排序）
    - gold: 标准答案
    - k: 只看前 k 条

    返回：
    - 0~1 的浮点数
    """
    used_gold = set()
    relevance_flags = []

    for retrieved_item in retrieved[:k]:
        matched_index = None

        # 找一条还没被命中过的 gold
        for gold_index, gold_item in enumerate(gold):
            if gold_index not in used_gold and is_match(retrieved_item, gold_item):
                matched_index = gold_index
                break

        if matched_index is None:
            relevance_flags.append(0.0)
        else:
            used_gold.add(matched_index)
            relevance_flags.append(1.0)

    dcg = sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(relevance_flags)
    )

    # 理想情况下，前 min(len(gold), k) 个位置全是命中的 DCG
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg else 0.0


def item_name_hit(predicted_names, gold_names) -> float:
    """
    主体名命中：预测的主体名和标准主体名有没有交集。

    例子：
    - 预测 ["成都"]，标准 ["成都"] -> 1.0
    - 预测 ["成都"]，标准 ["成都交通指南"] -> 0（这里不做模糊匹配）
    - 预测 ["成都", "重庆"]，标准 ["成都"] -> 1.0

    这里只做去空白的精确比较，不做同义词/别名扩展；
    别名匹配属于更上层的业务逻辑，不应该混进评估指标。
    """
    predicted = {str(name).strip() for name in (predicted_names or []) if str(name).strip()}
    gold = {str(name).strip() for name in (gold_names or []) if str(name).strip()}

    return 1.0 if predicted & gold else 0.0


def refusal_correct(answer: str, expect_refusal: bool) -> float:
    """
    拒答正确率。

    用大白话解释：
    - 如果这条题本来就应该拒答（比如问知识库里没有的内容），
      答案里要明确说“没有找到 / 资料中没有 / 不知道”；
    - 如果这条题不该拒答，答案不能是空的。

    参数：
    - answer: 系统最终回答
    - expect_refusal: 测试集里标注的“是否应该拒答”

    返回：
    - 1.0 表示符合预期，0.0 表示不符合
    """
    answer = (answer or "").strip()

    if expect_refusal:
        markers = (
            "没有找到",
            "未找到",
            "没有相关",
            "资料中没有",
            "无法回答",
            "不知道",
            "没有识别到",
        )
        return 1.0 if any(marker in answer for marker in markers) else 0.0

    return 1.0 if answer else 0.0


def average(values) -> float:
    """
    计算平均值，自动忽略 None。

    为什么忽略 None：
    - judge 调用失败、或者 retrieval 模式没有 judge 分数时，
      对应字段是 None；
    - 直接 sum(values) 会报错，所以先过滤掉非数字。
    """
    numbers = [value for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else 0.0
