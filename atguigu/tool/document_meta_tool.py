# atguigu/tool/document_meta_tool.py
"""
通用的 Markdown 元数据解析工具。

# ============================================================================
# 【改造说明：元数据通用化】
#
# 为什么需要：
# - 之前的切分把 `## 元数据` 当成普通正文，导致“城市：成都”这种字段
#   被当成可检索内容，既污染召回，又拿不到结构化字段；
# - 不同领域（旅游/听书/教育/产品手册）的元数据字段名不一样，
#   如果为每个领域写一套解析逻辑，代码就失去通用性。
#
# 这个模块只做一件通用的事：
# 1. 找到标题为“元数据/元信息/metadata/meta/文档信息/基本信息”的章节；
# 2. 把它下面的 `- key：value` / `- key: value` / `- **key**: value`
#    解析成一个普通字典；
# 3. 不关心 key 叫什么，也不关心 value 是什么领域的内容。
#
# 这样旅游的“城市：成都”、听书的“书名：三体”、教育的“课程名称：xxx”
# 都能用同一套代码解析，后续由节点决定哪些 key 有用。
# ============================================================================
"""

import re

# 这些标题会被当成“元数据章节”，大小写不敏感。
# 只放通用叫法，不放某个领域特有的标题。
METADATA_HEADING_NAMES = {
    "元数据",
    "元信息",
    "metadata",
    "meta",
    "文档信息",
    "基本信息",
}

# 匹配 Markdown 列表项里的键值对，例如：
# - 城市：成都
# - 书名: 三体
# - **系列编码**: general_purpose_programming_foundation
_METADATA_LINE_PATTERN = re.compile(
    r"^[-*]\s*\**([^：:*]+?)\**\s*[：:]\s*(.*)$"
)


def parse_metadata_section(lines):
    """
    从 Markdown 行列表里解析元数据章节。

    参数：
    - lines: 按行拆分后的 Markdown 文本

    返回：
    - dict: 元数据键值对；没有元数据章节时返回空字典

    解析规则：
    - 遇到标题，判断它是不是“元数据”类标题；
    - 在元数据章节内，只解析 `- key：value` 形式的列表项；
    - 遇到下一个标题，元数据章节结束。
    """
    metadata = {}
    in_metadata_section = False

    for line in lines:
        stripped = line.strip()

        heading_match = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading_match:
            heading_text = heading_match.group(1).strip().lower()
            in_metadata_section = heading_text in METADATA_HEADING_NAMES
            continue

        if not in_metadata_section:
            continue

        match = _METADATA_LINE_PATTERN.match(stripped)
        if not match:
            continue

        key = match.group(1).strip()
        value = match.group(2).strip()

        # 空 key 或空 value 不记录，避免把普通列表项误当元数据
        if key and value:
            metadata[key] = value

    return metadata
