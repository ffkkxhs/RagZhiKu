# atguigu/import_process/nodes/node_document_split.py
"""
文档切分节点（通用版）。

# ============================================================================
# 【改造说明：从“按每个标题切”改成“标题树 + 完整路径 + 旅游标签”】
#
# 原设计的问题：
# - 每遇到一个 # 标题就切一刀，H3 子标题也被当成独立章节；
# - chunk 的 title 只有当前标题，丢了 H1/H2 的父级上下文；
# - 短章节不带标题前缀、长章节带，chunk 结构不一致；
# - `## 元数据` 被当成正文入库，污染检索；
# - 不同领域（旅游/听书/教育/产品手册）的层级结构不同，
#   代码里没有通用的“标题路径”概念。
#
# 这次改造的原则：
# 1. 用“标题栈”维护当前所处的层级，生成 title_path（例如：
#    “成都景点推荐 > 核心景点推荐 > 都江堰”）；
# 2. 每个 chunk 都带完整 title_path，并把 title_path 拼进 content 前缀，
#    保证短块和长块的语义结构一致；
# 3. 不解析、不依赖源文档的 `## 元数据`：只跳过标题为“元数据/元信息/metadata”
#    的章节，不读取字段；旅游标签由主体识别节点从标题/正文抽取；
# 4. 参数（chunk 大小、重叠、整节不切的上限）都从环境变量读，
#    不同文档类型可以调整，不在代码里写死某个领域的规则。
#
# 注意：
# - 这里只负责“切分”，不关心 item_name 是城市、书名还是商品名；
# - 领域相关的判断放到主体识别节点，切分节点保持通用。
# ============================================================================
"""

import hashlib
import json
import os
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger

# 可以通过环境变量调整，避免把某个领域的经验写死在代码里：
# - CHUNK_SIZE：长章节二次切分时的目标大小（字符）
# - CHUNK_OVERLAP：相邻 chunk 的重叠字符数
# - MAX_SECTION_CHARS：章节正文小于该值时整节作为一个 chunk
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MAX_SECTION_CHARS = int(os.getenv("MAX_SECTION_CHARS", "800"))

# 不解析元数据字段，只把“标题恰好是这些名字”的章节整体跳过，
# 避免把 `## 元数据` 里的键值对当正文检索。
# 如果文档没有这个章节，这段逻辑不会生效。
METADATA_SECTION_TITLES = {"元数据", "元信息", "metadata", "meta"}

# 父块层级：默认把 H2 作为父块（H1 > H2 下的所有 H3 合进 H2）。
# 如果文档没有 H2，会自动退到 H1。
PARENT_LEVEL = int(os.getenv("PARENT_LEVEL", "2"))
# 父块正文的最大字节数，避免一次回捞太长
PARENT_MAX_BYTES = int(os.getenv("PARENT_MAX_BYTES", "6000"))


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：把 Markdown 切成带层级路径的 chunk。

    产出的每个 chunk 至少包含：
    - title：当前小标题
    - title_path：从 H1 到当前标题的完整路径
    - level：标题层级（H1=1, H2=2 ...）
    - parent_id：父 section 的稳定 ID，用于后续父块回捞
    - section_id：当前章节的稳定 ID
    - chunk_id：当前 chunk 的稳定 ID
    - file_title：文档名
    - content：带【title_path】前缀的正文
    - part：章节内的第几片
    - labels：由主体识别节点补充的旅游标签（entity/category/granularity）
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        """
        节点入口：读取 Markdown，按标题树切章节，再切成 chunk。

        参数：
        - state: 导入图状态，至少包含 md_path 和 file_title

        返回：
        - {"chunks": final_chunks}
        """
        file_title = state.get("file_title")
        if not file_title:
            logger.error("file_title 不能为空")
            raise ValueError("file_title 不能为空")

        md_path = state.get("md_path")
        if not md_path:
            logger.error("md_path 不能为空")
            raise ValueError("md_path 不能为空")

        md_path_obj = Path(md_path)
        if not md_path_obj.is_file():
            logger.error(f"md 文件不存在: {md_path}")
            raise FileNotFoundError(f"md 文件不存在: {md_path}")

        with open(md_path_obj, "r", encoding="utf-8") as file:
            md_content = file.read()

        # 统一换行，后面按行处理
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        lines = md_content.split("\n")

        # 不再解析源文档里的 `## 元数据`：文档结构不是契约，
        # 不假设一定存在“城市/书名/课程名称”这类字段。
        # 旅游标签（entity/category/granularity）由后续主体识别节点
        # 从标题和正文中抽取。
        sections = self._split_sections(lines, file_title)

        # 第二步：构建父块，并给每个 section 标上 parent_id
        parents = self._build_parents(sections, file_title)

        # 第三步：章节 -> 子 chunk
        final_chunks = []
        for section in sections:
            final_chunks.extend(self._section_to_chunks(section))

        # 调试产物跟着当前文档走，避免覆盖其他文档
        out_path = md_path_obj.parent / "split_chunks.json"
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(final_chunks, ensure_ascii=False, indent=4))

        parent_path = md_path_obj.parent / "split_parents.json"
        with open(parent_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(parents, ensure_ascii=False, indent=4))

        logger.info(
            f"切分完成: {len(sections)} 个章节 -> {len(final_chunks)} 个 chunk，"
            f"{len(parents)} 个父块"
        )
        return {
            "chunks": final_chunks,
            "parents": parents,
        }

    def _split_sections(self, lines, file_title):
        """
        用“标题栈”把 Markdown 切成章节树。

        为什么用栈：
        - H1/H2/H3 是嵌套关系，不是平铺关系；
        - 遇到 H3 时，它属于最近的 H2；遇到新的 H2 时，之前的 H3 应该出栈；
        - 用栈可以随时得到“当前完整路径”，例如：
          成都景点推荐 > 核心景点推荐 > 都江堰

        参数：
        - lines: Markdown 行列表
        - file_title: 文档名

        返回：
        - list[dict]：每个章节的 title/title_path/level/parent_path/content
        """
        sections = []
        heading_stack = []          # 元素是 (level, title)
        current_lines = []          # 当前章节的正文行
        current_title_path = ""     # 当前完整标题路径
        current_level = 0           # 当前标题层级
        current_is_metadata = False # 当前章节是否是“元数据”章节（只跳过，不解析）
        in_code_block = False
        code_marker = ""

        def flush_current_section():
            """
            把当前累积的正文落成一个 section。

            为什么单独抽成函数：
            - 遇到新标题、文件结束时都要执行同样的“收尾”逻辑；
            - 抽出来避免重复代码。
            """
            nonlocal current_lines

            content = "\n".join(current_lines).strip()

            # 只跳过标题为“元数据/元信息/metadata”的章节，
            # 不解析其中的字段，也不依赖字段名。
            # 注意：空章节也要保留（例如只有 H3 子节的 H2），
            # 否则父块构建时找不到父节点。
            if not current_is_metadata:
                title = heading_stack[-1][1] if heading_stack else ""
                parent_path = " > ".join(
                    item_title for _, item_title in heading_stack[:-1]
                )

                # 用 file_title + title_path 生成稳定 ID，
                # 同一份文档重复切分时 ID 不变，方便增量更新和去重
                section_id = hashlib.sha1(
                    f"{file_title}|{current_title_path}".encode("utf-8")
                ).hexdigest()[:16]

                sections.append({
                    "title": title,
                    "title_path": current_title_path,
                    "level": current_level,
                    "parent_path": parent_path,
                    "section_id": section_id,
                    "content": content,
                    "file_title": file_title,
                })

            current_lines = []

        for line in lines:
            stripped = line.strip()

            # 代码围栏里的 # 不是标题，需要跳过
            fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
            if fence_match:
                marker = fence_match.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_marker = marker
                elif code_marker == marker:
                    in_code_block = False
                    code_marker = ""

                current_lines.append(line)
                continue

            heading_match = None
            if not in_code_block:
                heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)

            if heading_match:
                # 遇到新标题：先把上一节收尾
                flush_current_section()

                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # 维护标题栈：弹出层级 >= 当前标题的旧标题
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))

                current_level = level
                current_title_path = " > ".join(
                    item_title for _, item_title in heading_stack
                )
                current_is_metadata = (
                    title.strip().lower() in METADATA_SECTION_TITLES
                )

            else:
                current_lines.append(line)

        # 文件结束，收尾最后一节
        flush_current_section()
        return sections

    def _build_parents(self, sections, file_title):
        """
        构建父块，并给每个 section 标上 parent_id。

        规则：
        - 默认把 H2 作为父块；
        - H2 下面的所有 H3 子节都合进这个 H2 的父块内容；
        - 如果文档没有 H2，就退到最近的 H1；
        - 找不到合适的祖先时，父块就是自己。

        参数：
        - sections: _split_sections 产出的章节列表
        - file_title: 文档名

        返回：
        - list[dict]：父块列表，每条包含 parent_id/title_path/level/content
        """
        if not sections:
            return []

        section_by_id = {section["section_id"]: section for section in sections}

        # 1. 给每个 section 找父 section（最近的、level <= PARENT_LEVEL 的祖先）
        for section in sections:
            candidates = []
            for candidate in sections:
                if candidate["level"] > PARENT_LEVEL:
                    continue
                if (
                    candidate["title_path"] == section["title_path"]
                    or section["title_path"].startswith(candidate["title_path"] + " > ")
                ):
                    candidates.append(candidate)

            if candidates:
                # 路径越长，层级越深，取最近的祖先
                parent_section = max(candidates, key=lambda item: len(item["title_path"]))
            else:
                # 没有合适祖先（例如只有 H1/H3），父块用自己
                parent_section = section

            section["parent_id"] = parent_section["section_id"]

        # 2. 生成父块：父 section + 它的所有后代 section
        parent_map = {}

        for section in sections:
            parent_id = section["parent_id"]
            parent_section = section_by_id[parent_id]

            content_parts = []
            for candidate in sections:
                if (
                    candidate["title_path"] == parent_section["title_path"]
                    or candidate["title_path"].startswith(parent_section["title_path"] + " > ")
                ):
                    if candidate["content"].strip():
                        title = candidate["title"]
                        content_parts.append(
                            f"{title}\n{candidate['content']}" if title else candidate["content"]
                        )

            content = "\n\n".join(content_parts).strip()
            if not content:
                continue

            parent_map[parent_id] = {
                "parent_id": parent_id,
                "file_title": file_title,
                "title_path": parent_section["title_path"] or file_title,
                "level": parent_section["level"],
                "content": self._clip_bytes(content, PARENT_MAX_BYTES),
                # 标签由主体识别节点统一补充
                "labels": {},
                "item_name": "",
            }

        return list(parent_map.values())

    def _section_to_chunks(self, section):
        """
        把一个章节切成一个或多个 chunk。

        切分策略：
        1. 正文 <= MAX_SECTION_CHARS：整节作为一个 chunk，避免把短小节切碎；
        2. 含 HTML 表格：按 </tr> 切，尽量保持表格行完整；
        3. 其余长正文：用 RecursiveCharacterTextSplitter 按 CHUNK_SIZE/OVERLAP 切。

        每个 chunk 的 content 都会加上【title_path】前缀，
        这样即使是一个很短的块，模型也知道它属于哪一章、哪一节。
        """
        content = section["content"]

        # 空章节（例如只有子标题的 H2）不产生子 chunk，
        # 但它在 _build_parents 里仍然作为父节点存在。
        if not content.strip():
            return []

        # 按 Milvus VARCHAR 的字节上限做保护，避免长标题/长路径导致插入失败。
        # 这些字段是展示和引用用的，截断不会影响正文检索。
        safe_title = self._clip_bytes(section["title"], 100)
        safe_title_path = self._clip_bytes(
            section["title_path"] or section["file_title"], 1000
        )
        # parent_id 是父 section 的稳定 ID（16 位 hash），不需要再截断
        parent_id = section["parent_id"]

        prefix = f"【{safe_title_path}】\n" if safe_title_path else ""

        if len(content) <= MAX_SECTION_CHARS:
            pieces = [content]
        elif "<table" in content:
            pieces = self._split_big_table(content)
        else:
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", " "],
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                length_function=len,
            )
            pieces = splitter.split_text(content)

        chunks = []
        for index, piece in enumerate(pieces, start=1):
            # chunk_id 用 section_id + part 生成，重复导入时保持稳定
            chunk_id = hashlib.sha1(
                f"{section['section_id']}|{index}".encode("utf-8")
            ).hexdigest()[:16]

            chunks.append({
                "title": safe_title,
                "title_path": safe_title_path,
                "level": section["level"],
                "parent_id": parent_id,
                "section_id": section["section_id"],
                "chunk_id": chunk_id,
                "file_title": section["file_title"],
                # 安全网：content 字段是 VARCHAR(5000)，按字节截到 4800，
                # 避免极端长表格/长段落直接插入失败。
                "content": self._clip_bytes(prefix + piece, 4800),
                "part": index,
                # 旅游标签由 node_item_name_recognition 统一补充
                "labels": {},
            })

        return chunks

    @staticmethod
    def _clip_bytes(text, max_bytes):
        """
        按 UTF-8 字节数截断文本。

        为什么需要：
        - Milvus 的 VARCHAR 长度单位是字节，不是字符；
        - 一个中文占 3 个字节，100 字节大约只能放 33 个中文；
        - 直接按字符截断可能仍然超限，按字节截断后再解码更稳。
        """
        raw = (text or "").encode("utf-8")

        if len(raw) <= max_bytes:
            return text or ""

        # errors="ignore" 防止把多字节字符截成半个
        return raw[:max_bytes].decode("utf-8", errors="ignore")

    def _split_big_table(self, content, max_len=3000):
        """
        大表格切分：HTML 表格按 </tr> 切，Markdown 管道表格按行切。

        为什么不能直接按字符切：
        - 表格切碎后表头和数据行会分离，列含义就丢了；
        - 按 </tr> 切至少保证每一片是完整的表格行。
        """
        if "</tr>" in content.lower():
            units = re.split(r"(?<=</tr>)", content, flags=re.IGNORECASE)
        else:
            units = content.splitlines(keepends=True)

        pieces = []
        buffer = ""

        for unit in units:
            if buffer and len(buffer) + len(unit) > max_len:
                pieces.append(buffer)
                buffer = unit
            else:
                buffer += unit

        if buffer.strip():
            pieces.append(buffer)

        return pieces or [content[:max_len]]


if __name__ == "__main__":
    # 单节点调试入口
    node = NodeDocumentSplit()
    init_state = {
        "md_path": r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
        "file_title": "hak180产品安全手册",
    }
    result = node(init_state)
    logger.info(json.dumps(result, ensure_ascii=False, indent=4)[:1000])
