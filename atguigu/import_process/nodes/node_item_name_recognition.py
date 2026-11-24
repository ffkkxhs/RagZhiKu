# atguigu/import_process/nodes/node_item_name_recognition.py
"""
旅游领域的主体/标签识别节点。

# ============================================================================
# 【改造说明：从“通用主体名 + 元数据优先”改成“旅游标签抽取”】
#
# 原设计的问题：
# - 假设文档一定有 `## 元数据`，并从“城市/书名/课程名称”等字段取主体；
# - 把所有成都文档都压成 item_name=成都，丢掉了交通/美食/住宿/线路/景点；
# - 一个全局主体优先级表无法覆盖不同领域；
# - 元数据值可能是“云南多城市”“昆明、大理、丽江”，直接当主体名会污染。
#
# 现在的做法：
# 1. 完全不读取源文档的 `## 元数据`；
# 2. 只根据文档标题 + 开头正文，抽取旅游领域标签：
#    - entity：文档主要覆盖的目的地/区域
#    - category：景点/交通/美食/住宿/线路/综合
#    - granularity：city/region/poi/route/unknown
#    - mentioned_entities：正文提到但非主体的地点
# 3. 标签存进 Milvus 的 labels JSON 字段；
# 4. item_name 仍然保留，值等于 entity，主要用于展示/兼容旧代码；
#    category 等维度放在 labels 里，不把交通/美食/住宿压成同一个主体。
#
# 这是旅游领域的特化；换领域时替换 TRAVEL_LABEL_* 提示词即可。
# ============================================================================
"""

import json
from pathlib import Path

from langchain.chat_models import init_chat_model
from pymilvus import DataType

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import TRAVEL_LABEL_SYSTEM_PROMPT, TRAVEL_LABEL_TEMPLATE
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_embedding_tool import get_bgem3_embeddings
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.llm_tool import invoke_with_retry
from atguigu.tool.logger import logger
from atguigu.tool.milvus_client_tool import get_milvus_client

# 旅游领域的受控枚举。模型输出不在枚举里时回退到默认值。
VALID_CATEGORIES = {"景点", "交通", "美食", "住宿", "线路", "综合"}
VALID_GRANULARITIES = {"city", "region", "poi", "route", "unknown"}


class NodeItemNameRecognition(NodeBase):
    """
    主体/标签识别节点：给文档和 chunk 打上旅游领域标签。
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState):
        """
        节点入口。

        流程：
        1. 校验 chunks/file_title；
        2. 用标题 + 开头正文让 LLM 抽旅游标签；
        3. 把标签回填到每个 chunk；
        4. 写入 kb_item_names（item_name=entity，labels=完整标签）；
        5. 返回 chunks 和 entity。
        """
        chunks = state.get("chunks")
        file_title = state.get("file_title")

        if not chunks:
            logger.error("chunks 不能为空")
            raise ValueError("chunks 不能为空")

        if not file_title:
            logger.error("file_title 不能为空")
            raise ValueError("file_title 不能为空")

        labels = self.get_labels(chunks, file_title)

        # item_name 用 entity，截到 Milvus VARCHAR(100) 的字节上限；
        # category/granularity 等放在 labels JSON 里，不丢维度。
        entity = self._clip_bytes(labels.get("entity") or file_title, 100)
        labels["entity"] = entity

        # 调试文件跟着当前文档走
        md_path = state.get("md_path")
        out_dir = Path(md_path).parent if md_path else Path(state.get("local_dir") or ".")

        # 父块也要带上同一份标签，查询侧回捞后才能做同样的软加权
        parents = state.get("parents") or []

        self.back_labels_to_chunk(chunks, labels, entity, out_dir)
        self.back_labels_to_parents(parents, labels, entity, out_dir)

        collection_name, client = self.create_milvus_collection()
        self.insert_item_name(collection_name, file_title, entity, labels, client)

        return {
            "chunks": chunks,
            "parents": parents,
            "item_name": entity,
        }

    def get_chunks_str(self, chunks, file_title):
        """
        把“文档标题 + 前几个 chunk”拼成抽取标签的上下文。

        为什么不读元数据：
        - 源文档不一定有 `## 元数据`；
        - 即使有，字段名也不固定；
        - 只使用标题、标题路径和正文，才是通用输入。
        """
        parts = [f"[文档标题] {file_title}\n"]
        total = len(parts[0])

        for idx, chunk in enumerate(chunks[:5], start=1):
            title_path = chunk.get("title_path") or chunk.get("title") or ""
            content = chunk.get("content") or ""
            text = f"[片段{idx}] {title_path}\n{content}\n\n"

            # 控制 prompt 长度，避免把整篇文档塞进去
            if total + len(text) > 8000:
                break

            parts.append(text)
            total += len(text)

        return "".join(parts)

    def get_labels(self, chunks, file_title):
        """
        调用 LLM 抽取旅游标签。

        参数：
        - chunks: 切分后的 chunk 列表
        - file_title: 文档标题

        返回：
        - dict: entity/category/granularity/mentioned_entities/confidence
        - LLM 失败时返回一个保守的 fallback，不阻断导入。
        """
        context = self.get_chunks_str(chunks, file_title)

        llm = init_chat_model(
            model=LLMConfig.item_model,
            model_provider="openai",
            base_url=LLMConfig.openai_api_base,
            api_key=LLMConfig.openai_api_key,
            temperature=LLMConfig.llm_default_temperature,
            timeout=60,
        )

        messages = [
            {"role": "system", "content": TRAVEL_LABEL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": TRAVEL_LABEL_TEMPLATE.format(
                    file_title=file_title,
                    context=context,
                ),
            },
        ]

        try:
            response = invoke_with_retry(llm, messages)
            return self._parse_labels(response.content, file_title)
        except Exception as error:
            logger.warning(f"标签抽取失败，使用 fallback: {error}")
            return self._fallback_labels(file_title)

    def _parse_labels(self, raw, file_title):
        """
        解析 LLM 返回的 JSON，并做枚举/类型校验。

        模型可能返回 Markdown 代码块、缺字段、枚举值不对，
        这里统一兜底，保证后续写入不会因为一个字段崩掉。
        """
        text = (raw or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            data = json.loads(text)
        except Exception as error:
            logger.warning(f"标签 JSON 解析失败，使用 fallback: {error}")
            return self._fallback_labels(file_title)

        entity = str(data.get("entity") or "").strip()
        category = str(data.get("category") or "综合").strip()
        granularity = str(data.get("granularity") or "unknown").strip()

        if category not in VALID_CATEGORIES:
            category = "综合"
        if granularity not in VALID_GRANULARITIES:
            granularity = "unknown"

        mentioned_entities = data.get("mentioned_entities") or []
        if not isinstance(mentioned_entities, list):
            mentioned_entities = []
        mentioned_entities = [
            str(item).strip()
            for item in mentioned_entities
            if str(item).strip()
        ][:10]

        try:
            confidence = float(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        # entity 为空时用文档标题兜底，保证展示字段不为空；
        # 但 category/granularity 仍然保留，用于软加权。
        if not entity:
            entity = file_title

        return {
            "entity": entity,
            "category": category,
            "granularity": granularity,
            "mentioned_entities": mentioned_entities,
            "confidence": max(0.0, min(1.0, confidence)),
        }

    @staticmethod
    def _fallback_labels(file_title):
        """LLM 完全失败时的保守标签，避免导入中断。"""
        return {
            "entity": file_title,
            "category": "综合",
            "granularity": "unknown",
            "mentioned_entities": [],
            "confidence": 0.0,
        }

    @staticmethod
    def _clip_bytes(text, max_bytes=100):
        """
        按 UTF-8 字节数截断，避免超过 Milvus VARCHAR(100) 的字节上限。

        Milvus 的 max_length 单位是字节，一个中文占 3 个字节；
        直接按字符截断仍然可能超限。
        """
        raw = (text or "").encode("utf-8")

        if len(raw) <= max_bytes:
            return text or ""

        return raw[:max_bytes].decode("utf-8", errors="ignore")

    def create_milvus_collection(self):
        """
        创建/检查 kb_item_names 集合。

        schema 变化：
        - 原来的 metadata 字段改成 labels，存旅游标签 JSON；
        - item_name 仍然存 entity，用于展示/兼容；
        - 旧集合缺少 labels 字段时，Milvus 不能直接加字段，提示删除重建。
        """
        milvus_client = get_milvus_client()
        collection_name = MilvusConfig.item_name_collection

        if not milvus_client.has_collection(collection_name):
            schema = milvus_client.create_schema(auto_id=True)

            schema.add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
            ).add_field(
                field_name="item_name",
                datatype=DataType.VARCHAR,
                max_length=100,
            ).add_field(
                field_name="file_title",
                datatype=DataType.VARCHAR,
                max_length=100,
            ).add_field(
                # 旅游标签 JSON：entity/category/granularity/mentioned_entities/confidence
                field_name="labels",
                datatype=DataType.JSON,
            ).add_field(
                field_name="dense_vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=1024,
            ).add_field(
                field_name="sparse_vector",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            index = milvus_client.prepare_index_params()
            index.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",
                index_name="dense_vector_index",
                metric_type="COSINE",
                params={"nlist": 128},
            )
            index.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_vector_index",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )

            milvus_client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index,
            )

        else:
            fields = {
                field["name"]
                for field in milvus_client.describe_collection(collection_name).get("fields", [])
            }
            if "labels" not in fields:
                raise RuntimeError(
                    f"集合 {collection_name} 缺少 labels 字段；"
                    "Milvus 不支持直接加字段，请删除该集合后重新导入。"
                )

        return collection_name, milvus_client

    def insert_item_name(self, collection_name, file_title, entity, labels, milvus_client):
        """
        把 entity 和标签写入 kb_item_names。

        仍然按 file_title 幂等删除，避免重复导入产生重复行。
        """
        vec = get_bgem3_embeddings([entity])
        dense_vec = vec.get("dense")[0]
        sparse_vec = vec.get("sparse")[0]

        filter_file_title = file_title.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

        milvus_client.load_collection(collection_name=collection_name)

        milvus_client.delete(
            collection_name=collection_name,
            filter=f"file_title == '{filter_file_title}'",
        )

        data = {
            "file_title": file_title,
            "item_name": entity,
            "dense_vector": dense_vec,
            "sparse_vector": sparse_vec,
            "labels": labels,
        }

        milvus_client.insert(collection_name=collection_name, data=data)

    def back_labels_to_chunk(self, chunks, labels, entity, out_dir):
        """
        把标签回填到每个 chunk，并落一份调试 json。

        参数：
        - chunks: 切分后的 chunk 列表
        - labels: 文档级旅游标签
        - entity: 主体名（item_name）
        - out_dir: 调试文件输出目录
        """
        for chunk in chunks:
            chunk["labels"] = labels
            # 兼容旧字段：item_name 仍然等于 entity，只用于展示/软加权
            chunk["item_name"] = entity

        out_path = Path(out_dir) / "item_name_chunks.json"
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(json_format(chunks))

    def back_labels_to_parents(self, parents, labels, entity, out_dir):
        """
        把标签回填到父块，并落一份调试 json。

        父块和子块共享同一份文档级标签：
        - entity/category/granularity 相同；
        - 查询侧回捞父块后，仍然可以用 labels 做软加权。
        """
        if not parents:
            return

        for parent in parents:
            parent["labels"] = labels
            parent["item_name"] = entity

        out_path = Path(out_dir) / "item_name_parents.json"
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(json_format(parents))


if __name__ == "__main__":
    # 单节点调试入口
    node = NodeItemNameRecognition()

    with open(
        r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\split_chunks.json",
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    init_state = {
        "chunks": chunks,
        "file_title": "hak180产品安全手册",
        "md_path": r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
    }

    result = node(init_state)
    logger.info(json_format(result))
