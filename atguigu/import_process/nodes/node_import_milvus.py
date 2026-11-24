# atguigu/import_process/nodes/node_import_milvus.py
import json

from pymilvus import DataType

from atguigu.config.config import MilvusConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.milvus_client_tool import get_milvus_client
from atguigu.tool.mongo_tool import save_parents
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger


class NodeImportMilvus(NodeBase):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):

        chunks = state.get("chunks")
        file_title = state.get("file_title")

        if not file_title:
            logger.error("文件名不能为空")
            raise ValueError("文件名不能为空")

        if not chunks:
            logger.error("chunks不能为空")
            raise ValueError("chunks不能为空")


        client = get_milvus_client()

        collection_name = MilvusConfig.chunks_collection

        if not client.has_collection(collection_name):


            schema = client.create_schema(
                auto_id = True,
            )

            schema.add_field(
                field_name = "id",
                datatype = DataType.INT64,
                is_primary = True,
            ).add_field(
                field_name="title",
                datatype=DataType.VARCHAR,
                max_length=100,
            ).add_field(
                field_name = "file_title",
                datatype = DataType.VARCHAR,
                max_length = 100,
            ).add_field(
                field_name = "content",
                datatype = DataType.VARCHAR,
                max_length = 5000,
            ).add_field(
                field_name = "part",
                datatype = DataType.INT64,
            ).add_field(
                field_name = "item_name",
                datatype = DataType.VARCHAR,
                max_length = 100,
            ).add_field(
                # ============================================================
                # 【改造说明：把“标题层级”和“通用元数据”变成一等字段】
                #
                # 原 schema 只有 title/content/part，chunk 丢了父级标题路径，
                # 也拿不到城市/书名/课程名等结构化信息。
                # 新 schema 增加：
                # - title_path：完整标题路径，例如“成都景点推荐 > 核心景点推荐 > 都江堰”
                # - level：标题层级，H1=1 / H2=2 / H3=3
                # - parent_id：父级路径，后续做父块回捞时用
                # - section_id：章节稳定 ID
                # - chunk_id：chunk 稳定 ID，方便去重和引用
                # - labels：JSON 字段，存旅游领域标签：
                #   entity/category/granularity/mentioned_entities/confidence
                #
                # 注意：这里不是源文档的 `## 元数据`，而是我们自己的标注结果；
                # 切分/识别节点不读取源文档元数据。
                # 注意：Milvus 不支持直接给已有集合加字段，
                # 旧集合需要删除后重新导入（下面有明确检查）。
                # ============================================================
                field_name="title_path",
                datatype=DataType.VARCHAR,
                max_length=1000,
            ).add_field(
                field_name="level",
                datatype=DataType.INT64,
            ).add_field(
                field_name="parent_id",
                datatype=DataType.VARCHAR,
                max_length=500,
            ).add_field(
                field_name="section_id",
                datatype=DataType.VARCHAR,
                max_length=100,
            ).add_field(
                field_name="chunk_id",
                datatype=DataType.VARCHAR,
                max_length=100,
            ).add_field(
                field_name="labels",
                datatype=DataType.JSON,
            ).add_field(
                field_name = "dense_vector",
                datatype = DataType.FLOAT_VECTOR,
                dim = 1024
            ).add_field(
                field_name = "sparse_vector",
                datatype = DataType.SPARSE_FLOAT_VECTOR,
            )

            index = client.prepare_index_params()

            index.add_index(
                field_name = "sparse_vector",
                index_type = "SPARSE_INVERTED_INDEX",
                index_name = "sparse_vector_index",
                metric_type = "IP",
                params = {
                    "inverted_index_algo":"DAAT_MAXSCORE"
                }

            )
            index.add_index(
                field_name = "dense_vector",
                index_type = "IVF_FLAT",
                index_name = "dense_vector_index",
                metric_type = "COSINE",
                params = {
                    # nprobe 是检索参数，不是建索引参数；
                    # 建 IVF_FLAT 索引只需要 nlist，nprobe 应放到查询侧。
                    "nlist": 128,
                }
            )

            client.create_collection(
                collection_name = collection_name,
                schema = schema,
                index_params =index
            )

        else:
            # 【改造说明】旧集合没有新字段时，insert 会报错。
            # 这里提前检查并给出可操作的提示，而不是让用户看 Milvus 底层异常。
            fields = {
                field["name"]
                for field in client.describe_collection(collection_name).get("fields", [])
            }
            required_fields = {
                "title_path", "level", "parent_id",
                "section_id", "chunk_id", "labels",
            }
            missing = required_fields - fields
            if missing:
                raise RuntimeError(
                    f"集合 {collection_name} 缺少字段 {sorted(missing)}；"
                    "Milvus 不支持直接加字段，请删除该集合后重新导入。"
                )

        client.load_collection(collection_name)

        fft = file_title.replace(
            "\\","\\\\"
        ).replace(
            "\'","\\'"
        ).replace(
            '\"','\\"'
        )

        filter = f"file_title == '{fft}'"

        client.delete(
            collection_name = collection_name,
            filter = filter
        )

        client.insert(
            collection_name = collection_name,
            data=chunks,
        )

        # ================================================================
        # 【父子切块：父块写入 Mongo】
        #
        # 父块不参与向量检索，只用于“命中子块后回捞上下文”，
        # 所以放在 Mongo 的 doc_parents 集合里，按 file_title 幂等覆盖。
        # ================================================================
        parents = state.get("parents") or []

        # 父块是“增强项”，不是主链路：
        # Mongo 写失败时记录警告，但不让整个导入失败；
        # 查询侧回捞不到父块会自动降级为子块。
        try:
            save_parents(file_title, parents)
        except Exception as error:
            logger.warning(f"父块写入 Mongo 失败，已跳过: {error}")

        return {
            "chunks": chunks,
            "parents": parents,
        }


if __name__ == '__main__':
    node = NodeImportMilvus()
    with open(r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\embedding_chunks.json","r",encoding="utf-8") as f:
        chunks = json.load(f)

    init_state = {
        "chunks":chunks,
        "file_title":"hak180产品安全手册"
    }

    result = node(init_state)
    logger.info(json_format(result))