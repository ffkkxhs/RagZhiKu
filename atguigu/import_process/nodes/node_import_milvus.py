# atguigu/import_process/nodes/node_import_milvus.py
import json

from pymilvus import DataType

from atguigu.config.config import MilvusConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.milvus_client_tool import get_milvus_client
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
                    "nprobe": 10,
                    "nlist": 128,
                }
            )

            client.create_collection(
                collection_name = collection_name,
                schema = schema,
                index_params =index
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


        return {
            "chunks": chunks,
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