import json
from pathlib import Path

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_embedding_tool import embedding_model, get_embedding_model, get_bgem3_embeddings
from atguigu.tool.milvus_client_tool import get_milvus_client
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger
from langchain.chat_models import init_chat_model
from pymilvus import DataType


class NodeItemNameRecognition(NodeBase):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    # LLM 在“识别不到”时经常直接返回这些占位词，不能当成真正的主体名
    _EMPTY_MARKERS = {
        "", "空字符串", "空", "无", "未知", "无法识别", "没有",
        "none", "null", "n/a", "na", "undefined",
    }

    # 文件名/模型输出里常见的类别后缀，收敛主体名时去掉
    _CATEGORY_SUFFIXES = (
        "交通指南", "景点攻略", "景点推荐", "线路推荐", "美食推荐",
        "住宿推荐", "酒店信息", "使用手册", "产品安全手册", "用户手册",
        "说明书", "简介", "的使用", "多城市",
    )

    def process(self, state: ImportGraphState):

        #todo 0、获取切分的chunks和文件名校验

        chunks = state.get("chunks")

        file_title = state.get("file_title")

        if not chunks:
            logger.error("chunks不能为空")
            raise ValueError("chunks不能为空")

        if not file_title:
            logger.error("file_title not exist")
            raise Exception("file_title not exist")

        chunk_str = self.get_chunks_str(chunks,file_title)

        raw_item_name = self.get_item_name(chunk_str,file_title)
        item_name = self.normalize_item_name(raw_item_name, file_title, chunks)

        collection_name,client =  self.create_milvus_collection()

        self.insert_item_name(collection_name,file_title,item_name,client)

        # 调试 json 跟着当前文档走，不再写死到 hak180 目录
        md_path = state.get("md_path")
        out_dir = Path(md_path).parent if md_path else Path(state.get("local_dir") or ".")

        self.back_item_name_to_chunk(chunks,item_name,out_dir)

        return{
            "chunks":chunks,
            "item_name":item_name,
        }


        #todo 1、把chunks的前10个取出来，拼接成一个大串
    def get_chunks_str(self,chunks, file_title):
        chunk_list = chunks[:10]
        chunk_str = ""

        for idx,chunk in enumerate(chunk_list,start=1):
            chunk_title = chunk.get("title")
            chunk_content = chunk.get("content")

            temp = f"[切片:{idx}]\n[切片名称：{chunk_title}]\n[文件名称：{file_title}]\n[切片文本内容：{chunk_content}]\n\n"
            chunk_str += temp
            if len(chunk_str) > 10000:
                break
        chunks_K_str = chunk_str[:10000]
        return chunks_K_str



        #todo 2、初始化大模型，对大串进行总结，识别出来我们的主体名（商品名）

    def get_item_name(self,chunk_str,file_title):

        llm = init_chat_model(
            model = LLMConfig.item_model,
            model_provider= "openai",
            base_url = LLMConfig.openai_api_base,
            api_key = LLMConfig.openai_api_key,
            temperature = LLMConfig.llm_default_temperature,
        )

        msg =[
            {"role":"system","content":ITEM_NAME_SYSTEM_PROMPT},
            {"role":"user","content":
                ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=chunk_str)},
        ]


        res = llm.invoke(msg)

        # 这里只做基础清洗，空值和占位词交给 normalize_item_name 统一处理
        return res.content.replace("\t", "").replace("\n", "").replace("\r", "").replace(" ", "")


    def _clean_item_name(self, raw):
        """
        清理 LLM 的原始输出，把“看起来有值、实际是占位符”的结果归一为空字符串。

        为什么需要这一步：
        - 提示词写的是“无法识别时返回空字符串”，但模型经常把“空字符串”
          这几个字原样返回，甚至带引号或反引号；
        - 之前代码只判断 `if not item_name`，所以“空字符串”会被当成合法主体名
          写进 Milvus，最终所有 chunk 的 item_name 都是“空字符串”，
          查询时按城市/商品过滤必然失效。

        参数：
        - raw: LLM 返回的原始文本

        返回：
        - 清理后的主体名；命中占位词则返回空字符串
        """
        name = (raw or "").strip()

        # 去掉模型可能加上的引号、反引号、首尾空白
        name = name.strip("`\"'“”‘’ \t\r\n")

        # 精确匹配占位词，避免把“无锡”“无限”这类正常名字误判为空
        if name.lower() in self._EMPTY_MARKERS:
            return ""
        return name

    def _metadata_item_name(self, chunks):
        """
        从切片开头的“元数据”里抽取主体名，作为 LLM 输出为空时的兜底。

        为什么需要这一步：
        - 旅游数据每个文件都有 `- 城市：成都`；
        - 听书数据有 `- 书名：三体`；
        - 教育数据有 `- 课程名称：xxx`；
        - 这些字段比 LLM 自由发挥更稳定，而且在同一实体的多份文档里是一致的。
          例如成都的 5 份文档都能抽到“成都”，查询时才能用同一个 item_name 过滤。

        参数：
        - chunks: 切分后的切片列表

        返回：
        - 元数据里的主体名；没有则返回空字符串
        """
        prefixes = (
            "城市：", "书名：", "课程名称：", "商品名称：",
            "景点名称：", "主体名称：",
        )

        # 元数据只在文档开头，查前 3 个切片就够了，避免无谓遍历
        for chunk in chunks[:3]:
            for line in (chunk.get("content") or "").splitlines():
                # Markdown 列表项是“- 城市：成都”，先去掉前面的短横线
                line = line.strip().lstrip("-").strip()
                for prefix in prefixes:
                    if line.startswith(prefix):
                        return line[len(prefix):].strip()
        return ""

    def _strip_category_suffix(self, name):
        """
        去掉主体名后面附带的类别后缀，让同一实体的多份文档收敛成同一个名字。

        为什么需要这一步：
        - 文件名和 LLM 输出经常是“成都交通指南”“成都景点推荐”“成都美食推荐”，
          如果直接入库，同一座城市会得到 5 个不同的 item_name，
          查询时 `item_name in ["成都"]` 一个都匹配不上；
        - 去掉“交通指南/景点推荐/…”后，5 份文档都变成“成都”，过滤才有效。

        例子：
        - 成都交通指南 -> 成都
        - 云南多城市   -> 云南
        - 三体简介     -> 三体
        - 万用表RS-12的使用 -> 万用表RS-12
        """
        for suffix in self._CATEGORY_SUFFIXES:
            # 加长度判断，避免 name 本身就等于后缀时被截成空串
            if name.endswith(suffix) and len(name) > len(suffix):
                return name[:-len(suffix)].strip(" -_")
        return name

    @staticmethod
    def _clip_bytes(text, max_bytes=100):
        """
        按 UTF-8 字节数截断文本，避免超过 Milvus VARCHAR 的字节上限。

        为什么不能按字符截断：
        - Milvus 的 max_length 单位是字节，不是字符；
        - 一个中文占 3 个字节，100 字节大约只能放 33 个中文；
        - 直接按字符截断可能仍然超限，按字节截断后再解码更稳。

        参数：
        - text: 原始文本
        - max_bytes: 允许的最大字节数，默认对应 schema 里的 VARCHAR(100)
        """
        raw = text.encode("utf-8")
        if len(raw) <= max_bytes:
            return text

        # errors="ignore" 防止把多字节字符截成半个导致解码异常
        return raw[:max_bytes].decode("utf-8", errors="ignore")

    def normalize_item_name(self, raw, file_title, chunks):
        """
        把 LLM 输出收敛成“可入库、可过滤”的主体名。

        为什么需要这一层：
        1. LLM 可能返回“空字符串”“无”“未知”等占位词，不能直接入库；
        2. LLM 可能返回带类别后缀的文件名，导致同一实体的多份文档名字不一致；
        3. 主体名要写进 Milvus 的 VARCHAR(100)，必须按字节截断；
        4. 以上任何一种失败都不应该让整个导入流程中断，所以要有兜底。

        兜底顺序（从可靠到不可靠）：
        LLM 结果 -> 切片元数据 -> 文件名 -> 最后截断到 100 字节。

        参数：
        - raw: LLM 原始输出
        - file_title: 文件名（去后缀）
        - chunks: 切分后的切片列表，用于元数据兜底

        返回：
        - 最终要写入 Milvus、并回填到每个 chunk 的主体名
        """
        # 第一步：清理引号/空白，把“空字符串”之类的占位词变成真正的空
        name = self._clean_item_name(raw)

        # 第二步：LLM 没给出有效名字时，优先从元数据里抽（旅游/听书/教育数据都有）
        if not name:
            name = self._metadata_item_name(chunks)
            if name:
                logger.warning(f"LLM 未识别出主体名，使用元数据兜底: {name}")

        # 第三步：元数据也没有，再退回文件名，保证 item_name 不为空
        if not name:
            name = file_title
            logger.warning(f"LLM 和元数据都没有主体名，回退文件名: {name}")

        # 第四步：去掉“交通指南”这类后缀，让同一实体的多份文档名字一致
        name = self._strip_category_suffix(name)

        # 第五步：按 Milvus 的字节上限截断，避免插入时报 VARCHAR 超长
        name = self._clip_bytes(name, 100) or file_title

        logger.info(f"最终主体名: {name}")
        return name


    #todo 3、创建向量数据库的表
    def create_milvus_collection(self):
        milvus_client = get_milvus_client()
        collection_name = MilvusConfig.item_name_collection

        if not milvus_client.has_collection(collection_name):
            schema = milvus_client.create_schema(
                auto_id = True,
            )
            schema.add_field(
                field_name = "id",
                datatype = DataType.INT64,
                is_primary = True,
            ).add_field(
                field_name = "item_name",
                datatype=DataType.VARCHAR,
                max_length=100
            ).add_field(
                field_name = "file_title",
                datatype = DataType.VARCHAR,
                max_length = 100
            ).add_field(
                field_name = "dense_vector",
                datatype = DataType.FLOAT_VECTOR,
                dim = 1024  # 必须加维度
            ).add_field(
                field_name = "sparse_vector",
                datatype = DataType.SPARSE_FLOAT_VECTOR,
            )

            index = milvus_client.prepare_index_params()
            index.add_index(
                field_name="dense_vector",
                index_type="IVF_FLAT",
                index_name="dense_vector_index",
                metric_type="COSINE",
                params={
                    "nlist": 128,
                }
            )
            index.add_index(
                field_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_vector_index",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"}
            )

            milvus_client.create_collection(
                collection_name = collection_name,
                schema = schema,
                index_params = index
            )

        return collection_name,milvus_client


    #todo 4、把识别的商品名称进行幂等删除并插入到milvus表当中
    def insert_item_name(self,collection_name, file_title, item_name, milvus_client):

        vec = get_bgem3_embeddings([item_name])
        dense_vec = vec.get("dense")[0]
        sparse_vec = vec.get("sparse")[0]

        filter_file_title = file_title.replace("\\","\\\\").replace("\'","\\'").replace('\"','\\"')

        milvus_client.load_collection(collection_name = collection_name,)

        filter =f"file_title == '{filter_file_title}'"

        milvus_client.delete(
            collection_name = collection_name,
            filter = filter
        )

        data = {
            "file_title" : file_title,
            "item_name" : item_name,
            "sparse_vector" : sparse_vec,
            "dense_vector" : dense_vec
        }

        milvus_client.insert(collection_name = collection_name,data = data)


    #todo 5、把识别出来的商品名回填给个chunk(提升语义)，保存chunks到一个新的json文件，下个节点使用
    def back_item_name_to_chunk(self,chunks, item_name, out_dir):
        """
        把识别出的商品名回填到每个 chunk，并落一份调试 json。

        参数：
        - chunks: 切分后的切片列表
        - item_name: 识别出的商品名
        - out_dir: 调试文件输出目录（当前文档所在目录）
        """
        for chunk in chunks:
            chunk["item_name"] = item_name

        out_path = Path(out_dir) / "item_name_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_format(chunks))


if __name__ == '__main__':

    node = NodeItemNameRecognition()
    with open(r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\split_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    init_state = {
        "chunks": chunks,
        "file_title": "hak180产品安全手册",
        "md_path": r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
    }
    result = node(init_state)
    logger.info(json_format(result))
