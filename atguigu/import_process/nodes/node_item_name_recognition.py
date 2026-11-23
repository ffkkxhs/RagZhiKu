import json
from pathlib import Path

from atguigu.config.config import LLMConfig, MilvusConfig
from atguigu.config.prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.bgem3_embedding_tool import embedding_model, get_embedding_model, get_bgem3_embeddings
from atguigu.tool.milvus_client_tool import get_milvus_client
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.llm_tool import invoke_with_retry
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

    # ========================================================================
    # 【改造说明：去掉领域特定的主体名后缀规则】
    #
    # 原设计里有一组“交通指南/景点推荐/美食推荐/多城市”等后缀，
    # 这是旅游领域特有的规则。换到听书、教育、产品手册就不成立，
    # 而且会让代码越来越像一堆 if。
    #
    # 现在改成：
    # 1. 优先相信文档自己的元数据（“城市：成都”“书名：三体”“课程名称：xxx”）；
    # 2. 元数据没有时，再用 LLM 输出；
    # 3. 最后才回退文件名。
    # 代码只保留“通用字段名”的优先级，不再写死任何领域的后缀。
    #
    # 下面这组 key 只是“常见叫法”，不是某个领域的业务规则：
    # 文档里出现哪个，就用哪个；没有就继续往下找。
    # ========================================================================
    # 顺序很重要：
    # - “主体/商品名称/城市/书名/课程名称”这类字段才是主体本身；
    # - “名称/标题/文档名称”经常只是文件名，应该最后才用。
    # 例如旅游文档同时有“城市：成都”和“文档名称：成都美食推荐”，
    # 必须优先“城市”，否则 item_name 会变成文件名。
    _SUBJECT_KEYS = (
        "主体", "主体名称",
        "商品名称", "产品名称", "服务名称",
        "城市", "地区", "目的地", "景点名称",
        "书名", "课程名称", "项目名称",
        "主题",
        "名称", "标题", "文档名称",
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

        # 每个 chunk 都带着文档级元数据；这里取第一份作为文档级 metadata 入库，
        # 方便查询时按“城市/课程/书名”等字段做过滤，而不是只依赖一个字符串。
        metadata = (chunks[0].get("metadata") or {}) if chunks else {}
        self.insert_item_name(collection_name,file_title,item_name,client,metadata)

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

            # 把元数据也放进提示词：不同领域的元数据字段不同，
            # 但模型都能从“key：value”里找到主体线索。
            chunk_metadata = chunk.get("metadata") or {}
            temp = (
                f"[切片:{idx}]\n"
                f"[元数据：{json.dumps(chunk_metadata, ensure_ascii=False)}]\n"
                f"[切片名称：{chunk_title}]\n"
                f"[文件名称：{file_title}]\n"
                f"[切片文本内容：{chunk_content}]\n\n"
            )
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
            # 单次请求超时，避免网络异常时一直挂着
            timeout = 60,
        )

        msg =[
            {"role":"system","content":ITEM_NAME_SYSTEM_PROMPT},
            {"role":"user","content":
                ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=chunk_str)},
        ]


        # 公网调用偶发连接重置/超时/429，用统一封装重试几次；
        # 只有重试耗尽才抛异常，避免一次网络抖动打断整个导入。
        res = invoke_with_retry(llm, msg)

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
        从切片携带的 metadata 字典里抽取主体名。

        为什么优先用元数据：
        - 元数据是文档作者/解析器明确给出的字段，比 LLM 自由发挥稳定；
        - 同一实体的多份文档（例如成都的 5 份旅游文档）通常有相同的
          “城市：成都”，用元数据能保证它们的 item_name 完全一致；
        - 听书数据有“书名：三体”，教育数据有“课程名称：xxx”，
          代码不需要知道这些字段属于哪个领域，只按通用 key 优先级找。

        兼容性：
        - 新切分节点会把 metadata 作为字典挂在 chunk 上，优先读字典；
        - 旧 chunk 没有 metadata 字段时，退回到解析正文里的“key：value”行。

        参数：
        - chunks: 切分后的切片列表

        返回：
        - 元数据里的主体名；没有则返回空字符串
        """
        # 优先：新切分节点产出的结构化 metadata
        for chunk in chunks[:3]:
            metadata = chunk.get("metadata") or {}
            for key in self._SUBJECT_KEYS:
                value = metadata.get(key)
                if value:
                    return str(value).strip()

        # 兼容：旧 chunk 没有 metadata，只有正文里的“- key：value”
        for chunk in chunks[:3]:
            for line in (chunk.get("content") or "").splitlines():
                line = line.strip().lstrip("-").strip()
                for key in self._SUBJECT_KEYS:
                    for separator in ("：", ":"):
                        prefix = f"{key}{separator}"
                        if line.startswith(prefix):
                            return line[len(prefix):].strip()

        return ""

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
        把主体名收敛成“可入库、可过滤”的最终结果。

        # ====================================================================
        # 【改造说明：兜底顺序从“LLM 优先”改成“元数据优先”】
        #
        # 原顺序：LLM -> 元数据 -> 文件名。
        # 问题是 LLM 会把“成都”写成“成都美食”“成都3日经典游”，
        # 同一座城市的 5 份文档因此得到 5 个不同的 item_name，
        # 查询时无法用一个城市名过滤。
        #
        # 新顺序：元数据 -> LLM -> 文件名。
        # 元数据是文档作者明确写的“城市：成都”“书名：三体”，
        # 比 LLM 猜的更稳定；没有元数据的文档才用 LLM，再没有才回退文件名。
        # 这个顺序不依赖具体领域，旅游/听书/教育/产品手册都适用。
        # ====================================================================

        参数：
        - raw: LLM 原始输出
        - file_title: 文件名（去后缀）
        - chunks: 切分后的切片列表，用于读取 metadata

        返回：
        - 最终要写入 Milvus、并回填到每个 chunk 的主体名
        """
        name = ""

        # 第一步：优先从文档元数据里取（结构化字段最可靠）
        metadata_name = self._metadata_item_name(chunks)
        if metadata_name:
            name = self._clean_item_name(metadata_name)
            if name:
                logger.info(f"使用元数据中的主体名: {name}")

        # 第二步：元数据没有，再用 LLM 的结果
        if not name:
            name = self._clean_item_name(raw)

        # 第三步：都没有，回退文件名，保证 item_name 不为空
        if not name:
            name = file_title
            logger.warning(f"元数据和 LLM 都没有主体名，回退文件名: {name}")

        # 第四步：按 Milvus 的字节上限截断，避免插入时报 VARCHAR 超长
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
                # 【改造说明】用 JSON 字段存通用元数据（城市/书名/课程名/别名等）。
                # 不同领域可以存不同字段，不需要为每个领域改一次 schema。
                field_name = "metadata",
                datatype = DataType.JSON,
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

        else:
            # 【改造说明】Milvus 不支持直接给已有集合加字段。
            # 如果集合是旧版本建的、没有 metadata 字段，插入会失败；
            # 这里提前给出明确提示，避免用户看到一堆底层报错。
            fields = {
                field["name"]
                for field in milvus_client.describe_collection(collection_name).get("fields", [])
            }
            if "metadata" not in fields:
                raise RuntimeError(
                    f"集合 {collection_name} 缺少 metadata 字段；"
                    "Milvus 不支持直接加字段，请删除该集合后重新导入。"
                )

        return collection_name,milvus_client


    #todo 4、把识别的商品名称进行幂等删除并插入到milvus表当中
    def insert_item_name(self,collection_name, file_title, item_name, milvus_client, metadata=None):
        """
        把主体名和元数据写入 kb_item_names。

        # 【改造说明】增加 metadata 参数：
        # 以前只存 item_name/file_title，查询时只能按一个字符串过滤；
        # 现在把“城市/书名/课程名/别名”等通用字段一起存进 JSON 字段，
        # 查询侧可以做更细的过滤，而不需要为每个领域加列。
        """

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
            "dense_vector" : dense_vec,
            # JSON 字段：没有元数据时存空字典，保证 schema 一致
            "metadata": metadata or {},
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
