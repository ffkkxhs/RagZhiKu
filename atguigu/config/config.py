
import os

from dotenv import load_dotenv
from pygments.lexers import q

load_dotenv(override=True)

class MineruConfig:
    mineru_token: str = os.getenv("MINERU_TOKEN")


class LLMConfig:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    llm_default_model = os.getenv("LLM_DEFAULT_MODEL")
    llm_default_temperature = float(os.getenv("LLM_DEFAULT_TEMPERATURE"))
    vl_model = os.getenv("VL_MODEL")
    item_model = os.getenv("ITEM_MODEL")



class MinioConfig:
    minio_access_key = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key = os.getenv("MINIO_SECRET_KEY")
    minio_endpoint = os.getenv("MINIO_ENDPOINT")
    minio_bucket_name = os.getenv("MINIO_BUCKET_NAME")
    minio_img_dir = os.getenv("MINIO_IMG_DIR")



class BGEM3Config:
    bge_m3_path = os.getenv("BGE_M3_PATH")
    bge_m3 = os.getenv("BGE_M3")
    bge_device = os.getenv("BGE_DEVICE")
    bge_fp16 = os.getenv("BGE_FP16") in ("True","true","1","on")



class MilvusConfig:
    milvus_url = os.getenv("MILVUS_URL")
    chunks_collection = os.getenv("CHUNKS_COLLECTION")
    item_name_collection = os.getenv("ITEM_NAME_COLLECTION")



class MongoConfig:
    mongo_url = os.getenv("MONGO_URL")
    mongo_db_name = os.getenv("MONGO_DB_NAME")

class MCPConfig:
    mcp_dashscope_base_url = os.getenv("MCP_DASHSCOPE_BASE_URL")
    mcp_dashscope_api_key = os.getenv("MCP_DASHSCOPE_API_KEY")




class RerankConfig:
    rerank_work_space = os.getenv("RERANK_WORK_SPACE")
    rerank_api_key = os.getenv("RERANK_API_KEY")