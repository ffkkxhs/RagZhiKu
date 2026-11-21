# atguigu/import_process/nodes/node_entry.py
from pathlib import Path

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger


class NodeEntry(NodeBase):
    """
    入口节点：任务分发
    """
    name = "node_entry"

    # todo 拿state，检验文件和路径，后缀，去向，设定一些state消息
    def process(self, state: ImportGraphState):
        local_file_path = state.get("local_file_path")

        if not local_file_path:
            logger.error("local_file_path is None")
            raise ValueError("local_file_path is None")

        local_file_path_obj = Path(local_file_path)
        if not local_file_path_obj.exists():
            logger.error("local_file_path_obj is None")
            raise ValueError("local_file_path_obj is None")

        stem =  local_file_path_obj.stem
        suffix = local_file_path_obj.suffix

        if suffix.lower() == ".pdf":
            state["file_title"] = stem
            state["pdf_path"] = local_file_path
            state["is_pdf_read_enabled"] = True

        elif suffix.lower() == ".md":
            state["file_title"] = stem
            state["md_path"] = local_file_path
            state["is_md_read_enabled"] = True

        else:
            logger.error("file is not pdf or md")
            raise Exception("file is not pdf or md")

        return state

if __name__ == "__main__":
    node = NodeEntry()
    init_state = {
        "local_file_path":r"E:\sgg\projects\RagZhiKu\res\doc\hak180产品安全手册.pdf"
    }
    res = node(init_state)
    logger.debug(res)
    logger.info(res)