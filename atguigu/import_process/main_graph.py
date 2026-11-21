from langgraph.constants import START, END
# from langgraph.graph import StateGraph
#
from atguigu.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from atguigu.import_process.nodes.node_document_split import NodeDocumentSplit
from atguigu.import_process.nodes.node_entry import NodeEntry
from atguigu.import_process.nodes.node_import_milvus import NodeImportMilvus
from atguigu.import_process.nodes.node_item_name_recognition import NodeItemNameRecognition
from atguigu.import_process.nodes.node_md_img import NodeMDImg
from atguigu.import_process.nodes.node_pdf_to_md import NodePDFToMD
# from atguigu.import_process.state import ImportGraphState
# from atguigu.tool.logger import logger

from atguigu.import_process.state import ImportGraphState

from langgraph.graph import StateGraph



class MainGraphRunner:

    #todo 初始化
    def __init__(self):
        self.builder= StateGraph(state_schema=ImportGraphState)

        self.add_node()
        self.add_edge()
        # self.add_file_router()
        self.graph = None
        # self.run()

    #todo 添加节点
    def add_node(self):
        self.builder.add_node(NodePDFToMD.name,NodePDFToMD())
        self.builder.add_node(NodeBGEEmbedding.name,NodeBGEEmbedding())
        self.builder.add_node(NodeDocumentSplit.name,NodeDocumentSplit())
        self.builder.add_node(NodeItemNameRecognition.name,NodeItemNameRecognition())
        self.builder.add_node(NodeMDImg.name,NodeMDImg())
        self.builder.add_node(NodeImportMilvus.name,NodeImportMilvus())
        self.builder.add_node(NodeEntry.name,NodeEntry())

    #todo 路由 去向节点
    def add_file_router(self,state: ImportGraphState):
        md = state.get("is_md_read_enabled")
        pdf = state.get("is_pdf_read_enabled")

        if md :
            return "to_md"
        elif pdf :
            return "to_pdf"
        else:
            return "to_nowhere"

    #todo 边
    def add_edge(self,):
        self.builder.set_entry_point(NodeEntry.name)

        self.builder.add_conditional_edges(NodeEntry.name, self.add_file_router,{
            "to_md":NodeMDImg.name,
            "to_pdf":NodePDFToMD.name,
            "to_nowhere":END,
        })

        self.builder.add_edge(NodePDFToMD.name, NodeMDImg.name)
        self.builder.add_edge(NodeMDImg.name, NodeDocumentSplit.name)
        self.builder.add_edge(NodeDocumentSplit.name, NodeItemNameRecognition.name)
        self.builder.add_edge(NodeItemNameRecognition.name, NodeBGEEmbedding.name)
        self.builder.add_edge(NodeBGEEmbedding.name, NodeImportMilvus.name)
        self.builder.add_edge(NodeImportMilvus.name, END)


    #todo 编译并执行
    def run(self,state):
        #1、懒加载
        if self.graph is None:
            self.graph = self.builder.compile()
        result = self.graph.invoke(state)
        return result

    @classmethod
    def create_and_run(cls,state):
        return cls().run(state)

if __name__ == "__main__":
    pass
    # runner = MainGraphRunner()
    # init_state = {
    #     "local_dir":r"E:\sgg\projects\RagZhiKu\res\output",
    #     "local_file_path":r"E:/20286/output/hak180产品安全手册.pdf"
    # }
    # runner.run(init_state)

