from atguigu.import_process.base import NodeBase
from atguigu.tool.logger import logger


class TestNode(NodeBase):

    name = "test_node"

    def process(self,state):
        # print("测试节点执行中")
        logger.info("测试节点执行中")
        return 100


if __name__ == '__main__':
    node = TestNode()
    init_state = {
        "task_id":"1111"
    }
    result = node(init_state)
    print(result)