import time
from abc import ABC, abstractmethod

from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger
from atguigu.tool.task_utils import add_running_task, add_done_task, add_node_duration


class NodeBase(ABC):

    name ="node_base"

    # todo   init 检验节点名称
    def __init__(self, **kwargs):
        if self.name == "node_base":
            logger.error("节点名称错误")
            raise Exception("节点名称错误:为配置节点名称！")

    # todo  call 执行process，抛异常
    def __call__(self, state: ImportGraphState):
        try:
            startTime = time.time()

            task_id = state.get("task_id")
            add_running_task(task_id,self.name)
            logger.info(f"节点{self.name}的process开始执行")

            res = self.process(state)

            logger.info(f"节点{self.name}的process结束执行")
            add_done_task(task_id,self.name)
            endTime = time.time()

            add_node_duration(task_id,self.name,endTime-startTime)

            return res
        except Exception as e:
            logger.error(f"节点{self.name}执行失败：{e}")
            raise

    # todo 抽象process
    @abstractmethod
    def process(self, state: ImportGraphState):
        pass


if __name__ == "__main__":


    pass