# atguigu/query_process/nodes/node_web_search_mcp.py
import asyncio
import json
import os

from atguigu.config.config import MCPConfig
from atguigu.query_process.base import NodeBase
from atguigu.query_process.state import QueryGraphState
from atguigu.tool.json_format_tool import json_format
from atguigu.tool.logger import logger

from agents.mcp import MCPServerStreamableHttp

class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState):
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 评估时默认关闭 Web 搜索：
        # - 它会带来额外的外部调用成本；
        # - 网页结果噪音较大，会干扰“检索改造是否有效”的判断。
        # run_eval.py 会设置 EVAL_DISABLE_WEB=1；生产/手动调用不受影响。
        if os.getenv("EVAL_DISABLE_WEB") == "1":
            return {"web_search_docs": []}

        rewritten_query = state.get("rewritten_query")

        result = asyncio.run(self.run_mcp(rewritten_query))

        # print(json_format(json.loads(result.content[0].text).get("pages")))

        data = json.loads(result.content[0].text).get("pages")

        chunks = [
            {
                "content": item.get("snippet"),
                "title": item.get("title"),
                "url": item.get("url"),
                "source": "web"
            }
            for item in data
        ]

        return {
            "web_search_docs": chunks
        }



    async def run_mcp(self,query,limit=10) -> None:
        token = MCPConfig.mcp_dashscope_api_key
        async with MCPServerStreamableHttp(
                name="网络搜索chunks的mcp工具",
                params={
                    "url": MCPConfig.mcp_dashscope_base_url,
                    "headers": {"Authorization": f"Bearer {token}"},
                    "timeout": 10,
                },
                cache_tools_list=True,
                max_retry_attempts=3,
        ) as server:
            result = await server.call_tool(
                tool_name="bailian_web_search",
                arguments={
                    "query":query,
                    "count":limit
                }
            )
            return result


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于RSPRORS-12数字万用表如何使用"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(json_format(result))