
import base64
import datetime
import os

import re
import time
from collections import deque

from pathlib import Path


from langchain.chat_models import init_chat_model
from minio.deleteobjects import DeleteObject

from atguigu.config.config import MinioConfig, LLMConfig
from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool import minio_client_tool
from atguigu.tool.logger import logger

from rich import print as rprint


class NodeMDImg(NodeBase):
    """
    MarkDown图片处理节点：多模态图片理解
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):

        image_with_summary_list = []


        #todo 获取md路径 md_path_obj
        # todo 检验路径和文件
        md_path = state.get("md_path")
        if not md_path:
            logger.error("路径不能为空")
            raise Exception("路径不能为空")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error("文件找不到")
            raise Exception("文件找不到")


        #todo md读到内存里 md_content
        with open(md_path_obj, "r",encoding="utf-8") as f:
            md_content = f.read()

        if not md_content:
            logger.warning("md内容为空")


        #todo 根据md的path，获取同级下的图片文件夹目录images，检验，没有就返回content    image_dir_path_obj
        image_dir_path_obj = md_path_obj.parent / "images"
        if not image_dir_path_obj.exists():
            logger.error("")
            return {
                "md_content": md_content,
            }

        #todo 获取所有图片的名称(含目录)，放到一个列表image_names，检验，没有就返回content
        image_names = os.listdir(image_dir_path_obj)
        if not image_names:
            logger.error("")
            return {
                "md_content": md_content,
            }

        #todo 拿图片的上下文  image_with_context_list
        image_with_context_list = []

        #todo 循环拿图，检验后缀
        #   todo 常用图片后缀列表 IMAGE_EXTENSIONS{}
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

        for image_name in image_names:
        #   todo 遍历，
        #       todo 拼接path，拿后缀小写，看在不在列表里
        #       todo 正则表达式( r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)" )，搜索md里的对应的图片文件路径，得到一个对象match

        #       todo span() + 拆包
        #       todo 取图片位置的前后上下文，注意边界
        #       todo 添加到image_with_context_list
            image_path_obj = image_dir_path_obj / image_name # 图片的本地完整路径
            suffix = image_path_obj.suffix.lower()
            if not suffix in IMAGE_EXTENSIONS:
                logger.error("md文件不能引用这种文件")
                continue

            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r"\)")
            match = pattern.search(md_content)

            if not match:
                logger.warning("内容没有引用这个图片")
                continue

            start,end = match.span()
            start_pos = max(0, start-100)
            end_pos = min(len(md_content), end+100)
            pre_text = md_content[start_pos:start]
            post_text = md_content[end:end_pos]

            image_with_context_list.append(
                {
                    "pre_text": pre_text,
                    "post_text": post_text,
                    "image_path": image_path_obj,
                    "image_name": image_name,
                }
            )


        #todo 初始化视觉模型对象  找key/baseurl
        # todo +++++++++++++++++++++++++++++++++++++++++++==============================================================
        llm = init_chat_model(
            model=LLMConfig.vl_model,
            model_provider="openai",
            api_key=LLMConfig.openai_api_key,
            base_url=LLMConfig.openai_api_base,
            temperature=LLMConfig.llm_default_temperature
        )

        #todo 双端队列 deque 30
        #todo 遍历image_with_context_list
        #   todo 时间作为标记，每个图片有1分钟的时间来调用大模型
        #   todo 计算时间，dp满了就等，等待第一个任务的剩余时间结束，再往dp里加当前的时间作为新来的任务的起始时间
        #   todo 没满，当前任务的时间就直接加到dp
        dp = deque(maxlen=30)

        for image_with_context in image_with_context_list:
            current_time = time.time()

            if len(dp) >= 30:
                wait_time = 60 - (current_time - dp[0])
                if wait_time > 0:
                    time.sleep(wait_time)

            dp.append(current_time)

        #   todo 获取图片路径，读取图片，转base64
            with open(image_with_context.get("image_path"), "rb") as f:
                base64_image_data = base64.b64encode(f.read()).decode("utf-8")

        #   todo 拼接message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                # 我们要把图片传递给大模型：1、可以传递网络路径 2、可以传递编码后的图片内容（base64编码）
                                "url": f"data:image/jpeg;base64,{base64_image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""这是一张图片，图片上文部分为"{image_with_context.get("pre_text")}"，
                                               下文部分为"{image_with_context.get("post_text")}"，
                                               请用中文简要总结这张图片的摘要,字数在50字以内。"""
                        }
                    ],
                },
            ]

        #   todo 调用模型得到回复
            try:
                res = llm.invoke(messages)
                summary = res.content
            except Exception as e:
                logger.error(f"图片 {image_with_context.get("image_name")} 解析失败: {e}")
                summary = "图片解析失败"

        #   todo 拼接结果
            image_with_summary_list.append(
                {
                    "image_name": image_with_context.get("image_name"),
                    "image_path": str(image_with_context.get("image_path")),
                    "summary":summary,
                }
            )

        #todo 返回得到的图片摘要列表 image_with_summary_list
        # return image_with_summary_list



        #todo 将图片存到云端minio，拿到minio里图片的url
        image_with_summary_url_list = []
        minio_client = minio_client_tool.get_minio_client()
        # todo 拼接upload_dir(文件夹+日期+图片名称)  云端minio里的桶里的路径，桶名称在配置里
        upload_dir = f"{MinioConfig.minio_img_dir}/{datetime.datetime.now().strftime("%Y%m%d")}/{md_path_obj.stem}"

        # todo client调用list_objects，传入桶名称和路径，得到old_file_generator老文件列表，client调用
        old_file_generator = minio_client.list_objects(
            bucket_name=MinioConfig.minio_bucket_name,
            prefix=upload_dir + "/",
            recursive=True,
        )

        # todo 得到删除列表  delete_object_list 、DeleteObject(item.object_name)
        delete_object_list = [DeleteObject(item.object_name) for item in old_file_generator]

        # todo 假删除 remove_object传入桶名称和删除列表，得到一个生成器对象
        errors = minio_client.remove_objects(
            bucket_name=MinioConfig.minio_bucket_name,
            delete_object_list = delete_object_list,
        )

        # todo 真删除
        for error in errors:
            logger.warning(f"删除失败: 对象={error.object_name}, 原因={error.message}")

        # todo 遍历image_with_summary_list
        #       todo 上传图片到minio、fput_object（桶名称，目标路径(拼文件名称)，图片文件路径）
        #       todo
        #       todo 拼接url。服务器+桶+路径+名称
        #       todo
        #       todo image_with_summary_url_list.append()
        for image_with_summary in image_with_summary_list:
            minio_client.fput_object(
                bucket_name=MinioConfig.minio_bucket_name,
                object_name= upload_dir +"/"+ image_with_summary.get("image_name"),
                file_path = image_with_summary.get("image_path"),
            )

            url = (
                f"http://"
                f"{MinioConfig.minio_endpoint}"
                f"/{MinioConfig.minio_bucket_name}"
                f"/{upload_dir}"
                f"/{image_with_summary.get('image_name')}"
            )

            image_with_summary_url_list.append({
                **image_with_summary,
                "url": url
            })

        #todo 现在有图片在云端的url：image_with_summary_url_list，有图片的摘要信息summary也在image_with_summary_url_list里·，把两者都替换到md里

        for image_with_summary_url in image_with_summary_url_list:
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_with_summary_url["image_name"]) + r"\)")

            md_content = pattern.sub(
                lambda _: f"![{image_with_summary_url.get("summary")}]({image_with_summary_url.get("url")})",
                md_content)

        new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new.md"
        with open(new_md_path_obj, "w", encoding="utf-8") as f:
            f.write(md_content)


        return {
            "md_path": str(new_md_path_obj),
            "md_content": md_content,
        }



if __name__ == "__main__":

    node = NodeMDImg()
    init_state = {
        "md_path":r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册.md"
    }

    result = node(init_state)
    rprint(result)
