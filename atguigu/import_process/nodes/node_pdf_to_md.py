# atguigu/import_process/nodes/node_pdf_to_md.py
import shutil
from pathlib import Path

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger


# import shutil
import time
# from pathlib import Path
#
from atguigu.config.config import MineruConfig
# from atguigu.import_process.base import NodeBase
# from atguigu.import_process.state import ImportGraphState
# from atguigu.tool.logger import logger

class NodePDFToMD(NodeBase):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        # todo 一、从state当中获取初始化数据
        # todo path()

        local_dir = state.get("local_dir")
        local_dir_obj = Path(local_dir)
        pdf_path = state.get("pdf_path")
        pdf_path_obj  = Path(pdf_path)

        if not pdf_path:
            logger.error("pdf is not exist")
            raise Exception("pdf is not exist")

        if not local_dir:
            logger.error("local_dir is not exist")
            raise Exception("local_dir is not exist")

        if not pdf_path_obj.exists():
            logger.error("pdf_path_obj is not exist")
            raise Exception("pdf_path_obj is not exist")

        if not local_dir_obj.exists():
            local_dir_obj.mkdir(parents=True, exist_ok=True)

        # todo 二、上传文件到mineru  就是发送http请求
        # todo 拼接请求、上传检验成功、数据检验、urls的回应
        import requests  # 发请求的工具库
        token = MineruConfig.mineru_token
        url = "https://mineru.net/api/v4/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": f"{pdf_path_obj.name}", "data_id": f"{pdf_path_obj.stem}"}
            ],
            "model_version": "pipeline"
        }
        file_paths = [pdf_path]

        response = requests.post(url, headers=header, json=data)
        if response.status_code >= 300:
            logger.debug(MineruConfig.mineru_token)
            logger.debug(response)
            logger.error(f"上传请求失败")
            raise Exception("上传请求失败")

        result = response.json()
        if result["code"] != 0:
            logger.error("上传文件请求的数据出错")
            raise Exception("上传文件请求的数据出错")

        batch_id = result["data"]["batch_id"]
        urls = result["data"]["file_urls"]

        for i in range(0, len(urls)):
            url = urls[i]
            file_path = file_paths[i]
            with open(file_path, "rb") as f:
                response = requests.put(url, data=f)
                if response.status_code == 200:
                    logger.info(f"{file_path}成功上传")
                else:
                    logger.error(f"{file_path}上传失败")
                    raise Exception("上传失败")

        #
        # todo 拼接请求，带batch_id、算时间轮询、抛异常statuscode/code/data.extract_result、
        token = MineruConfig.mineru_token
        url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        time_limit = 300
        start_time = time.time()

        while True:
            if time.time() - start_time > time_limit:
                raise Exception("超时")
            time.sleep(2)

            try:
                response = requests.get(url, headers=header,timeout=(5,15))
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常，重试中: {e}")
                continue

            if response.status_code in (401,403):
                logger.error(f"鉴权失败401/403{response.text}")
                raise Exception(f"鉴权失败401/403, MinerU Token 无效或过期,response信息:{response.text}")

            if response.status_code >= 500:
                logger.error("5xx，重试")
                continue

            if response.json()["code"] != 0:
                logger.error("code = 0")
                continue

            data = response.json().get("data")
            results = data.get("extract_result") or []
            if not results or results[0].get("state") != "done":
                logger.error("results[0].get(state) != done")
                continue

            if data.get("extract_result")[0].get("state") == "failed":
                logger.error("")
                raise Exception("下载失败")

            zip_url = data.get("extract_result")[0].get("full_zip_url")
            break


        # todo 下载zip压缩包保存到本地
        # todo 拿结果、判断status、拿内容、拼接地址、保存到本地指定路径
        import requests
        response = requests.get(zip_url)
        if response.status_code != 200:
            logger.error("下载zip失败")
            raise Exception("下载zip失败")

        zip_content = response.content
        zip_path_obj = local_dir_obj / f"{pdf_path_obj.stem}.zip"
        with open(zip_path_obj, 'wb') as f:
            f.write(zip_content)


        # todo 解压压缩包解压到一个位置
        # todo 创建指定文件夹、解压到指定地点、ZipFile/extractall
        import zipfile
        unzip_path = local_dir_obj / f"{pdf_path_obj.stem}"
        if unzip_path.exists():
            shutil.rmtree(unzip_path)
        unzip_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path_obj) as zip_ref:
            zip_ref.extractall(unzip_path)

        # todo 找full.md 改文件名
        # todo
        old_file_path_obj = unzip_path / "full.md"
        new_file_path_obj = old_file_path_obj.with_name(f"{pdf_path_obj.stem}.md")
        old_file_path_obj.rename(new_file_path_obj)


        # todo 读解压文件的md的内容，写入内存
        with open(new_file_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()


        # 删除zip
        # zip_path_obj.unlink()

        return {
            "md_path": str(new_file_path_obj),
            "md_content": md_content
        }



if __name__ == '__main__':
    node = NodePDFToMD()
    init_state = {
        "local_dir":r"E:\sgg\projects\RagZhiKu\res\output",  #输出的文件放哪
        "pdf_path":r"E:\sgg\projects\RagZhiKu\res\doc\hak180产品安全手册.pdf" #上传给mineru解析的文件路径
    }
    result = node(init_state)
    logger.info(result)