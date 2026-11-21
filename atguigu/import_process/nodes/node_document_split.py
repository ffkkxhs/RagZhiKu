# atguigu/import_process/nodes/node_document_split.py
import json
import re
from pathlib import Path

from atguigu.import_process.base import NodeBase
from atguigu.import_process.state import ImportGraphState
from atguigu.tool.logger import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter


class NodeDocumentSplit(NodeBase):
    """
    文档切分节点：智能文档切片
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):


        file_title = state.get("file_title")
        if not file_title:
            logger.error("文件不存在")
            raise Exception("文件不存在")

        md_path = state.get("md_path")
        if not md_path:
            logger.error("md文件路径不能为空")
            raise Exception("md文件路径不能为空")

        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            logger.error("md 文件不存在")
            raise FileNotFoundError(f"md 文件不存在: {md_path}")


        with open(md_path_obj,"r",encoding="utf-8") as f:
            md_content = f.read()

        md_content = md_content.replace("\r\n","\n").replace("\r","\n")

        spilt_lines = md_content.split("\n")

        codePattern = r"^(`{3}|~{3})"
        title_pattern = r'^\s*#{1,6}\s+.+'

        is_block = False

        temp_list = []

        start_idx = 0
        marker = ""

        for idx,spilt_line in enumerate(spilt_lines):
            
            spilt_line = spilt_line.strip()

            match = re.search(codePattern,spilt_line)

            if match:
                if is_block == False:
                    is_block = True
                    marker = match.group(1)
                elif marker == match.group(1):
                    is_block = False
                    marker = ""


            if re.search(title_pattern,spilt_line) and not is_block:
                
                if spilt_lines[start_idx:idx]:
                    temp_list.append(spilt_lines[start_idx:idx])
                    start_idx = idx

        temp_list.append(spilt_lines[start_idx:])


        section_list = []


        for t in temp_list:
            
            first_line = t[0].strip()
            title = first_line if first_line.startswith("#") else "无标题"  


            section_list.append({
                "title":title,
                "file_title":file_title,
                "content":"\n".join(t),
            })

        spliter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", " "],
            chunk_size=300,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )

        final_chunks = []

        for section in section_list:
            section_content =section.get("content")
            section_title = section.get("title")
            section_file_title = section.get("file_title")

            if section_title.startswith("#"):
                parts = section_content.split("\n", 1)
                section_real_content = parts[1].lstrip("\n") if len(parts) > 1 else ""
            else:
                section_real_content = section_content


            if not section_real_content.strip():
                continue

            if len(section_real_content)<=300:
                final_chunks.append({
                    "title":section_title,
                    "file_title":section_file_title,
                    "content":section_real_content,
                    "part" : 0
                })
                continue

            if "<table" in section_real_content:
                for i, table_chunk in enumerate(self._split_big_table(section_real_content), start=1):
                    final_chunks.append({
                        "title": section_title[:100],          # title 也是 VARCHAR(100)
                        "file_title": section_file_title[:100],
                        "content": table_chunk,
                        "part": i,
                    })
                continue
                

            chunk_list = spliter.split_text(section_real_content)

            for id, chunk in enumerate(chunk_list):
                final_chunks.append({
                    "title":section_title,
                    "file_title":section_file_title,
                    "content":section_title+"\n\n"+chunk,
                    "part" : id+1,
                })

        # 调试产物跟着当前文档走，避免处理新文档时覆盖 hak180 的文件
        out_path = md_path_obj.parent / "split_chunks.json"
        with open(out_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(final_chunks, ensure_ascii=False, indent=4))

        return {
            "chunks": final_chunks,
        }
        
    def _split_big_table(self, content, max_len=3000):
        """HTML 表格按 </tr> 切，Markdown 管道表格按行切"""
        if "</tr>" in content.lower():
            units = re.split(r"(?<=</tr>)", content, flags=re.IGNORECASE)
        else:
            units = content.splitlines(keepends=True)

        pieces, buf = [], ""
        for unit in units:
            if buf and len(buf) + len(unit) > max_len:
                pieces.append(buf)
                buf = unit
            else:
                buf += unit
        if buf.strip():
            pieces.append(buf)
        return pieces or [content[:max_len]]


if __name__ == "__main__":

    node = NodeDocumentSplit()
    init_state = {
        "md_path":r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
        "file_title":"hak180产品安全手册"
    }
    result = node(init_state)
    logger.info(json.dumps(result,ensure_ascii=False,indent=4))





# class NodeDocumentSplit(NodeBase):
#     """
#     文档切分节点：智能文档切片
#     """
#
#     name = "node_document_split"
#
#     def process(self, state: ImportGraphState):
#
#         md_path = state.get("md_path")
#         file_title = state.get("file_title")
#         md_path_obj = Path(md_path)
#
#         if not md_path:
#             logger.error("文件路径不能为空")
#             raise Exception("文件路径不能为空")
#
#         if not file_title:
#             logger.error("文件不存在")
#             raise Exception("文件不存在")
#
#         if not md_path_obj.exists():
#             logger.error("文件名不能为空")
#             raise Exception("文件名不能为空")
#
#         with open (md_path_obj, "r",encoding="utf-8") as file:
#             md_content = file.read()
#
#         md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
#
#         print(type(md_content),file_title)
#
#
#
#
#         #todo
#
#         # todo 标题正则
#         # todo 代码围栏is_block
#         # todo 代码块边界符号标记
#         # todo 遍历 行
#         #   todo 去白符
#         #   todo 正则拿match对象
#
#         split_lines = md_content.split("\n")
#
#         codePattern = r"^(`{3,}|~{3,})"
#         title_pattern = r'^\s*#{1,6}\s+.+'
#
#         is_block = False
#         marker = None
#
#         start_idx = 0
#
#         temp_list = []
#
#         for idx,s_line in enumerate(split_lines) :
#
#             s_line = s_line.strip()
#
#             match = re.match(codePattern,s_line)
#
#             if match:
#                 if not is_block:
#                     is_block = True
#                     marker = match.group(1)
#                 elif marker == match.group(1):
#                     is_block = False
#                     marker = None
#
#             if re.match(title_pattern, s_line) and not is_block:
#
#                 temp_list.append(split_lines[start_idx:idx])
#                 start_idx = idx
#
#         temp_list.append(split_lines[start_idx:])
#
#
#         section_list = []
#
#         for t in temp_list:
#
#             section_list.append({
#                 "title": t[0] if t[0].startswith("#") else "无标题",
#                 "file_title": file_title,
#                 "content": "\n".join(t),
#             })
#
#         print(section_list[0])
#
#
#         spliter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", " "],
#             chunk_size=300,
#             chunk_overlap=50,
#             length_function=len,
#             add_start_index=True
#         )
#
#
#         final_chunks = []
#
#         for section in section_list:
#
#             section_title = section.get("title")
#             section_content = section.get("content")
#             section_file_title = section.get("file_title")
#
#             section_real_content = section_content[len(section_title)+2:] if section_content.startswith("#") else section_content
#
#             if len(section_real_content) <= 300:
#                 final_chunks.append({
#                     "title": section_title,
#                     "file_title": section_file_title,
#                     "content": section_real_content,
#                     "part": 0
#                 })
#
#                 continue
#
#             # todo table
#             if "<table" in section_real_content:
#                 final_chunks.append({
#                     "title": section_title,
#                     "file_title": section_file_title,
#                     "content": section_real_content,
#                     # **section
#                     "part": 0
#                 })
#                 continue
#
#             for chunk_id,chunk in enumerate(spliter.split_text(section_real_content)):
#
#                 final_chunks.append({
#                     "title": section_title,
#                     "file_title": section_file_title,
#                     "content": section_title + "\n\n" +chunk,   #把标题拼到chunk最前面，做上下文增强
#                     "part": chunk_id+1,
#                 })
#
#         with open(r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\split_chunks.json","w",encoding="utf-8") as file:
#             file.write(json.dumps(final_chunks,ensure_ascii=False,indent=4))
#
#         return {
#             "chunks": final_chunks,
#         }
#
#
# if __name__ == "__main__":
#
#     node = NodeDocumentSplit()
#     init_state = {
#         "md_path":r"E:\sgg\projects\RagZhiKu\res\output\hak180产品安全手册\hak180产品安全手册_new.md",
#         "file_title":"hak180产品安全手册"
#     }
#     result = node(init_state)
#     logger.info(json.dumps(result,ensure_ascii=False,indent=4))
