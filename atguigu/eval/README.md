# RAG 评估模块（轻量版）

这套东西解决一个问题：**你改了一个地方，怎么知道效果是变好还是变差。**

不要凭感觉看一两条回答。正确做法是：

1. 准备一套固定的问题（测试集）；
2. 每条问题人工标好“应该命中哪份文档/哪一节”；
3. 每次改造后跑同一套问题，看指标和 bad case 有没有变化。

---

## 一、先理解四个指标

### Recall@k（召回率）

标准答案里一共有 3 个“应该被找到”的片段，系统返回的前 k 条里找到了 2 个：

```
Recall@k = 2 / 3 ≈ 0.67
```

它回答的是：**该找到的，有没有漏？**

- `Recall@5`：只看前 5 条
- `Recall@10`：只看前 10 条
- 1.0 表示全找到，0 表示一个都没找到

### MRR（Mean Reciprocal Rank）

第一个正确结果排第几：

- 排第 1 -> 1 分
- 排第 2 -> 1/2 分
- 排第 5 -> 1/5 分
- 没找到 -> 0 分

它回答的是：**第一个正确答案出现得早不早。**

### nDCG@k

和 MRR 类似，但它会考虑**多个**正确结果的整体排序：

- 命中的位置越靠前，贡献越大；
- 最后除以“理想排序的得分”，压到 0~1。

只有一个标准答案时，nDCG 和 MRR 很接近；有多个标准答案时，nDCG 更能反映整体排序质量。

### Faithfulness / Relevance / Citation

这三个是 judge 给“答案”打的分，不是检索指标：

- **Faithfulness（忠实度）**：答案里的事实能不能在检索资料里找到依据，有没有编造；
- **Relevance（相关性）**：有没有答非所问；
- **Citation（引用）**：答案里的来源/章节和资料对不对得上。

每项 1~5 分。judge 的评分理由会写进报告，方便你抽检。

---

## 二、测试集长什么样

`dataset/travel_eval.jsonl` 每行一个 case：

```json
{
  "id": "cd_transport_001",
  "type": "transport",
  "question": "成都双流机场到市区怎么走",
  "gold_item_names": ["成都"],
  "gold_docs": [
    {"file_title": "成都交通指南"}
  ],
  "gold_facts": [],
  "expect_refusal": false
}
```

字段说明：

| 字段 | 作用 |
|---|---|
| `id` | 唯一标识，方便定位 bad case |
| `type` | 题型，报告会按题型分组 |
| `question` | 用户问题 |
| `gold_item_names` | 应该识别出的主体名（旅游场景就是城市） |
| `gold_docs` | 应该命中的文档/章节 |
| `gold_facts` | 答案里必须出现的关键事实，给 judge 参考；可以先留空 |
| `expect_refusal` | 是否应该拒答（库外问题） |

当前测试集是**文档级标注**：只写 `file_title`，同一文件的任意章节都算命中。
等你把切分改造成带 `title_path` 后，可以这样升级成**章节级**：

```json
"gold_docs": [
  {"file_title": "成都景点推荐", "title_path": "成都景点推荐 > 核心景点推荐 > 都江堰"}
]
```

---

## 三、怎么跑

先确保 Milvus 里已经导入了测试集需要的文档。

### 只测检索（推荐先跑这个）

```bash
python -m atguigu.eval.run_eval --mode retrieval
```

特点：

- 不调用答案生成模型，速度快、成本低；
- 默认关闭 Web 搜索，避免外部噪音；
- 只看 Recall / MRR / nDCG / 主体名命中。

### 端到端（检索 + 生成 + judge）

```bash
python -m atguigu.eval.run_eval --mode full
```

特点：

- 会生成最终答案；
- 会调用 LLM judge 打 faithfulness / relevance / citation；
- token 成本和耗时更高。

### 只跑前 5 条

```bash
python -m atguigu.eval.run_eval --mode retrieval --limit 5
```

### 需要把 Web 搜索也纳入评估

```bash
python -m atguigu.eval.run_eval --mode full --enable-web
```

默认关闭 Web，是因为网页结果噪音大，会干扰“检索改造是否有效”的判断。

---

## 四、输出在哪里

每次运行会在 `atguigu/eval/results/<时间>_<模式>/` 下生成：

| 文件 | 内容 |
|---|---|
| `results.jsonl` | 每条 case 的召回结果、答案、各项指标 |
| `summary.json` | 总体指标、分题型指标、跳过/失败数量 |
| `report.html` | 可直接双击打开的轻量可视化报告 |

报告里有：

- 指标卡片：总 case 数、已评估、跳过、Recall@5、MRR、Faithfulness；
- 分题型条形图；
- 逐条明细：标准答案、实际召回、识别出的主体名、最终回答、judge 理由；
- Recall@5 为 0 或 faithfulness 低于 3 的 case 会标红。

---

## 五、文档还没导全怎么办

评估脚本启动时会先查 Milvus 里已经有哪些 `file_title`：

- 如果某条 case 需要的文档还没导入，会标记为 `skipped`，并在报告里写明缺哪份文档；
- 这样不会把“没导入”误判成“检索效果差”。

所以你可以**一份一份导入**，每导入几份就跑一次，看对应题型的指标。

---

## 六、这套评估的边界

- 测试集目前只有 20 条，属于冒烟级，不是严谨的统计评估；后续可以按城市/题型扩充；
- `gold_facts` 目前大多为空，judge 会更多依赖“资料是否自洽”，建议你读文档后补上关键事实；
- judge 和生成模型可能同源，存在自评偏袒；生产环境应换一个模型，并人工抽检 10%~20%；
- `retrieval` 模式不生成答案，所以拒答正确率会显示为 `-`，这是预期行为；
- 这不是企业级评估平台，只是本地可跑的最小闭环。企业里还会加：数据集版本管理、CI 回归门槛、实验跟踪（MLflow/LangSmith/Langfuse）、线上 A/B、Grafana 监控等。

---

## 七、建议的使用节奏

1. 导入成都的 5 份文档，先跑 `--mode retrieval`，看成都相关 case 的指标；
2. 再导入三亚、厦门等城市，逐步扩大测试范围；
3. 每次改切分/embedding/检索，都跑同一套测试集，记录 before/after；
4. 重点看两类信息：
   - 哪个题型最差；
   - 哪些 case 的 Recall@5 是 0，为什么没召回。
5. 等检索稳定后，再跑 `--mode full` 看答案质量。
