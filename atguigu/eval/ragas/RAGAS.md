# RAGAS 评估方案（独立）

这是和 `custom/run_eval.py` 并行的一套评估，不改动原有方案。

## 1. 安装

RAGAS 是可选依赖，不安装不影响项目运行：

```bash
uv sync --extra ragas
```

或者：

```bash
uv pip install "ragas>=0.2,<0.3"
```

建议锁在 `0.2.x`，避免 RAGAS API 变动导致代码不兼容。

## 2. 配置

RAGAS 的 judge 模型默认使用项目默认模型。建议在 `.env` 里换一个更强的模型：

```env
EVAL_JUDGE_MODEL=qwen-plus
```

原因：RAGAS 会做 claim 拆分、NLI、answer correctness 等复杂判断，flash 级别模型容易不稳定。

## 3. 运行

```bash
# 只跑前 5 条（成都）
python -m atguigu.eval.ragas.ragas_eval --limit 5

# 跑完整测试集
python -m atguigu.eval.ragas.ragas_eval
```

注意：

- RAGAS 需要 `reference_answer`；没有参考答案的 case 会跳过；
- 默认 `max_workers=1`，避免并发调用 LLM 被限流；
- 不要和导入服务/`custom/run_eval` 同时跑，BGE 和 LLM 会抢资源；
- 默认关闭 Web 搜索，需要时加 `--enable-web`。

## 4. 输出

每次运行会在 `atguigu/eval/ragas/results/<时间>_ragas/` 下生成：

| 文件 | 内容 |
|---|---|
| `ragas_samples.jsonl` | 每条 case 的 question/answer/reference/contexts |
| `ragas_scores.jsonl` | 每条 case 的 RAGAS 五项指标 |
| `ragas_summary.json` | 五项指标平均值、跳过原因、样本数 |
| `ragas_report.html` | 轻量 HTML 报告 |

五项指标：

- `context_recall`
- `context_precision`
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`

## 5. 和现有 run_eval 的区别

| | `custom/run_eval.py` | `ragas_eval.py` |
|---|---|---|
| 检索指标 | 自己实现，确定性 | RAGAS 指标 |
| judge | 自己的中文 rubric | RAGAS 内置 prompt/claim 拆分 |
| 依赖 | 无额外依赖 | 需要 `ragas` |
| 成本 | 较低 | 较高（claim/NLI/多指标） |
| 结果 | `custom/results/<time>_retrieval/full` | `ragas/results/<time>_ragas` |

建议：

- 日常改切分/检索时用 `custom/run_eval.py`，快、便宜；
- 需要和行业标准对齐、看 claim 级指标时用 `ragas_eval.py`；
- 两者结果分开看，不要混成一个分数。
