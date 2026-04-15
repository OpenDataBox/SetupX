# 综合实验报告 (2026-04-14T17:04:07)

## 范围说明
- 数据集：整合后的最终300条结果（open_code/qwen_code/claude_code 各100）。
- 本报告仅描述最终结果，不包含基线对比。

## 总体结果
- 总条目：300
- 成功：252 (84.00%)
- 失败：48 (16.00%)
- 超时：1
- 平均耗时：512.51s

## 分工具结果
- claude_code: 成功 83/100 (83.00%), 失败 17, 超时 1, 平均耗时 544.38s
- open_code: 成功 82/100 (82.00%), 失败 18, 超时 0, 平均耗时 499.88s
- qwen_code: 成功 87/100 (87.00%), 失败 13, 超时 0, 平均耗时 493.28s

## 主要失败原因
- claude_code: phase2_guilty=15; timeout=1; container_id_parse_failure=1
- open_code: container_id_parse_failure=16; phase2_guilty=2
- qwen_code: phase2_guilty=10; io_timeout=2; _ssl.c:1015: The handshake operation timed out=1

## Token统计（来自 run.log）
- 口径：若日志为累计快照（如 step-finish total 递增），按会话最大值计，不再逐步求和。
- 总token：12,051,107
- claude_code: 2,185,575
- open_code: 4,134,992
- qwen_code: 5,730,540

## 图表
- [成功率对比](./charts/chart_success_rate.png)
- [耗时分布（avg/median/p90）](./charts/chart_runtime.png)
- [Token消耗对比](./charts/chart_tokens.png)
- [失败原因Top5](./charts/chart_failure_reasons.png)

## 明细文件
- `tool_metrics.csv`
- `failure_reasons.csv`
- `repo_details.csv`