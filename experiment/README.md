# CLI 仓库配置实验流程

这个目录用于评估不同 CLI Agent 在“仓库配置”任务上的成功率，当前默认覆盖：

- `claude_code`
- `open_code`
- `qwen_code`

目标不是比较单次样例，而是用统一仓库清单、统一任务描述、统一验证口径，统计不同工具在仓库配置任务上的通过率。

## 目录结构

- `configs/tools.example.json`：工具配置样例
- `configs/repos_example.jsonl`：仓库清单样例
- `prompts/repo_setup_task.txt`：统一任务提示词
- `results/`：实验输出目录
- `run_cli_benchmark.py`：批量运行实验
- `summarize_results.py`：汇总成功率

## 实验流程

完整流程分成 4 步：

1. 读取仓库清单  
   默认从 `data/python329.jsonl` 或自定义 JSONL 中读取仓库，字段至少包含 `repository`，可选 `revision`。

2. 调用外部 CLI 工具执行仓库配置任务  
   `run_cli_benchmark.py` 会读取 `tools.json`，按工具逐个运行 `command_template`。模板变量由脚本统一替换，包括仓库名、仓库 URL、任务提示词路径和任务全文。

3. 复用当前仓库已有的 Docker 和 verify 流程做统一验证  
   这是这套实验的关键：
   - 如果外部 CLI 最终产出一个容器，并在输出中打印 `container_id=<容器ID>`，实验脚本会调用 `EnvironmentManager.from_container()` 接管这个容器。
   - 如果外部 CLI 最终产出一个 Dockerfile 目录，并打印 `dockerfile_dir=<目录路径>`，实验脚本会调用 `EnvironmentManager.from_dockerfile()` 构建并启动容器。
   - 接管成功后，统一调用 `VerifierAgent.verify()` 在容器内做黑箱验证。

4. 记录原始结果并汇总成功率  
   每次运行会在 `results/<run_id>/` 下生成一套独立结果，并按 `run_id / tool / repo` 隔离落盘。脚本默认不覆盖历史实验。

## Docker 与 Verify 复用约定

如果要走统一验证链路，外部 CLI 需要满足下面至少一种输出协议：

- 容器模式：stdout 或 stderr 中包含 `container_id=<容器 ID>`
- Dockerfile 模式：stdout 或 stderr 中包含 `dockerfile_dir=<Dockerfile 目录>`

对应配置项如下：

- `output_mode`：`container` 或 `dockerfile`
- `verify_enabled`：是否启用统一验证
- `container_id_pattern`：从输出中提取容器 ID 的正则
- `dockerfile_dir_pattern`：从输出中提取 Dockerfile 目录的正则
- `container_work_dir` / `dockerfile_work_dir`：验证时进入的项目目录

如果工具没有产出可接管的容器或 Dockerfile，脚本仍可运行，但只能按 `judge_mode` 用退出码或输出关键字做粗粒度判定，不能复用当前仓库的 verify 流程。

## 配置方式

先复制配置样例：

```bash
cp experiment/configs/tools.example.json experiment/configs/tools.json
```

然后填写每个工具的真实命令模板。示例字段：

```json
{
  "name": "claude_code",
  "enabled": true,
  "command_template": "your-cli --repo {repo_url} --task-file {task_prompt_path}",
  "output_mode": "container",
  "verify_enabled": true,
  "container_id_pattern": "container_id=([a-f0-9]{12,64})",
  "container_work_dir": "/workspace"
}
```

模板支持以下变量：

- `{repository}`
- `{repo_url}`
- `{revision}`
- `{task_prompt_path}`
- `{task_prompt}`

## 运行命令

运行 benchmark：

```bash
python experiment/run_cli_benchmark.py \
  --tools-config experiment/configs/tools.json \
  --repo-list data/python329.jsonl \
  --limit 10
```

汇总结果：

```bash
python experiment/summarize_results.py \
  --result-dir experiment/results/<run_id>
```

## 输出结果

每次运行会生成一套独立目录，结构如下：

```text
experiment/results/<run_id>/
  raw_results.jsonl
  summary.json
  <tool>/
    summary.json
    <repo>/
      run.log
      result.json
```

其中：

- `summary.json`：run 级汇总
- `<tool>/summary.json`：tool 级汇总
- `<tool>/<repo>/run.log`：单仓库原始日志
- `<tool>/<repo>/result.json`：单仓库结构化结果

当启用统一验证时，`raw_results.jsonl` 里还会包含：

- `verify_attempted`
- `verify_success`
- `verify_result`
- `verify_error`

## 建议口径

为了让不同工具的成功率具备可比性，建议固定以下条件：

- 使用同一份仓库清单
- 使用同一份任务提示词
- 使用同样的超时时间
- 尽量统一容器工作目录约定
- 成功判定优先采用 `VerifierAgent` 的结果，而不是工具自报成功

## 结果隔离约定

实验结果默认单独存放，不覆盖已有结果：

- `run_cli_benchmark.py` 每次运行都会创建新的结果目录
- 单仓库结果固定写到 `run_id/tool/repo/`
- 每个 tool 自动生成自己的 `summary.json`
- 每个 run 自动生成自己的 `summary.json`
- `summarize_results.py` 默认会额外写新的 `run_summary_<时间戳-微秒>.json`，不覆盖已有结果
- 如果你确实要写到固定文件，只能显式传 `--output`
