# experiment 中三个 CLI 工具的实现架构

本文总结 `experiment/` 目录下三个 CLI 工具包装器的实现方式。这里的三个工具分别是：

- `claude_code`
- `open_code`
- `qwen_code`

它们的目标一致：在统一实验框架里，接收同一份仓库配置任务，在隔离容器中完成仓库 setup，并把最终容器交给主项目已有的验证链路复用。

## 1. 总体分层

从整体上看，这三套实现都遵循相同的四层架构：

1. 实验调度层  
   由 [`experiment/run_cli_benchmark.py`](experiment/run_cli_benchmark.py) 负责。它读取工具配置、仓库列表和统一 prompt，按工具逐个触发 launcher。

2. Python launcher 层  
   分别由下面三个脚本负责：
   - [`experiment/claude_code/launch_claude_code.py`](experiment/claude_code/launch_claude_code.py)
   - [`experiment/opencode/launch_opencode.py`](experiment/opencode/launch_opencode.py)
   - [`experiment/qwen_code/launch_qwen_code.py`](experiment/qwen_code/launch_qwen_code.py)

   这一层负责加载 `.env`、构建或复用基础镜像、启动独立 Docker 容器，并通过 `docker exec` 进入容器执行内部 runner。

3. 容器内 TypeScript runner 层  
   分别由以下文件实现：
   - [`experiment/claude_code/src/in_container.ts`](experiment/claude_code/src/in_container.ts)
   - [`experiment/opencode/src/in_container.ts`](experiment/opencode/src/in_container.ts)
   - [`experiment/qwen_code/src/in_container.ts`](experiment/qwen_code/src/in_container.ts)

   这一层负责在容器内 clone 目标仓库、拼接实验 prompt，并调用具体 CLI/SDK 执行仓库配置任务。

4. 统一验证层  
   launcher 成功后会打印 `container_id=<id>`，随后 [`experiment/run_cli_benchmark.py`](experiment/run_cli_benchmark.py) 通过 `EnvironmentManager.from_container()` 接管该容器，再调用 `VerifierAgent.verify()` 做统一黑箱验证。

这意味着：三个工具虽然调用方式不同，但都被收敛到“输出可接管容器”这一统一协议上。

## 2. 统一执行链路

一次单仓库实验的调用链基本如下：

1. `run_cli_benchmark.py` 读取 `tools.json` 中的 `command_template`。
2. 调度脚本将 `{repo_url}`、`{revision}`、`{task_prompt_path}` 等变量渲染到命令模板。
3. 对应 launcher 启动：
   - 校验所需环境变量
   - 检查基础镜像是否存在，不存在则 `docker build`
   - `docker run` 启动一个长期存活的 sandbox 容器（`sleep infinity`）
   - `docker exec <container> npm run benchmark-internal -- ...` 执行容器内 runner
4. 容器内 runner：
   - 解析 CLI 参数
   - 清空并重建工作目录
   - `git clone` 目标仓库并可选 checkout revision
   - 将统一任务 prompt 追加实验约束
   - 调用对应 CLI/SDK 执行任务
5. launcher 若执行成功，则输出 `container_id=<container.id>`。
6. 主调度脚本提取 `container_id`，进入统一 verifier 流程。

因此，这套设计的核心不是直接比较不同 CLI 的原始输出，而是比较“它们最终是否能把仓库配置到可通过统一验证”的能力。

## 3. 三个工具的共同实现模式

### 3.1 共同的基础镜像策略

三个工具都提供各自的 Dockerfile：

- [`experiment/claude_code/Dockerfile`](experiment/claude_code/Dockerfile)
- [`experiment/opencode/Dockerfile`](experiment/opencode/Dockerfile)
- [`experiment/qwen_code/Dockerfile`](experiment/qwen_code/Dockerfile)

共同点：

- 基础镜像统一为 `node:20-bookworm`
- 预装 `git`、`python3`、`make`、`g++` 等常见构建依赖
- 把本工具目录下的 `package.json`、`tsconfig.json`、`src/` 拷入 `/runner`
- 通过 `npm install` 安装内部 runner 依赖
- 通过 `npm install -g` 安装对应 CLI
- 默认容器命令为 `sleep infinity`，便于后续 `docker exec`

这说明基础镜像只负责提供“可运行该 CLI agent 的标准沙箱”，而不是直接在镜像构建阶段执行实验任务。

### 3.2 共同的 launcher 模式

三个 launcher 都采用 Python 负责外部编排，原因很明确：

- 更容易复用当前项目里已有的 `.env`、Docker SDK 和 Python 验证逻辑
- 更适合作为 `run_cli_benchmark.py` 的直接子进程被调度
- 成功后能够稳定输出 `container_id=...` 供主流程解析

launcher 的共同行为包括：

- `load_dotenv(PROJECT_ROOT / ".env", override=True)`
- `ensure_base_image(...)` 负责镜像存在性检查和必要时重建
- `client.containers.run(...)` 启动隔离容器
- `run_exec(...)` 用 `docker exec` 调用容器内的 `npm run benchmark-internal`
- 失败时删除当前 sandbox 容器，成功时保留容器并把容器 ID 暴露给 verifier

### 3.3 共同的容器内 runner 模式

三个 `src/in_container.ts` 的结构非常接近，基本都有以下职责：

- `parseArgs`：解析 `--repository`、`--repo-url`、`--revision`、`--task-prompt`
- `prepareWorkspace`：创建工作目录，clone 仓库并 checkout 指定 revision
- `buildPrompt`：将统一任务提示词和额外实验要求拼接
- `runSession` / `runClaude`：调用目标 agent
- 顶层 `main()`：串起整个容器内流程

三者都把“仓库准备”和“agent 调用”清晰分离，因此替换底层模型工具时，不需要改动调度层和验证层。

## 4. Claude Code 架构

Claude Code 的特点是“Python launcher + 容器内直接调用 claude CLI”。

### 4.1 外层 launcher

[`experiment/claude_code/launch_claude_code.py`](experiment/claude_code/launch_claude_code.py) 的关键职责：

- 从 `.env` 中读取 `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL`，并支持回退到 `OPENCODE_API_KEY` / `OPENCODE_BASE_URL`
- 统一整理模型相关环境变量，如 `CLAUDE_CODE_MODEL`、`ANTHROPIC_MODEL`
- 构建 `claude-code-benchmark:latest` 之类的基础镜像
- 以 `user="node"` 启动容器，并设置 `BENCHMARK_WORKSPACE_DIR`
- 通过 `docker exec` 执行容器内 `npm run benchmark-internal`

这里还专门实现了 `get_exec_timeout_sec()` 和临时文件收集 stdout/stderr 的逻辑，说明 Claude CLI 输出较长、运行时更需要稳妥的超时与日志处理。

### 4.2 容器内 runner

[`experiment/claude_code/src/in_container.ts`](experiment/claude_code/src/in_container.ts) 的关键点：

- 在容器内 clone 仓库到工作目录
- 调用 `writeClaudeSettings()` 生成 `~/.claude/settings.json`
- 通过 `buildClaudeEnv()` 把 Anthropic 兼容网关、模型映射、关闭自动更新等配置写入 settings
- 最终用 `claude -p` 无头模式执行 prompt

Claude Code 不是通过 SDK API 驱动，而是通过 CLI 二进制直接运行，参数包括：

- `--output-format`
- `--dangerously-skip-permissions`
- `--max-turns`
- 可选 `--model`

所以 Claude Code 这一套更像“CLI 包装器模式”：Python 负责容器编排，TypeScript 负责仓库准备和本地配置，真正的 agent 执行落在 `claude` 命令本身。

## 5. OpenCode 架构

OpenCode 的特点是“Python launcher + 容器内通过 TypeScript SDK 启动本地服务并轮询 session”。

### 5.1 外层 launcher

[`experiment/opencode/launch_opencode.py`](experiment/opencode/launch_opencode.py) 的关键职责：

- 强制要求 `OPENCODE_API_KEY`、`OPENCODE_BASE_URL`、`OPENCODE_MODEL`
- 通过 `choose_server_port()` 动态分配一个本地端口，避免多个实验同时运行时端口冲突
- 构建并复用 OpenCode 基础镜像
- 将 API Key、Base URL、Model 和服务端口注入容器环境变量

与另外两个工具相比，OpenCode launcher 额外负责“端口选择”，因为容器内 SDK 会起本地服务。

### 5.2 容器内 runner

[`experiment/opencode/src/in_container.ts`](experiment/opencode/src/in_container.ts) 不是直接调用 `opencode` CLI，而是使用 `@opencode-ai/sdk`：

- `createOpencode(...)` 启动一个本地 OpenCode server
- `session.create(...)` 创建会话
- `prompt_async` 异步发送 prompt
- 周期性轮询 `/session/status` 和 `/session/<id>/message`
- 等待 session 进入终态后提取新的 assistant 消息并输出

它的实现实际上是一个“小型会话驱动器”，核心逻辑包括：

- `sendPromptAsync`：发起任务
- `fetchSessionStatuses` / `fetchSessionMessages`：轮询状态和消息
- `waitForAssistantMessage`：等待完成并拿到最终 assistant 输出

因此 OpenCode 的架构不是简单“执行一个命令”，而是“在容器内启动 SDK server，再通过 HTTP/session 协议驱动 agent”。这是三者里中间控制逻辑最重的一套。

## 6. Qwen Code 架构

Qwen Code 的特点是“Python launcher + 容器内通过 SDK 直接流式 query”。

### 6.1 外层 launcher

[`experiment/qwen_code/launch_qwen_code.py`](experiment/qwen_code/launch_qwen_code.py) 的职责与 OpenCode 类似，但更轻：

- 要求 `QWEN_CODE_API_KEY` 和 `QWEN_CODE_BASE_URL`
- 允许通过 `QWEN_CODE_MODEL` 指定模型，默认 `qwen3-coder-plus`
- 可选透传 `QWEN_CODE_CLI_PATH`
- 复用或构建 Qwen Code 基础镜像
- 启动容器后执行 `npm run benchmark-internal`

它没有 OpenCode 那样的本地 server 端口管理，也没有 Claude 那样的本地配置文件生成。

### 6.2 容器内 runner

[`experiment/qwen_code/src/in_container.ts`](experiment/qwen_code/src/in_container.ts) 使用 `@qwen-code/sdk` 的 `query(...)`：

- 指定 `cwd`
- 设置 `model`
- 使用 `authType: "openai"` 对接 OpenAI 兼容接口
- 使用 `permissionMode: "yolo"` 放开权限
- 通过 `pathToQwenExecutable` 指向容器内 `qwen` CLI
- 把 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 注入到执行环境

随后它通过：

- `for await (const message of session)`  
  按流式方式持续消费 agent 输出并逐条打印 JSON

所以 Qwen Code 是三者里最轻量的一套：没有 OpenCode 的本地 session server，也没有 Claude Code 的本地 settings 写入，直接使用 SDK 暴露的 query 流完成任务。

## 7. 三者的主要差异

从实现架构上看，三者最大的区别集中在“容器内如何驱动 agent”：

- `claude_code`：直接调用 `claude` CLI，无头模式执行
- `open_code`：通过 OpenCode SDK 启动本地服务，再走 session + polling
- `qwen_code`：通过 Qwen SDK 的 `query()` 直接流式执行

从工程复杂度看：

- `claude_code` 的复杂点在配置兼容层，尤其是 Anthropic 兼容网关和 `~/.claude/settings.json`
- `open_code` 的复杂点在 session 生命周期管理和状态轮询
- `qwen_code` 的复杂点最少，更多依赖 SDK 本身的流式封装

从外部实验框架角度看，它们又被有意做成一致：

- 输入协议一致：都接收 `repository / repo-url / revision / task-prompt`
- 输出协议一致：成功时都输出 `container_id=<id>`
- 隔离方式一致：每个仓库运行一个独立 Docker 容器
- 验证方式一致：都交给现有 `EnvironmentManager + VerifierAgent`

这使得实验框架只需要替换工具配置，不需要改动主评测逻辑。

## 8. 设计价值

这套实现架构的价值主要有三点：

- 统一评测口径  
  三个工具最终都被转换成“是否能产出一个可验证容器”，避免直接比较不同 CLI 的文本输出。

- 解耦工具实现与验证实现  
  各工具只负责完成仓库配置，验证完全复用主项目已有能力。

- 易于扩展  
  如果后续接入新的 CLI agent，只要补齐：
  - 一个 launcher
  - 一个容器内 runner
  - 一个 `container_id` 输出协议
  
  就可以直接挂进现有 benchmark 框架。

## 9. 一句话总结

`experiment/` 下三个 CLI 工具本质上都是“面向统一 verifier 的 Docker 化 agent 适配层”：

- `run_cli_benchmark.py` 负责统一调度
- Python launcher 负责镜像和容器生命周期
- TypeScript runner 负责容器内仓库准备和 agent 调用
- 主项目的 `EnvironmentManager` 与 `VerifierAgent` 负责统一验收

差异只存在于最内层的 agent 驱动方式，而外层实验接口被刻意设计成一致。
