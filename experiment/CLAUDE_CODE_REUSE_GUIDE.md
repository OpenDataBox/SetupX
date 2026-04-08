# Claude Code Docker 复用整理

这份说明只整理“另一个项目也要起 Docker 容器，在容器里安装并运行 Claude Code，再把结果交给现有 verify 流程”这一条链路。

## 先说结论

当前仓库里可直接复用的不是单个脚本，而是下面 4 层：

1. `experiment/claude_code/Dockerfile`
   负责做 Claude Code runner 基础镜像。
2. `experiment/claude_code/launch_claude_code.py`
   负责确保基础镜像存在、起一个独立 sandbox 容器、执行容器内入口，并在成功时输出 `container_id=...`。
3. `experiment/claude_code/src/in_container.ts`
   负责在 sandbox 容器里 clone 目标仓库、写 `~/.claude/settings.json`、调用 `claude -p`。
4. `src/environment_manager.py` + `src/verifier_agent.py`
   负责接管外部工具产出的容器，复用现有黑盒验证逻辑。

如果你另一个项目也需要这条能力，最小复用单元建议就是这 4 层，不要只拷 `Dockerfile`。

## 现有调用链

### 1. benchmark 调度层

`[experiment/run_cli_benchmark.py](/home/yukai/project/setUpAgentOurs/experiment/run_cli_benchmark.py#L193)` 约定：

- 外部工具如果产出容器，需要在输出里打印 `container_id=<id>`
- 然后脚本会调用 `[src/environment_manager.py](/home/yukai/project/setUpAgentOurs/src/environment_manager.py#L125)` 的 `from_container()`
- 再调用 `[src/verifier_agent.py](/home/yukai/project/setUpAgentOurs/src/verifier_agent.py)` 做统一验证

这意味着你另一个项目如果也想复用 verify，不需要重新设计接口，只要继续遵守 `container_id=...` 输出协议即可。

### 2. launcher 层

`[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py#L68)` 做三件事：

- 确保基础镜像存在，不存在就 `docker build`
- 用这个镜像启动一个长期存活容器：`sleep infinity`
- 执行 `docker exec <container> npm run benchmark-internal -- ...`

成功后在 `[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py#L225)` 输出 `container_id=<id>`，失败则删除容器。

这里是整个方案最适合抽成“通用 runner”的地方。

### 3. 容器内入口层

`[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L155)` 的顺序很清晰：

1. 清空工作目录
2. `git clone` 目标仓库
3. 如果不是 `HEAD`，就 checkout 指定 revision
4. 写 `~/.claude/settings.json`
5. 构造 prompt
6. 调 `claude -p ... --dangerously-skip-permissions`

真正和“Claude Code”强绑定的逻辑，基本只有两块：

- `[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L180)` 的 `buildClaudeEnv()`
- `[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L214)` 的 `writeClaudeSettings()`

其他诸如 clone 仓库、拼 prompt、执行 CLI，本质上都可以做成通用框架。

### 4. verifier 接管层

`[src/environment_manager.py](/home/yukai/project/setUpAgentOurs/src/environment_manager.py#L125)` 的 `from_container()` 已经把“接管外部容器”抽象好了。

`[src/environment_manager.py](/home/yukai/project/setUpAgentOurs/src/environment_manager.py#L142)` 的 `from_dockerfile()` 还支持接管外部工具产出的 Dockerfile。

所以另一个项目如果最终产物是：

- 一个已经配好的容器：继续输出 `container_id=...`
- 一个 Dockerfile 目录：输出 `dockerfile_dir=...`

这两种都能直接接现有统一验证链路。

## 哪些逻辑可以原样复用

下面这些建议直接复制过去：

- `[experiment/claude_code/Dockerfile](/home/yukai/project/setUpAgentOurs/experiment/claude_code/Dockerfile)` 的基础 runner 形态
- `[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py#L152)` 的环境变量收集逻辑
- `[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py#L110)` 的超时执行方式
- `[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L155)` 的仓库准备逻辑
- `[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L96)` 的 Claude CLI 调用方式
- `[experiment/run_cli_benchmark.py](/home/yukai/project/setUpAgentOurs/experiment/run_cli_benchmark.py#L193)` 的输出协议解析和 verify 接管逻辑

## 哪些逻辑是当前项目耦合点

这些地方不要无脑照搬：

### 1. benchmark 任务定义

`[experiment/prompts/repo_setup_task.txt](/home/yukai/project/setUpAgentOurs/experiment/prompts/repo_setup_task.txt)` 明显是“仓库配置实验”场景专用。

如果另一个项目不是 repo setup benchmark，而是固定仓库上的实验任务，应该把 prompt 换成你那边的任务模板。

### 2. `container_id` 输出协议

这套协议对接当前 benchmark 很合适，但如果新项目不走 `[experiment/run_cli_benchmark.py](/home/yukai/project/setUpAgentOurs/experiment/run_cli_benchmark.py)`，可以不保留 benchmark 目录结构。

真正必须保留的是：

- 外层能拿到 sandbox 容器 ID
- 后续验证模块知道容器内项目根目录

### 3. Claude 配置回退策略

`[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py#L154)` 和
`[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts#L193)` 里都做了：

- `ANTHROPIC_*` 缺失时回退到 `OPENCODE_*`

这是为了当前实验统一配置方便，不一定适合另一个项目。新项目如果只跑 Claude Code，建议取消这层回退，避免配置语义混乱。

### 4. host 网络模式

`launch_claude_code.py` 和 `EnvironmentManager` 都大量使用 `network_mode="host"`。

这在当前机器上是为了降低拉依赖、连兼容网关时的网络问题，但它不是跨环境都稳妥的默认值。另一个项目如果部署环境更严格，这里可能需要改成 bridge 网络 + 显式代理配置。

## 推荐的抽法

如果你准备长期复用，建议抽成下面三个模块：

### A. 通用 docker sandbox runner

职责：

- 确保基础镜像存在
- 起容器
- 把任务参数传进容器
- 超时控制
- 成功输出 `container_id`
- 失败自动清理

可直接从 `[experiment/claude_code/launch_claude_code.py](/home/yukai/project/setUpAgentOurs/experiment/claude_code/launch_claude_code.py)` 演化。

### B. tool-specific in-container adapter

职责：

- clone repo
- 写工具配置
- 调具体 CLI

Claude Code 对应当前的：

- `[experiment/claude_code/src/in_container.ts](/home/yukai/project/setUpAgentOurs/experiment/claude_code/src/in_container.ts)`

如果以后接别的 CLI，只替换这一层即可。

### C. verify adapter

职责：

- 从 `container_id` 或 `dockerfile_dir` 接管环境
- 调统一 verifier

这层已经基本现成，在：

- `[experiment/run_cli_benchmark.py](/home/yukai/project/setUpAgentOurs/experiment/run_cli_benchmark.py#L193)`
- `[src/environment_manager.py](/home/yukai/project/setUpAgentOurs/src/environment_manager.py#L125)`

## 另一个项目里最小接入步骤

如果你只想尽快跑起来，按这个顺序最省事：

1. 复制 `experiment/claude_code/` 整个目录
2. 保留 `launch_claude_code.py` 和 `src/in_container.ts` 的主体结构
3. 把任务 prompt 改成新项目自己的任务模板
4. 把 `.env` 里和 Claude 相关的 key 独立出来，不再回退到 `OPENCODE_*`
5. 新项目执行完成后继续打印 `container_id=<id>`
6. 如果要复用当前 verify，再从新项目调用 `[src/environment_manager.py](/home/yukai/project/setUpAgentOurs/src/environment_manager.py#L125)` 风格的接管接口

## 我对当前实现的判断

这套逻辑现在已经够复用，但还没有真正抽成“通用 runner”。

当前最值得抽象的公共接口是：

- `ensure_base_image(...)`
- `run_containerized_task(...)`
- `emit_container_handle(...)`
- `attach_and_verify(...)`

而不是继续复制 `launch_claude_code.py`、`launch_opencode.py`、`launch_qwen_code.py` 三份近似脚本。

如果你要，我下一步可以继续直接帮你做两件事里的一个：

1. 把当前仓库整理成一个可复用的 `docker_cli_runner` 公共模块
2. 按你另一个项目的目录结构，给你列一份“哪些文件直接复制、哪些地方改变量名和 env”的迁移清单
