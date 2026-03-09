一、调研背景
我们的 Agent 目标是自动化配置 GitHub 仓库的运行环境（克隆仓库 → 安装依赖 → 修复构建错误 → 验证可运行）。现有的 Code CLI / AI 编程工具是否能直接胜任这个任务？可以作为我们的 baseline 对比。
我们对 10 款主流 AI 编程工具进行了详细调研，从"能否自动配置仓库环境"的角度进行分析。
二、详细分析
1. Claude Code（Anthropic）

核心能力：
完整的终端访问权限：可执行任意 bash 命令（git clone、pip install、apt-get 等）
文件读写与编辑：可直接创建/修改代码文件
Git 集成：自动 commit、创建 PR
MCP支持：可扩展自定义工具
上下文窗口 200K tokens
Headless / 批处理模式：
支持 -p 参数做非交互式运行：claude -p "clone this repo and set up the environment"
可集成进 CI/CD 管道、bash 脚本
支持管道输入输出：echo "fix the build" | claude -p
支持并发会话
能否用于仓库环境配置：
可以，Claude Code 有完整的 bash 执行能力，可以克隆仓库、读取 requirements.txt、安装依赖、运行构建、分析报错并自动修复
headless 模式下可以脚本化批量运行
但成本较高（每次调用消耗大量 token，Opus 4.5 输出 $25/M tokens），可能得用sonnet或者haiku来跑
Docker 支持：可以在 Docker 容器内运行
局限性：
需要 Anthropic API key 或订阅
重度使用有周限流
成本随任务复杂度线性增长

2. Aider
暂时无法在飞书文档外展示此内容
核心能力：
多文件编辑：可以同时修改多个文件
Git 自动提交：每次编辑自动 commit，方便回滚
代码检查：每次 LLM 编辑后自动 lint + 修复
支持 /run 命令执行 shell 命令并将输出反馈给 LLM
Headless / 批处理模式：
支持 --message 参数非交互执行：aider --message "install dependencies and fix build errors"
支持 --yes-always 自动确认所有操作
Docker 官方镜像：docker pull paulgauthier/aider
能否用于仓库环境配置：
部分可以。Aider 的核心定位是代码编辑，不是环境配置
可以通过 /run 执行 shell 命令，但不是自动循环执行
适合"修改代码 → 测试 → 修复"的循环，但对"安装系统依赖 → 解决环境冲突"这类纯运维任务不如 Claude Code 直接
可以配合脚本封装实现环境配置
模型灵活性：
最大优势：支持几乎所有 LLM（OpenAI、Anthropic、Google、DeepSeek）
有官方模型排行榜（aider.chat/docs/leaderboards）
可以用便宜模型（DeepSeek）降低成本
局限性：
不是为环境配置任务设计的
shell 命令执行需手动触发（/run）或在 headless 模式下有限支持
没有内置的"检测错误 → 自动修复"循环

3. Cursor
暂时无法在飞书文档外展示此内容
核心能力：
Tab 补全：智能代码补全
Cmd+K 内联编辑：选中代码直接修改
Chat：上下文感知的对话
Composer（Agent Mode）：多文件自主编辑 + 终端命令执行
支持最多 8 个 agent 并行工作（各自独立 git worktree）
.cursorrules 自定义规则文件
Agent Mode（Composer）：
可以自主规划多步骤任务
可以运行终端命令、分析输出、迭代修复
支持浏览器操作（调试 UI）
能否用于仓库环境配置：
不太可以，Cursor 是纯 GUI 工具，没有 CLI/headless 模式
无法脚本化、无法批量运行、无法集成到 CI/CD
适合开发者手动打开项目后让 Agent 辅助配置，不适合自动化 pipeline
Docker/远程支持：支持 SSH 远程开发、WSL
局限性：
没有 CLI 模式
需要 GUI 环境
Pro 订阅制，成本固定

4. GitHub Copilot CLI
暂时无法在飞书文档外展示此内容
核心能力：
Plan Mode：Shift+Tab 切换，先分析后执行
Autopilot Mode：完全自主执行，不需逐步确认
专用 Agent：Explore（代码分析）、Task（构建测试）、Code Review、Plan
文件编辑、终端命令执行、迭代修复
能否用于仓库环境配置：
可以。有完整的终端执行能力和自主循环
Autopilot 模式可以让 agent 自主完成环境配置
可集成到 CI/CD（GitHub Actions 原生集成）

局限性：
需要 GitHub 订阅
比较新，生态还在完善

5. OpenAI Codex CLI
暂时无法在飞书文档外展示此内容
核心能力：
本地终端执行：读、改、运行代码
轻量级，启动快
GPT-5.x-Codex 系列模型专门针对编码优化
多级自主度控制
能否用于仓库环境配置：
可以。有终端执行能力，可以自主配置环境
是开源的
局限性：
依赖 OpenAI API
SWE-bench 分数不高，看起来能力稍差

6. Google Gemini CLI
暂时无法在飞书文档外展示此内容
核心能力：
ReAct 循环：推理 + 行动
内置工具：Google Search、文件操作、shell 命令执行、Web Fetch
MCP Server 支持
1M token 长上下文窗口
能否用于仓库环境配置：
可以。有 shell 命令执行能力和 ReAct 循环
最大优势：完全免费，适合大规模批量测试
1M token 上下文可以处理非常长的构建日志
局限性：
Gemini 在编码任务上的表现不如 Claude/GPT（SWE-bench 分数较低）
免费额度有每日上限

7. OpenHands（原 OpenDevin）
暂时无法在飞书文档外展示此内容
核心能力：
完整的 Docker 沙箱执行环境
Agent 可以写代码、运行命令、浏览网页
支持多 Agent 协作
丰富的评估框架
SDK 可扩展
能否用于仓库环境配置：
可以。OpenHands 的架构天然适合环境配置任务：
  Docker 沙箱隔离
  自主执行命令和修复错误的循环
  可以直接用来跑 SWE-bench 式的仓库配置任务
可以作为我们最直接的对比 baseline
局限性：
暂无

8. Cline（原 Claude Dev）
暂时无法在飞书文档外展示此内容
核心能力：
自主编码 Agent：创建/编辑文件、执行终端命令、浏览器操作
每步需用户确认（安全优先）
MCP 支持：可扩展自定义工具
CLI 2.0：支持 headless 模式、并行 Agent
支持 Agent Client Protocol
能否用于仓库环境配置：
可以。有终端执行能力、headless 模式
CLI 2.0 的 headless 模式可以集成到 CI/CD
9. Windsurf（原 Codeium）
暂时无法在飞书文档外展示此内容

核心能力：
Cascade：Agentic AI 助手，多步编辑 + 工具调用 + 深度仓库上下文
终端集成
可保存工作流为可复用 markdown 命令
能否用于仓库环境配置：
不太可以。纯 GUI 工具，类似 Cursor
没有 CLI/headless 模式，无法自动化

三、对比表
暂时无法在飞书文档外展示此内容

四、下一步
选择baseline
设计统一评估指标：成功率、平均步数、耗时、API 成本
构建测试集：从 python329.jsonl 中选取 50-100 个仓库，所有工具跑同一批
对照实验：baseline 工具 vs 我们的 Agent（无 XPU）vs 我们的 Agent（有 XPU），三组对比
