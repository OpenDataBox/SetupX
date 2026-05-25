# Implementation Architecture of the Three CLI Tools in experiment

This document summarizes how the three CLI tool wrappers under the `experiment/` directory are implemented. The three tools are:

- `claude_code`
- `open_code`
- `qwen_code`

They share the same goal: within a unified experiment framework, accept the same repository setup task, complete the repository setup in an isolated container, and hand off the final container for reuse by the main project's existing verification pipeline.

## 1. Overall Layering

At a high level, all three implementations follow the same four-layer architecture:

1. Experiment scheduling layer
   Handled by [`experiment/run_cli_benchmark.py`](experiment/run_cli_benchmark.py). It reads the tool configuration, repository list, and unified prompt, and triggers launchers one tool at a time.

2. Python launcher layer
   Handled respectively by the following three scripts:
   - [`experiment/claude_code/launch_claude_code.py`](experiment/claude_code/launch_claude_code.py)
   - [`experiment/opencode/launch_opencode.py`](experiment/opencode/launch_opencode.py)
   - [`experiment/qwen_code/launch_qwen_code.py`](experiment/qwen_code/launch_qwen_code.py)

   This layer is responsible for loading `.env`, building or reusing the base image, starting an isolated Docker container, and entering the container via `docker exec` to run the internal runner.

3. In-container TypeScript runner layer
   Implemented respectively by the following files:
   - [`experiment/claude_code/src/in_container.ts`](experiment/claude_code/src/in_container.ts)
   - [`experiment/opencode/src/in_container.ts`](experiment/opencode/src/in_container.ts)
   - [`experiment/qwen_code/src/in_container.ts`](experiment/qwen_code/src/in_container.ts)

   This layer is responsible for cloning the target repository inside the container, assembling the experiment prompt, and invoking the specific CLI/SDK to run the repository setup task.

4. Unified verification layer
   After the launcher succeeds, it prints `container_id=<id>`. [`experiment/run_cli_benchmark.py`](experiment/run_cli_benchmark.py) then takes over the container via `EnvironmentManager.from_container()` and calls `VerifierAgent.verify()` to run unified black-box verification.

This means: although the three tools are invoked differently, they all converge onto the unified protocol of "producing a container that can be taken over".

## 2. Unified Execution Pipeline

The call chain for a single-repository experiment is essentially as follows:

1. `run_cli_benchmark.py` reads the `command_template` from `tools.json`.
2. The scheduling script renders variables such as `{repo_url}`, `{revision}`, and `{task_prompt_path}` into the command template.
3. The corresponding launcher starts:
   - Validates the required environment variables
   - Checks whether the base image exists, running `docker build` if not
   - `docker run` starts a long-lived sandbox container (`sleep infinity`)
   - `docker exec <container> npm run benchmark-internal -- ...` runs the in-container runner
4. The in-container runner:
   - Parses CLI arguments
   - Cleans and rebuilds the working directory
   - Runs `git clone` on the target repository and optionally checks out the revision
   - Appends experiment constraints to the unified task prompt
   - Invokes the corresponding CLI/SDK to run the task
5. If the launcher executes successfully, it prints `container_id=<container.id>`.
6. The main scheduling script extracts `container_id` and enters the unified verifier flow.

Therefore, the core of this design is not to directly compare the raw output of different CLIs, but to compare "whether they can ultimately configure the repository to a state that passes unified verification".

## 3. Shared Implementation Patterns Across the Three Tools

### 3.1 Shared Base Image Strategy

All three tools provide their own Dockerfile:

- [`experiment/claude_code/Dockerfile`](experiment/claude_code/Dockerfile)
- [`experiment/opencode/Dockerfile`](experiment/opencode/Dockerfile)
- [`experiment/qwen_code/Dockerfile`](experiment/qwen_code/Dockerfile)

Common points:

- The base image is uniformly `node:20-bookworm`
- Common build dependencies such as `git`, `python3`, `make`, and `g++` are preinstalled
- The tool directory's `package.json`, `tsconfig.json`, and `src/` are copied into `/runner`
- Internal runner dependencies are installed via `npm install`
- The corresponding CLI is installed via `npm install -g`
- The default container command is `sleep infinity`, to facilitate subsequent `docker exec`

This shows that the base image is only responsible for providing a "standard sandbox capable of running the CLI agent", rather than executing the experiment task during the image build stage.

### 3.2 Shared Launcher Pattern

All three launchers use Python for external orchestration, for clear reasons:

- It is easier to reuse the project's existing `.env`, Docker SDK, and Python verification logic
- It is better suited to being scheduled as a direct subprocess of `run_cli_benchmark.py`
- It can reliably print `container_id=...` on success for the main pipeline to parse

The launchers share the following behavior:

- `load_dotenv(PROJECT_ROOT / ".env", override=True)`
- `ensure_base_image(...)` handles image existence checks and rebuilds when necessary
- `client.containers.run(...)` starts the isolated container
- `run_exec(...)` calls the in-container `npm run benchmark-internal` via `docker exec`
- On failure, deletes the current sandbox container; on success, keeps the container and exposes the container ID to the verifier

### 3.3 Shared In-Container Runner Pattern

The structure of the three `src/in_container.ts` files is very similar, generally with the following responsibilities:

- `parseArgs`: parses `--repository`, `--repo-url`, `--revision`, `--task-prompt`
- `prepareWorkspace`: creates the working directory, clones the repository, and checks out the specified revision
- `buildPrompt`: concatenates the unified task prompt with additional experiment requirements
- `runSession` / `runClaude`: invokes the target agent
- Top-level `main()`: ties the entire in-container flow together

All three cleanly separate "repository preparation" from "agent invocation", so replacing the underlying model tool does not require changes to the scheduling layer or verification layer.

## 4. Claude Code Architecture

Claude Code is characterized by "Python launcher + directly invoking the claude CLI inside the container".

### 4.1 Outer Launcher

Key responsibilities of [`experiment/claude_code/launch_claude_code.py`](experiment/claude_code/launch_claude_code.py):

- Reads `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL` from `.env`, with support for falling back to `OPENCODE_API_KEY` / `OPENCODE_BASE_URL`
- Uniformly organizes model-related environment variables, such as `CLAUDE_CODE_MODEL` and `ANTHROPIC_MODEL`
- Builds a base image such as `claude-code-benchmark:latest`
- Starts the container with `user="node"` and sets `BENCHMARK_WORKSPACE_DIR`
- Runs the in-container `npm run benchmark-internal` via `docker exec`

It also specifically implements `get_exec_timeout_sec()` and logic for collecting stdout/stderr through temporary files, which indicates that the Claude CLI produces longer output and requires more robust timeout and log handling at runtime.

### 4.2 In-Container Runner

Key points of [`experiment/claude_code/src/in_container.ts`](experiment/claude_code/src/in_container.ts):

- Clones the repository into the working directory inside the container
- Calls `writeClaudeSettings()` to generate `~/.claude/settings.json`
- Uses `buildClaudeEnv()` to write configuration such as the Anthropic-compatible gateway, model mapping, and disabling auto-update into the settings
- Finally runs the prompt using `claude -p` headless mode

Claude Code is not driven by an SDK API, but runs directly via the CLI binary, with arguments including:

- `--output-format`
- `--dangerously-skip-permissions`
- `--max-turns`
- An optional `--model`

So Claude Code's setup is more of a "CLI wrapper pattern": Python handles container orchestration, TypeScript handles repository preparation and local configuration, and the actual agent execution falls to the `claude` command itself.

## 5. OpenCode Architecture

OpenCode is characterized by "Python launcher + starting a local service inside the container via the TypeScript SDK and polling the session".

### 5.1 Outer Launcher

Key responsibilities of [`experiment/opencode/launch_opencode.py`](experiment/opencode/launch_opencode.py):

- Strictly requires `OPENCODE_API_KEY`, `OPENCODE_BASE_URL`, and `OPENCODE_MODEL`
- Dynamically allocates a local port via `choose_server_port()` to avoid port conflicts when multiple experiments run simultaneously
- Builds and reuses the OpenCode base image
- Injects the API key, base URL, model, and service port into the container's environment variables

Compared with the other two tools, the OpenCode launcher additionally handles "port selection", because the in-container SDK starts a local service.

### 5.2 In-Container Runner

[`experiment/opencode/src/in_container.ts`](experiment/opencode/src/in_container.ts) does not directly invoke the `opencode` CLI, but instead uses `@opencode-ai/sdk`:

- `createOpencode(...)` starts a local OpenCode server
- `session.create(...)` creates a session
- `prompt_async` sends the prompt asynchronously
- Periodically polls `/session/status` and `/session/<id>/message`
- After the session enters a terminal state, extracts the new assistant messages and outputs them

Its implementation is effectively a "small session driver", with core logic including:

- `sendPromptAsync`: initiates the task
- `fetchSessionStatuses` / `fetchSessionMessages`: polls status and messages
- `waitForAssistantMessage`: waits for completion and obtains the final assistant output

Therefore, OpenCode's architecture is not simply "running a command", but "starting an SDK server inside the container and then driving the agent via an HTTP/session protocol". This is the one with the heaviest intermediate control logic among the three.

## 6. Qwen Code Architecture

Qwen Code is characterized by "Python launcher + directly streaming the query via the SDK inside the container".

### 6.1 Outer Launcher

The responsibilities of [`experiment/qwen_code/launch_qwen_code.py`](experiment/qwen_code/launch_qwen_code.py) are similar to OpenCode's, but lighter:

- Requires `QWEN_CODE_API_KEY` and `QWEN_CODE_BASE_URL`
- Allows specifying the model via `QWEN_CODE_MODEL`, defaulting to `qwen3-coder-plus`
- Optionally passes through `QWEN_CODE_CLI_PATH`
- Reuses or builds the Qwen Code base image
- Runs `npm run benchmark-internal` after starting the container

It has neither OpenCode's local server port management nor Claude's local configuration file generation.

### 6.2 In-Container Runner

[`experiment/qwen_code/src/in_container.ts`](experiment/qwen_code/src/in_container.ts) uses `query(...)` from `@qwen-code/sdk`:

- Specifies `cwd`
- Sets `model`
- Uses `authType: "openai"` to connect to the OpenAI-compatible interface
- Uses `permissionMode: "yolo"` to open up permissions
- Points `pathToQwenExecutable` to the in-container `qwen` CLI
- Injects `OPENAI_API_KEY` and `OPENAI_BASE_URL` into the execution environment

It then uses:

- `for await (const message of session)`
  to continuously consume the agent's output in a streaming fashion and print JSON line by line

So Qwen Code is the lightest of the three: there is no OpenCode local session server, nor Claude Code's local settings writing; it simply uses the query stream exposed by the SDK to complete the task.

## 7. Main Differences Among the Three

From an implementation-architecture perspective, the biggest difference among the three centers on "how the agent is driven inside the container":

- `claude_code`: directly invokes the `claude` CLI in headless mode
- `open_code`: starts a local service via the OpenCode SDK, then uses session + polling
- `qwen_code`: directly streams execution via the Qwen SDK's `query()`

From an engineering-complexity perspective:

- `claude_code`'s complexity lies in the configuration compatibility layer, especially the Anthropic-compatible gateway and `~/.claude/settings.json`
- `open_code`'s complexity lies in session lifecycle management and status polling
- `qwen_code` has the least complexity, relying more on the SDK's own streaming abstraction

From the external experiment framework's perspective, they are deliberately made consistent:

- Consistent input protocol: all accept `repository / repo-url / revision / task-prompt`
- Consistent output protocol: all print `container_id=<id>` on success
- Consistent isolation approach: each repository runs in an isolated Docker container
- Consistent verification approach: all hand off to the existing `EnvironmentManager + VerifierAgent`

This means the experiment framework only needs to swap the tool configuration, without changing the main evaluation logic.

## 8. Design Value

The value of this implementation architecture lies mainly in three points:

- Unified evaluation criteria
  All three tools are ultimately reduced to "whether they can produce a verifiable container", avoiding direct comparison of the text output of different CLIs.

- Decoupling tool implementation from verification implementation
  Each tool is only responsible for completing the repository setup, while verification fully reuses the main project's existing capabilities.

- Easy to extend
  To integrate a new CLI agent later, you only need to provide:
  - A launcher
  - An in-container runner
  - A `container_id` output protocol

  and it can be plugged directly into the existing benchmark framework.

## 9. One-Sentence Summary

The three CLI tools under `experiment/` are essentially "Dockerized agent adaptation layers oriented toward a unified verifier":

- `run_cli_benchmark.py` handles unified scheduling
- The Python launcher handles image and container lifecycle
- The TypeScript runner handles in-container repository preparation and agent invocation
- The main project's `EnvironmentManager` and `VerifierAgent` handle unified acceptance

The differences exist only in the innermost agent driving method, while the outer experiment interface is deliberately designed to be consistent.
