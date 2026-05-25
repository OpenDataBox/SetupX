# CLI Repository Setup Experiment Pipeline

This directory is used to evaluate the success rate of different CLI agents on "repository setup" tasks. The current default coverage includes:

- `claude_code`
- `open_code`
- `qwen_code`

The goal is not to compare individual examples, but to measure the pass rate of different tools on repository setup tasks using a unified repository list, unified task description, and unified verification criteria.

## Directory Structure

- `configs/tools.example.json`: example tool configuration
- `configs/repos_example.jsonl`: example repository list
- `prompts/repo_setup_task.txt`: unified task prompt
- `claude_code/`: Claude Code headless-mode wrapper
- `opencode/`: OpenCode TypeScript SDK wrapper
- `qwen_code/`: Qwen Code TypeScript SDK wrapper
- `results/`: experiment output directory
- `run_cli_benchmark.py`: batch experiment runner
- `summarize_results.py`: success-rate aggregator

## Experiment Pipeline

The full pipeline has 4 steps:

1. Read the repository list
   By default, repositories are read from `data/python329.jsonl` or a custom JSONL file. Each entry must contain at least a `repository` field, with an optional `revision`.

2. Invoke the external CLI tool to run the repository setup task
   `run_cli_benchmark.py` reads `tools.json`. By default it runs "multiple tools concurrently per repository", meaning a single repository simultaneously launches `claude_code`, `open_code`, and `qwen_code`. Template variables are substituted uniformly by the script, including the repository name, repository URL, task prompt path, and full task text.

3. Reuse the repository's existing Docker, verify, and adjudication phase2 flow for unified verification
   This is the key part of the experiment:
   - If the external CLI ultimately produces a container and prints `container_id=<container ID>` in its output, the experiment script calls `EnvironmentManager.from_container()` to take over that container.
   - If the external CLI ultimately produces a Dockerfile directory and prints `dockerfile_dir=<directory path>`, the experiment script calls `EnvironmentManager.from_dockerfile()` to build and start the container.
   - After takeover succeeds, it first calls `VerifierAgent.verify()` to run black-box verification inside the container, then reuses the same prosecutor/judge phase2 adjudication used by the main pipeline.

4. Record raw results and aggregate the success rate
   Each run generates a separate set of results under `results/<run_id>/`, isolated by `run_id / tool / repo`. By default the script does not overwrite previous experiments. After each single-repository result completes, it immediately refreshes `raw_results.jsonl`, the tool-level `summary.json`, and the run-level `summary.json`, so that even if the process exits unexpectedly midway, the completed portion is preserved.

## Docker and Verify Reuse Conventions

To use the unified verification pipeline, the external CLI must satisfy at least one of the following output protocols:

- Container mode: stdout or stderr contains `container_id=<container ID>`
- Dockerfile mode: stdout or stderr contains `dockerfile_dir=<Dockerfile directory>`

The corresponding configuration options are:

- `output_mode`: `container` or `dockerfile`
- `verify_enabled`: whether to enable unified verification
- `container_id_pattern`: regex used to extract the container ID from output
- `dockerfile_dir_pattern`: regex used to extract the Dockerfile directory from output
- `container_work_dir` / `dockerfile_work_dir`: the project directory to enter during verification
- `phase2_enabled`: whether to continue running adjudication phase2 after verify; enabled by default

If a tool does not produce a container or Dockerfile that can be taken over, the script can still run, but it can only make coarse-grained judgments based on the exit code or output keywords according to `judge_mode`, and cannot reuse the repository's verify/phase2 flow.

## Configuration

First, copy the example configuration:

```bash
cp experiment/configs/tools.example.json experiment/configs/tools.json
```

The default example already invokes each launcher using `.venv/bin/python` under the repository root.
If your virtual environment directory is different, update `command_template` accordingly.

Then fill in the real command template for each tool. Example fields:

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

The template supports the following variables:

- `{repository}`
- `{repo_url}`
- `{revision}`
- `{task_prompt_path}`
- `{task_prompt}`
- `{repo_dir}`

## Qwen Code Integration

`qwen_code` is currently integrated via the official TypeScript SDK, with code in `experiment/qwen_code/`. It is always invoked with:

- Node.js `>= 20.0.0`
- A Qwen Code CLI accessible inside the sandbox container
- `permissionMode: "yolo"`
- `authType: "openai"`
- `QWEN_CODE_API_KEY` from the root `.env`
- `QWEN_CODE_BASE_URL` from the root `.env`
- An optional `QWEN_CODE_MODEL`, defaulting to `qwen3-coder-plus`
- Qwen Code runs only inside an isolated Docker sandbox container
- On success, it directly prints `container_id=<id>` for the unified verifier to take over

First, prepare the runner image dependencies:

```bash
cd experiment/qwen_code
npm install
```

The root `.env` requires at least:

```bash
QWEN_CODE_API_KEY=your_api_key
QWEN_CODE_BASE_URL=https://your-openai-compatible-endpoint/v1
QWEN_CODE_MODEL=qwen3-coder-plus
QWEN_CODE_BASE_IMAGE=qwen-code-benchmark:latest
QWEN_CODE_CLI_NPM_SPEC=@qwen-code/qwen-code@latest
# If qwen is not directly discoverable on PATH, specify the full path explicitly
QWEN_CODE_CLI_PATH=/full/path/to/qwen
```

For each repository, the wrapper will:

1. First build and cache a `qwen_code` base image
2. Start a new, isolated Docker container from that base image for each repository experiment
3. Clone the repository, run Qwen Code, and perform dependency installation and verification only inside that container
4. On success, print `container_id=<id>`, handed off to the existing `EnvironmentManager.from_container()` and `VerifierAgent.verify()` for reuse
5. On failure, automatically delete that sandbox container to avoid polluting the host machine

In other words, images are reused at the tool level, while containers are isolated at the `tool + repo + run` level.

The official SDK documentation requires:

- Node.js `>= 20.0.0`
- Qwen Code `>= 0.4.0` (stable release) installed and accessible on PATH
- If you use `nvm`, you should explicitly set `pathToQwenExecutable` to the full path of the `qwen` binary

## Claude Code Integration

`claude_code` is currently integrated via an OpenAI-compatible gateway, with code in `experiment/claude_code/`. It is always invoked with:

- Node.js `>= 20.0.0`
- Claude Code installed inside the base image via `npm install -g @anthropic-ai/claude-code`
- Repository setup run inside a Docker container
- `claude -p` headless mode
- `--output-format json` by default, to reduce instability at completion caused by long streaming logs
- `--dangerously-skip-permissions` to skip permission confirmation and avoid the experiment stalling on interactive prompts
- `ANTHROPIC_AUTH_TOKEN` from the root `.env`
- `ANTHROPIC_BASE_URL` from the root `.env`
- An optional `CLAUDE_CODE_MODEL` or `ANTHROPIC_MODEL`
- If no separate Claude configuration is provided, it falls back to reusing `OPENCODE_API_KEY`, `OPENCODE_BASE_URL`, and `OPENCODE_MODEL` by default
- On success, it directly prints `container_id=<id>` for the unified verifier to take over

First, prepare the runner image dependencies:

```bash
cd experiment/claude_code
npm install
```

The root `.env` requires at least:

```bash
ANTHROPIC_AUTH_TOKEN=your_api_key
ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
CLAUDE_CODE_MODEL=qwen3-coder-plus
CLAUDE_CODE_BASE_IMAGE=claude-code-benchmark:latest
CLAUDE_CODE_CLI_NPM_SPEC=@anthropic-ai/claude-code
CLAUDE_CODE_NPM_REGISTRY=https://registry.npmjs.org
CLAUDE_CODE_OUTPUT_FORMAT=json
```

If you prefer to directly reuse your existing OpenCode configuration, you can avoid adding Claude-specific keys; `claude_code` will automatically fall back to:

```bash
OPENCODE_API_KEY=...
OPENCODE_BASE_URL=...
OPENCODE_MODEL=...
```

If you want strict control via Claude Code's model environment variables, you can also add:

```bash
ANTHROPIC_MODEL=qwen3-coder-plus
ANTHROPIC_SMALL_FAST_MODEL=qwen3-coder-plus
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3-coder-plus
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3-coder-plus
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3-coder-plus
```

For each repository, the wrapper will:

1. First build and cache a `claude_code` base image
2. Start a new, isolated Docker container from that base image for each repository experiment
3. Clone the repository, write `~/.claude/settings.json`, run Claude Code, and perform dependency installation and verification only inside that container
4. On success, print `container_id=<id>`, handed off to the existing `EnvironmentManager.from_container()` and `VerifierAgent.verify()` for reuse
5. On failure, automatically delete that sandbox container to avoid polluting the host machine

What is measured here is the combined effect of "Claude Code frontend + a third-party Anthropic-compatible gateway + target model", not the native results of the official Claude model.

## OpenCode Integration

`open_code` is currently integrated via the official JavaScript/TypeScript SDK, with code in `experiment/opencode/`. It is always invoked with:

- Node.js `>= 20.0.0`
- The OpenCode SDK inside the base image
- Repository setup run inside a Docker container
- OpenCode permissions are open by default, corresponding to the `yolo` semantics in the experiment
- `OPENCODE_API_KEY` from the root `.env`
- `OPENCODE_BASE_URL` from the root `.env`
- `OPENCODE_MODEL` from the root `.env`
- On success, it directly prints `container_id=<id>` for the unified verifier to take over

First, prepare the runner image dependencies:

```bash
cd experiment/opencode
npm install
```

The root `.env` requires at least:

```bash
OPENCODE_API_KEY=your_api_key
OPENCODE_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENCODE_MODEL=qwen3-coder-plus
OPENCODE_BASE_IMAGE=opencode-benchmark:latest
OPENCODE_CLI_NPM_SPEC=opencode-ai@latest
# Optional; if omitted, the launcher will automatically pick a free port
OPENCODE_SERVER_PORT=
```

For each repository, the wrapper will:

1. First build and cache an `open_code` base image
2. Start a new, isolated Docker container from that base image for each repository experiment
3. Clone the repository, run OpenCode, and perform dependency installation and verification only inside that container
4. On success, print `container_id=<id>`, handed off to the existing `EnvironmentManager.from_container()` and `VerifierAgent.verify()` for reuse
5. On failure, automatically delete that sandbox container to avoid polluting the host machine

OpenCode official SDK notes:

- The SDK package name is `@opencode-ai/sdk`
- `createOpencode()` starts both the server and the client
- Permissions are allowed by default; the experiment explicitly sets `bash` and `edit` to `allow` as a `yolo`-equivalent configuration
- The OpenAI-compatible interface should be integrated via `npm: "@ai-sdk/openai-compatible"`, `options.baseURL`, and `options.apiKey` in the `provider` configuration

## Running Commands

Run the benchmark:

```bash
.venv/bin/python experiment/run_cli_benchmark.py \
  --tools-config experiment/configs/tools.json \
  --repo-list data/python329.jsonl \
  --limit 10 \
  --tool-parallelism 3
```

Aggregate the results:

```bash
.venv/bin/python experiment/summarize_results.py \
  --result-dir experiment/results/<run_id>
```

## Output Results

Each run generates a separate directory with the following structure:

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

Where:

- `summary.json`: run-level summary
- `<tool>/summary.json`: tool-level summary
- `<tool>/<repo>/run.log`: raw log for a single repository
- `<tool>/<repo>/result.json`: structured result for a single repository

Additional notes:

- `--tool-parallelism` controls how many tools run simultaneously per repository; the default `0` means it automatically uses the number of enabled tools, i.e. with the current configuration it runs 3 CLIs concurrently by default.
- The benchmark refreshes `raw_results.jsonl` and the summary files in real time after each task completes, to avoid losing an entire round's results if it crashes midway.
- If a repository throws an uncaught exception during the launcher, verify, or phase2 process, the script falls back to writing out that repository's `run.log` and `result.json`, and continues processing other tasks.

When unified verification is enabled, `raw_results.jsonl` will also include:

- `verify_attempted`
- `verify_success`
- `verify_result`
- `verify_error`

## Recommended Criteria

To make success rates comparable across different tools, it is recommended to fix the following conditions:

- Use the same repository list
- Use the same task prompt
- Use the same timeout
- Keep the container working-directory convention as consistent as possible
- Prefer the `VerifierAgent` result for the success judgment, rather than the tool's self-reported success

## Result Isolation Conventions

Experiment results are stored separately by default and do not overwrite existing results:

- `run_cli_benchmark.py` creates a new result directory on each run
- Single-repository results are always written to `run_id/tool/repo/`
- Each tool automatically generates its own `summary.json`
- Each run automatically generates its own `summary.json`
- By default, `summarize_results.py` additionally writes a new `run_summary_<timestamp-microseconds>.json`, without overwriting existing results
- If you really need to write to a fixed file, you must pass `--output` explicitly
