import { mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import process from "node:process";

type CliArgs = {
  repository: string;
  repoUrl: string;
  revision: string;
  taskPrompt: string;
};

type ClaudeSettings = {
  env: Record<string, string>;
  hasCompletedOnboarding: boolean;
};

function parseArgs(argv: string[]): CliArgs {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (!item.startsWith("--")) {
      continue;
    }
    const key = item.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`参数缺少取值: --${key}`);
    }
    args.set(key, value);
    i += 1;
  }

  const repository = args.get("repository");
  const repoUrl = args.get("repo-url");
  const revision = args.get("revision") ?? "HEAD";
  const taskPrompt = args.get("task-prompt");

  if (!repository || !repoUrl || !taskPrompt) {
    throw new Error("缺少必需参数: --repository --repo-url --task-prompt");
  }

  return {
    repository,
    repoUrl,
    revision,
    taskPrompt,
  };
}

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(`缺少环境变量: ${name}`);
  }
  return value;
}

function getEnvValue(...names: string[]): string {
  for (const name of names) {
    const value = process.env[name]?.trim();
    if (value) {
      return value;
    }
  }
  throw new Error(`缺少环境变量，至少需要配置其中之一: ${names.join(", ")}`);
}

function runCommand(command: string, args: string[], cwd?: string): void {
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    env: process.env,
  });
  if (result.error) {
    throw result.error;
  }
  if (typeof result.status === "number" && result.status !== 0) {
    throw new Error(`命令失败: ${command} ${args.join(" ")}`);
  }
}

function toBool(value: string | undefined, defaultValue: boolean): boolean {
  if (!value) {
    return defaultValue;
  }
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function resetDirectory(dir: string): void {
  mkdirSync(dir, { recursive: true });
  for (const entry of readdirSync(dir)) {
    rmSync(`${dir}/${entry}`, { recursive: true, force: true });
  }
}

async function runClaude(prompt: string, cwd: string): Promise<void> {
  const format = process.env.CLAUDE_CODE_OUTPUT_FORMAT?.trim() || "stream-json";
  const maxTurns = process.env.CLAUDE_CODE_MAX_TURNS?.trim() || "200";
  const verbose = toBool(process.env.CLAUDE_CODE_VERBOSE, format === "stream-json");
  const includePartialMessages = toBool(process.env.CLAUDE_CODE_INCLUDE_PARTIAL_MESSAGES, false);
  const model = process.env.CLAUDE_CODE_MODEL?.trim() || process.env.ANTHROPIC_MODEL?.trim() || "";
  const commandEnv = {
    ...process.env,
    HOME: process.env.HOME || "/root",
  };
  const cliArgs = [
    "-p",
    prompt,
    "--output-format",
    format,
    "--dangerously-skip-permissions",
    "--max-turns",
    maxTurns,
  ];
  if (verbose) {
    cliArgs.push("--verbose");
  }
  if (format === "stream-json" && verbose && includePartialMessages) {
    cliArgs.push("--include-partial-messages");
  }
  if (model) {
    cliArgs.push("--model", model);
  }

  await new Promise<void>((resolve, reject) => {
    const child = spawn(
      "claude",
      cliArgs,
      {
        cwd,
        env: commandEnv,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    child.stdout.on("data", (chunk: Buffer | string) => {
      process.stdout.write(chunk);
    });
    child.stderr.on("data", (chunk: Buffer | string) => {
      process.stderr.write(chunk);
    });
    child.on("error", (error) => {
      reject(error);
    });
    child.on("close", (code) => {
      if (typeof code === "number" && code !== 0) {
        reject(new Error(`Claude Code 执行失败，退出码=${code}`));
        return;
      }
      resolve();
    });
  });
}

function prepareWorkspace(repoUrl: string, revision: string): string {
  const workspaceDir = process.env.BENCHMARK_WORKSPACE_DIR?.trim() || "/tmp/workspace";
  resetDirectory(workspaceDir);

  runCommand("git", ["clone", repoUrl, workspaceDir]);
  if (revision !== "HEAD") {
    runCommand("git", ["checkout", revision], workspaceDir);
  }
  return workspaceDir;
}

function buildPrompt(taskPrompt: string, repository: string): string {
  return [
    taskPrompt.trim(),
    "",
    "额外实验要求：",
    `1. 当前工作目录就是目标仓库 ${repository}，所有仓库配置、依赖安装和验证都必须在当前 Docker 容器内完成。`,
    "2. 不要依赖宿主机已有环境，也不要假设仓库外部目录可写。",
    "3. 优先根据仓库内 README、requirements、pyproject、package.json、Dockerfile、Makefile 等信息完成配置。",
    "4. 尽可能运行仓库自带测试或最小可行验证命令，并如实报告结果。",
    "5. 在结束前，明确说明你做了哪些关键改动、运行了哪些验证命令，以及是否仍有失败项。",
    "6. 不要输出虚假的验证结论；如果失败，直接说明失败原因。",
  ].join("\n");
}

function buildClaudeEnv(): Record<string, string> {
  const model =
    process.env.CLAUDE_CODE_MODEL?.trim() ||
    process.env.ANTHROPIC_MODEL?.trim() ||
    process.env.OPENCODE_MODEL?.trim() ||
    "";
  const smallFastModel =
    process.env.CLAUDE_CODE_SMALL_FAST_MODEL?.trim() ||
    process.env.ANTHROPIC_SMALL_FAST_MODEL?.trim() ||
    process.env.OPENCODE_MODEL?.trim() ||
    model ||
    "";

  const env: Record<string, string> = {
    ANTHROPIC_AUTH_TOKEN: getEnvValue("ANTHROPIC_AUTH_TOKEN", "OPENCODE_API_KEY"),
    ANTHROPIC_BASE_URL: getEnvValue("ANTHROPIC_BASE_URL", "OPENCODE_BASE_URL"),
    DISABLE_AUTOUPDATER: "1",
  };

  if (model) {
    env.ANTHROPIC_MODEL = model;
    env.ANTHROPIC_DEFAULT_OPUS_MODEL = process.env.ANTHROPIC_DEFAULT_OPUS_MODEL?.trim() || model;
    env.ANTHROPIC_DEFAULT_SONNET_MODEL = process.env.ANTHROPIC_DEFAULT_SONNET_MODEL?.trim() || model;
    env.ANTHROPIC_DEFAULT_HAIKU_MODEL =
      process.env.ANTHROPIC_DEFAULT_HAIKU_MODEL?.trim() || smallFastModel || model;
  }

  if (smallFastModel) {
    env.ANTHROPIC_SMALL_FAST_MODEL = smallFastModel;
  }

  return env;
}

function writeClaudeSettings(): void {
  const configDir = `${process.env.HOME || "/root"}/.claude`;
  mkdirSync(configDir, { recursive: true });

  const settings: ClaudeSettings = {
    env: buildClaudeEnv(),
    hasCompletedOnboarding: true,
  };
  writeFileSync(
    `${configDir}/settings.json`,
    `${JSON.stringify(settings, null, 2)}\n`,
    "utf-8",
  );
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const workspaceDir = prepareWorkspace(args.repoUrl, args.revision);
  writeClaudeSettings();
  const prompt = buildPrompt(args.taskPrompt, args.repository);
  await runClaude(prompt, workspaceDir);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
