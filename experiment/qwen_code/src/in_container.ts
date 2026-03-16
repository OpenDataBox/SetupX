import { mkdirSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";

import { query } from "@qwen-code/sdk";

type CliArgs = {
  repository: string;
  repoUrl: string;
  revision: string;
  taskPrompt: string;
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

function prepareWorkspace(repoUrl: string, revision: string): string {
  const workspaceDir = "/workspace";
  rmSync(workspaceDir, { recursive: true, force: true });
  mkdirSync(workspaceDir, { recursive: true });

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

async function runSession(prompt: string, cwd: string): Promise<void> {
  const model = process.env.QWEN_CODE_MODEL?.trim() || process.env.OPENAI_MODEL?.trim() || "qwen3-coder-plus";
  const executablePath = process.env.QWEN_CODE_CLI_PATH?.trim();
  const apiKey = requireEnv("QWEN_CODE_API_KEY");
  const baseUrl = requireEnv("QWEN_CODE_BASE_URL");

  const session = query({
    prompt,
    options: {
      cwd,
      model,
      authType: "openai",
      permissionMode: "yolo",
      pathToQwenExecutable: executablePath || "qwen",
      env: {
        OPENAI_API_KEY: apiKey,
        OPENAI_BASE_URL: baseUrl,
      },
    },
  });

  for await (const message of session) {
    process.stdout.write(`${JSON.stringify(message)}\n`);
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const workspaceDir = prepareWorkspace(args.repoUrl, args.revision);
  const prompt = buildPrompt(args.taskPrompt, args.repository);
  await runSession(prompt, workspaceDir);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
