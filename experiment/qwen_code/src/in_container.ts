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
      throw new Error(`Argument is missing a value: --${key}`);
    }
    args.set(key, value);
    i += 1;
  }

  const repository = args.get("repository");
  const repoUrl = args.get("repo-url");
  const revision = args.get("revision") ?? "HEAD";
  const taskPrompt = args.get("task-prompt");

  if (!repository || !repoUrl || !taskPrompt) {
    throw new Error("Missing required arguments: --repository --repo-url --task-prompt");
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
    throw new Error(`Missing environment variable: ${name}`);
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
    throw new Error(`Command failed: ${command} ${args.join(" ")}`);
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
    "Additional experiment requirements:",
    `1. The current working directory is the target repository ${repository}; all repository setup, dependency installation, and verification must be done inside the current Docker container.`,
    "2. Do not rely on any pre-existing environment on the host, and do not assume that directories outside the repository are writable.",
    "3. Prefer to complete the setup based on information inside the repository, such as README, requirements, pyproject, package.json, Dockerfile, Makefile, etc.",
    "4. Run the repository's own tests or the minimal viable verification commands whenever possible, and report the results truthfully.",
    "5. Before finishing, clearly state which key changes you made, which verification commands you ran, and whether any failures remain.",
    "6. Do not output false verification conclusions; if something fails, state the cause of failure directly.",
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
