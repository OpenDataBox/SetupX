import { mkdirSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import process from "node:process";

import { createOpencode } from "@opencode-ai/sdk";

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

function printResponse(response: unknown): void {
  process.stdout.write(`${JSON.stringify(response)}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function getPollIntervalMs(): number {
  const raw = process.env.OPENCODE_POLL_INTERVAL_MS?.trim();
  const parsed = raw ? Number.parseInt(raw, 10) : 2000;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 2000;
}

function getTimeoutMs(): number {
  const raw = process.env.OPENCODE_SESSION_TIMEOUT_MS?.trim();
  const parsed = raw ? Number.parseInt(raw, 10) : 15 * 60 * 1000;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 15 * 60 * 1000;
}

function getServerPort(): number {
  const raw = process.env.OPENCODE_SERVER_PORT?.trim();
  const parsed = raw ? Number.parseInt(raw, 10) : 4096;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 4096;
}

function describeStatus(status: unknown): string {
  if (!status || typeof status !== "object") {
    return "unknown";
  }

  const type = "type" in status ? String(status.type) : "unknown";
  if (type !== "retry") {
    return type;
  }

  const attempt = "attempt" in status ? String(status.attempt) : "?";
  const message = "message" in status ? String(status.message) : "";
  return `retry(attempt=${attempt}, message=${message})`;
}

function isTerminalStatus(status: unknown): boolean {
  if (!status || typeof status !== "object") {
    return true;
  }

  const type = "type" in status ? String(status.type) : "";
  return type === "idle";
}

async function parseJsonResponse(response: Response): Promise<unknown> {
  if (!response.ok) {
    throw new Error(`OpenCode HTTP 请求失败: ${response.status} ${response.statusText}`);
  }

  const text = await response.text();
  if (!text.trim()) {
    return null;
  }
  return JSON.parse(text) as unknown;
}

async function sendPromptAsync(
  serverUrl: string,
  sessionId: string,
  cwd: string,
  providerId: string,
  model: string,
  prompt: string,
): Promise<void> {
  const url = new URL(`/session/${sessionId}/prompt_async`, serverUrl);
  url.searchParams.set("directory", cwd);

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: {
        providerID: providerId,
        modelID: model,
      },
      parts: [{ type: "text", text: prompt }],
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`OpenCode prompt_async 失败: ${response.status} ${response.statusText} ${body}`);
  }
}

async function fetchSessionStatuses(serverUrl: string): Promise<Record<string, unknown>> {
  const url = new URL("/session/status", serverUrl);
  const data = await parseJsonResponse(await fetch(url));
  return (data as Record<string, unknown> | null) ?? {};
}

async function fetchSessionMessages(serverUrl: string, sessionId: string, cwd: string): Promise<Array<{ info: { id: string; role: string }; parts: unknown[] }>> {
  const url = new URL(`/session/${sessionId}/message`, serverUrl);
  url.searchParams.set("directory", cwd);
  const data = await parseJsonResponse(await fetch(url));
  return (data as Array<{ info: { id: string; role: string }; parts: unknown[] }> | null) ?? [];
}

async function waitForAssistantMessage(
  serverUrl: string,
  sessionId: string,
  cwd: string,
  previousAssistantIds: Set<string>,
): Promise<unknown> {
  const deadline = Date.now() + getTimeoutMs();
  const pollIntervalMs = getPollIntervalMs();

  while (Date.now() < deadline) {
    const [statusResult, messagesResult] = await Promise.all([
      fetchSessionStatuses(serverUrl),
      fetchSessionMessages(serverUrl, sessionId, cwd),
    ]);

    const currentStatus = statusResult[sessionId];
    const messages = messagesResult;
    const assistantMessage = [...messages]
      .reverse()
      .find((message) => message.info.role === "assistant" && !previousAssistantIds.has(message.info.id));

    if (assistantMessage && isTerminalStatus(currentStatus)) {
      return assistantMessage;
    }

    process.stderr.write(`等待 OpenCode 完成，当前状态: ${describeStatus(currentStatus)}\n`);
    await sleep(pollIntervalMs);
  }

  throw new Error(`OpenCode 执行超时，超过 ${getTimeoutMs()} ms 仍未完成`);
}

async function runSession(prompt: string, cwd: string): Promise<void> {
  const apiKey = requireEnv("OPENCODE_API_KEY");
  const baseUrl = requireEnv("OPENCODE_BASE_URL");
  const model = requireEnv("OPENCODE_MODEL");
  const providerId = "benchmark";
  const serverPort = getServerPort();

  process.chdir(cwd);

  const opencode = await createOpencode({
    hostname: "127.0.0.1",
    port: serverPort,
    config: {
      model: `${providerId}/${model}`,
      provider: {
        [providerId]: {
          npm: "@ai-sdk/openai-compatible",
          name: "Benchmark Provider",
          options: {
            baseURL: baseUrl,
            apiKey,
          },
          models: {
            [model]: {
              name: model,
            },
          },
        },
      },
      permission: {
        bash: "allow",
        edit: "allow",
      },
      autoupdate: false,
    },
  });

  try {
    const session = await opencode.client.session.create({
      body: {
        title: `benchmark:${model}`,
      },
      query: {
        directory: cwd,
      },
    });
    if (!session.data) {
      throw new Error("创建 OpenCode session 失败，未返回 session 数据");
    }

    const existingMessages = await fetchSessionMessages(opencode.server.url, session.data.id, cwd);
    const previousAssistantIds = new Set(
      existingMessages
        .filter((message) => message.info.role === "assistant")
        .map((message) => message.info.id),
    );

    await sendPromptAsync(opencode.server.url, session.data.id, cwd, providerId, model, prompt);

    const result = await waitForAssistantMessage(
      opencode.server.url,
      session.data.id,
      cwd,
      previousAssistantIds,
    );
    printResponse(result);
  } finally {
    opencode.server.close();
  }
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const workspaceDir = prepareWorkspace(args.repoUrl, args.revision);
  const prompt = buildPrompt(args.taskPrompt, args.repository);
  await runSession(prompt, workspaceDir);
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch((error: unknown) => {
    const message = error instanceof Error ? error.stack || error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exit(1);
  });
