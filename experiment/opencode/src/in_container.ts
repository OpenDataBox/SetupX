import { mkdirSync, rmSync } from "node:fs";
import { spawnSync } from "node:child_process";
import process from "node:process";

import { createOpencode } from "@opencode-ai/sdk/v2";

type CliArgs = {
  repository: string;
  repoUrl: string;
  revision: string;
  taskPrompt: string;
};

type SessionMessage = {
  info: {
    id: string;
    role: string;
  };
  parts: unknown[];
};

type OpencodeEvent = {
  type: string;
  properties?: Record<string, unknown>;
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
  const parsed = raw ? Number.parseInt(raw, 10) : 60 * 60 * 1000;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 60 * 60 * 1000;
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
  client: Awaited<ReturnType<typeof createOpencode>>["client"],
  sessionId: string,
  cwd: string,
  providerId: string,
  model: string,
  prompt: string,
): Promise<void> {
  const response = await client.session.promptAsync({
    sessionID: sessionId,
    directory: cwd,
    model: {
      providerID: providerId,
      modelID: model,
    },
    parts: [{ type: "text", text: prompt }],
  });
  if (response.error) {
    throw new Error(`OpenCode promptAsync 失败: ${JSON.stringify(response.error)}`);
  }
}

async function fetchSessionMessages(serverUrl: string, sessionId: string, cwd: string): Promise<Array<{ info: { id: string; role: string }; parts: unknown[] }>> {
  const url = new URL(`/session/${sessionId}/message`, serverUrl);
  url.searchParams.set("directory", cwd);
  const data = await parseJsonResponse(await fetch(url));
  return (data as SessionMessage[] | null) ?? [];
}

function emitSessionMessage(sessionId: string, message: SessionMessage): void {
  process.stdout.write(`${JSON.stringify({
    type: "session_message",
    session_id: sessionId,
    role: message.info.role,
    message_id: message.info.id,
    parts: message.parts,
  })}\n`);
}

function emitSessionPart(sessionId: string, role: string, messageId: string, part: unknown): void {
  process.stdout.write(`${JSON.stringify({
    type: "session_message",
    session_id: sessionId,
    role,
    message_id: messageId,
    parts: [part],
  })}\n`);
}

function emitCommandExecuted(sessionId: string, properties: Record<string, unknown>): void {
  process.stdout.write(`${JSON.stringify({
    type: "command_executed",
    session_id: sessionId,
    message_id: properties.messageID,
    name: properties.name,
    arguments: properties.arguments,
  })}\n`);
}

function getEventSessionId(event: OpencodeEvent): string | undefined {
  const properties = event.properties ?? {};
  const info = properties.info;
  if (typeof properties.sessionID === "string") {
    return properties.sessionID;
  }
  if (info && typeof info === "object" && typeof (info as Record<string, unknown>).sessionID === "string") {
    return String((info as Record<string, unknown>).sessionID);
  }
  if (properties.part && typeof properties.part === "object") {
    const partSessionId = (properties.part as Record<string, unknown>).sessionID;
    if (typeof partSessionId === "string") {
      return partSessionId;
    }
  }
  return undefined;
}

async function waitForSessionCompletion(
  client: Awaited<ReturnType<typeof createOpencode>>["client"],
  serverUrl: string,
  sessionId: string,
  cwd: string,
  signal: AbortSignal,
): Promise<SessionMessage | null> {
  const subscription = await client.event.subscribe(
    { directory: cwd },
    {
      signal,
      sseMaxRetryAttempts: 5,
    },
  );

  let lastStatus = "unknown";

  try {
    for await (const event of subscription.stream as AsyncGenerator<OpencodeEvent>) {
      const eventSessionId = getEventSessionId(event);
      if (eventSessionId !== sessionId) {
        continue;
      }

      const properties = event.properties ?? {};
      if (event.type === "message.part.updated") {
        const part = properties.part;
        if (!part || typeof part !== "object") {
          continue;
        }
        const partRecord = part as Record<string, unknown>;
        const messageId = String(partRecord.messageID ?? "");
        if (!messageId) {
          continue;
        }

        const messages = await fetchSessionMessages(serverUrl, sessionId, cwd);
        const message = messages.find((item) => item.info.id === messageId);
        const role = message?.info.role ?? "assistant";
        emitSessionPart(sessionId, role, messageId, partRecord);
      } else if (event.type === "command.executed") {
        emitCommandExecuted(sessionId, properties);
      } else if (event.type === "session.status") {
        const status = properties.status;
        lastStatus =
          status && typeof status === "object" && typeof (status as Record<string, unknown>).type === "string"
            ? String((status as Record<string, unknown>).type)
            : "unknown";
        process.stderr.write(`等待 OpenCode 完成，当前状态: ${describeStatus(status)}\n`);
      } else if (event.type === "session.idle") {
        const messages = await fetchSessionMessages(serverUrl, sessionId, cwd);
        const lastAssistantMessage = [...messages].reverse().find((message) => message.info.role === "assistant") ?? null;
        return lastAssistantMessage;
      } else if (event.type === "session.error") {
        throw new Error(`OpenCode session 出错: ${JSON.stringify(properties)}`);
      }
    }
  } catch (error) {
    if (signal.aborted) {
      throw new Error(`OpenCode 执行超时，超过 ${getTimeoutMs()} ms 仍未完成`);
    }
    throw error;
  }

  if (lastStatus !== "idle") {
    throw new Error(`OpenCode 事件流提前结束，最后状态=${lastStatus}`);
  }

  return null;
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
      title: `benchmark:${model}`,
      directory: cwd,
    });
    if (!session.data) {
      throw new Error("创建 OpenCode session 失败，未返回 session 数据");
    }

    const abortController = new AbortController();
    const timeoutId = setTimeout(() => {
      abortController.abort();
    }, getTimeoutMs());

    await sendPromptAsync(opencode.client, session.data.id, cwd, providerId, model, prompt);

    const result = await waitForSessionCompletion(
      opencode.client,
      opencode.server.url,
      session.data.id,
      cwd,
      abortController.signal,
    );
    clearTimeout(timeoutId);
    printResponse({
      type: "result",
      session_id: session.data.id,
      message: result,
    });
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
