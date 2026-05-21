FROM node:20-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_CLI_NPM_SPEC=@anthropic-ai/claude-code
ARG CLAUDE_CODE_NPM_REGISTRY=https://registry.npmjs.org

WORKDIR /runner

COPY package.json tsconfig.json ./
COPY src ./src

RUN mkdir -p /workspace \
    && chown -R node:node /runner /workspace

RUN npm config set registry https://registry.npmmirror.com \
    && npm install
RUN npm install -g --registry "${CLAUDE_CODE_NPM_REGISTRY}" "${CLAUDE_CODE_CLI_NPM_SPEC}"
RUN claude --version

CMD ["sleep", "infinity"]
