FROM node:20-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates python3 make g++ \
    && rm -rf /var/lib/apt/lists/*

ARG OPENCODE_CLI_NPM_SPEC=opencode-ai@latest

WORKDIR /runner

COPY package.json tsconfig.json ./
COPY src ./src

RUN npm config set registry https://registry.npmmirror.com \
    && npm install
RUN npm config set registry https://registry.npmmirror.com \
    && npm install -g "${OPENCODE_CLI_NPM_SPEC}"
RUN opencode --version

CMD ["sleep", "infinity"]
