"""
统一配置注入模块
从 .env 文件加载所有配置项
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 加载 .env 文件
load_dotenv(PROJECT_ROOT / ".env", override=True)


@dataclass(frozen=True)
class ARKConfig:
    """字节 ARK API 配置"""
    api_key: str
    base_url: str
    deployment: str


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI 兼容 API 配置（支持 Kimi/DeepSeek/Qwen 等）"""
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class DockerConfig:
    """Docker 配置"""
    base_image: str
    work_dir: str
    timeout: int


@dataclass(frozen=True)
class XPUConfig:
    """XPU 服务配置"""
    base_url: str
    enabled: bool
    disabled: bool
    db_dns: str | None      # PostgreSQL 向量数据库连接串（env: dns 或 XPU_DB_DNS）
    vector_enabled: bool    # 启用 VectorXPUClient（env: XPU_VECTOR_ENABLED）


@dataclass(frozen=True)
class Config:
    """全局配置"""
    ark: ARKConfig
    openai: OpenAIConfig | None
    docker: DockerConfig
    xpu: XPUConfig
    llm_provider: str  # "ark" 或 "openai"
    log_dir: Path


def _get_env(key: str, default: str | None = None) -> str:
    """获取环境变量，不存在时抛出异常（除非有默认值）"""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"缺少必需的环境变量: {key}")
    return value


def _get_env_bool(key: str, default: bool = False) -> bool:
    """获取布尔类型环境变量"""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """获取整数类型环境变量"""
    return int(os.getenv(key, str(default)))


def load_config() -> Config:
    """加载并返回全局配置"""

    # ARK 配置（必需）
    ark = ARKConfig(
        api_key=_get_env("ARK_API_KEY"),
        base_url=_get_env("ARK_BASE_URL"),
        deployment=_get_env("ARK_DEPLOYMENT"),
    )

    # OpenAI 兼容配置（可选）
    openai_key = os.getenv("OPENAI_API_KEY")
    openai = None
    if openai_key:
        openai = OpenAIConfig(
            api_key=openai_key,
            base_url=_get_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_get_env("OPENAI_MODEL", "gpt-4o"),
        )

    # Docker 配置
    docker = DockerConfig(
        base_image=_get_env("DOCKER_BASE_IMAGE", "ubuntu:22.04"),
        work_dir=_get_env("DOCKER_WORK_DIR", "/workspace"),
        timeout=_get_env_int("DOCKER_TIMEOUT", 300),
    )

    # XPU 配置
    xpu = XPUConfig(
        base_url=_get_env("XPU_BASE_URL", "http://localhost:8080"),
        enabled=_get_env_bool("XPU_ENABLED", False),
        disabled=_get_env_bool("XPU_DISABLED", False),
        db_dns=os.getenv("dns") or os.getenv("XPU_DB_DNS"),
        vector_enabled=_get_env_bool("XPU_VECTOR_ENABLED", False),
    )

    # LLM 提供商选择
    llm_provider = _get_env("LLM_PROVIDER", "ark")
    if llm_provider not in ("ark", "openai"):
        raise ValueError(f"不支持的 LLM 提供商: {llm_provider}，仅支持 ark 或 openai")

    # 日志目录
    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(exist_ok=True)

    return Config(
        ark=ark,
        openai=openai,
        docker=docker,
        xpu=xpu,
        llm_provider=llm_provider,
        log_dir=log_dir,
    )


# 全局配置实例（延迟加载）
_config: Config | None = None


def get_config() -> Config:
    """获取全局配置实例（单例模式）"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
