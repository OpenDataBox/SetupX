"""
统一日志系统
- 文件记录完整日志（DEBUG 级别）
- 屏幕仅显示 ERROR 级别
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import get_config


class LoggerSetup:
    """日志系统配置"""

    _initialized = False

    @classmethod
    def setup(cls) -> logging.Logger:
        """初始化日志系统，返回根 logger"""
        if cls._initialized:
            return logging.getLogger("setup_agent")

        config = get_config()
        env_log_file = os.getenv("LOG_FILE")
        env_log_dir = os.getenv("LOG_DIR")
        env_prefix = os.getenv("LOG_FILE_PREFIX", "").strip()

        log_dir = Path(env_log_dir) if env_log_dir else config.log_dir

        # 创建日志文件名
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if env_log_file:
            log_file = Path(env_log_file)
        else:
            prefix = f"{env_prefix}_" if env_prefix else ""
            log_file = log_dir / f"{prefix}{timestamp}.log"

        # 创建根 logger
        logger = logging.getLogger("setup_agent")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # 文件处理器：记录所有级别
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 控制台处理器：仅 ERROR 级别
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter(
            "%(levelname)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        cls._initialized = True
        logger.info(f"日志系统初始化完成，日志文件: {log_file}")

        return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 logger 实例

    Args:
        name: 子模块名称，如 "agent", "docker"。为 None 时返回根 logger

    Returns:
        配置好的 logger 实例
    """
    # 确保日志系统已初始化
    LoggerSetup.setup()

    if name:
        return logging.getLogger(f"setup_agent.{name}")
    return logging.getLogger("setup_agent")
