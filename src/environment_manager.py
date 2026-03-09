"""
Docker 交互层（按 blueprint 1.1 节定义）
管理容器生命周期、命令执行、快照和回滚
"""

import docker
from docker.models.containers import Container
from docker.errors import DockerException, ImageNotFound

from .config import get_config
from .logger import get_logger
from .models import CommandResult

logger = get_logger("docker")

# 输出截断阈值（按 blueprint 4.1 节：头部 1000 + 尾部 1000）
MAX_HEAD_LENGTH = 1000
MAX_TAIL_LENGTH = 1000


def truncate_output(text: str) -> tuple[str, bool]:
    """截断过长输出（保留头部 1000 + 尾部 1000）"""
    if len(text) <= MAX_HEAD_LENGTH + MAX_TAIL_LENGTH:
        return text, False

    head = text[:MAX_HEAD_LENGTH]
    tail = text[-MAX_TAIL_LENGTH:]
    return f"{head}\n...[Truncated {len(text) - MAX_HEAD_LENGTH - MAX_TAIL_LENGTH} chars]...\n{tail}", True


class EnvironmentManager:
    """Docker 环境管理器（按 blueprint 1.1 节定义）"""

    def __init__(self):
        self._client = docker.from_env()
        self._container: Container | None = None
        self._config = get_config().docker
        self._env_vars: dict[str, str] = {}
        # 使用栈结构存储快照（按 blueprint 要求）
        self._history_snapshots: list[str] = []

    @property
    def container(self) -> Container:
        """获取当前容器，不存在时抛出异常"""
        if self._container is None:
            raise RuntimeError("容器未初始化，请先调用 create_container()")
        return self._container

    @property
    def container_id(self) -> str | None:
        """获取容器 ID"""
        return self._container.id if self._container else None

    @property
    def image_name(self) -> str:
        """获取基础镜像名"""
        return self._config.base_image

    @property
    def history_snapshots(self) -> list[str]:
        """获取快照历史栈"""
        return self._history_snapshots.copy()

    def create_container(self, repo_url: str) -> str:
        """创建并启动容器"""
        logger.info(f"创建容器，基础镜像: {self._config.base_image}")

        # 检查镜像是否存在
        try:
            self._client.images.get(self._config.base_image)
        except ImageNotFound:
            logger.info(f"拉取镜像: {self._config.base_image}")
            self._client.images.pull(self._config.base_image)

        # 创建容器
        self._container = self._client.containers.run(
            self._config.base_image,
            command="sleep infinity",
            detach=True,
            working_dir=self._config.work_dir,
            environment=self._env_vars.copy(),
        )

        logger.info(f"容器已创建: {self._container.id[:12]}")

        # 安装基础工具并克隆仓库
        self._setup_container(repo_url)

        return self._container.id

    def _setup_container(self, repo_url: str) -> None:
        """初始化容器环境"""
        logger.info("初始化容器环境...")

        root = "/"

        # 安装 git
        result = self.exec_run(
            "which git || (apt-get update && apt-get install -y git)",
            work_dir=root,
        )
        if not result.success:
            raise RuntimeError(f"安装 git 失败: {result.stderr}")

        # 创建工作目录
        self.exec_run(f"mkdir -p {self._config.work_dir}", work_dir=root)

        # 克隆仓库（配置 HTTP/1.1 避免 HTTP2 framing 问题，带重试）
        logger.info(f"克隆仓库: {repo_url}")
        self.exec_run("git config --global http.version HTTP/1.1", work_dir=root)
        clone_cmd = f"git clone --depth=1 {repo_url} {self._config.work_dir}/repo"
        result = None
        for attempt in range(3):
            result = self.exec_run(clone_cmd, work_dir=root)
            if result.success:
                break
            logger.warning(f"克隆仓库第 {attempt+1} 次失败，重试...")
        if not result.success:
            raise RuntimeError(f"克隆仓库失败: {result.stderr}")

        # 验证仓库目录存在
        result = self.exec_run("pwd")
        logger.info(f"仓库目录: {result.stdout.strip()}")

        # 拍初始快照：clone 完成、尚未执行任何操作的干净状态
        # 这样 ROLLBACK_ENV 连续多次调用最终能回到此初始状态
        self.create_checkpoint("initial_clone")

        logger.info("容器环境初始化完成")

    def exec_run(
        self,
        command: str,
        timeout: int | None = None,
        work_dir: str | None = None,
    ) -> CommandResult:
        """在容器中执行命令（按 blueprint 1.1 节定义）"""
        if timeout is None:
            timeout = self._config.timeout

        # 确定工作目录
        if work_dir is None:
            work_dir = f"{self._config.work_dir}/repo"

        # 构造执行命令
        if work_dir == "/":
            exec_command = command
        else:
            exec_command = f"cd {work_dir} && {command}"

        logger.debug(f"执行命令: {command}")

        # 通过 Docker 原生 environment 参数注入环境变量
        # 相比 shell prefix（KEY=VAL cmd），此方式对整个 bash 进程及其所有子命令均可见，
        # 不受 && 链分隔符影响，且 echo $KEY 可正确返回值
        exit_code, output = self.container.exec_run(
            cmd=["timeout", str(timeout), "bash", "-c", exec_command],
            demux=True,
            environment=self._env_vars if self._env_vars else None,
        )

        stdout_raw = (output[0] or b"").decode("utf-8", errors="replace")
        stderr_raw = (output[1] or b"").decode("utf-8", errors="replace")

        # 截断过长输出（按 blueprint 4.1 节：头部 1000 + 尾部 1000）
        stdout, stdout_truncated = truncate_output(stdout_raw)
        stderr, stderr_truncated = truncate_output(stderr_raw)
        truncated = stdout_truncated or stderr_truncated

        result = CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
        )

        logger.debug(f"命令结果: exit_code={exit_code}, truncated={truncated}")
        if not result.success:
            logger.warning(f"命令执行失败: {command}, exit_code={exit_code}")

        return result

    def read_file(self, path: str) -> str:
        """读取容器内文件内容（按 blueprint 1.1 节定义）"""
        result = self.exec_run(f"cat {path}")
        if not result.success:
            raise RuntimeError(f"读取文件失败 {path}: {result.stderr}")
        return result.stdout

    def write_file(self, path: str, content: str) -> bool:
        """向容器内写入文件（按 blueprint 1.1 节定义）"""
        # 转义特殊字符
        escaped_content = content.replace("\\", "\\\\").replace("'", "'\\''")
        result = self.exec_run(f"echo '{escaped_content}' > {path}")
        return result.success

    def set_env(self, key: str, value: str) -> None:
        """设置环境变量。变量存入 _env_vars，后续每次 exec_run 均通过
        Docker 原生 environment 参数注入，对整个 bash 进程及所有子命令可见。"""
        self._env_vars[key] = value
        logger.info(f"设置环境变量: {key}={value}")

    def get_env(self, key: str) -> str | None:
        """获取环境变量"""
        return self._env_vars.get(key)

    def create_checkpoint(self, tag: str) -> str:
        """创建快照（按 blueprint 1.1 节定义）"""
        logger.info(f"创建快照: {tag}")

        # 使用 docker commit 创建镜像
        image = self.container.commit(repository="setup_agent_checkpoint", tag=tag)

        # 压入快照栈
        self._history_snapshots.append(tag)
        logger.info(f"快照已创建: {image.id[:12]}，栈深度: {len(self._history_snapshots)}")

        return image.id

    def rollback_to_checkpoint(self) -> bool:
        """回滚到最近的快照（按 blueprint 1.1 节定义：弹出栈顶）"""
        if not self._history_snapshots:
            logger.warning("没有可用的快照，无法回滚")
            return False

        # 弹出最近的快照
        tag = self._history_snapshots.pop()
        logger.info(f"回滚到快照: {tag}")

        # 停止并删除当前容器
        old_container_id = self.container.id
        self.container.stop()
        self.container.remove()
        logger.debug(f"已删除旧容器: {old_container_id[:12]}")

        # 从快照镜像创建新容器
        image_name = f"setup_agent_checkpoint:{tag}"
        self._container = self._client.containers.run(
            image_name,
            command="sleep infinity",
            detach=True,
            working_dir=self._config.work_dir,
            environment=self._env_vars.copy(),
        )

        logger.info(f"已回滚到快照，新容器: {self._container.id[:12]}")
        return True

    def list_checkpoints(self) -> list[str]:
        """列出所有快照"""
        return self._history_snapshots.copy()

    def cleanup_snapshots(self) -> None:
        """仅删除快照镜像（释放磁盘），保留容器"""
        logger.info("清理快照镜像...")
        for tag in self._history_snapshots:
            try:
                image_name = f"setup_agent_checkpoint:{tag}"
                self._client.images.remove(image_name, force=True)
                logger.debug(f"已删除快照镜像: {tag}")
            except DockerException as e:
                logger.warning(f"删除快照镜像失败 {tag}: {e}")
        self._history_snapshots.clear()
        logger.info("快照镜像清理完成")

    def destroy(self) -> None:
        """停止并删除容器"""
        logger.info("销毁容器...")
        if self._container:
            try:
                self._container.stop()
                self._container.remove()
                logger.debug(f"已删除容器: {self._container.id[:12]}")
            except DockerException as e:
                logger.warning(f"删除容器失败: {e}")
            self._container = None
        logger.info("容器已销毁")

    def cleanup(self) -> None:
        """清理所有资源：停止容器 + 删除快照镜像"""
        logger.info("清理全部资源...")
        self.destroy()
        self.cleanup_snapshots()
        logger.info("资源清理完成")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
