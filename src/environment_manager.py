"""
Docker 交互层（按 blueprint 1.1 节定义）
管理容器生命周期、命令执行、快照和回滚

本模块是 Setup Agent 与 Docker 容器交互的核心层，提供：
- 容器创建与销毁（create_container / destroy）
- 外部容器连接（from_container / from_dockerfile）
- 容器内命令执行（exec_run）
- 文件读写（read_file / write_file）
- 环境变量管理（set_env / get_env）
- 快照与回滚机制（create_checkpoint / rollback_to_checkpoint）

快照机制说明：
  使用 docker commit 将当前容器状态保存为镜像，回滚时从镜像重建容器。
  快照使用栈结构（LIFO），rollback 弹出栈顶快照。
"""

import docker  # Docker SDK for Python，用于操作 Docker 容器
from docker.models.containers import Container  # Docker 容器模型类
from docker.errors import DockerException, ImageNotFound  # Docker 异常类

from .config import get_config, DockerConfig  # 全局配置和 Docker 配置数据类
from .logger import get_logger  # 统一日志系统
from .models import CommandResult  # 命令执行结果数据类

logger = get_logger("docker")  # 创建 docker 模块专用日志记录器

# 输出截断阈值（按 blueprint 4.1 节：头部 1000 字符 + 尾部 1000 字符）
# 容器命令输出可能非常长（如编译日志），截断后保留头尾关键信息
MAX_HEAD_LENGTH = 1000
MAX_TAIL_LENGTH = 1000


def truncate_output(text: str) -> tuple[str, bool]:
    """截断过长输出（保留头部 1000 + 尾部 1000 字符）

    Args:
        text: 原始输出文本

    Returns:
        (截断后的文本, 是否发生了截断)
    """
    # 如果文本长度在阈值内，原样返回
    if len(text) <= MAX_HEAD_LENGTH + MAX_TAIL_LENGTH:
        return text, False

    head = text[:MAX_HEAD_LENGTH]  # 保留头部 1000 字符
    tail = text[-MAX_TAIL_LENGTH:]  # 保留尾部 1000 字符
    # 中间用截断提示连接，标明被省略的字符数
    return f"{head}\n...[Truncated {len(text) - MAX_HEAD_LENGTH - MAX_TAIL_LENGTH} chars]...\n{tail}", True


class EnvironmentManager:
    """Docker 环境管理器（按 blueprint 1.1 节定义）

    负责 Agent 与 Docker 容器的所有交互。支持三种初始化方式：
    1. create_container()：从基础镜像创建新容器并克隆仓库
    2. from_container()：连接已有容器（如外部工具产出的容器）
    3. from_dockerfile()：从 Dockerfile 构建镜像并启动容器
    """

    def __init__(self):
        self._client = docker.from_env()  # 从环境变量连接 Docker daemon
        self._container: Container | None = None  # 当前管理的容器实例
        self._config = get_config().docker  # 获取 Docker 相关配置（镜像、工作目录、超时等）
        self._env_vars: dict[str, str] = {}  # 存储通过 set_env() 设置的环境变量
        self._repo_dir_override: str | None = None  # attach() 时指定的工作目录覆盖
        # 快照栈：使用列表模拟栈结构，存储快照标签（tag）
        # rollback 时弹出栈顶，可连续回滚到更早的状态
        self._history_snapshots: list[str] = []
        # _repo_subdir 标志：控制 exec_run 的默认工作目录
        # True = create_container() 场景：仓库克隆到 {work_dir}/repo，exec_run 默认在 {work_dir}/repo
        # False = from_dockerfile()/from_container() 场景：仓库直接在 work_dir 下，不追加 /repo
        self._repo_subdir = True
        # 启动时清理上次进程崩溃遗留的 checkpoint 镜像
        self._cleanup_stale_checkpoints()

    def _cleanup_stale_checkpoints(self) -> None:
        """清理上次进程异常退出遗留的 setup_agent_checkpoint 镜像"""
        try:
            stale = self._client.images.list(name="setup_agent_checkpoint")
            if not stale:
                return
            logger.info(f"发现 {len(stale)} 个遗留 checkpoint 镜像，清理中...")
            for img in stale:
                try:
                    self._client.images.remove(img.id, force=True)
                    logger.debug(f"已清理遗留镜像: {img.tags}")
                except DockerException as e:
                    logger.warning(f"清理遗留镜像失败 {img.tags}: {e}")
            logger.info("遗留 checkpoint 镜像清理完成")
        except DockerException as e:
            logger.warning(f"查询遗留 checkpoint 镜像失败: {e}")

    @classmethod
    def _create_with_work_dir(cls, work_dir: str) -> "EnvironmentManager":
        """内部工厂方法：创建实例并覆盖工作目录

        用于 from_container() 和 from_dockerfile() 场景，
        这些场景的工作目录可能不同于默认配置。

        使用 __new__ 跳过 __init__，然后手动初始化所有属性，
        并用自定义 work_dir 构造新的 DockerConfig（因为 DockerConfig 是 frozen dataclass，不能直接修改）。

        Args:
            work_dir: 自定义工作目录路径（如 /repo、/workspace）
        """
        instance = cls.__new__(cls)  # 跳过 __init__，直接创建实例
        instance._client = docker.from_env()  # 连接 Docker daemon
        instance._container = None
        instance._env_vars = {}
        instance._repo_dir_override = None
        instance._history_snapshots = []
        instance._repo_subdir = False  # 外部容器/Dockerfile 场景不追加 /repo 子目录
        # 用原始配置的 base_image 和 timeout，但替换 work_dir
        orig = get_config().docker
        instance._config = DockerConfig(
            base_image=orig.base_image,
            work_dir=work_dir,  # 使用自定义工作目录
            timeout=orig.timeout,
        )
        return instance

    @classmethod
    def from_container(cls, container_id: str, work_dir: str = "/workspace") -> "EnvironmentManager":
        """连接到已有的 Docker 容器（不创建新容器）

        用于 Verifier 独立验证外部工具（OpenHands、Repo2Run 等）产出的容器，
        与 Setup Agent 的容器创建流程完全解耦。

        Args:
            container_id: 目标容器 ID 或名称（支持短 ID）
            work_dir: 容器内项目根目录（默认 /workspace，Repo2Run 用 /repo）
        """
        instance = cls._create_with_work_dir(work_dir)  # 创建实例，设置工作目录
        # 通过 container_id 获取已存在的容器实例
        instance._container = instance._client.containers.get(container_id)
        logger.info(f"已连接到已有容器: {container_id[:12]}，工作目录: {work_dir}")
        return instance

    @classmethod
    def from_dockerfile(cls, dockerfile_dir: str, work_dir: str = "/repo") -> "EnvironmentManager":
        """从 Dockerfile 构建镜像、启动容器并连接

        用于验证 Repo2Run 等工具产出的 Dockerfile 是否能正确搭建环境。
        流程：构建镜像 → 启动容器（sleep infinity 保持运行）→ 返回连接好的实例

        Args:
            dockerfile_dir: 包含 Dockerfile 的目录路径
            work_dir: 容器内项目根目录（Repo2Run 默认 /repo）
        """
        instance = cls._create_with_work_dir(work_dir)  # 创建实例，设置工作目录
        logger.info(f"从 Dockerfile 构建镜像: {dockerfile_dir}")
        # 调用 Docker SDK 构建镜像，rm=True 表示构建完删除中间层容器
        # network_mode="host" 使构建过程能访问宿主机网络（下载依赖用）
        image, build_logs = instance._client.images.build(
            path=dockerfile_dir,
            rm=True,
            network_mode="host",
        )
        # 逐行打印构建日志（仅 debug 级别）
        for chunk in build_logs:
            if "stream" in chunk:
                line = chunk["stream"].strip()
                if line:
                    logger.debug(f"[build] {line}")
        logger.info(f"镜像构建完成: {image.id[:12]}")

        # 从构建好的镜像启动容器
        # sleep infinity 使容器保持运行状态，等待 exec_run 执行命令
        instance._container = instance._client.containers.run(
            image.id,
            command="sleep infinity",
            detach=True,  # 后台运行
            network_mode="host",  # 使用宿主机网络
        )
        logger.info(f"容器已启动: {instance._container.id[:12]}，工作目录: {work_dir}")
        return instance

    @property
    def container(self) -> Container:
        """获取当前容器实例

        如果容器未初始化（未调用 create_container / from_container / from_dockerfile），
        抛出 RuntimeError。
        """
        if self._container is None:
            raise RuntimeError("容器未初始化，请先调用 create_container()")
        return self._container

    @property
    def container_id(self) -> str | None:
        """获取容器 ID，容器不存在时返回 None"""
        return self._container.id if self._container else None

    @property
    def image_name(self) -> str:
        """获取配置中指定的基础镜像名"""
        return self._config.base_image

    @property
    def history_snapshots(self) -> list[str]:
        """获取快照历史栈的副本（防止外部修改内部状态）"""
        return self._history_snapshots.copy()

    def attach(self, container_id: str, repo_dir: str | None = None) -> None:
        """接管已有容器（不创建新容器、不克隆仓库）。
        repo_dir: 仓库在容器内的完整路径，若指定则 exec_run 默认 cd 到此路径。
        """
        self._container = self._client.containers.get(container_id)
        if repo_dir:
            self._repo_dir_override = repo_dir
        logger.info(f"已接管容器: {self._container.id[:12]}, repo_dir={repo_dir or 'default'}")

    def create_container(self, repo_url: str) -> str:
        """创建并启动容器，克隆目标仓库

        这是 Setup Agent 主流程使用的初始化方式。
        流程：拉取基础镜像 → 创建容器 → 安装 git → 克隆仓库 → 拍初始快照

        Args:
            repo_url: 要克隆的 Git 仓库 URL

        Returns:
            容器 ID
        """
        logger.info(f"创建容器，基础镜像: {self._config.base_image}")

        # 检查基础镜像是否已存在于本地，不存在则拉取
        try:
            self._client.images.get(self._config.base_image)
        except ImageNotFound:
            logger.info(f"拉取镜像: {self._config.base_image}")
            self._client.images.pull(self._config.base_image)

        # 创建并启动容器
        # sleep infinity 使容器保持运行，detach=True 后台运行
        # network_mode="host" 使用宿主机网络栈，避免 Docker 默认桥接网络的 DNS 解析问题
        self._container = self._client.containers.run(
            self._config.base_image,
            command="sleep infinity",
            detach=True,
            working_dir=self._config.work_dir,  # 设置容器默认工作目录
            environment=self._env_vars.copy(),  # 注入环境变量
            network_mode="host",  # 使用宿主机网络
        )

        logger.info(f"容器已创建: {self._container.id[:12]}")

        # 安装基础工具（git）并克隆仓库
        self._setup_container(repo_url)

        return self._container.id

    def _setup_container(self, repo_url: str) -> None:
        """初始化容器环境：安装 git、克隆仓库、拍初始快照

        Args:
            repo_url: 要克隆的 Git 仓库 URL
        """
        logger.info("初始化容器环境...")

        root = "/"  # 在根目录执行初始化命令，避免工作目录不存在的问题

        # 检查 git 是否已安装，未安装则通过 apt-get 安装
        result = self.exec_run(
            "which git || (apt-get update && apt-get install -y git)",
            work_dir=root,
        )
        if not result.success:
            raise RuntimeError(f"安装 git 失败: {result.stderr}")

        # 创建工作目录（如果不存在）
        self.exec_run(f"mkdir -p {self._config.work_dir}", work_dir=root)

        # 克隆仓库（带重试机制）
        logger.info(f"克隆仓库: {repo_url}")
        # 配置 HTTP/1.1 避免某些代理环境下的 HTTP2 framing 问题
        self.exec_run("git config --global http.version HTTP/1.1", work_dir=root)
        # 仓库克隆到 {work_dir}/repo 子目录
        clone_cmd = f"git clone --depth=1 {repo_url} {self._config.work_dir}/repo"
        result = None
        for attempt in range(3):  # 最多重试 3 次（网络不稳定时）
            result = self.exec_run(clone_cmd, work_dir=root)
            if result.success:
                break
            logger.warning(f"克隆仓库第 {attempt+1} 次失败，重试...")
        if not result.success:
            raise RuntimeError(f"克隆仓库失败: {result.stderr}")

        # 验证仓库目录存在并打印路径
        result = self.exec_run("pwd")
        logger.info(f"仓库目录: {result.stdout.strip()}")

        # 拍初始快照：记录 clone 完成后的干净状态
        # 这样 ROLLBACK_ENV 连续多次调用最终能回到此初始状态
        self.create_checkpoint("initial_clone")

        logger.info("容器环境初始化完成")

    def exec_run(
        self,
        command: str,
        timeout: int | None = None,
        work_dir: str | None = None,
    ) -> CommandResult:
        """在容器中执行命令（按 blueprint 1.1 节定义）

        核心执行逻辑：
        1. 确定工作目录（默认根据 _repo_subdir 标志决定）
        2. 构造 bash -c 命令，用 timeout 包裹防止超时
        3. 通过 Docker SDK 的 exec_run 执行
        4. 解码并截断输出
        5. 返回 CommandResult

        Args:
            command: 要执行的 shell 命令字符串
            timeout: 超时秒数（默认使用配置值）
            work_dir: 指定工作目录（默认根据 _repo_subdir 决定）

        Returns:
            CommandResult 包含 exit_code、stdout、stderr、truncated 等信息
        """
        if timeout is None:
            timeout = self._config.timeout  # 使用配置中的默认超时

        # 确定工作目录
        if work_dir is None:
            if self._repo_dir_override:
                work_dir = self._repo_dir_override
            elif self._repo_subdir:
                # create_container() 场景：仓库在 {work_dir}/repo
                work_dir = f"{self._config.work_dir}/repo"
            else:
                # from_dockerfile/from_container() 场景：仓库直接在 work_dir
                work_dir = self._config.work_dir

        # 构造执行命令：先 cd 到工作目录，再执行命令
        if work_dir == "/":
            exec_command = command  # 根目录不需要 cd
        else:
            exec_command = f"cd {work_dir} && {command}"

        logger.debug(f"执行命令: {command}")

        # 通过 Docker SDK 执行命令
        # timeout 命令包裹防止命令运行超时
        # demux=True 使 stdout 和 stderr 分开返回
        # environment 参数注入 set_env() 设置的环境变量
        exit_code, output = self.container.exec_run(
            cmd=["timeout", str(timeout), "bash", "-c", exec_command],
            demux=True,  # 分离 stdout 和 stderr
            environment=self._env_vars if self._env_vars else None,
        )

        # 解码二进制输出为 UTF-8 字符串（errors="replace" 替换无法解码的字节）
        stdout_raw = (output[0] or b"").decode("utf-8", errors="replace")
        stderr_raw = (output[1] or b"").decode("utf-8", errors="replace")

        # 截断过长输出（保留头部 1000 + 尾部 1000 字符）
        stdout, stdout_truncated = truncate_output(stdout_raw)
        stderr, stderr_truncated = truncate_output(stderr_raw)
        truncated = stdout_truncated or stderr_truncated  # 任一被截断则标记

        # 构造命令结果对象
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
        """读取容器内文件内容（按 blueprint 1.1 节定义）

        Args:
            path: 容器内的文件绝对路径

        Returns:
            文件内容字符串

        Raises:
            RuntimeError: 文件不存在或读取失败
        """
        result = self.exec_run(f"cat {path}")
        if not result.success:
            raise RuntimeError(f"读取文件失败 {path}: {result.stderr}")
        return result.stdout

    def write_file(self, path: str, content: str) -> bool:
        """向容器内写入文件（按 blueprint 1.1 节定义）

        Args:
            path: 容器内的文件绝对路径
            content: 要写入的内容

        Returns:
            是否写入成功
        """
        # 转义反斜杠和单引号，防止 shell 注入
        escaped_content = content.replace("\\", "\\\\").replace("'", "'\\''")
        result = self.exec_run(f"echo '{escaped_content}' > {path}")
        return result.success

    def set_env(self, key: str, value: str) -> None:
        """设置环境变量

        变量存入 _env_vars 字典，后续每次 exec_run 均通过
        Docker 原生 environment 参数注入，对整个 bash 进程及所有子命令可见。
        相比 shell prefix（KEY=VAL cmd），此方式不受 && 链分隔符影响。

        注意：Docker environment 不会展开 shell 变量引用（如 $PATH），
        因此需要在设置前先解析容器内的实际值进行替换。
        """
        # 展开 value 中引用的 shell 变量（如 $PATH、${HOME}）
        resolved_value = self._resolve_env_refs(value)
        self._env_vars[key] = resolved_value
        logger.info(f"设置环境变量: {key}={resolved_value}")

    def _resolve_env_refs(self, value: str) -> str:
        """展开 value 中的 $VAR / ${VAR} 引用为容器内实际值"""
        import re
        # 匹配 $VAR 或 ${VAR}
        pattern = re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?')
        refs = pattern.findall(value)
        if not refs:
            return value

        for var_name in set(refs):
            # 优先从已设置的环境变量中取，否则从容器内读取
            actual = self._env_vars.get(var_name)
            if actual is None:
                result = self.container.exec_run(
                    cmd=["bash", "-c", f"echo -n ${var_name}"],
                    demux=True,
                )
                actual = (result[1][0] or b"").decode("utf-8", errors="replace").strip()
            # 替换 ${VAR} 和 $VAR 两种形式
            value = value.replace(f"${{{var_name}}}", actual)
            value = value.replace(f"${var_name}", actual)
        return value

    def get_env(self, key: str) -> str | None:
        """获取已设置的环境变量值，不存在返回 None"""
        return self._env_vars.get(key)


    def get_env_snapshot(self) -> str:
        """获取容器环境快照，供检察官/法官在调查前快速感知环境。"""
        cmd = (
            "echo '--- Python 解释器 ---' && "
            "which python3 python 2>/dev/null; python3 --version 2>/dev/null; "
            "echo '--- 虚拟环境 ---' && "
            "ls -d */venv */env */.venv venv .venv env 2>/dev/null || echo '(未发现)'; "
            "echo 'VIRTUAL_ENV='$VIRTUAL_ENV; "
            "echo 'CONDA_PREFIX='$CONDA_PREFIX; "
            "echo '--- PATH ---' && echo \"$PATH\"; "
            "echo '--- 项目标记文件 ---' && "
            "ls pyproject.toml setup.py setup.cfg requirements.txt "
            "CMakeLists.txt Makefile configure meson.build "
            "package.json Cargo.toml go.mod pom.xml build.gradle 2>/dev/null || echo '(无)'; "
            "echo '--- 工作目录 ---' && pwd && ls | head -30"
        )
        result = self.exec_run(cmd, timeout=15)
        if result.success:
            return result.stdout[:2000]
        return f"(快照获取失败: exit_code={result.exit_code})"

    # ========================================================================
    # 快照与回滚机制
    # 使用 docker commit 将容器当前状态保存为镜像，回滚时从镜像重建容器
    # ========================================================================

    def create_checkpoint(self, tag: str) -> str:
        """创建快照（按 blueprint 1.1 节定义）

        使用 docker commit 将当前容器状态保存为镜像，tag 压入快照栈。

        Args:
            tag: 快照标签（如 "initial_clone"、"step_3_pre_xpu"）

        Returns:
            快照镜像 ID
        """
        logger.info(f"创建快照: {tag}")

        # 使用 docker commit 将容器当前状态保存为镜像
        # repository 固定为 "setup_agent_checkpoint"，tag 为快照标签
        image = self.container.commit(repository="setup_agent_checkpoint", tag=tag)

        # 将快照标签压入栈顶
        self._history_snapshots.append(tag)
        logger.info(f"快照已创建: {image.id[:12]}，栈深度: {len(self._history_snapshots)}")

        return image.id

    def rollback_to_checkpoint(self) -> bool:
        """回滚到最近的快照（按 blueprint 1.1 节定义：弹出栈顶）

        流程：
        1. 弹出栈顶快照标签
        2. 停止并删除当前容器
        3. 从快照镜像创建新容器
        4. 新容器继承原有的环境变量

        Returns:
            是否回滚成功
        """
        if not self._history_snapshots:
            logger.warning("没有可用的快照，无法回滚")
            return False

        # 弹出最近的快照标签
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
            command="sleep infinity",  # 保持运行
            detach=True,
            working_dir=self._config.work_dir,
            environment=self._env_vars.copy(),  # 继承环境变量
        )

        logger.info(f"已回滚到快照，新容器: {self._container.id[:12]}")
        return True

    def list_checkpoints(self) -> list[str]:
        """列出所有快照标签（返回副本，防止外部修改）"""
        return self._history_snapshots.copy()

    def cleanup_snapshots(self) -> None:
        """仅删除快照镜像（释放磁盘空间），保留容器运行"""
        logger.info("清理快照镜像...")
        for tag in self._history_snapshots:
            try:
                image_name = f"setup_agent_checkpoint:{tag}"
                self._client.images.remove(image_name, force=True)  # 强制删除
                logger.debug(f"已删除快照镜像: {tag}")
            except DockerException as e:
                logger.warning(f"删除快照镜像失败 {tag}: {e}")
        self._history_snapshots.clear()  # 清空快照栈
        logger.info("快照镜像清理完成")

    # ========================================================================
    # 资源清理
    # ========================================================================

    def destroy(self) -> None:
        """停止并删除容器（不删除快照镜像）"""
        logger.info("销毁容器...")
        if self._container:
            try:
                self._container.stop()  # 先停止容器
                self._container.remove()  # 再删除容器
                logger.debug(f"已删除容器: {self._container.id[:12]}")
            except DockerException as e:
                logger.warning(f"删除容器失败: {e}")
            self._container = None  # 清空引用
        logger.info("容器已销毁")

    def cleanup(self) -> None:
        """清理所有资源：停止容器 + 删除快照镜像

        通常在 Agent 任务完成或异常退出时调用。
        """
        logger.info("清理全部资源...")
        self.destroy()  # 销毁容器
        self.cleanup_snapshots()  # 清理快照镜像
        logger.info("资源清理完成")

    def get_env_snapshot(self) -> str:
        """获取容器环境快照，供检察官/法官了解当前环境状态。"""
        cmd = (
            "echo '--- Python 解释器 ---' && "
            "which python3 python 2>/dev/null; python3 --version 2>/dev/null; "
            "echo '--- 虚拟环境 ---' && "
            "ls -d */venv */env */.venv venv .venv env 2>/dev/null || echo '(未发现)'; "
            "echo \"VIRTUAL_ENV=$VIRTUAL_ENV\"; "
            "echo \"CONDA_PREFIX=$CONDA_PREFIX\"; "
            "echo '--- PATH ---' && echo \"$PATH\"; "
            "echo '--- 项目标记文件 ---' && "
            "ls pyproject.toml setup.py setup.cfg requirements.txt "
            "CMakeLists.txt Makefile configure meson.build "
            "package.json Cargo.toml go.mod pom.xml build.gradle 2>/dev/null || echo '(无)'; "
            "echo '--- 工作目录 ---' && pwd && ls | head -30"
        )
        result = self.exec_run(cmd, timeout=15)
        if result.success:
            return result.stdout[:2000]
        return f"(快照获取失败: exit_code={result.exit_code})"

    def __enter__(self):
        """支持 with 语句（上下文管理器入口）"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句（上下文管理器出口，自动清理资源）"""
        self.cleanup()
        return False  # 不吞异常，让异常继续传播
