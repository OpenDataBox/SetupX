"""
测试 Verifier 通过 from_dockerfile() 验证 Repo2Run 产出的 Dockerfile
"""
import sys
import shutil
import tempfile
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment_manager import EnvironmentManager
from src.verifier_agent import VerifierAgent


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_verifier_dockerfile.py <repo_output_dir>")
        print("Example: python test_verifier_dockerfile.py /tmp/repo2run_output/amperser/proselint")
        sys.exit(1)

    repo_dir = Path(sys.argv[1])
    dockerfile_path = repo_dir / "Dockerfile"
    if not dockerfile_path.exists():
        print(f"错误: {dockerfile_path} 不存在")
        sys.exit(1)

    # 创建临时构建目录，复制 Dockerfile 和必要文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shutil.copy(dockerfile_path, tmp_path / "Dockerfile")

        # Repo2Run 的 Dockerfile 需要 code_edit.py
        code_edit_src = PROJECT_ROOT.parent / "Repo2Run" / "build_agent" / "tools" / "code_edit.py"
        if code_edit_src.exists():
            shutil.copy(code_edit_src, tmp_path / "code_edit.py")
        else:
            # 创建空的 code_edit.py 占位（Dockerfile COPY 需要）
            (tmp_path / "code_edit.py").touch()

        print(f"构建目录: {tmp_path}")
        print(f"Dockerfile: {dockerfile_path}")
        print("=" * 50)

        # 从 Dockerfile 构建并启动容器
        print("阶段1: 构建镜像 + 启动容器...")
        env = EnvironmentManager.from_dockerfile(str(tmp_path), work_dir="/repo")

        # 简单检查容器是否能执行命令
        result = env.exec_run("ls /repo")
        print(f"容器内 /repo 内容: {result.stdout[:200]}")

        # 启动 Verifier
        print("=" * 50)
        print("阶段2: 运行 Verifier...")
        verifier = VerifierAgent(env)
        verify_result = verifier.verify()

        # 输出结果
        print("=" * 50)
        print(f"验证结果: {'通过' if verify_result.success else '失败'}")
        print(f"测试框架: {verify_result.test_framework}")
        print(f"收集测试数: {verify_result.collect_count}")
        print(f"退出码: {verify_result.exit_code}")
        print(f"输出: {verify_result.stdout[:500]}")
        if verify_result.stderr:
            print(f"错误: {verify_result.stderr[:500]}")

        # 清理
        env.destroy()
        print("容器已销毁")


if __name__ == "__main__":
    main()
