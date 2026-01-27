#!/usr/bin/env python3
"""
Qwen3-TTS Skill 依赖安装脚本
适配 CUDA 11.6 环境

依赖安装到用户级别（--user），无需每次重新安装
"""
import subprocess
import sys
import os
from pathlib import Path


SKILL_DIR = Path(__file__).parent


def run_cmd(cmd: list, desc: str):
    """运行命令并打印状态"""
    print(f"[*] {desc}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] 错误: {result.stderr[:200] if len(result.stderr) > 200 else result.stderr}")
        return False
    print(f"[+] {desc} 完成")
    return True


def pip_install(packages: list, desc: str, extra_args: list = None):
    """使用 pip 安装包，自动处理 externally-managed-environment"""
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append("--break-system-packages")  # Python 3.12+ 需要
    cmd.extend(packages)
    return run_cmd(cmd, desc)


def check_installed(package: str) -> bool:
    """检查包是否已安装"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def setup_shell_wrapper():
    """设置 shell wrapper 到 PATH"""
    wrapper_src = SKILL_DIR / "qwen-tts"
    wrapper_dst = Path("/usr/local/bin/qwen-tts")

    # 设置执行权限
    os.chmod(wrapper_src, 0o755)

    # 创建符号链接到 PATH
    if wrapper_dst.exists() or wrapper_dst.is_symlink():
        wrapper_dst.unlink()

    try:
        wrapper_dst.symlink_to(wrapper_src)
        print(f"[+] 已创建命令链接: qwen-tts -> {wrapper_src}")
        return True
    except PermissionError:
        print(f"[!] 无法创建 /usr/local/bin/qwen-tts (需要 sudo)")
        print(f"    可手动执行: sudo ln -sf {wrapper_src} /usr/local/bin/qwen-tts")
        return False


def main():
    print("=" * 50)
    print("Qwen3-TTS Skill 依赖安装")
    print("=" * 50)

    # 设置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[*] HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
    print(f"[*] Skill 目录: {SKILL_DIR}")

    # 检查 CUDA 可用性
    print("\n[1/5] 检查 CUDA 环境...")
    pytorch_installed = False
    try:
        import torch
        pytorch_installed = True
        if torch.cuda.is_available():
            print(f"[+] CUDA 可用: {torch.cuda.get_device_name(0)}")
            print(f"[+] PyTorch 版本: {torch.__version__}")
        else:
            print("[!] CUDA 不可用，将尝试安装 GPU 版本 PyTorch")
            pytorch_installed = False
    except ImportError:
        print("[!] PyTorch 未安装")

    # 安装 PyTorch GPU 版本 (CUDA 11.6 兼容)
    print("\n[2/5] 检查/安装 PyTorch (CUDA 11.8 兼容 11.6)...")
    if pytorch_installed:
        print("[+] PyTorch 已安装，跳过")
    else:
        # CUDA 11.8 向下兼容 11.6
        if not pip_install(
            ["torch", "torchaudio"],
            "PyTorch GPU 版本安装",
            ["--index-url", "https://download.pytorch.org/whl/cu118"]
        ):
            print("[!] PyTorch 安装失败，请手动安装")

    # 安装 qwen-tts 及依赖
    print("\n[3/5] 检查/安装 qwen-tts 及依赖...")
    deps = [
        "qwen-tts",
        "soundfile",
        "huggingface_hub",
        "pydub",  # 用于 mp3 转换
    ]
    for dep in deps:
        if check_installed(dep):
            print(f"[+] {dep} 已安装，跳过")
        else:
            pip_install([dep], f"安装 {dep}")

    # 设置 shell wrapper
    print("\n[4/5] 设置命令行工具...")
    setup_shell_wrapper()

    # 验证安装
    print("\n[5/5] 验证安装...")
    try:
        import torch
        import soundfile
        from qwen_tts import Qwen3TTSTokenizer, Qwen3TTSModel
        print(f"[+] PyTorch: {torch.__version__}")
        print(f"[+] CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[+] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("[+] qwen-tts: OK")
        print("[+] soundfile: OK")
    except ImportError as e:
        print(f"[!] 验证失败: {e}")
        return 1

    print("\n" + "=" * 50)
    print("安装完成！")
    print("=" * 50)
    print("\n使用方法:")
    print("  # 方式1: 直接命令 (任意目录)")
    print("  qwen-tts --list-voices")
    print("  qwen-tts -v '自己的声音' -t '你好' -o /tmp/out.wav")
    print("")
    print("  # 方式2: Python 脚本")
    print(f"  python3 {SKILL_DIR}/tts_cli.py --list-voices")
    print("")
    print("提示: 首次运行会自动下载模型 (~2GB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
