#!/usr/bin/env python3
"""
Qwen3-TTS 客户端封装
支持 GPU 加速、内置音色、自定义参考音频
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, List
import warnings

# 抑制 flash-attn 警告
warnings.filterwarnings("ignore", message=".*flash-attn.*")

# 设置 HuggingFace 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Skill 根目录
SKILL_DIR = Path(__file__).parent
VOICES_DIR = SKILL_DIR / "assets" / "voices"


class Qwen3TTSClient:
    """Qwen3-TTS 语音合成客户端"""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda",
    ):
        """
        初始化客户端

        Args:
            model_id: HuggingFace 模型 ID
            device: 设备类型，仅支持 "cuda"（GPU 模式强制）
        """
        self.model_id = model_id
        self.device = device
        self._model = None
        self._voices_cache = {}

    def _ensure_model_loaded(self):
        """懒加载模型"""
        if self._model is not None:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        # 检查 CUDA 可用性
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用！本 Skill 仅支持 GPU 模式。\n"
                "请检查:\n"
                "  1. NVIDIA 驱动是否安装\n"
                "  2. CUDA 是否正确配置\n"
                "  3. PyTorch GPU 版本是否安装 (运行 install.py)"
            )

        print(f"[*] 加载模型: {self.model_id}")
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # 使用 from_pretrained 加载模型
        # 注意：使用 dtype 而非 torch_dtype（后者已废弃）
        self._model = Qwen3TTSModel.from_pretrained(
            self.model_id,
            device_map="cuda",
            dtype=torch.float32,  # float32 更稳定
        )

        print("[+] 模型加载完成")

    def list_voices(self) -> List[str]:
        """列出所有内置音色"""
        voices = []
        if VOICES_DIR.exists():
            for f in VOICES_DIR.iterdir():
                if f.suffix in (".mp3", ".wav"):
                    voices.append(f.stem)
        return sorted(set(voices))

    def _get_voice_files(self, voice: str) -> tuple[str, Optional[str]]:
        """获取音色的音频和文本文件路径"""
        if voice in self._voices_cache:
            return self._voices_cache[voice]

        audio_path = None
        ref_text = None

        # 查找音频文件
        for ext in (".wav", ".mp3"):
            p = VOICES_DIR / f"{voice}{ext}"
            if p.exists():
                audio_path = str(p)
                break

        # 查找文本文件
        text_file = VOICES_DIR / f"{voice}.txt"
        if text_file.exists():
            ref_text = text_file.read_text(encoding="utf-8").strip()

        if audio_path is None:
            raise ValueError(f"未找到音色: {voice}\n可用音色: {self.list_voices()}")

        self._voices_cache[voice] = (audio_path, ref_text)
        return audio_path, ref_text

    def _prepare_voice_clone_prompt(
        self,
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only: bool = True,
    ):
        """
        准备语音克隆提示（可复用）

        Returns:
            (prompt_items, audio_path) 元组
        """
        # 获取参考音频
        if voice is not None:
            audio_path, text_from_file = self._get_voice_files(voice)
            if ref_text is None:
                ref_text = text_from_file
        else:
            audio_path = ref_audio
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"参考音频不存在: {audio_path}")

        # 确保模型加载
        self._ensure_model_loaded()

        # 创建 voice clone prompt
        prompt_items = self._model.create_voice_clone_prompt(
            ref_audio=audio_path,
            ref_text=ref_text if not x_vector_only else None,
            x_vector_only_mode=x_vector_only,
        )

        return prompt_items, audio_path

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Chinese",
        output_path: Optional[str] = None,
        x_vector_only: bool = True,
        max_tokens: int = 2048,
    ) -> str:
        """
        合成语音

        Args:
            text: 要合成的文本
            voice: 内置音色名称（与 ref_audio 二选一）
            ref_audio: 自定义参考音频路径（与 voice 二选一）
            ref_text: 参考音频对应的文字（可选，用于更精确的克隆）
            language: 语言，默认 "Chinese"
            output_path: 输出文件路径，默认生成临时文件
            x_vector_only: 仅使用音色向量（快速模式）
            max_tokens: 最大生成 token 数

        Returns:
            输出文件路径
        """
        import soundfile as sf

        # 参数检查
        if voice is None and ref_audio is None:
            raise ValueError("必须指定 voice 或 ref_audio")
        if voice is not None and ref_audio is not None:
            raise ValueError("voice 和 ref_audio 不能同时指定")

        # 准备语音克隆提示
        prompt_items, audio_path = self._prepare_voice_clone_prompt(
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
        )

        # 生成输出路径
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".wav")

        print(f"[*] 合成文本: {text[:50]}..." if len(text) > 50 else f"[*] 合成文本: {text}")
        print(f"[*] 参考音频: {audio_path}")
        print(f"[*] x_vector_only: {x_vector_only}")

        # 生成语音
        wavs, sample_rate = self._model.generate_voice_clone(
            text=text,
            voice_clone_prompt=prompt_items,
            language=language,
            max_new_tokens=max_tokens,
        )

        # 保存音频
        if wavs and len(wavs) > 0:
            sf.write(output_path, wavs[0], sample_rate)
            print(f"[+] 输出文件: {output_path}")
        else:
            raise RuntimeError("语音生成失败：输出为空")

        return output_path

    def synthesize_batch(
        self,
        texts: List[str],
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Chinese",
        output_dir: Optional[str] = None,
        output_prefix: str = "tts",
        x_vector_only: bool = True,
        max_tokens: int = 2048,
    ) -> List[str]:
        """
        批量合成语音（模型只加载一次）

        Args:
            texts: 要合成的文本列表
            voice: 内置音色名称（与 ref_audio 二选一）
            ref_audio: 自定义参考音频路径（与 voice 二选一）
            ref_text: 参考音频对应的文字（可选）
            language: 语言，默认 "Chinese"
            output_dir: 输出目录，默认为临时目录
            output_prefix: 输出文件前缀，默认 "tts"
            x_vector_only: 仅使用音色向量（快速模式）
            max_tokens: 最大生成 token 数

        Returns:
            输出文件路径列表
        """
        import soundfile as sf

        # 参数检查
        if voice is None and ref_audio is None:
            raise ValueError("必须指定 voice 或 ref_audio")
        if voice is not None and ref_audio is not None:
            raise ValueError("voice 和 ref_audio 不能同时指定")
        if not texts:
            raise ValueError("文本列表不能为空")

        # 准备输出目录
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="qwen3_tts_batch_")
        else:
            os.makedirs(output_dir, exist_ok=True)

        # 准备语音克隆提示（只执行一次）
        print(f"[*] 批量模式：共 {len(texts)} 条文本")
        prompt_items, audio_path = self._prepare_voice_clone_prompt(
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only=x_vector_only,
        )
        print(f"[*] 参考音频: {audio_path}")
        print(f"[*] x_vector_only: {x_vector_only}")
        print(f"[*] 输出目录: {output_dir}")
        print("-" * 50)

        output_files = []
        for i, text in enumerate(texts, 1):
            text = text.strip()
            if not text:
                print(f"[{i}/{len(texts)}] 跳过空行")
                continue

            output_path = os.path.join(output_dir, f"{output_prefix}_{i:04d}.wav")
            print(f"[{i}/{len(texts)}] {text[:30]}..." if len(text) > 30 else f"[{i}/{len(texts)}] {text}")

            try:
                # 生成语音
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=prompt_items,
                    language=language,
                    max_new_tokens=max_tokens,
                )

                # 保存音频
                if wavs and len(wavs) > 0:
                    sf.write(output_path, wavs[0], sample_rate)
                    output_files.append(output_path)
                else:
                    print(f"  [!] 生成失败：输出为空")
            except Exception as e:
                print(f"  [!] 生成失败：{e}")

        print("-" * 50)
        print(f"[+] 完成！成功生成 {len(output_files)}/{len(texts)} 条音频")
        print(f"[+] 输出目录: {output_dir}")

        return output_files

    def synthesize_from_file(
        self,
        txt_file: str,
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "Chinese",
        output_dir: Optional[str] = None,
        output_prefix: str = "tts",
        x_vector_only: bool = True,
        max_tokens: int = 2048,
    ) -> List[str]:
        """
        从文本文件批量合成语音（每行一条）

        Args:
            txt_file: 文本文件路径（每行一条文本）
            voice: 内置音色名称（与 ref_audio 二选一）
            ref_audio: 自定义参考音频路径（与 voice 二选一）
            ref_text: 参考音频对应的文字（可选）
            language: 语言，默认 "Chinese"
            output_dir: 输出目录，默认与 txt 文件同目录
            output_prefix: 输出文件前缀，默认使用 txt 文件名
            x_vector_only: 仅使用音色向量（快速模式）
            max_tokens: 最大生成 token 数

        Returns:
            输出文件路径列表
        """
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"文本文件不存在: {txt_file}")

        # 读取文本文件
        with open(txt_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            raise ValueError(f"文本文件为空: {txt_file}")

        print(f"[*] 读取文件: {txt_file}")
        print(f"[*] 共 {len(texts)} 行有效文本")

        # 设置默认输出目录和前缀
        if output_dir is None:
            output_dir = os.path.dirname(txt_file) or "."
        if output_prefix == "tts":
            output_prefix = os.path.splitext(os.path.basename(txt_file))[0]

        return self.synthesize_batch(
            texts=texts,
            voice=voice,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            output_dir=output_dir,
            output_prefix=output_prefix,
            x_vector_only=x_vector_only,
            max_tokens=max_tokens,
        )

    def __del__(self):
        """清理资源"""
        try:
            import torch
            if self._model is not None:
                del self._model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass


# 单例客户端（避免重复加载模型）
_client: Optional[Qwen3TTSClient] = None


def get_client() -> Qwen3TTSClient:
    """获取单例客户端"""
    global _client
    if _client is None:
        _client = Qwen3TTSClient()
    return _client
