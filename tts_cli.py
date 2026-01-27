#!/usr/bin/env python3
"""
Qwen3-TTS CLI 命令行工具
支持单条合成和批量合成
"""
import argparse
import sys
import os

# 添加 skill 目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_tts_client import Qwen3TTSClient, VOICES_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 语音合成 CLI（支持批量生成）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用内置音色（单条）
  %(prog)s --voice "自己的声音" --text "你好" --out /tmp/output.wav

  # 使用自定义参考音频
  %(prog)s --ref-audio /path/to/ref.wav --text "你好" --out /tmp/output.wav

  # 批量生成（从 txt 文件读取，每行一条）
  %(prog)s --voice "赵信" --batch-file /path/to/texts.txt --out-dir /tmp/outputs/

  # 批量生成（多个文本参数）
  %(prog)s --voice "赵信" --texts "第一句话" "第二句话" "第三句话" --out-dir /tmp/outputs/

  # 列出内置音色
  %(prog)s --list-voices
        """,
    )

    parser.add_argument(
        "--voice", "-v",
        type=str,
        help="内置音色名称",
    )
    parser.add_argument(
        "--ref-audio", "-r",
        type=str,
        help="自定义参考音频路径",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        help="参考音频对应的文字（可选）",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="要合成的文本（单条模式）",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="+",
        help="要合成的多条文本（批量模式）",
    )
    parser.add_argument(
        "--batch-file", "-b",
        type=str,
        help="批量文本文件路径（每行一条，批量模式）",
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        help="输出文件路径（单条模式）",
    )
    parser.add_argument(
        "--out-dir", "-d",
        type=str,
        help="输出目录（批量模式）",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="tts",
        help="输出文件前缀（批量模式，默认: tts）",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="Chinese",
        help="语言 (默认: Chinese)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大生成 token 数 (默认: 2048)",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="列出所有内置音色",
    )
    parser.add_argument(
        "--full-clone",
        action="store_true",
        help="完整克隆模式（更慢但更精确）",
    )

    args = parser.parse_args()

    # 列出音色
    if args.list_voices:
        print("内置音色:")
        print("-" * 40)
        client = Qwen3TTSClient.__new__(Qwen3TTSClient)
        client._voices_cache = {}
        for voice in client.list_voices():
            # 读取文本预览
            text_file = VOICES_DIR / f"{voice}.txt"
            preview = ""
            if text_file.exists():
                text = text_file.read_text(encoding="utf-8").strip()
                preview = text[:30] + "..." if len(text) > 30 else text
            print(f"  {voice}: {preview}")
        return 0

    # 检查必要参数
    if not args.voice and not args.ref_audio:
        parser.error("必须指定 --voice 或 --ref-audio")
    if args.voice and args.ref_audio:
        parser.error("--voice 和 --ref-audio 不能同时指定")

    # 判断模式：批量 vs 单条
    is_batch_mode = args.batch_file or args.texts

    if is_batch_mode:
        # 批量模式
        if args.text:
            parser.error("批量模式下不能使用 --text，请使用 --texts 或 --batch-file")

        try:
            client = Qwen3TTSClient()

            if args.batch_file:
                # 从文件批量生成
                outputs = client.synthesize_from_file(
                    txt_file=args.batch_file,
                    voice=args.voice,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    language=args.language,
                    output_dir=args.out_dir,
                    output_prefix=args.out_prefix,
                    x_vector_only=not args.full_clone,
                    max_tokens=args.max_tokens,
                )
            else:
                # 从参数批量生成
                outputs = client.synthesize_batch(
                    texts=args.texts,
                    voice=args.voice,
                    ref_audio=args.ref_audio,
                    ref_text=args.ref_text,
                    language=args.language,
                    output_dir=args.out_dir,
                    output_prefix=args.out_prefix,
                    x_vector_only=not args.full_clone,
                    max_tokens=args.max_tokens,
                )

            print(f"\n完成！生成了 {len(outputs)} 个音频文件")
            return 0

        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1

    else:
        # 单条模式
        if not args.text:
            parser.error("必须指定 --text（单条模式）或 --batch-file/--texts（批量模式）")

        try:
            client = Qwen3TTSClient()
            output = client.synthesize(
                text=args.text,
                voice=args.voice,
                ref_audio=args.ref_audio,
                ref_text=args.ref_text,
                language=args.language,
                output_path=args.out,
                x_vector_only=not args.full_clone,
                max_tokens=args.max_tokens,
            )
            print(f"\n完成！输出文件: {output}")
            return 0

        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
