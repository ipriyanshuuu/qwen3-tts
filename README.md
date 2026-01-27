# Qwen3-TTS

本地 GPU 加速语音克隆工具，基于 [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) 模型。

## 特性

- **零样本语音克隆**：只需一段参考音频即可克隆任意声音
- **批量生成**：模型只加载一次，高效处理多条文本
- **内置音色**：提供多种预置音色，开箱即用
- **简单易用**：CLI 命令行和 Python API 双支持

## 环境要求

- NVIDIA GPU（6GB+ 显存）
- CUDA 11.6+
- Python 3.10+

## 安装

```bash
# 克隆仓库
git clone https://github.com/wangwangit/qwen3-tts.git
cd qwen3-tts

# 安装依赖
python3 install.py

# （可选）添加到 PATH，全局使用
echo 'export PATH="$PATH:'$(pwd)'"' >> ~/.bashrc
source ~/.bashrc
```

## 使用方法

### CLI 命令

```bash
# 列出内置音色
./qwen-tts --list-voices

# 使用内置音色合成
./qwen-tts -v "寒冰射手" -t "你好，这是测试" -o /tmp/output.wav

# 使用自定义参考音频克隆
./qwen-tts -r /path/to/reference.wav -t "你好，这是测试" -o /tmp/output.wav

# 批量生成：从 txt 文件读取（每行一条文本）
./qwen-tts -v "赵信" -b /path/to/texts.txt -d /tmp/outputs/

# 批量生成：多个文本参数
./qwen-tts -v "赵信" --texts "第一句话" "第二句话" "第三句话" -d /tmp/outputs/
```

### Python API

```python
from qwen3_tts_client import Qwen3TTSClient

client = Qwen3TTSClient()

# 使用内置音色
client.synthesize(
    text="你好，世界！",
    voice="寒冰射手",
    output_path="/tmp/output.wav"
)

# 使用自定义参考音频
client.synthesize(
    text="你好，世界！",
    ref_audio="/path/to/my_voice.wav",
    output_path="/tmp/output.wav"
)

# 批量生成（模型只加载一次，效率更高）
outputs = client.synthesize_batch(
    texts=["第一句话", "第二句话", "第三句话"],
    voice="赵信",
    output_dir="/tmp/outputs/"
)
```

## 内置音色

| 名称 | 说明 |
|------|------|
| 寒冰射手 | 游戏角色配音风格 |
| 布里茨 | 机器人风格配音 |
| 赵信 | 游戏角色配音风格 |

## 添加自定义音色

将参考音频（.mp3/.wav）和对应文本（.txt）放入 `assets/voices/` 目录：

```
assets/voices/
├── 你的音色名.mp3   # 参考音频
└── 你的音色名.txt   # 音频对应的文字内容
```

## 配置说明

- **HuggingFace 镜像**：默认使用 `hf-mirror.com`，可通过 `HF_ENDPOINT` 环境变量修改
- **模型**：首次运行自动下载 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

## License

MIT
