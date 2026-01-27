# Qwen3-TTS Skill

本地 GPU 加速语音克隆工具，基于 Qwen3-TTS-12Hz-0.6B-Base 模型。支持单条合成和批量生成。

## 触发条件

- 用户请求使用 `/qwen-tts` 命令
- 用户需要本地 GPU 语音合成/克隆功能

## 内置音色

| 名称 | 说明 |
|------|------|
| 寒冰射手 | 游戏角色配音风格 |
| 布里茨 | 机器人风格配音 |
| 赵信 | 游戏角色配音风格 |

## 使用方法

### 全局命令（任意目录可用）

安装后可直接使用 `qwen-tts` 命令：

```bash
# 列出内置音色
qwen-tts --list-voices

# 使用内置音色合成（单条）
qwen-tts -v "寒冰射手" -t "你好，这是测试" -o /tmp/output.wav

# 使用自定义参考音频
qwen-tts -r /path/to/reference.wav -t "你好，这是测试" -o /tmp/output.wav

# 批量生成：从 txt 文件读取（每行一条文本）
qwen-tts -v "赵信" -b /path/to/texts.txt -d /tmp/outputs/

# 批量生成：多个文本参数
qwen-tts -v "赵信" --texts "第一句话" "第二句话" "第三句话" -d /tmp/outputs/
```

### CLI 命令行（完整路径）

```bash
# 使用内置音色合成（单条）
python3 tts_cli.py \
  --voice "寒冰射手" \
  --text "你好，这是测试" \
  --out /tmp/output.wav

# 使用自定义参考音频
python3 tts_cli.py \
  --ref-audio /path/to/reference.wav \
  --text "你好，这是测试" \
  --out /tmp/output.wav

# 批量生成：从 txt 文件读取
python3 tts_cli.py \
  --voice "赵信" \
  --batch-file /path/to/texts.txt \
  --out-dir /tmp/outputs/

# 批量生成：多个文本参数
python3 tts_cli.py \
  --voice "赵信" \
  --texts "第一句话" "第二句话" "第三句话" \
  --out-dir /tmp/outputs/

# 列出内置音色
python3 tts_cli.py --list-voices
```

### Python API

```python
from qwen3_tts_client import Qwen3TTSClient

client = Qwen3TTSClient()

# 使用内置音色（单条）
client.synthesize(
    text="你好，世界！",
    voice="寒冰射手",
    output_path="/tmp/output.wav"
)

# 使用自定义音频
client.synthesize(
    text="你好，世界！",
    ref_audio="/path/to/my_voice.wav",
    output_path="/tmp/output.wav"
)

# 批量生成（模型只加载一次）
outputs = client.synthesize_batch(
    texts=["第一句话", "第二句话", "第三句话"],
    voice="赵信",
    output_dir="/tmp/outputs/"
)

# 从 txt 文件批量生成
outputs = client.synthesize_from_file(
    txt_file="/path/to/texts.txt",
    voice="赵信",
    output_dir="/tmp/outputs/"
)
```

## 批量生成说明

批量模式的优势：
- **模型只加载一次**：避免每条文本都重新加载模型，大幅提升效率
- **语音克隆提示复用**：参考音频只处理一次，后续生成直接复用
- **自动编号输出**：文件自动命名为 `前缀_0001.wav`, `前缀_0002.wav` ...

txt 文件格式：
```
第一行是第一句话
第二行是第二句话
空行会被自动跳过

第四行是第四句话
```

## 环境要求

- NVIDIA GPU (6GB+ 显存)
- CUDA 11.6+
- Python 3.10+

## 安装依赖

```bash
python3 install.py
```

## 配置

- HuggingFace 镜像: `hf-mirror.com` (自动配置)
- 显存策略: 仅 GPU 模式，显存不足时报错提示
- 模型: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

## 工作流程

1. 确认用户需求（文本内容、音色选择、单条/批量）
2. 执行 CLI 命令或 Python API 调用
3. 返回生成的音频文件路径
4. 如需播放验证：`ffplay /path/to/output.wav`
