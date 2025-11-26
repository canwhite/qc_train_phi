# ==================== 标准库 ====================
import os  # 操作系统接口，用于文件路径操作和环境变量获取
import json  # JSON数据处理库，用于配置文件和数据序列化

# ==================== AI训练相关核心包 ====================
import torch  # PyTorch深度学习框架，用于张量计算和神经网络训练
# 基础用法：创建张量 tensor = torch.tensor([1, 2, 3])
# 主要作用：提供GPU加速的张量运算和自动求导功能

import datasets  # Hugging Face数据集库，用于加载和处理大规模数据集
# 基础用法：dataset = load_dataset("csv", data_files="data.csv")
# 主要作用：统一数据集格式，支持流式处理和高效存储
from datasets import load_dataset, Dataset  # 加载预置数据集和创建自定义数据集

import transformers  # Hugging Face变换器库，提供各种预训练模型
# 基础用法：model = AutoModel.from_pretrained("bert-base-uncased")
# 主要作用：加载和训练Transformer架构的NLP模型
from transformers import (
    AutoModelForCausalLM,     # 自动选择因果语言模型（如GPT系列）
    AutoTokenizer,           # 自动选择对应的分词器
    TrainingArguments,       # 训练参数配置类
    BitsAndBytesConfig       # 4bit/8bit量化配置，减少显存使用
)

# ==================== 高效微调相关包 ====================
from peft import LoraConfig  # 参数高效微调库的LoRA配置
# PEFT(Parameter-Efficient Fine-Tuning)主要用于在少量数据上高效微调大模型
# LoRA(Low-Rank Adaptation)通过添加小规模适配器层实现高效微调
# 基础用法：config = LoraConfig(r=16, lora_alpha=32)

from trl import SFTTrainer  # 监督微调训练器，专为语言模型优化
# TRL(Transformer Reinforcement Learning)库的监督微调工具
# 基础用法：trainer = SFTTrainer(model, dataset, args=training_args)
# 主要作用：简化语言模型监督微调流程，支持各种训练技巧


def setup_logging():
    """设置日志"""
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

def load_model_and_tokenizer(model_name="microsoft/Phi-3.5-mini-instruct"):
    """加载模型和tokenizer"""

    # 模型配置参数
    model_kwargs = {
        "use_cache": False,        # 训练时关闭KV缓存，节省内存；推理时可设为True加速
        "trust_remote_code": True, # 信任模型的远程代码，某些自定义模型架构需要
        "torch_dtype": torch.bfloat16,  # 使用bfloat16精度，减少显存占用，精度损失比float16小
        "device_map": "auto"       # 自动将模型层分布到可用设备(CPU/GPU)，支持大模型分层加载
    }

    # 加载tokenizer（文本预处理工具）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # 如果没有填充token，用未知token代替
    tokenizer.padding_side = 'right'  # 在序列右侧进行填充，符合训练惯例

    # 加载预训练的语言模型
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model, tokenizer


def create_sample_alpaca_data():
    """创建示例Alpaca格式数据"""
    
    alpaca_data = [
        {
            "instruction": "写一个Python函数计算斐波那契数列",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "The weather is really nice today, let's go for a walk.",
            "output": "今天天气真好，我们出去散步吧。"
        },
        {
            "instruction": "解释机器学习的基本概念",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。"
        },
        {
            "instruction": "写一封求职信",
            "input": "申请软件工程师职位，有3年Python经验",
            "output": "尊敬的招聘经理：\n\n我写信申请贵公司的软件工程师职位...\n\n此致\n敬礼"
        },
        {
            "instruction": "总结以下文章的主要内容",
            "input": "人工智能正在改变世界。从医疗诊断到自动驾驶，AI技术正在各个领域产生深远影响...",
            "output": "文章主要讨论了人工智能技术在各行各业的广泛应用和深远影响。"
        }
    ]
    
    # 保存为JSON文件
    os.makedirs("./data", exist_ok=True)
    with open("./data/alpaca_sample.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)
    
    return "./data/alpaca_sample.json"


def main():
    setup_logging()
    load_model_and_tokenizer()
    create_sample_alpaca_data()






if __name__ == "__main__":
    main()
