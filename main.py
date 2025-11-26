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

def main():
    print("Hello from qc-train-phi!")

    # ==================== 基础用法示例 ====================

    # 1. PyTorch 张量操作示例
    print("\n=== PyTorch 基础用法 ===")
    # 创建张量
    tensor_data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"创建张量:\n{tensor_data}")

    # GPU检查（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 2. 数据集加载示例
    print("\n=== Datasets 基础用法 ===")
    # 创建示例数据集
    sample_data = {
        "text": ["你好世界", "机器学习很有趣", "人工智能改变生活"],
        "label": [1, 0, 1]
    }
    dataset = Dataset.from_dict(sample_data)
    print(f"数据集大小: {len(dataset)}")
    print(f"第一条数据: {dataset[0]}")

    # 3. Transformers 模型加载示例
    print("\n=== Transformers 基础用法 ===")
    # 注意：这里只是示例，实际运行需要下载模型
    try:
        # 加载一个小模型作为示例
        model_name = "microsoft/DialoGPT-small"  # 小型对话模型
        print(f"尝试加载模型: {model_name}")
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        print("模型加载示例代码已准备，实际运行时会下载模型")
    except Exception as e:
        print(f"模型加载示例跳过: {e}")

    # 4. LoRA配置示例
    print("\n=== PEFT LoRA 配置示例 ===")
    lora_config = LoraConfig(
        r=16,  # LoRA注意力矩阵的秩
        lora_alpha=32,  # 缩放参数
        target_modules=["q_proj", "v_proj"],  # 应用LoRA的模块
        lora_dropout=0.1,  # dropout率
        bias="none",  # bias设置
        task_type="CAUSAL_LM"  # 任务类型
    )
    print(f"LoRA配置创建完成: {lora_config}")

    # 5. 训练参数配置示例
    print("\n=== 训练参数配置示例 ===")
    training_args = TrainingArguments(
        output_dir="./results",  # 输出目录
        num_train_epochs=3,  # 训练轮数
        per_device_train_batch_size=4,  # 每设备批次大小
        warmup_steps=100,  # 预热步数
        weight_decay=0.01,  # 权重衰减
        logging_dir="./logs",  # 日志目录
        learning_rate=5e-5,  # 学习率
    )
    print(f"训练参数配置: {training_args}")

    print("\n=== 基础用法示例完成 ===")


if __name__ == "__main__":
    main()
