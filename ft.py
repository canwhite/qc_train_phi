# ==================== 标准库 ====================
import json  # JSON数据处理库，用于配置文件和数据序列化
import logging
import os  # 操作系统接口，用于文件路径操作和环境变量获取


# ==================== AI训练相关核心包 ====================
import torch  # PyTorch深度学习框架，用于张量计算和神经网络训练

# 基础用法：dataset = load_dataset("csv", data_files="data.csv")
# 主要作用：统一数据集格式，支持流式处理和高效存储
from datasets import Dataset  # 加载预置数据集和创建自定义数据集

# ==================== 高效微调相关包 ====================

# 基础用法：model = AutoModel.from_pretrained("bert-base-uncased")
# 主要作用：加载和训练Transformer架构的NLP模型
from transformers import AutoModelForCausalLM  # 自动选择因果语言模型（如GPT系列）
from transformers import AutoTokenizer  # 自动选择对应的分词器

# 基础用法：创建张量 tensor = torch.tensor([1, 2, 3])
# 主要作用：提供GPU加速的张量运算和自动求导功能

# PEFT(Parameter-Efficient Fine-Tuning)主要用于在少量数据上高效微调大模型
# LoRA(Low-Rank Adaptation)通过添加小规模适配器层实现高效微调
# 基础用法：config = LoraConfig(r=16, lora_alpha=32)


# TRL(Transformer Reinforcement Learning)库的监督微调工具
# 基础用法：trainer = SFTTrainer(model, dataset, args=training_args)
# 主要作用：简化语言模型监督微调流程，支持各种训练技巧

logger = logging.getLogger(__name__)


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def load_model_and_tokenizer(model_name="microsoft/Phi-3.5-mini-instruct"):
    """加载模型和tokenizer"""

    # 模型配置参数
    model_kwargs = {
        "use_cache": False,  # 训练时关闭KV缓存，节省内存；推理时可设为True加速
        "trust_remote_code": True,  # 信任模型的远程代码，某些自定义模型架构需要
        "dtype": torch.bfloat16,  # 使用bfloat16精度，减少显存占用，精度损失比float16小
        "device_map": "auto",  # 自动将模型层分布到可用设备(CPU/GPU)，支持大模型分层加载
        "attn_implementation": "eager",  # 使用eager attention避免flash attention警告
    }

    # 加载tokenizer（文本预处理工具）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # 如果没有填充token，用未知token代替
    tokenizer.padding_side = "right"  # 在序列右侧进行填充，符合训练惯例

    # 加载预训练的语言模型
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    return model, tokenizer


def create_sample_alpaca_data():
    """创建示例Alpaca格式数据"""

    alpaca_data = [
        {
            "instruction": "写一个Python函数计算斐波那契数列",
            "input": "",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)",
        },
        {
            "instruction": "将以下英文翻译成中文",
            "input": "The weather is really nice today, let's go for a walk.",
            "output": "今天天气真好，我们出去散步吧。",
        },
        {
            "instruction": "解释机器学习的基本概念",
            "input": "",
            "output": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。",
        },
        {
            "instruction": "写一封求职信",
            "input": "申请软件工程师职位，有3年Python经验",
            "output": "尊敬的招聘经理：\n\n我写信申请贵公司的软件工程师职位...\n\n此致\n敬礼",
        },
        {
            "instruction": "总结以下文章的主要内容",
            "input": "人工智能正在改变世界。从医疗诊断到自动驾驶，AI技术正在各个领域产生深远影响...",
            "output": "文章主要讨论了人工智能技术在各行各业的广泛应用和深远影响。",
        },
    ]

    # 保存为JSON文件
    os.makedirs("./data", exist_ok=True)
    with open("./data/alpaca_sample.json", "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    return "./data/alpaca_sample.json"


def convert_alpaca_to_messages_format(data_path):
    """将Alpaca格式转换为消息格式"""

    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        alpaca_data = json.load(f)

    messages_data = []

    for example in alpaca_data:
        # 构建用户消息
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += "\n" + example["input"]

        # 构建消息格式
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]

        messages_data.append({"messages": messages})

    return messages_data


def prepare_dataset(tokenizer, data_path=None, use_sample_data=True):
    """准备训练数据集"""

    # 步骤1：决定使用什么数据
    if use_sample_data:
        # 如果use_sample_data=True，使用内置的示例数据
        data_path = create_sample_alpaca_data()  # 创建5个样本的JSON文件

    # 步骤2：转换数据格式
    # 从Alpaca格式 {"instruction": "...", "input": "...", "output": "..."}
    # 转换为聊天格式 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages_data = convert_alpaca_to_messages_format(data_path)
    logger.info(f"message_data: {len(messages_data)} items")

    # 步骤3：创建Hugging Face数据集对象
    raw_dataset = Dataset.from_list(messages_data)  # 把Python列表转为HF数据集格式
    logger.info(f"raw_dataset: {raw_dataset}")

    # 步骤4：定义内部函数 - 应用聊天模板
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],  # 聊天消息列表
            tokenize=False,  # 不进行分词，返回字符串
            add_generation_prompt=False,  # 不添加生成提示符
        )
        return example

    # 步骤5：批量处理数据集
    processed_dataset = raw_dataset.map(
        apply_chat_template,  # 对每个样本应用这个函数
        remove_columns=raw_dataset.column_names,  # 删除原来的"messages"列，只保留"text"列
    )

    # 步骤6：分割数据集
    if len(processed_dataset) > 1:  # 如果数据多于1条
        dataset = processed_dataset.train_test_split(
            test_size=0.2, seed=42
        )  # 80%训练，20%验证
        train_dataset = dataset["train"]  # 训练集
        eval_dataset = dataset["test"]  # 验证集
    else:  # 如果只有1条数据
        train_dataset = processed_dataset  # 全部作为训练集
        eval_dataset = None  # 没有验证集

    return train_dataset, eval_dataset  # 返回训练集和验证集


def main():
    setup_logging()

    model, tokenizer = load_model_and_tokenizer()
    data_path = create_sample_alpaca_data()
    # data = convert_alpaca_to_messages_format(data_path)
    # logger.info(data)

    data = prepare_dataset(tokenizer, data_path)
    logger.info(data)


if __name__ == "__main__":
    main()
