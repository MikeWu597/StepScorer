# StepScorer

[English](README.md) | [中文](README_zh.md)

StepScorer 是一个为序列数据中的步骤评分而设计的机器学习项目。该模型可以应用于各种特定任务，如评估语法正确性、检测歧视性内容以及其他文本评估任务，具体取决于所提供的训练数据。它的使用非常灵活，支持广泛的顺序评分应用。这个仓库提供了用于训练模型、执行推理和可视化结果的工具。

## 目录
- [数据集准备](#数据集准备)
- [训练](#训练)
- [推理](#推理)
- [调试](#调试)
- [模型结构](#模型结构)

## 数据集准备

在训练或运行推理之前，您需要将数据集准备成适当的格式：

项目期望的CSV数据包含三列：standard、object和score。

示例数据结构：

```markdown
data/
└── sample_data.csv
```

CSV中的每一行应该包含：
- standard：评估标准
- object：被评估的项目
- score：真实分数（0-5分制）

示例行：
```markdown
standard,object,score
"诗歌应押韵且有情感深度","秋叶飘零水自流，孤舟独坐忆故友。十年生死两茫茫，不思量，自难忘。",4.9
```

## 训练

要训练StepScorer模型，请直接修改[train.py](train.py)中的CONFIG字典：

CONFIG中的关键训练配置选项：
- `data_path`：训练数据的路径（默认：'data/sample_data.csv'）
- `model_save_path`：保存训练模型的文件路径（默认：'scoring_model.pt'）
- `epochs`：训练轮数（默认：20）
- `batch_size`：训练批次大小（默认：24）
- `lr`：优化器的学习率（默认：0.002）

在训练过程中，模型将：
1. 加载并预处理训练数据
2. 初始化模型参数
3. 运行指定轮数的训练循环
4. 根据验证损失保存最佳模型

只需运行：
```bash
python train.py
```

## 推理

训练后，您可以使用[inference.py](inference.py)脚本来为新序列评分。

在运行推理之前，请更新[inference.py](inference.py)中的CONFIG字典：
- `model_path`：训练模型检查点的路径（默认：'scoring_model.pt'）

然后运行：
```bash
python inference.py
```

推理脚本将：
1. 加载训练好的模型
2. 提示输入标准和对象
3. 为序列生成步骤分数
4. 将结果以结构化格式保存到'scoring_steps.json'

## 调试

出于调试和可视化目的，使用[figure.py](figure.py)脚本将模型输出转换为图像：

首先，确保您有一个评分结果文件（由inference.py生成）：
- 默认情况下，这将是'scoring_steps.json'

然后运行：
```bash
python figure.py
```

此脚本有助于：
- 可视化预测结果
- 显示随步骤演变的累积分数
- 显示每个步骤的增量值
- 生成用于分析和演示的图表

可视化结果将保存为'scoring_evolution.png'。

## 模型结构

模型架构在[model.py](model.py)中定义。关键组件包括：

### 核心架构
- **BERT编码器**：使用预训练的BERT模型处理输入文本（冻结）
- **GRU模块**：建模顺序评分过程
- **增量预测器**：预测每一步的增量分数变化
- **累加器**：从增量计算累积分数

### 关键特性
- 使用BERT对标准和对象进行语义理解
- 采用GRU建模逐步评分过程
- 预测每步的分数增量而不是绝对分数
- 应用掩码正确处理可变长度序列
- 使用来自标准的全局上下文初始化GRU状态

### 模型参数
可以在相应CONFIG字典中调整的主要超参数：
- 隐藏层维度（默认：128）
- 最大步数（默认：100）
- 学习率（默认：0.002）
- 批次大小（默认：24）

有关详细实现，请参阅[model.py](model.py)。

## 依赖

所有必需的包都列在[requirements.txt](requirements.txt)中。使用以下命令安装它们：
```bash
pip install -r requirements.txt
```

## 贡献

欢迎提交问题和拉取请求来改进这个项目。