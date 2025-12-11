# 📊StepScorer

[English](README.md) | [中文](README_zh.md)

StepScorer 是一个为文本评分而设计的机器学习项目。该模型可以应用于各种特定任务，如评估语法正确性、检测歧视性内容以及其他文本评估任务，具体取决于所提供的训练数据。它的使用非常灵活，支持广泛的评分应用。这个仓库提供了用于训练模型、执行推理和可视化结果的工具。

与产生单一整体评估的传统方法不同，StepScorer 采用分解策略，在保持整个序列上下文关系的同时生成可解释的逐步评估。模型在每轮推理中输出的是delta值，这些值代表了当前步骤对总分的贡献。随着序列的推进，这些delta值逐渐减小并趋近于0，表明模型对最终评分越来越确定。这种方法的核心创新在于其增量评分机制：增量值随着时间步的推移逐渐趋近于0，最终累加值达到极限，即模型认为的最终结果。模型正是利用这一数学原理完成了训练和推理过程。

模型具有轻量化的特点。开发过程中采用仓库中尺寸为1500左右的歧视性语句数据集，使用Nvidia GeForce RTX 5060 Ti显卡进行训练和推理，模型训练耗时约40秒，单次推理模型耗时约1秒。

## 🔍 示例

检测对象： **The sentence contains race-color related discriminary content.**

| 输入                                               | 输出   |
|----------------------------------------------------|--------|
| PolyU is an outstanding university.                | 0.401  |
| White Americans aren't smart enough to be scientists. | 4.912  |

当输出值高于一定阈值时，可以认为对象被检测为包含种族歧视性内容。

## 📁 目录
- [数据集准备](#数据集准备)
- [训练](#训练)
- [推理](#推理)
- [调试](#调试)
- [模型结构](#模型结构)

## 🗃️ 数据集准备

在训练或运行推理之前，您需要将数据集准备成适当的格式：

项目期望的CSV数据包含三列：standard、object和score。

示例数据结构：

```markdown
data/
└── sample_data.csv
```

仓库已提供2份用于检测歧视性内容的训练数据集，来自[CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models](https://aclanthology.org/2020.emnlp-main.154/) (Nangia et al., EMNLP 2020)，用户可自行选用。

CSV中的每一行应该包含：
- standard：评估标准
- object：被评估的项目
- score：真实分数（0-5分制，也可使用其他分制，但需保持统一）

示例行：
```markdown
standard,object,score
"诗歌应押韵且有情感深度","秋叶飘零水自流，孤舟独坐忆故友。十年生死两茫茫，不思量，自难忘。",4.9
```

## 🔧 依赖准备

所有必需的包都列在[requirements.txt](requirements.txt)中。使用以下命令安装它们：
```bash
pip install -r requirements.txt
```

## 🏋️ 训练

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

## 🔮 推理

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

## 🐞 调试

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

## 🧩 模型结构

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

### 模型参数
可以在相应CONFIG字典中调整的主要超参数：
- 隐藏层维度（默认：128）
- 最大步数（默认：100）
- 学习率（默认：0.002）
- 批次大小（默认：24）

有关详细实现，请参阅[model.py](model.py)。


## 🤝 法律声明

- 本项目所公开的源代码、文档及相关技术资料（统称“本技术”）仅为技术展示与学术交流之目的提供，并不构成对本技术相关知识产权的放弃、转让或许可。
- 开发者/申请人保留在中华人民共和国及其他司法管辖区就本技术申请专利、商标、著作权或其他知识产权的一切权利。
- 任何个人或组织不得基于本公开内容主张本技术已进入公有领域，亦不得在未获得书面授权的情况下将本技术用于商业目的、专利规避设计或妨碍本技术的专利授权进程。
- 本公开不构成对任何第三方的明示或暗示许可，所有权利保留。