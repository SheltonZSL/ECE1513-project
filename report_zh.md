# BERT对抗鲁棒性研究：第一阶段报告

---

## 1. 引言

BERT等大规模预训练语言模型在众多NLP基准任务上取得了最先进的结果，包括情感分类任务。然而，近期研究表明这些模型极易受到对抗攻击的影响——即对输入文本进行微小的、保持语义的扰动，即可导致模型产生错误预测。

本项目研究了在SST-2（斯坦福情感树库）二分类情感分析任务上微调的`bert-base-uncased`的对抗鲁棒性。我们使用TextFooler攻击（Jin等人，2019）评估模型的脆弱性，并探索对抗数据增强——使用WordNet同义词替换和字符级交换生成的训练数据——能否提升模型的鲁棒性。

## 2. 方法

### 2.1 基线模型

我们使用以下超参数在SST-2训练集（约67,349个样本）上微调`bert-base-uncased`：

| 超参数 | 取值 |
|---|---|
| 优化器 | AdamW |
| 学习率 | 2e-5 |
| 批大小 | 32 |
| 最大序列长度 | 128 |
| 训练轮数 | 3 |
| 权重衰减 | 0.01 |

### 2.2 对抗攻击：TextFooler

TextFooler（Jin等人，2019）是一种词级对抗攻击方法，通过将单词替换为语义相似的替代词来最大化模型的预测误差。该攻击使用：
- 反向拟合（Counter-fitted）词嵌入生成同义词候选
- 通用句子编码器（USE）进行语义相似度约束
- 词性一致性检查

我们对每个模型的200个正确分类的验证样本进行了攻击。

### 2.3 对抗数据增强

我们通过对67,349个原始训练样本交替应用两种增强策略来生成增强训练数据：

1. **WordNet增强器**：替换20%的词为WordNet同义词
2. **字符交换增强器**：交换10%词中的字符

共生成67,349个增强样本，与原始数据合并后总计134,698个训练样本。

### 2.4 增强模型

使用相同的超参数，在合并数据集（134,698个样本）上从头微调一个新的`bert-base-uncased`模型，训练3个轮次。

## 3. 实验结果

### 3.1 结果汇总

| 指标 | 基线模型 | 增强模型 | 变化 |
|---|---|---|---|
| 干净准确率 | 93.23% | 93.12% | -0.11 pp |
| TextFooler攻击成功率（ASR） | 93.50% | 94.50% | +1.00 pp |
| 攻击下准确率 | 6.06% | 5.12% | -0.94 pp |

### 3.2 可视化

对比柱状图见`results/comparison_chart.png`，攻击结果分布饼图见`results/attack_distribution.png`。

### 3.3 攻击示例

| 原始文本 | 扰动文本 | 标签翻转 |
|---|---|---|
| "it's a **charming** and often affecting **journey**" | "it's a **cutie** and often affecting **circuits**" | 正面 -> 负面 |
| "unflinchingly **bleak** and desperate" | "unflinchingly **eerie** and desperate" | 负面 -> 正面 |
| "allows us to hope ... a major **career** as a **commercial** yet **inventive** filmmaker" | "allows ourselves to hope ... a major **quarries** as a **commercialized** yet **imaginative** filmmaker" | 正面 -> 负面 |

## 4. 分析与讨论

### 4.1 主要发现

1. **BERT极易受到TextFooler攻击**：基线模型虽然达到了93.23%的干净准确率，但攻击成功率高达93.50%，意味着TextFooler在200个正确分类的样本中成功欺骗了187个。攻击下的有效准确率仅为6.06%。

2. **简单数据增强未能提升对抗鲁棒性**：增强模型保持了相当的干净准确率（93.12%），但鲁棒性没有改善——攻击成功率反而略微上升至94.50%。

3. **干净准确率得以保持**：增强训练没有降低模型在干净（非对抗）输入上的表现，证明增强方法没有引入有害噪声。

### 4.2 增强方法未能提升鲁棒性的原因分析

鲁棒性未能提升可归因于以下几个因素：

1. **增强策略与攻击策略不匹配**：TextFooler使用由反向拟合词嵌入和语义相似度约束（USE）引导的精密同义词替换，而我们的增强方法仅使用了较简单的WordNet同义词和字符交换。增强训练数据未能让模型接触到TextFooler所利用的特定类型扰动。

2. **TextFooler的自适应特性**：TextFooler是一种自适应、迭代式攻击，它查询模型并选择专门最大化预测误差的替换词。静态数据增强无法防御此类有针对性的扰动。

3. **增强强度不足**：WordNet同义词和字符交换产生的扰动相对温和。可能需要更激进的增强技术——如带模型反馈的对抗训练——才能有效提升鲁棒性。

4. **基线脆弱性过高**：在93.5%的ASR下，BERT对TextFooler的脆弱性非常严重，简单增强不足以应对。这表明需要架构层面的改变或对抗训练方法（如FreeLB、SMART）才能实现有意义的防御。

### 4.3 启示

这些结果揭示了一个重要教训：**并非所有数据增强都能提升对抗鲁棒性**。增强的类型和强度必须与威胁模型仔细匹配。对于防御TextFooler等复杂攻击，需要更有针对性的方法：

- **对抗训练**（在训练过程中生成对抗样本）
- **认证防御**（如针对NLP的随机平滑）
- **鲁棒微调**方法（FreeLB、SMART、R3F）

## 5. 结论

本项目表明，在SST-2上微调的`bert-base-uncased`虽然达到了较高的干净准确率（93.23%），但极易受到TextFooler对抗攻击（攻击成功率93.50%）。使用WordNet同义词和字符交换的对抗数据增强保持了干净准确率（93.12%），但未能提升对TextFooler的鲁棒性（攻击成功率94.50%）。

这一负面结果本身具有信息价值：它表明朴素的数据增强不足以对抗复杂的词级对抗攻击，需要更先进的防御机制。未来工作可以探索带模型反馈的对抗训练、认证鲁棒性方法或基于集成的防御策略。

## 6. 参考文献

1. Devlin, J.等 (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
2. Jin, D.等 (2019). Is BERT Really Robust? A Strong Baseline for Natural Language Attack and Defense. arXiv:1907.11932.
3. Socher, R.等 (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. EMNLP.
4. Zang, Y.等 (2020). Word-level Textual Adversarial Attacking as Combinatorial Optimization. ACL.

## 附录：可复现性

### 环境
- Python 3.12, PyTorch (CUDA), Transformers 4.57.6, TextAttack
- 硬件：NVIDIA GeForce RTX 4060 Laptop GPU, Windows 11

### 流水线命令
```bash
uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt
python src/train.py --mode baseline
python src/evaluate.py --model_dir results/baseline_model
python src/attack.py --model_dir results/baseline_model
python src/augment.py
python src/train.py --mode augmented
python src/evaluate.py --model_dir results/augmented_model
python src/attack.py --model_dir results/augmented_model
```
