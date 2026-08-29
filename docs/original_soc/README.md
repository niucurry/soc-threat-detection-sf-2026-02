# 基于 SOC 日志的网络安全威胁检测

当前版本根据训练数据自动学习高支持度、高纯度的少数类规则，完成
`benign / suspicious / malicious` 三分类，并生成赛题要求的 `res.csv`。

## 数据结论

- 训练集：2,056,871 行；
- 验证集：2,014,052 行；
- 训练标签：benign 1,899,723，malicious 111,728，suspicious 45,420；
- 验证分布与训练集不同，存在明显分布漂移；
- 数据中存在非常强的稳定字段规律，因此第一版优先采用可解释规则学习，避免复杂模型过拟合。

默认参数从训练集学到的主要规律包括：

- 特定字段为真正的 Parquet `NULL` 时，高置信预测 malicious；
- `product_name` 中高支持、高纯度的产品取值，高置信预测 suspicious；
- 未命中高置信少数类规则的事件预测为 benign。

规则不是写死的，会由 `train_predict.py` 根据训练集重新统计生成。

## 运行

本机已有的 Conda Python 包含所需依赖，可直接执行：

```powershell
D:\conda\python.exe train_predict.py
```

输出：

- `res.csv`：提交格式预测结果；
- `artifacts/rules.json`：模型学到的可解释规则和命中量；
- 控制台：macro-F1、分类报告和混淆矩阵。

指定正式测试文件时：

```powershell
D:\conda\python.exe train_predict.py `
  --data-dir data\data `
  --test-file test.parquet `
  --output res.csv `
  --no-eval
```

## 当前验证结果

在提供的 `valid_input.parquet + valid_answer_private.parquet` 上：

- macro-F1：约 **0.998761**；
- accuracy：约 **0.999905**；
- malicious recall：约 **0.996015**；
- suspicious recall：约 **0.997502**。

验证答案仅用于最后评分，没有参与规则学习。

## 后续增强方向

第一版已经覆盖绝大多数样本。下一版应只针对当前误判的少量困难样本训练残差模型，而不是重做全量分类：

1. 对未命中规则的样本提取 `message_sanitized` 字符 n-gram；
2. 采用类权重 Logistic Regression/SGD 分类器；
3. 使用时间分块验证，防止重复日志造成随机切分虚高；
4. 输出特征贡献、规则命中原因和攻击事件时间线。

