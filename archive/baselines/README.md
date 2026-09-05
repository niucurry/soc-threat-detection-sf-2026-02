# 归档基线

这里保留两条不属于当前神经网络主线、但曾用于建立参考结果的基线：

- `train_catboost.py`：b0.1 CatBoost 结构化基线；依赖
  `requirements-catboost.txt`；
- `train_predict_rules.py`：b0.2 训练集统计规则基线；依赖
  `requirements-rules.txt`。

它们没有被当前 v1–v5 训练入口调用。保留它们是为了追溯历史结果，不应把它们解释为
当前模型的组成部分。归档代码只做兼容维护，不再从这里派生新的主线版本。
