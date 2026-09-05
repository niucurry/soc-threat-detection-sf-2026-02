# SOC日志威胁检测

main是项目导航分支，保存训练环境、数据约定、评分公式、完整开发日志和各模型简要介绍。
训练代码与该版本的全部结果位于各自分支README。当前内容神经网络候选为v5.2-exp01，
完整验证2,014,052行，Score=0.9996698618032245，36错（FP1/FN11/子类型24）。
这是已反复用于开发选择的验证集成绩，尚无独立时间外推或OOF结果。

## 环境与评分

已核验Linux aarch64、Python3.10.16、PyTorch2.4.0、torch-npu2.4.0.post2、Ascend910B2。
完整依赖、数据列和操作约定见[训练环境](docs/ENVIRONMENT.md)。main不安装依赖、不提供训练脚本。

Competition Score = 0.40×Threat Binary F1 + 0.25×Threat Binary Recall + 0.15×两威胁类平均Recall
+ 0.10×Macro-F1 + 0.05×Soft Label Score + 0.05×Balanced Accuracy。
[完整计算公式](docs/SCORING.md)解释合并威胁类、子类型互错、三分类概率和Log Loss。

## 模型版本

| 版本与分支 | 简要训练方法 | 完整验证结果：错误数与Score |
|---|---|---|
| [v1.0](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v1.0-tabular) | 结构Embedding与MLP | 9,833错；0.9574032044 |
| [v1.1](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v1.1-drain) | 分组Drain与确定性解析 | 76错；0.9990226566 |
| [v1.2](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/experiment/v1.2-structured) | 深层结构解析与schema特征 | 142错；0.9984380474 |
| [v2.0](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v2.0-hybrid) | 结构基础模型与路由正文专模 | 保守10错/0.9998902649；可疑覆盖0错/1.0 |
| [v2.1](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v2.1-score) | 混合模型全量评分选择 | 保守10错/0.9998902649；可疑覆盖0错/1.0 |
| [v2.2](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v2.2-semantic) | 混合模型语义规则与泛化审计 | 保守10错/0.9998902649；可疑覆盖0错/1.0 |
| [v3.0](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v3.0-content) | 无模板内容神经网络 | 最好2,618错；0.9762209763 |
| [v4.0](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v4.0-hierarchical) | 威胁与子类型分层神经网络 | 初始46错/0.9993140394；可用seed29为57错/0.9993387402 |
| [v4.1](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/experiment/v4.1-multiview) | 多视图与证据保持消融 | 最好56错；0.9992727909 |
| [v5.0](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/experiment/v5.0-metadata-residual) | 冻结锚点的元数据冲突残差 | 回退57错；0.9993387402 |
| [v5.1](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/experiment/v5.1-content-gap24) | 冻结锚点的内容冲突残差 | 78错；0.9994966836 |
| [v5.2](https://github.com/niucurry/soc-threat-detection-sf-2026-02/tree/release/v5.2-content-rescue) | 内容救援与局部幅度限制 | 36错；0.9996698618 |

点击版本直接进入对应分支README，可查看完整输入、模型定义、全部对照实验、缺陷和后续解决方法。
可疑覆盖版满分包括验证错误启发的规则，不能与单神经网络成绩混为一谈；未获收益的实验也保留完整记录。

## 文档

- [完整开发日志](docs/DEVELOPMENT_LOG_STANDARD.md)：按模型组织的训练、实验、错误分析、真实日志案例、时间外推方案和云端复现记录。
- [完整特征字典](docs/FEATURE_REFERENCE.md)：所有实际模型输入列及定义。
- [版本与分支职责](docs/VERSIONING.md)：正式命名和维护规则。
- [机器可读版本登记](model_registry.json)：分支、当前模型和结果。

## 开始训练

先选具体版本分支。以v5.2为例：

```bash
git fetch origin
git switch release/v5.2-content-rescue
git pull --ff-only
```

随后按该分支README准备数据和锚点，再运行scripts/run.sh。main本身没有训练代码。
