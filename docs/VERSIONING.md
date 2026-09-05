# 版本命名与分支职责

主版本表示算法家族，次版本表示同一家族的方法变化。只改seed、epoch、阈值对照使用实验后缀。

- v1.x：结构Embedding与MLP。
- v2.x：结构神经网络加路由TF-IDF/SGD及规则。
- v3.x：无模板内容神经网络。
- v4.x：威胁检测与子类型分层神经网络。
- v5.x：冻结锚点的局部冲突残差。

完整实验写v主.次-expNN，重复写seedYYYYMMDD，checkpoint写epNN。次版本不能因为一次重训自动增加。
main只有环境、公式、开发日志和版本导航；模型代码分别在版本分支。
每个版本分支README必须独立说明全部输入、模型、训练命令、验证范围、实测指标、缺陷与后续方法。

| 版本 | 分支 | 模型 |
|---|---|---|
| v1.0 | `release/v1.0-tabular` | 结构Embedding与MLP |
| v1.1 | `release/v1.1-drain` | 分组Drain与确定性解析 |
| v1.2 | `experiment/v1.2-structured` | 深层结构解析与schema特征 |
| v2.0 | `release/v2.0-hybrid` | 结构基础模型与路由正文专模 |
| v2.1 | `release/v2.1-score` | 混合模型全量评分选择 |
| v2.2 | `release/v2.2-semantic` | 混合模型语义规则与泛化审计 |
| v3.0 | `release/v3.0-content` | 无模板内容神经网络 |
| v4.0 | `release/v4.0-hierarchical` | 威胁与子类型分层神经网络 |
| v4.1 | `experiment/v4.1-multiview` | 多视图与证据保持消融 |
| v5.0 | `experiment/v5.0-metadata-residual` | 冻结锚点的元数据冲突残差 |
| v5.1 | `experiment/v5.1-content-gap24` | 冻结锚点的内容冲突残差 |
| v5.2 | `release/v5.2-content-rescue` | 内容救援与局部幅度限制 |

每个模型分支统一入口scripts/run.sh只运行该版本默认实验，依赖的特征生成/锚点训练明确在该分支保留。
跨版本结果比较使用相同数据、评分与配套预测；单独报告随机重复差异。
