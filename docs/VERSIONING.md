# 版本、实验与分支规范

本规范从 2026-09-05 起生效。唯一的机器可读事实来源是仓库根目录的
`model_registry.json`；README 和开发日志都应与它一致。

## 1. 名称如何组成

模型版本写成 `v主版本.次版本`：

- 主版本变化表示基础算法族改变，例如单头三分类改为分层分类；
- 次版本变化表示算法族不变，但输入、门控或约束有实质变化；
- 只改变对照组、随机种子、epoch 或一个超参数，不升级版本。

实验统一写成：

```text
v4.1-exp01-multiview-standard
v4.1-exp02-raw-evidence
v4.0-exp02-seed20260829
v5.2-exp01-ep01
```

其中 `expNN` 是同一版本下的对照组，`seedNN` 是随机重复，`epNN` 只表示某个
训练轮次的权重。`recovery`、`cloud`、`oldv6` 都不能作为模型版本。

非主线基线使用 `b0.x`，避免 CatBoost 或规则脚本和神经网络主线争用 V1 名称。

## 2. 旧名称到标准名称

| 旧名称 | 标准名称 | 说明 |
|---|---|---|
| V1、V1.2 权重扫描 | v1.0 | 权重扫描只是实验，不是新模型 |
| V4 Drain | v1.1 | 仍是同一个类别 Embedding + MLP |
| V5/V5.1 structured | v1.2 | 仍是同一个 MLP，只深化了解析输入 |
| V2 | v2.0 | 两模型混合架构首次形成 |
| V2-F | v2.1 | 只改变正式评分和模型选择逻辑 |
| V3/V3-G | v2.2 | 沿用 V2 架构，增强语义和泛化审计 |
| V6 | v3.0 | 新的无模板内容神经网络族 |
| V7 H1/H2 | v4.0-exp01/exp02 | 分层威胁/子类型模型；H1/H2 是对照组 |
| V8 A1/B1/C1 | v4.1-exp01/02/03 | 同一分层模型的多视图实验 |
| V9 | v5.0 | metadata 证据锚定残差 |
| V10 | v5.1 | content 证据、gap=24 |
| V10.1 | v5.2 | content 证据、局部可信域 gap=2 |

旧名只出现在 `legacy_aliases`、历史结果说明和旧 checkpoint 兼容字段中；新代码、脚本、
产物目录不再用旧名。

## 3. 分支与标签

标准分支命名：

- `release/vM.m-short-name`：可复现版本快照；
- `experiment/vM.m-short-name`：已结束但未成为正式候选的实验；
- `maintenance/...`：不改变模型定义的文档、兼容或工程修正。

历史提交不会被改写。旧的 `feat/v4...feat/v9...` 分支暂时保留为兼容别名，防止云端
目录仍在跟踪旧分支时突然失去上游。它们在所有云环境切换到标准分支后才能删除。

当前标准分支与历史快照的关系：

| 标准分支 | 历史提交 | 含义 |
|---|---|---|
| `release/v1.0-tabular` | `6cf9854` | v1.0 训练与权重对照形成 |
| `release/v1.1-drain` | `d431e98` | 旧 V4 快照 |
| `experiment/v1.2-structured` | `261ef68` | 旧 V5.1 快照 |
| `release/v2.0-hybrid` | `3bd15ba` | 两模型混合与正式评分形成 |
| `release/v2.1-score` | `3bd15ba` | 与 v2.0 同一历史提交；登记当时只改变选择/评分策略的逻辑版本 |
| `release/v2.2-semantic` | `84a07f3` | 旧 V3-G 云端结果记录 |
| `release/v3.0-content` | `5e6c636` | 旧 V6 快照 |
| `release/v4.0-hierarchical` | `e75454e` | 旧 V7 快照 |
| `experiment/v4.1-multiview` | `65490ca` | 旧 V8 快照 |
| `experiment/v5.0-metadata-residual` | `e73742c` | 旧 V9 快照 |
| `experiment/v5.1-content-gap24` | `b4504e5` | 旧 V10 代码快照 |
| `release/v5.2-content-rescue` | 标准化提交 | 当前候选 |

历史分支里的文件仍使用当时名称，这是事实快照，不做破坏性重写。当前标准分支包含统一后的
全部可训练入口。

## 4. 产物规则

每个正式实验目录至少应包含：

```text
model.pt
metrics.json
valid_predictions.parquet
train_console.log
```

预处理目录应包含 manifest；manifest 记录模型版本、固定参数、输入输出行数、是否只在训练集
拟合以及旧版本别名。训练结束不能只看 `metrics.json`：只有模型、指标和逐行预测同时存在，
才视为完整。

大权重和 Parquet 不提交 Git。关机前应把 `model.pt`、`metrics.json`、预测和 manifest
一起复制到持久化对象存储，并记录 SHA-256；只保存权重而丢失预处理器或特征定义，仍然不能
保证可推理。
