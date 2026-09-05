# SOC 日志威胁检测

本仓库把安全日志分类为 `benign`、`malicious`、`suspicious`。当前正式候选是
`v5.2-exp01-anchor-content-gap2`：冻结 v4.0 分层神经网络，只在“锚点判 benign、独立内容
分支判 threat”的冲突中学习可信度，并把最大 logit 修正限制为 2。

完整官方验证集结果：Competition Score `0.9996698618`，36 个三分类错误，threat FP=1、
FN=11、子类型混淆=24；相对可复现 seed29 锚点修正 21 行、新增错误 0。

## 快速运行当前版本

原始数据目录必须同时包含 `train.parquet`、`valid_input.parquet`、
`valid_answer_private.parquet`。若数据在 `/root/work`，先得到 v4.0 seed29 锚点：

```bash
mkdir -p artifacts/v4_0_hierarchical
nohup bash scripts/run_cloud_v4_0_hierarchical.sh /root/work \
  > artifacts/v4_0_hierarchical/nohup.log 2>&1 &
echo $! > artifacts/v4_0_hierarchical/train.pid
tail -f artifacts/v4_0_hierarchical/nohup.log
```

基础训练完成后运行固定 seed 对照；当前 V5 默认锚点是 seed `20260829`：

```bash
nohup bash scripts/run_cloud_v4_0_seed_sweep.sh \
  data/processed/v3_0 artifacts/v4_0_hierarchical \
  > artifacts/v4_0_hierarchical/seed_sweep.log 2>&1 &
echo $! > artifacts/v4_0_hierarchical/seed_sweep.pid
tail -f artifacts/v4_0_hierarchical/seed_sweep.log
```

然后训练当前正式候选。默认只跑计算成本更低、hard result 最好的 exp01：

```bash
mkdir -p artifacts/v5_2_content_rescue
nohup bash scripts/run_cloud_v5_anchored_residual.sh /root/work \
  > artifacts/v5_2_content_rescue/nohup.log 2>&1 &
echo $! > artifacts/v5_2_content_rescue/train.pid
tail -f artifacts/v5_2_content_rescue/nohup.log
```

要额外跑不作为默认的多视图 exp02，设置 `V5_RUN_MULTIVIEW=1`。不要继续扫描 gap；当前默认
`V5_MAX_CONFLICT_GAP=2` 已经是验证集引导的消融，需要用时间外推/OOF 做下一次确认。

### 复用旧 V6/V7/V8 云端产物

旧目录不会自动改名或删除。假设旧产物在 `/root/soc-threat-detection-sf-2026-02-1`：

```bash
bash scripts/link_legacy_artifacts.sh \
  /root/soc-threat-detection-sf-2026-02-1

V5_REUSE_CONTENT_FILES=1 \
nohup bash scripts/run_cloud_v5_anchored_residual.sh /root/work \
  > artifacts/v5_2_content_rescue/nohup.log 2>&1 &
```

迁移脚本只建立符号链接，遇到已存在目标会保留，不复制或删除源权重。

## 版本总览

| 标准版本 | 旧名称 | 基础方法 | 完整验证结果 | 决策 |
|---|---|---|---:|---|
| v1.0 | V1 | 结构类别 Embedding + MLP | 9,833 错；Score 0.957403 | 基线 |
| v1.1 | V4 | 同 MLP + 分格式解析/分组 Drain | 76 错；0.999023 | 被后续替代 |
| v1.2 | V5.1 | 同 MLP + 深层结构/schema | 142 错；0.998438 | 否定实验 |
| v2.2 | V3-G | 表格模型 + 路由 TF-IDF/SGD + 规则 | 保守 10 错；调优 0 错 | 强混合基线；有验证规则风险 |
| v3.0 | V6 | 无模板内容神经网络 | 最佳 2,618 错；0.976221 | 发现单头任务混淆 |
| v4.0 | V7 | 分层 threat/subtype 神经网络 | 历史 46 错；可复现锚点 57 错 | 冻结锚点 |
| v4.1 | V8 | 多视图/证据保持 | 最佳 56 错；0.999273 | 未超过 v4.0 |
| v5.0 | V9 | metadata 锚定残差 | 57 错；无改动 | 候选条件错误 |
| v5.1 | V10 | content 残差，gap=24 | 78 错；0.999497 | 修 FN 但新增 42 FP |
| v5.2 | V10.1 | content 残差，gap=2 | 36 错；0.999670 | 当前候选 |

机器可读的全部实验、结果和历史提交在 `model_registry.json`。

## 文档导航

- `docs/DEVELOPMENT_LOG_STANDARD.md`：从数据、完整输入、模型、实验、失败原因到下一步的标准开发日志；
- `docs/VERSIONING.md`：版本、实验、seed、epoch、分支和产物命名规则；
- `docs/versions/V1_TABULAR_FAMILY.md`：v1.0/v1.1/v1.2；
- `docs/versions/V2_HYBRID_FAMILY.md`：v2.0/v2.1/v2.2；
- `docs/versions/V3_CONTENT_FAMILY.md`：v3.0 三组内容实验；
- `docs/versions/V4_HIERARCHICAL_FAMILY.md`：v4.0/v4.1；
- `docs/versions/V5_RESIDUAL_FAMILY.md`：v5.0/v5.1/v5.2。

## 代码入口

```text
scripts/run_cloud_v1_0_tabular.sh
scripts/run_cloud_v1_0_weight_sweep.sh
scripts/run_cloud_v1_1_drain.sh
scripts/run_cloud_v1_2_structured.sh
scripts/run_cloud_v2_2_hybrid.sh
scripts/run_cloud_v2_2_text_refresh.sh
scripts/run_cloud_v3_0_content.sh
scripts/run_cloud_v4_0_hierarchical.sh
scripts/run_cloud_v4_0_seed_sweep.sh
scripts/run_cloud_v4_1_multiview.sh
scripts/run_cloud_v5_anchored_residual.sh
```

所有训练入口都接受原始数据目录作为第一个参数。版本内重复实验使用 `expNN`，随机重复使用
`seedNN`，不再为一次轻微超参修改创建新的模型版本。
