# v5.1：冻结锚点的内容冲突残差

本分支：`experiment/v5.1-content-gap24`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v5_anchored_residual.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

冻结同一个v4.0锚点，候选证据改为content，max_conflict_gap=24。通过训练正例与高分困难正常样本学习可信度，只修正候选内威胁logit，保持子类型和候选外预测。

## 完整输入特征

### 类别输入（9个）

| 字段 | 定义 |
|---|---|
| `pipeline` | 采集管道；空串/NULL归为缺失类别。 |
| `product_name` | 产品名称；不直接决定威胁标签。 |
| `product_group` | 产品粗组：missing、asa、aws_vpc、other_suspicious_products（Precinct/Falcon）、other。 |
| `src_ip_kind` | 源地址外形类别：missing、ipv4_shape、host_token、other；仅按当前正则判外形。 |
| `port_bucket` | 源端口桶：missing、0–1023、1024–49151、49152以上；基于可解析整数。 |
| `message_length_bucket` | 正文长度桶：missing、1–120、121–180、181–300、301–1000、1001以上。 |
| `structure_combo` | pipeline、product_group、message_length_bucket和是否含deny拼接的类别。 |
| `network_missing_pattern` | 源IP/目的IP/源端口三个缺失位按顺序拼接，如111。 |
| `vendor_name` | 厂商名称；空值单独编码。 |

### 数值输入（23个）

| 字段 | 定义 |
|---|---|
| `src_port_number` | 原始src_port尝试转整数，缺失填-1。 |
| `src_ip_missing` | src_ip对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `dst_ip_missing` | dst_ip对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `src_port_missing` | src_port对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `src_host_missing` | src_host对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `dst_host_missing` | dst_host对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `username_missing` | username对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `product_missing` | product对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `message_missing` | message对应字段缺失为1，否则0；源端口按整数转换是否成功判断。 |
| `network_present_count` | src_ip、dst_ip、src_port三个字段存在数量，0–3。 |
| `message_length` | 原始正文字符长度，空正文为0。 |
| `src_ip_length` | src_ip对应字符串字符长度，空值按空串。 |
| `dst_ip_length` | dst_ip对应字符串字符长度，空值按空串。 |
| `src_host_length` | src_host对应字符串字符长度，空值按空串。 |
| `dst_host_length` | dst_host对应字符串字符长度，空值按空串。 |
| `username_length` | username对应字符串字符长度，空值按空串。 |
| `message_has_deny` | 小写正文是否含deny子串；不是完整语义判断。 |
| `message_has_allow` | 小写正文是否含allow子串。 |
| `message_has_accepted` | 小写正文是否含accepted子串。 |
| `message_has_failed` | 小写正文是否含failed子串。 |
| `message_has_blocked` | 小写正文是否含blocked子串。 |
| `message_starts_angle` | 去左空白后的正文首字符是否为<。 |
| `message_contains_json` | 正文是否包含{；并不表示JSON解析成功。 |

### 语义塔完整输入

### 类别输入（4个）

| 字段 | 定义 |
|---|---|
| `content_family` | 内容解析家族。 |
| `content_action` | 内容粗动作。 |
| `content_protocol` | 内容协议。 |
| `content_event_code` | 内容事件码。 |

### 数值输入（4个）

| 字段 | 定义 |
|---|---|
| `content_has_threat` | 内容安全词组的威胁信号；不是标签。 |
| `content_has_authentication` | 认证关键词信号。 |
| `content_has_potentially_harmful` | potentially harmful相关词组信号。 |
| `raw_token_count` | raw序列非padding数量，上限96。 |

另输入raw_token_ids（96位）；v4.1多视图实验用multiview_token_ids（head/middle/tail/key_value各64位）替代raw。频次键为product_name、content_family、content_action。

trust输入为三塔128/64/128维与metadata_margin、content_margin、novelty_gate、log1p(combo_count)，共324维。exp02另加128维四视图，共452维。anchor_margin只用于候选和gap，不输入trust。

## 训练、实验结果、缺陷与后续解决方法

残差训练AdamW lr=0.0005、weight_decay=0.0001、batch512、scan/valid_batch4096、最多8轮、patience3、seed20260904。正例选真实threat且证据分支判threat；benign按证据margin降序选最多正例2倍。损失=trust BCE+最终威胁CE+0.10锚点KL+0.01 trust_score平方均值；原锚点不更新。candidate=(anchor_margin<0且evidence_margin>0)，delta=candidate*max(0,tanh(trust_logit))*clamp(evidence_margin-anchor_margin,0,max_conflict_gap)。epoch0为锚点，只有验证选择指标改善才替换。

只将证据源改为content，保持gap24、冻结锚点和训练保护。反向冲突在全验证集上的审计：

| content阈值 | 候选 | 真threat | benign | 精度 |
|---:|---:|---:|---:|---:|
| 0.50 | 294 | 21 | 273 | 0.071429 |
| 0.70 | 174 | 21 | 153 | 0.120690 |
| 0.90 | 41 | 21 | 20 | 0.512195 |
| 0.91 | 36 | 21 | 15 | 0.583333 |
| 0.92 | 29 | 17 | 12 | 0.586207 |
| 0.94 | 20 | 16 | 4 | 0.800000 |
| 0.96 | 9 | 8 | 1 | 0.888889 |

高阈值仍有FP且会丢正例，所以正式模型保留0.5宽候选，用训练标签学trust。

| 实验 | Score | 错误 | FP | FN | subtype | epoch | 修复/新增 |
|---|---:|---:|---:|---:|---:|---:|---|
| exp01锚点内容 | 0.9994966836 | 78 | 43 | 11 | 24 | 1 | 21/42 |
| exp02附加四视图 | 0.9994142658 | 98 | 63 | 11 | 24 | 2 | 21/62 |

两组修复同21个FN（20个无产品恶意JSON、1个Duo可疑JSON），新增FP均集中Falcon；
exp01共改63行，正确修复比例21/63=33.33%；exp02改83行，比例25.30%。
exp01 Log Loss约0.0002541，高于锚点0.0002223。Score提高源于比赛更重视威胁召回，
并非所有业务指标均提高。四视图新增20个FP，未选为主候选。

真威胁20条anchor平均0.438、最小0.316，Duo0.253；Falcon新增FP anchor平均0.0135、
content平均0.899。gap24允许可信度高时推翻极高置信benign。下一次版本限制logit最大修正，
不加入厂商名规则。

## 云平台运行

```bash
git fetch origin
git switch experiment/v5.1-content-gap24
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

先准备内容特征和配套锚点（已存在且身份核对通过时可跳过）：

```bash
python -m pip install -r requirements-npu.txt
python src/prepare_features.py --data-dir /root/work --output-dir data/processed/v1_0
python src/prepare_content_features.py --data-dir /root/work --base-feature-dir data/processed/v1_0 --output-dir data/processed/v3_0
V4_0_SWEEP_SEEDS=20260829 bash scripts/run_cloud_v4_0_seed_sweep.sh
```

也可通过V5_ANCHOR_MODEL、V5_ANCHOR_PREDICTIONS、V5_ANCHOR_METRICS指定已核验的锚点绝对路径。固定seed重新训练不保证逐行复现，须重新评价epoch0。

```bash
mkdir -p artifacts/v5_1_content_rescue
nohup bash scripts/run.sh /root/work > artifacts/v5_1_content_rescue/nohup.log 2>&1 &
echo $! > artifacts/v5_1_content_rescue/train.pid
tail -f artifacts/v5_1_content_rescue/nohup.log
```

默认只跑exp01，V5_RUN_MULTIVIEW=1启用exp02。
模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
