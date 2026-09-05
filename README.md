# v5.0：冻结锚点的元数据冲突残差

本分支：`experiment/v5.0-metadata-residual`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v5_anchored_residual.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

冻结v4.0-exp02-seed20260829全部参数。仅在anchor判benign、metadata判threat时允许非负局部修正；最大logit gap=24。epoch0等于锚点，验证未改善则回退。子类型始终来自锚点。

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

冻结v4.0-exp02-seed20260829。候选是anchor margin<0且metadata margin>0；
margin是threat logit减benign logit。delta=candidate*max(0,tanh(trust_logit))*
clamp(evidence_margin-anchor_margin,0,24)。candidate外逐行不变，子类型不变；
零初始化trust输出层保证epoch0等于锚点。max(0,tanh)限制非负delta，最多靠近证据分支。

可信度输入包括metadata128、semantic64、raw content128以及metadata margin、content margin、
novelty gate、log1p(combo_count)四标量。exp02再加128维四视图向量，其encoder复制锚点初始化
后仍可训练；原锚点才被冻结。最终anchor margin只进入候选和gap计算，不输入trust网络。

训练正例为真实threat且指定证据分支判threat，困难负例从benign按证据margin降序选最多正例2倍。
若只选训练集实际冲突，烟雾测试出现0样本，无法训练，因此改为学习证据可信度；
推理仍严格限定真正冲突。总损失=trust二元CE+最终威胁CE+0.10*锚点KL蒸馏+
0.01*trust_score平方均值。AdamW lr=0.0005，weight_decay=0.0001，batch=512，
scan/valid_batch=4096，epochs≤8，patience=3，梯度裁剪5，seed=20260904。

| 实验 | Score | 错误 | FP | FN | subtype | best_epoch | 改变分类 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp01锚点表示 | 0.9993387401713992 | 57 | 1 | 32 | 24 | 0 | 0 |
| exp02附加四视图 | 0.9993387401713992 | 57 | 1 | 32 | 24 | 0 | 0 |

验证metadata判threat共55,450行，其中真threat54,439、benign1,011；
anchor benign/metadata threat候选恰是1,011个benign，真threat0。
32个FN全部被外层mask排除，故不存在可修正FN；增加视图也无法突破候选边界。
两组自动回退是有效否定结果，不代表训练后提高了模型。

恢复锚点的57错进一步分解：

- 20个产品缺失JSON恶意FN：final平均0.438180、metadata0.003641、content0.947359；
  content范围0.912818–0.961938。
- 1个Duo可疑FN：final0.253372、metadata0.000602、content0.938998。
- 11个全分支偏benign的FN：8个Symantec success、2个无产品WindowsJSON、1个Symantec其他事件。
- 24个malicious→suspicious：WindowsJSON/fail，final0.938884、metadata0.877243、content0.000036。
- 1个Duo正常误报：final0.515301、metadata0.034732、content0.466180。

21个FN存在独立内容证据，下一次版本只换证据源；24个子类型错误和1个已有FP不在向上残差可修范围。

## 云平台运行

```bash
git fetch origin
git switch experiment/v5.0-metadata-residual
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
mkdir -p artifacts/v5_0_metadata_residual
nohup bash scripts/run.sh /root/work > artifacts/v5_0_metadata_residual/nohup.log 2>&1 &
echo $! > artifacts/v5_0_metadata_residual/train.pid
tail -f artifacts/v5_0_metadata_residual/nohup.log
```

默认只跑exp01，V5_RUN_MULTIVIEW=1启用exp02。
模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
