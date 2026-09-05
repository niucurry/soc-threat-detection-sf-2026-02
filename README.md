# v2.1：混合模型全量评分选择

本分支：`release/v2.1-score`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v2_1_hybrid.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

同一双模型与basic规则；基础轮次与正文阈值按全量Competition Score选。它是选择策略变更，不是增加第三个模型。

## 完整输入特征

### 类别输入（8个）

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

正文专模另输入message_sanitized全文；路由使用pipeline和product_name；可疑覆盖读取产品和正文。event_id仅连接，label_binary仅训练/评价。

## 训练、实验结果、缺陷与后续解决方法

训练默认AdamW，lr=0.002，weight_decay=1e-5，batch=8192，num_workers=4，梯度范数裁剪5，seed=20260828，类别权重power=0。v1.x最多20轮/patience4；v2.x基础最多12轮/patience3。

正文设置：TF-IDF word(1,2)，lowercase=True，min_df=2，max_df=0.9999，max_features=200000，sublinear_tf=True，float32；SGD log_loss/L2，alpha=1e-6，max_iter=30，tol=1e-4，average=True，seed=20260828。

架构不变，基础checkpoint改按全量Competition Score选。正文阈值先计算路由外1,712,719行
固定混淆矩阵，再累加每个候选边界的路由TP/FP/FN/TN，最大化全量六项分数；
1e-12同分内偏向预测更多威胁的边界。

本地阈值0.056902363896369934，云端0.05870809219777584；
两者在各自概率下都将路由分对。阈值必须与配套模型一起保存，不能跨模型直接照搬。
云端结构训练227.10秒，正文训练与评价60.57秒，词表200,000。
正文修复基础分类9,823条：9,793 benign→malicious和30 malicious→benign；
再加可疑覆盖共修复9,833条。

| 实验 | 全量Score | Macro-F1 | 错误 | FP | FN | 子类型互错 |
|---|---:|---:|---:|---:|---:|---:|
| exp01保守 | 0.9998902649047847 | 0.9999579178042195 | 10 | 0 | 10 | 0 |
| exp02可疑覆盖 | 1.0 | 1.0 | 0 | 0 | 0 | 0 |

保守结果Threat-F1=0.9999082130924845，Threat-Recall=0.9998164430330954，
两威胁类平均召回=0.9998763202810004，Soft-Score=0.9999950348848987，
Balanced-Accuracy=0.9999175468540002。可疑覆盖版六项均1。
正文阈值和可疑规则依赖同一验证分布，下一步需跨家族检查。

## 云平台运行

```bash
git fetch origin
git switch release/v2.1-score
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v2_1_hybrid
nohup bash scripts/run.sh /root/work > artifacts/v2_1_hybrid/nohup.log 2>&1 &
echo $! > artifacts/v2_1_hybrid/train.pid
tail -f artifacts/v2_1_hybrid/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
