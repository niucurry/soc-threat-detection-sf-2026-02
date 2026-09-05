# v1.0：结构Embedding与MLP

本分支：`release/v1.0-tabular`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v1_0_tabular.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

目标是建立紧凑基线，正文只以长度和关键词指示进入，无法完整表达嵌套事件。

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

## 训练、实验结果、缺陷与后续解决方法

训练默认AdamW，lr=0.002，weight_decay=1e-5，batch=8192，num_workers=4，梯度范数裁剪5，seed=20260828，类别权重power=0。v1.x最多20轮/patience4；v2.x基础最多12轮/patience3。

在神经网络前曾准备 CatBoost 候选：MultiClass，350 iterations，depth=8，
learning_rate=0.12，l2_leaf_reg=5，Balanced class weights，random_strength=0.5，
seed=20260828。未留存完整验证结果，因此只作为方法探索记录，不列为有成绩的模型版本。

神经网络采用 AdamW，lr=0.002，weight_decay=1e-5，batch=8192，梯度范数裁剪5，
seed=20260828。类别权重为 [N/(K*n_class)]^power，正式使用power=0，即全部为1。
类别 Embedding 维数 min(24,max(3,round(2*cardinality^0.25)))。三类输出顺序固定为
benign、malicious、suspicious。两个 checkpoint 必须分别记录：

| 实验 | 选择指标 | epoch | Score | 错误 | FP | FN | 子类型互错 |
|---|---|---:|---:|---:|---:|---:|---:|
| exp01 | Macro-F1 | 1 | 0.9572098298 | 9,857 | 9,793 | 40 | 24 |
| exp02 | Competition Score | 3 | 0.9574032044 | 9,833 | 9,793 | 40 | 0 |

exp01混淆矩阵：[[1949780,9793,0],[30,13998,24],[10,0,40417]]；
exp02：[[1949780,9793,0],[30,14022,0],[10,0,40417]]。
exp01 Macro-F1约0.912286，exp02为0.9126510510016193。
exp02云端训练耗时227.10秒；初始记录每epoch约35–37秒，checkpoint为236,108字节。
这些耗时是单次环境测量，不能当作所有机器的保证。

权重对照计划在相同20万行上比较0、0.25、0.50、0.75。仅power=0全量结果可核实，
其他权重未留存完整成绩。训练损失下降但验证分类不动，说明继续增加训练轮次未解决问题。
9,793个正常误报集中在产品缺失syslog路由：该路由训练231,092 benign/111,728 malicious，
验证287,281 benign/14,052 malicious。攻击比例由32.59%变4.66%，原结构表示又把不同正文
压成同样输入，支持“信息不足并伴随先验漂移”的诊断。后续分别尝试解析特征和正文专模。

## 云平台运行

```bash
git fetch origin
git switch release/v1.0-tabular
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v1_0_tabular
nohup bash scripts/run.sh /root/work > artifacts/v1_0_tabular/nohup.log 2>&1 &
echo $! > artifacts/v1_0_tabular/train.pid
tail -f artifacts/v1_0_tabular/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
