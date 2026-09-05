# v2.2：混合模型语义规则与泛化审计

本分支：`release/v2.2-semantic`。统一入口`scripts/run.sh`调用`scripts/run_cloud_v2_2_hybrid.sh`。
完整验证范围为2,014,052行；路由和烟雾结果在表中单独标明。

## 模型定义与改动理由

类别字段各自Embedding（维数min(24,max(3,round(2*cardinality^0.25))))，数值按训练均值/标准差标准化并截断[-12,12]。拼接→Linear256→BatchNorm→SiLU→Dropout0.15→Linear128→BatchNorm→SiLU→Dropout0.10→Linear64→SiLU→Linear3，交叉熵训练。未知类别索引0。

保持全量Score选择，expanded规则增加两个精确语义条件；正文仍是TF-IDF/SGD。家族/字符实验独立于正式推理。

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

精确原文审计：954,292个验证benign在训练出现且标签一致；malicious和suspicious原文重叠为0。
长hex和数字归一化后，这两类完整模板重叠仍为0；归一化前48字符家族覆盖5,266个验证恶意。
验证209行来自2个未见产品，pipeline无新增枚举。正常原文重复与攻击格式漂移同时存在。

| 实验 | 实测结果 | 决策 |
|---|---|---|
| 模板首折 | 272,972训练/69,848留出；词80.73秒，字符263.83秒；内部均约0.999970 Macro-F1、2错 | 内部近满分未迁移外部 |
| 首折外部 | 词原始Macro-F1约0.75909、恶意召回0.36379；加基础规则约0.85828、召回0.57102 | 字符/词符融合无改善且更慢 |
| 五折家族隔离 | 内部错误0、0、0、2、24；外部原始恶意召回均约36.379%，加规则约57.102% | 当前隔离不足以模拟完整厂商迁移 |
| IP/hex/数字归一化 | 词表200,000→18,669；约38→24秒；外部只多识别2个恶意 | 未替换正式模型 |
| 无标签间隙阈值 | 五折错误4、1、0、38、287；阈值约0.0612、0.0413、0.0400、0.0713、0.1302 | 不稳定，未采用 |
| 五折概率融合 | 均值、中位数、最小、最大、logit均比较 | 未解决不同折压低不同家族 |

排除基础规则后，benign最大恶意概率约0.054065，malicious最小约0.059739，间隔0.005674；
6,028个未覆盖恶意的中位概率约0.21654。高权重有deny/protocol/dport，也有0x0/src/dst、
身份与时间片段；分类全对仍可能对漂移敏感。

| 规则候选 | 训练恶意 | 验证恶意 | benign命中 | 新增覆盖 |
|---|---:|---:|---:|---:|
| TRAFFIC,drop | 0 | 5,826 | 0 | 5,826 |
| THREAT,url且block-url | 0 | 200 | 0 | 200 |
| 协议字段后deny | 0 | 5,820 | 0 | 5,820 |
| 任意,deny, | 0 | 5,838 | 0 | 5,826 |
| decision=blocked | 260 | 0 | 0 | 260 |
| act=deny | 50 | 0 | 0 | 50 |

正式只新增,traffic,drop,和,threat,url,且block-url两条件。总计14,050恶意、0正常命中。
剩2条Windows4672由模型判断，没有硬写4672=恶意：授予特殊权限也可由正常系统账号触发。

本地阈值0.17455598153173923，云端0.1985421571880579，正文耗时59.31秒，
路由Log Loss=0.008998715901508317。

| 实验 | 全量Score | Macro-F1 | 错误 | 全量Log Loss |
|---|---:|---:|---:|---:|
| exp01保守 | 0.9998902649047847 | 0.9999579178042195 | 10 | 0.0014145733444869055 |
| exp02可疑覆盖 | 1.0 | 1.0 | 0 | 0.0013629193716559853 |

规则扩大当前概率间隔，但新规则纯度来自反复使用的验证集，尚无独立实验支持跨格式鲁棒性。
这一模型包含两个分类器与规则，之后的单神经网络路线单独报告。

## 云平台运行

```bash
git fetch origin
git switch release/v2.2-semantic
git pull --ff-only
```

/root/work应含train.parquet、valid_input.parquet、valid_answer_private.parquet。

```bash
mkdir -p artifacts/v2_2_hybrid
nohup bash scripts/run.sh /root/work > artifacts/v2_2_hybrid/nohup.log 2>&1 &
echo $! > artifacts/v2_2_hybrid/train.pid
tail -f artifacts/v2_2_hybrid/nohup.log
```


模型、预处理配置、metrics.json、valid_predictions.parquet、manifest、环境和提交号一起保存。

## 复现范围

上述指标来自已经返回的完整实验输出。本轮仓库整理未重新执行云端全量训练，本地检查不能证明新提交逐位复现原指标。改变特征列名后应重新生成对应特征并重训；不能只改文件名作为复现。

[训练环境](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/ENVIRONMENT.md) · [评分公式](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/SCORING.md) · [完整开发日志](https://github.com/niucurry/soc-threat-detection-sf-2026-02/blob/main/docs/DEVELOPMENT_LOG_STANDARD.md)
