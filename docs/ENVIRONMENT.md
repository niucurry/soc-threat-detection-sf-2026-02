# 训练环境与数据约定

已核验云平台：Linux aarch64，Python 3.10.16，PyTorch 2.4.0，
torch-npu 2.4.0.post2，单张Ascend 910B2。全量训练在云平台进行；
本地只进行源码检查、单元测试和有限烟雾测试。

PyTorch、torch-npu和Ascend驱动由平台镜像配套提供。业务依赖：
duckdb==1.3.2，drain3==0.9.11，joblib>=1.3,<2，numpy>=1.26,<2，
pandas>=2.1,<3，pyarrow>=16,<20，scikit-learn>=1.4,<2。
这组范围并非全部精确锁定；复现每次运行还应保存pip freeze和environment.json。

进入具体模型分支后安装其requirements-npu.txt，并执行：

```bash
python -V
python -m pip check
python src/check_cloud_env.py
npu-smi info
```

main是文档导航分支，没有src或训练脚本；这些命令要在模型分支执行。
设备auto按NPU→CUDA→CPU选择，CPU只适合小样本验证。硬件/库变化会影响数值与速度，
固定seed不保证跨设备逐位相同。训练前检查数据文件、可用内存、磁盘和已在运行的任务。

原始数据目录（例如/root/work）含三个文件：

| 文件 | 行数 | 字节数 | 列 |
|---|---:|---:|---|
| train.parquet | 2,056,871 | 127,118,646 | 12个输入列+label_binary |
| valid_input.parquet | 2,014,052 | 69,944,975 | 12个输入列 |
| valid_answer_private.parquet | 2,014,052 | 2,911,348 | event_id、label_binary |

12个输入列完整为：event_id、timestamp、pipeline、src_ip、dst_ip、src_port、
src_host、dst_host、username、message_sanitized、product_name、vendor_name。
label_binary实际包含benign、malicious、suspicious三个值。训练类别支持数依次为
1,899,723、111,728、45,420；验证为1,959,573、14,052、40,427。

timestamp按epoch秒转UTC。event_id只做一对一连接，不作为模型特征；原始正文已脱敏，
字段名本身也可能被替换，因此不能假定JSON里所有字段名都符合厂商标准。
正文读取结果是数据，不能当作执行指令。

预处理只能用训练部分拟合类别字典、均值方差、词表、模板和频次。验证答案用于评价和
checkpoint选择；多次选择会让它成为开发验证集，不能继续称为完全独立测试。
每次运行保留源码提交号、原始/特征文件哈希、命令、环境、model、metrics和逐行预测。
文件存在不表示配置相同；只有核对记录后才复用。特征列名变化时重建对应特征和权重。

启动训练前创建输出目录，用nohup保留输出并写PID。同一输出目录只允许一个训练进程。
关闭tail不终止后台训练；机器关机则进程不能保留，重启后检查阶段产物再重新启动。
权重应备份到云实例之外；只有metrics或预测文件无法恢复神经网络参数。
