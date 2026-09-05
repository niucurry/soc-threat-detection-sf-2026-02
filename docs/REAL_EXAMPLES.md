# 真实事件与当前确定性解析输出

以下四行于2026-09-05从train.parquet按event_id精确读取。input包含实际全部13列，message_sanitized完整保留；parsed为当前parse_log实际输出，未使用拟合模板模型，因此不伪造训练模板ID或频次。
这些样本不是新增训练实验，也不是模型预测。字段被脱敏时，其键名本身可能改变。

## EVT-0000113120 · ASA Firewall

明确格式识别为asa，直接得到deny、udp和目的端口53。可从语义字段建模，Drain不是字段抽取的来源。

完整原始输入：

```json
{
  "event_id": "EVT-0000113120",
  "timestamp": 1721992226.4811196,
  "pipeline": "syslog",
  "src_ip": "10.247.160.227",
  "dst_ip": "198.51.100.98",
  "src_port": "46422",
  "src_host": "10.232.175.9",
  "dst_host": "",
  "username": "",
  "message_sanitized": "<164>Jul 26 USER-9546 05:59:41: USER-0010-0324 Deny udp src dmz-2:10.202.238.40/46422 dst outside:100.64.54.208/53 by ORG-1738-group \"USER-6542\" [0x0, 0x0]\n",
  "product_name": "ASA Firewall",
  "vendor_name": "Cisco",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "asa",
  "semantic_action": "deny",
  "network_protocol": "udp",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": 46422,
  "dst_port": 53,
  "source_zone": "dmz-2",
  "destination_zone": "outside",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Cisco|ASA Firewall|asa",
  "normalized_message": "<<NUM>>Jul <NUM> <USER> <IPV6> <USER> Deny udp src dmz-<NUM>:<IP>/<NUM> dst outside:<IP>/<NUM> by <ORG> \"<USER>\" [0x0, 0x0]",
  "semantic_template": "format=asa action=deny protocol=udp",
  "semantic_field_count": 6,
  "is_auth_failure": 0,
  "is_network_denied": 1,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=53，field_token_count=59；每条编码上限96，原文长度156。这里的固定长度不代表全文所有字段均进入网络。

## EVT-0000134221 · Windows Active Directory

这是3893字符长日志，事件码4625与失败正文同时存在；原文字段名有脱敏扰动。完整原文与归一化片段不同，后者存在长度限制。

完整原始输入：

```json
{
  "event_id": "EVT-0000134221",
  "timestamp": 1721992263.8996067,
  "pipeline": "syslog",
  "src_ip": "192.168.147.151",
  "dst_ip": "",
  "src_port": "58382",
  "src_host": "USER-0010-0012.domain-0022.example.net",
  "dst_host": "",
  "username": "",
  "message_sanitized": "ORG-1657 ::: {\"@metaUSER-0010-54061\":{\"beat\":\"winlogbeat\",\"type\":\"_doc\",\"version\":\"8.2.2\"},\"@ORG-1526stamp\":\"USER-9546-07-26T11:10:08.448Z\",\"USER-0010-1577\":{\"ephemeral_id\":\"CRED-24CRED-25582381-3d10-4395-bc1c-5cdb52ecb35e\",\"id\":\"12e13a9a-252a-493f-9c35-fa4d61abd0bc\",\"name\":\"USER-0010-0206\",\"type\":\"winlogbeat\",\"version\":\"8.2.2\"},\"ecs\":{\"version\":\"8.0.0\"},\"USER-0010-56507\":{\"USER-0010-0121\":\"ORG-0106\",\"code\":\"4625\",\"created\":\"USER-9546-07-26T11:10:08.678Z\",\"kind\":\"USER-0010-56507\",\"outcome\":\"failure\",\"provider\":\"USER-8162-ORG-0407-Security-Auditing\"},\"USER-0010\":{\"name\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\"},\"log\":{\"level\":\"ORG-0706rmation\"},\"CRED-23501\":\"An account failed to log on.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tS-1-5-18\\n\\tAccount ORG-4643 ORG-CRED-25687\\USER-1430\\tORG-0106 ID:\\t\\t0x3E7\\n\\nORG-0106 Type:\\t\\t\\t3\\n\\nAccount For Which ORG-0106 Failed:\\n\\tSecurity ID:\\t\\tS-1-0-0\\n\\tAccount Name:\\t\\t\\n\\tAccount USER-24351:\\t\\t\\n\\USER-7624 ORG-0706rmation:\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\n\\tStatus:\\t\\t\\t0xC000006D\\n\\tSub Status:\\t\\t0xC0000064\\n\\USER-8912 ORG-0706rmation:\\n\\tCaller Process ID:\\t0x40c\\n\\tCaller Process Name:\\tC:\\\\ORG-0407\\\\USER-0010-1120\\\\ORG-3411\\USER-1430\\nORG-10379 ORG-0706rmation:\\n\\tWorkstation Name:\\tORG-0891-ORG-0023\\USER-1430\\USER-7625 ORG-10379 Address:\\t192.168.147.151\\n\\USER-7625 Port:\\t\\t58382\\n\\nDetailed ORG-0823 Process:\\t\\tSchannel\\n\\tAuthentication USER-0010-4214:\\tKerberos\\n\\tTransited USER-5935:\\t-\\n\\tUSER-0010-4214 Name (ORG-0504 only):\\t-\\n\\tKey Length:\\t\\t0\\n\\nUSER-9570 USER-0010-56507 is generated when a ORG-0106 USER-9484 fails. It is generated on the USER-3828 where ORG-1738 was atUSER-0010-1591ted.\\n\\nThe Subject fields indicate the account on the ORG-0462 USER-0086 which USER-9484ed the ORG-0106. USER-9570 is most USER-0010-54152ly a ORG-0090 such as the Server ORG-0090, or a ORG-0462 process such as ORG-0505 or USER-1847.\\n\\nThe ORG-0106 Type field indicates the kind of ORG-0106 that was USER-9484ed. The most USER-0010-54152 types are 2 (interCRED-24062) and 3 (ORG-10379).\\n\\nThe Process ORG-0706rmation fields indicate which account and process on the USER-0086 USER-9484ed ORG-2395 fields indicate where a ORG-4683 ORG-0106 USER-9484 originated. Workstation name is not always available and may be left blank in some cases.\\n\\nThe authentication ORG-0706rmation fields provide detailed ORG-0706rmation about USER-9570 specific ORG-0106 USER-9484.\\n\\t- Transited USER-5935 indicate which intermediate USER-5935 have participated in USER-9570 ORG-0106 ORG-0507 name indicates which sub-protocol was used among the ORG-0504 protocols.\\n\\t- Key length indicates the length of the generated session key. USER-9570 will be 0 if no session key was USER-9484ed.\",\"organization\":\"ORG-0003\",\"senderUSER-0010\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\",\"sensitivity\":\"normal\",\"winlog\":{\"api\":\"winUSER-0010-56507log\",\"channel\":\"Security\",\"USER-3828_name\":\"USER-0010-0012.USER-24351-0022.exampleUSER-8710\",\"USER-0010-56507_USER-0010-54061\":{\"AuthenticationUSER-0010-4214Name\":\"Kerberos\",\"FailureReason\":\"%%2313\",\"IpAddress\":\"192.168.141.126\",\"IpPort\":\"58382\",\"KeyLength\":\"0\",\"LmUSER-0010-4214Name\":\"-\",\"ORG-0106ProcessName\":\"Schannel\",\"ORG-0106Type\":\"3\",\"ProcessId\":\"0x40c\",\"ProcessName\":\"C:\\\\ORG-0407\\\\USER-0010-1120\\\\USER-0010-1155\",\"Status\":\"0xc000006d\",\"SubStatus\":\"0xc0000064\",\"SubjectUSER-24351Name\":\"ORG-0015\",\"SubjectORG-0106Id\":\"0x3e7\",\"SubjectUserName\":\"USER-0109\",\"SubjectCRED-0339id\":\"S-1-5-18\",\"USER-0010-4999CRED-0339id\":\"S-1-0-0\",\"TransmittedUSER-5935\":\"-\",\"WorkstationName\":\"USER-0010-0206\"},\"USER-0010-56507_id\":\"4625\",\"keywords\":[\"Audit Failure\"],\"opcode\":\"ORG-0706\",\"process\":{\"pid\":1036,\"thread\":{\"id\":8604}},\"provider_ORG-0515\":\"{54849625-5478-4994-a5ba-3e3b0328c30d}\",\"provider_name\":\"USER-8162-ORG-0407-Security-Auditing\",\"record_id\":6067660825,\"task\":\"ORG-0106\"}}",
  "product_name": "Windows Active Directory",
  "vendor_name": "Microsoft",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "windows_json",
  "semantic_action": "fail",
  "network_protocol": "__MISSING__",
  "event_code": "4625",
  "event_name": "logon_failure",
  "src_port_from_message": -1,
  "dst_port": -1,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Microsoft|Windows Active Directory|windows_json",
  "normalized_message": "<ORG> <IPV6> {\"@meta<USER>\":{\"beat\":\"winlogbeat\",\"type\":\"_doc\",\"version\":\"<NUM>.<NUM>\"},\"@<ORG>\":\"USER-<TIMESTAMP>\",\"<USER>\":{\"ephemeral_id\":\"<CREDENTIAL>-<UUID>\",\"id\":\"<UUID>\",\"name\":\"<USER>\",\"type\":\"winlogbeat\",\"version\":\"<NUM>.<NUM>\"},\"ecs\":{\"version\":\"<NUM>.<NUM>\"},\"<USER>\":{\"<USER>\":\"<ORG>\",\"code\":\"<NUM>\",\"created\":\"USER-<TIMESTAMP>\",\"kind\":\"<USER>\",\"outcome\":\"failure\",\"provider\":\"<USER>\"},\"<USER>\":{\"name\":\"<USER>.<USER>.example<USER>\"},\"log\":{\"level\":\"<ORG>\"},\"<CREDENTIAL>\":\"An account failed to log on.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tS-<NUM>-<NUM>-<NUM>\\n\\tAccount <ORG> <ORG>\\<USER>\\t<ORG> ID:\\t\\t0x3E7\\n\\n<ORG> Type:\\t\\t\\t3\\n\\nAccount For Which <ORG> Failed:\\n\\tSecurity ID:\\t\\tS-<NUM>-<NUM>-<NUM>\\n\\tAccount Name:\\t\\t\\n\\tAccount <USER>:\\t\\t\\n\\<USER> <ORG>:\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\n\\tStatus:\\t\\t\\t0xC000006D\\n\\tSub Status:\\t\\t0xC0000064\\n\\<USER> <ORG>:\\n\\tCaller Process ID:\\t0x40c\\n\\tCaller Process Name:\\tC:\\\\<ORG>\\\\<USER>\\\\<ORG>\\<USER>\\n<ORG> <ORG>:\\n\\tWorkstation Name:\\t<ORG>\\<USER>\\<USER> <ORG> Address:\\t<IP>\\n\\<USER> Port:\\t\\t58382\\n\\nDetailed <ORG> Process:\\t\\tSchannel\\n\\tAuthentication <USER>:\\tKerberos\\n\\tTransited <USER>:\\t-\\n\\t<USER> Name (<ORG> only):\\t-\\n\\tKey Length:\\t\\t0\\n\\n<USER> <USER> is generated when a <ORG> <USER> fails. It is generated on the <USER> where <ORG> was at<USER>.\\n\\nThe Subject fields indicate the account on the <ORG> <USER> which <USER> the <ORG>. <USER> is most <USER> a <ORG> such as the Server <ORG>, or a <ORG> process such as <ORG> or <USER>.\\n\\nThe <ORG> Type field indicates the kind of <ORG> that was <USER>. The most <USER> types are <NUM> (inter<CREDENTIAL>) and <NUM> (<ORG>).\\n\\nThe Process <ORG> fields indicate which account and process on the <USER> <USER> <ORG> fields indicate where a <ORG> <ORG> <USER> originated. Workstation name is not always available and may be left blank in some cases.\\n\\nThe authentication <ORG> fields provide detailed <ORG> about <USER> specific <ORG> <USER>.\\n\\t- Transited <USER> indicate which intermediate <U",
  "semantic_template": "format=windows_json event_code=4625 event_name=logon_failure action=fail keys=beat,type,version,ephemeral_id,id,name,ecs,code,created,kind",
  "semantic_field_count": 2,
  "is_auth_failure": 1,
  "is_network_denied": 0,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=96，field_token_count=96；每条编码上限96，原文长度3893。这里的固定长度不代表全文所有字段均进入网络。

## EVT-0000441159 · Duo

认证字段的实际键名含脱敏串，不能把它写成未出现的event_type/factor。no_response、duo_push、denied和auth_failure作为值仍存在，动作解析缺失并不表示正文无信息。

完整原始输入：

```json
{
  "event_id": "EVT-0000441159",
  "timestamp": 1721992859.0292988,
  "pipeline": "syslog",
  "src_ip": "",
  "dst_ip": "",
  "src_port": "",
  "src_host": "HOST-0019",
  "dst_host": "",
  "username": "USER-1343",
  "message_sanitized": "ORG-1780 ::: streamName=ORG-2927 ::: tags=[no_response] ::: CRED-23501={\"ORG-1738_USER-0010-1127\":{\"epkey\":null,\"USER-0010name\":null,\"ip\":\"100.64.52.151\",\"location\":{\"city\":null,\"country\":null,\"USER-0010-54774\":null}},\"alias\":\"\",\"application\":{\"key\":\"DIBE3VETN055FUSIDPC8\",\"name\":\"AWS USER-CRED-30678 VPN\"},\"auth_USER-0010-1127\":{\"ip\":null,\"key\":\"DPSH5S416UGQQL9GWA6G\",\"location\":{\"city\":null,\"country\":null,\"USER-0010-54774\":null},\"name\":\"412-370-9347\"},\"USER-0010-1CRED-23741\":\"user-0819@exampleUSER-8710\",\"USER-0010-56507_type\":\"authentication\",\"fUSER-0010-15196\":\"duo_push\",\"isoORG-1526stamp\":\"USER-9546-07-26T11:00:57.963780+00:00\",\"ood_USER-0010-1219\":null,\"reason\":\"no_response\",\"result\":\"denied\",\"ORG-1526stamp\":17219CRED-CRED-2980737,\"trusted_USER-0010-CRED-29699_status\":\"unknown\",\"txid\":\"67ca3e81-5815-4ed1-92a1-3616fa888a39\",\"user\":{\"groups\":[],\"key\":\"DUJCLW3N9SAXF4WG6Y0A\",\"name\":\"USER-1413\"}} ::: userName=user-0819@exampleUSER-8710 ::: USER-0010-1129=USER-5841 ::: CRED-23501Type=auth_failure",
  "product_name": "Duo",
  "vendor_name": "Cisco",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "json",
  "semantic_action": "__MISSING__",
  "network_protocol": "__MISSING__",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": -1,
  "dst_port": -1,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Cisco|Duo|json",
  "normalized_message": "<ORG> <IPV6> streamName=<ORG> <IPV6> tags=[no_response] <IPV6> <CREDENTIAL>={\"<ORG>_<USER>\":{\"epkey\":null,\"<USER>\":null,\"ip\":\"<IP>\",\"location\":{\"city\":null,\"country\":null,\"<USER>\":null}},\"alias\":\"\",\"application\":{\"key\":\"DIBE3VETN055FUSIDPC8\",\"name\":\"AWS <USER> VPN\"},\"auth_<USER>\":{\"ip\":null,\"key\":\"DPSH5S416UGQQL9GWA6G\",\"location\":{\"city\":null,\"country\":null,\"<USER>\":null},\"name\":\"<NUM>-<NUM>-<NUM>\"},\"<USER>\":\"<USER>@example<USER>\",\"<USER>_type\":\"authentication\",\"f<USER>\":\"duo_push\",\"iso<ORG>\":\"USER-<TIMESTAMP>\",\"ood_<USER>\":null,\"reason\":\"no_response\",\"result\":\"denied\",\"<ORG>\":<NUM><CREDENTIAL>,\"trusted_<USER>_status\":\"unknown\",\"txid\":\"<UUID>\",\"user\":{\"groups\":[],\"key\":\"DUJCLW3N9SAXF4WG6Y0A\",\"name\":\"<USER>\"}} <IPV6> userName=<USER>@example<USER> <IPV6> <USER>=<USER> <IPV6> <CREDENTIAL>=auth_failure",
  "semantic_template": "format=json keys=epkey,ip,location,city,country,alias,application,key,name,auth_user-0010-1127",
  "semantic_field_count": 0,
  "is_auth_failure": 0,
  "is_network_denied": 0,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=96，field_token_count=96；每条编码上限96，原文长度1007。这里的固定长度不代表全文所有字段均进入网络。

## EVT-0000600002 · AWS VPC Security

末尾REJECT OK中的OK是记录状态。当前粗解析得到reject，但协议仍为__MISSING__；不能因为第七列有6就声称此解析器实际输出tcp。

完整原始输入：

```json
{
  "event_id": "EVT-0000600002",
  "timestamp": 1721993140.7599907,
  "pipeline": "syslog",
  "src_ip": "100.64.0.237",
  "dst_ip": "10.216.192.18",
  "src_port": "48165",
  "src_host": "HOST-0031",
  "dst_host": "",
  "username": "",
  "message_sanitized": "2 100000013063 ORG-1504 100.64.0.237 10.182.224.117 48165 39878 6 1 40 1721992874 1721CRED-2CRED-3023300 REJECT OK",
  "product_name": "AWS VPC Security",
  "vendor_name": "Amazon Web Services",
  "label_binary": "suspicious"
}
```

全部确定性解析输出：

```json
{
  "message_format": "vpc_flow",
  "semantic_action": "reject",
  "network_protocol": "__MISSING__",
  "event_code": "__MISSING__",
  "event_name": "__MISSING__",
  "src_port_from_message": 48165,
  "dst_port": 39878,
  "source_zone": "__MISSING__",
  "destination_zone": "__MISSING__",
  "http_method": "__MISSING__",
  "http_status": -1,
  "parser_group": "syslog|Amazon Web Services|AWS VPC Security|vpc_flow",
  "normalized_message": "<NUM> <NUM> <ORG> <IP> <IP> <NUM> <NUM> <NUM> <NUM> <NUM> <NUM> <NUM><CREDENTIAL> REJECT OK",
  "semantic_template": "format=vpc_flow action=reject",
  "semantic_field_count": 3,
  "is_auth_failure": 0,
  "is_network_denied": 1,
  "is_process_creation": 0,
  "is_privileged_logon": 0
}
```

内容序列审计：raw_token_count=37，field_token_count=41；每条编码上限96，原文长度114。这里的固定长度不代表全文所有字段均进入网络。
