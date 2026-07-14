# Chart map

| Report section | Analytical question | Chart family | Fields | Supported claim |
| --- | --- | --- | --- | --- |
| 核心系统性能 | 哪个策略排空最快、吞吐最高？ | Grouped small-multiple bars | makespan, token/s, wait, service, completion | 系统效率与请求时间的权衡 |
| 请求完成时间与连续性 | 请求启动后是否被搁置？ | Grouped small-multiple bars | P95 service, max gap, dormant fraction, overhead | RH-SAIL 显著改善 SAILP 连续性 |
| Device 占用与负载 | GPU 是否长时间空闲？ | Interval timeline | op start/end by device | 差异主要不来自整体 GPU 空闲 |
| Device 占用与负载 | 各卡运行时间是否均衡？ | Grouped bars | device busy ratio | 五种策略均保持较高设备占用 |
| 各数据集 Service Time | 改进是否局限于特定任务？ | Grouped log-scale bars | average service by dataset | 不同 DAG 结构对策略反应不同 |
| 新策略 GPU 硬件采样 | 两个新策略的硬件负载是否异常？ | Grouped bars | utilization, memory, power | 两次运行均保持高负载且无 OOM |
