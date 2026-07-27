# 合同第二部分对应关系

1. 多源融合与时空同步：`data_fusion` 读取DAS分簇监测、施工压力和井轨迹。
2. 裂缝几何正演：`forward_models` 提供增强PKN、reduced BEM和数据代理统一接口。
3. EnKF实时修正：`inversion` 在物理参数空间更新后重新调用正演模型。
4. 双向闭环：`validate_direct_observations.py` 和 `visualization/digital_twin_3d.py` 完成验证与展示。

当前正式验证满足观测空间15%误差和15秒计算指标，但没有独立真实缝长标签。
