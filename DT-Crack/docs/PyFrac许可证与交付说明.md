# PyFrac 许可证与交付说明

本项目交付包允许包含 PyFrac 源码，但 PyFrac 是第三方 GPLv3 软件，不属于本项目原创代码。源码、许可证和版权信息放在：

```text
DT-Crack/third_party/PyFrac/
  GPL.txt
  LICENSE.TXT
  PROJECT_METADATA.json
  src/
  examples/
  docs/
```

本项目的适配代码放在 `DT-Crack/forward_models/pyfrac_adapter.py`，不修改 PyFrac 原始文件。交付或再分发时应保留 GPLv3 全文、原作者版权声明、版本/commit 信息，并由项目法务确认整体交付方式是否触发 GPLv3 的相应义务。

PyFrac在本系统中的用途是离线参考、教师数据和模型对照，不是现场控制器。在线闭环使用 PKN 或经验证的残差代理模型，EnKF更新物理参数后重新正演。
