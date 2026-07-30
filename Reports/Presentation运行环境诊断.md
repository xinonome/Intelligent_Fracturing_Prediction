# Presentation 运行环境诊断

## 结论

- 当前任务已加载 `Presentations` Skill，Skill目录、模板跟随脚本、渲染工具和API文档均存在。
- `@oai/artifact-tool` 已由平台预装，并非项目依赖缺失，也不是npm公共包安装问题。
- 预装位置：`C:\Users\xinonome\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\@oai\artifact-tool`。
- 版本：`2.8.33`；主入口 `dist\artifact_tool.mjs` 存在，依赖目录完整。
- 初次失败属于Windows平台运行时路径解析问题：当前进程的 `HOME` 为空，辅助脚本采用 `process.env.HOME || process.cwd()`，因而误查 `C:\Workspace\Intelligent_Fracturing_Prediction\.cache\codex-runtimes`，没有回退到有效的 `USERPROFILE=C:\Users\xinonome`。
- 临时设置进程级 `HOME=C:\Users\xinonome` 后，workspace初始化、`require.resolve()`和ESM导入全部成功。未修改永久环境变量。

## 环境信息

```text
Node: v24.12.0
npm: 11.6.2
工作目录: C:\Workspace\Intelligent_Fracturing_Prediction
NODE_PATH: 空
npm root -g: C:\Users\xinonome\AppData\Roaming\npm\node_modules
HOME: 空
USERPROFILE: C:\Users\xinonome
```

默认模块搜索路径：

```text
C:\Workspace\Intelligent_Fracturing_Prediction\node_modules
C:\Workspace\node_modules
C:\node_modules
```

因此默认执行：

```powershell
node -p "require.resolve('@oai/artifact-tool')"
```

会返回 `MODULE_NOT_FOUND`。使用平台预装运行时路径后可解析为：

```text
C:\Users\xinonome\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\@oai\artifact-tool\dist\artifact_tool.mjs
```

## Skill与工具状态

已加载Skill：

```text
C:\Users\xinonome\.codex\plugins\cache\openai-primary-runtime\presentations\26.727.11326\skills\presentations
```

包含：

- `SKILL.md`
- `style_guidelines.md`
- `artifact_tool_docs/`
- `container_tools/`
- `template_following_scripts/`
- `assets/`

当前没有连接中的PowerPoint Document Control会话，因此本地PPTX编辑应使用Presentation Skill的本地artifact-tool流程。

## 引用搜索结果

`@oai/artifact-tool`主要被以下预装文件引用：

- Presentation Skill规范与API示例。
- `container_tools/artifact_tool_utils.mjs`运行时定位和workspace初始化逻辑。
- `template_following_scripts`模板检查、复制和导出流程。
- Codex Grid 26种版式模块。

项目业务源码中未发现对该包的直接引用，说明它属于Codex平台演示文稿运行时，而非智能压裂项目依赖。

## 后续正确启动方式

在PowerShell进程内临时设置HOME，再调用Skill脚本：

```powershell
$env:HOME = $env:USERPROFILE
node "$skill\container_tools\setup_artifact_tool_workspace.mjs" --workspace "$tmp"
```

该方式仅修正当前进程解析路径，不安装npm包，不修改系统或用户永久环境变量。
