from __future__ import annotations

from datetime import date
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path.home() / ".codex" / "skills" / "frontend-design" / "assets" / "preview-template.html"
TOPIC = "application-oriented-native-pyfrac"
OUTPUT = ROOT / ".frontend-design" / TOPIC / f"{date.today().isoformat()}-application-oriented-native-pyfrac.html"


EXTRA_CSS = r"""
  .theme-light { --background:#F3F7FB; --topbar:#FFFFFF; --sidebar:#EAF2FA; --panel:#FFFFFF; --panel-alt:#F7FAFD; --border:#C9D9E8; --cyan:#0B9EA5; --blue:#2878C8; --orange:#D78900; --red:#D24B4B; --text:#17324D; --muted:#607C96; --surface-page:var(--background); --surface-panel:var(--panel); --surface-subtle:var(--sidebar); --border-subtle:var(--border); --text-primary:var(--text); --text-muted:var(--muted); --accent-primary:var(--blue); --accent-success:var(--cyan); --accent-warning:var(--orange); --accent-danger:var(--red); }
  .theme-dark { --background:#0D151D; --topbar:#101F2B; --sidebar:#132633; --panel:#1B2A36; --panel-alt:#223542; --border:#344B5A; --cyan:#20C7C2; --blue:#4D9DE0; --orange:#F2A93B; --red:#E05252; --text:#E8F0F5; --muted:#9EB2C1; --surface-page:var(--background); --surface-panel:var(--panel); --surface-subtle:var(--sidebar); --border-subtle:var(--border); --text-primary:var(--text); --text-muted:var(--muted); --accent-primary:var(--blue); --accent-success:var(--cyan); --accent-warning:var(--orange); --accent-danger:var(--red); }
  .app-shell { border:1px solid var(--border-subtle); border-radius:var(--radius-shell); overflow:hidden; background:var(--surface-page); min-height:760px; }
  .product-bar { display:flex; align-items:center; justify-content:space-between; padding:10px 14px; background:var(--surface-panel); border-bottom:1px solid var(--border-subtle); }
  .product-actions { display:flex; align-items:center; gap:8px; }
  .global-dataset-control { display:flex; align-items:center; gap:5px; color:var(--text-muted); font-size:var(--font-meta); }
  .global-dataset-control select { min-width:190px; }
  .product-name { color:var(--accent-primary); font-weight:750; font-size:var(--font-panel-title); }
  .product-meta { color:var(--text-muted); font-size:var(--font-meta); }
  .theme-toggle { border:1px solid var(--border-subtle); border-radius:999px; padding:5px 9px; background:var(--surface-subtle); color:var(--text-muted); font:inherit; font-size:var(--font-meta); cursor:pointer; }
  .theme-toggle:hover { background:var(--surface-page); color:var(--text-primary); border-color:var(--accent-primary); }
  .product-body { display:grid; grid-template-columns:154px 1fr; min-height:710px; }
  .app-nav { background:var(--surface-subtle); border-right:1px solid var(--border-subtle); padding:10px 8px; }
  .app-nav button { display:block; width:100%; text-align:left; border:0; border-left:3px solid transparent; background:transparent; color:var(--text-muted); padding:10px 9px; border-radius:6px; margin-bottom:5px; font:inherit; font-size:var(--font-meta); cursor:pointer; }
  .app-nav button.active { color:var(--accent-primary); background:var(--surface-panel); border-left-color:var(--accent-primary); font-weight:700; }
  .page-stack { padding:14px; min-width:0; }
  .app-page { display:none; }
  .app-page.active { display:block; }
  .page-head { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; margin-bottom:10px; }
  .page-name { font-size:var(--font-app-title); font-weight:750; line-height:1.15; }
  .page-flow { color:var(--text-muted); font-size:var(--font-meta); margin-top:3px; }
  .toolbar { background:var(--surface-panel); border:1px solid var(--border-subtle); border-radius:var(--radius-default); padding:9px; display:flex; flex-wrap:wrap; gap:7px; align-items:center; margin-bottom:9px; }
  .toolbar select,.toolbar input { border:1px solid var(--border-subtle); border-radius:6px; padding:5px 7px; background:var(--surface-page); color:var(--text-primary); font:inherit; font-size:var(--font-meta); }
  .tool-label { color:var(--text-muted); font-size:var(--font-meta); }
  .work-grid { display:grid; grid-template-columns:2fr 1fr; gap:8px; margin-bottom:8px; }
  .work-grid.equal { grid-template-columns:1fr 1fr; }
  .work-panel { background:var(--surface-panel); border:1px solid var(--border-subtle); border-radius:var(--radius-default); padding:10px; min-height:170px; }
  .panel-title { font-size:var(--font-dense); font-weight:700; margin-bottom:7px; }
  .chart { height:150px; border-radius:6px; background:linear-gradient(var(--border-subtle) 1px,transparent 1px) 0 0/100% 37px; position:relative; overflow:hidden; }
  .chart svg { width:100%; height:100%; }
  .legend { display:flex; gap:10px; flex-wrap:wrap; color:var(--text-muted); font-size:10px; margin-bottom:4px; }
  .legend i { display:inline-block; width:13px; height:2px; margin-right:4px; vertical-align:middle; }
  .timeline-bar { height:6px; background:var(--border-subtle); border-radius:999px; flex:1; min-width:100px; overflow:hidden; }
  .timeline-bar::before { content:""; display:block; width:42%; height:100%; background:var(--accent-primary); }
  .event-band { position:absolute; top:0; bottom:0; background:color-mix(in srgb,var(--accent-warning) 18%,transparent); border-left:1px dashed var(--accent-warning); }
  .risk-band { height:82px; display:grid; grid-template-rows:repeat(3,1fr); gap:3px; background:var(--surface-subtle); padding:6px; border-radius:6px; }
  .risk-row { position:relative; color:var(--text-muted); font-size:10px; padding-left:40px; }
  .risk-row span { position:absolute; top:2px; bottom:2px; border-radius:3px; opacity:.75; }
  .native-state { display:grid; grid-template-columns:1.1fr 1fr; gap:8px; }
  .field-map { min-height:220px; border-radius:6px; background:radial-gradient(ellipse at center,var(--accent-warning) 0 2%,transparent 3%),radial-gradient(ellipse at center,color-mix(in srgb,var(--accent-primary) 65%,transparent) 0 18%,transparent 42%),var(--surface-subtle); border:1px solid var(--border-subtle); position:relative; }
  .field-map::after { content:'真实压力场采样与裂缝前缘 · 当前内部点 370/370'; position:absolute; left:10px; bottom:8px; color:var(--text-muted); font-size:10px; }
  .fracture-stage { height:230px; display:flex; align-items:flex-end; justify-content:space-around; padding:18px 28px; background:linear-gradient(180deg,var(--surface-subtle),var(--surface-panel)); border-radius:6px; }
  .cluster { width:8px; height:44px; background:var(--accent-primary); border-radius:99px; position:relative; }
  .cluster::before,.cluster::after { content:""; position:absolute; bottom:4px; width:32px; height:70px; border:2px solid var(--accent-success); border-radius:55% 45% 50% 50%; }
  .cluster::before { right:5px; transform:skewY(-18deg); } .cluster::after { left:5px; transform:skewY(18deg); }
  .cluster:nth-child(2n)::before,.cluster:nth-child(2n)::after { height:95px; }
  .record-table { width:100%; border-collapse:collapse; font-size:10px; }
  .record-table th,.record-table td { border-bottom:1px solid var(--border-subtle); padding:5px; text-align:left; }
  .record-table th { color:var(--text-muted); font-weight:600; }
  .state-note { padding:7px 9px; border:1px solid var(--border-subtle); background:var(--surface-subtle); border-radius:6px; color:var(--text-muted); font-size:var(--font-meta); }
  .button-row { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .disabled { opacity:.48; }
  .inner-tabs { display:flex; gap:4px; margin-bottom:9px; border-bottom:1px solid var(--border-subtle); }
  .inner-tab { border:0; border-bottom:2px solid transparent; background:transparent; color:var(--text-muted); padding:7px 12px; font:inherit; font-size:var(--font-meta); cursor:pointer; }
  .inner-tab.active { color:var(--accent-primary); border-bottom-color:var(--accent-primary); font-weight:700; }
  .kg-preview { height:330px; display:grid; grid-template-columns:1fr 180px; gap:8px; }
  .kg-canvas { position:relative; overflow:hidden; border:1px solid var(--border-subtle); border-radius:6px; background:var(--surface-subtle); }
  .kg-canvas svg { width:100%; height:100%; }
  .kg-canvas circle { cursor:grab; }
"""

REAL_TOKENS = r"""const TOKENS = {
  surface: [
    { key:'--background', intent:'theme.py / background', kind:'color', options:[
      {id:'A',light:'#F3F7FB',dark:'#0D151D',note:'当前正式值 (default)'},
      {id:'B',light:'#EEF5FC',dark:'#0B1722',note:'对比方案'}] },
    { key:'--panel', intent:'theme.py / panel', kind:'color', options:[
      {id:'A',light:'#FFFFFF',dark:'#1B2A36',note:'当前正式值 (default)'},
      {id:'B',light:'#F8FBFE',dark:'#20313E',note:'弱化面板'}] },
    { key:'--sidebar', intent:'theme.py / sidebar', kind:'color', options:[
      {id:'A',light:'#EAF2FA',dark:'#132633',note:'当前正式值 (default)'},
      {id:'B',light:'#E3EEF8',dark:'#162B39',note:'增强层级'}] },
    { key:'--border', intent:'theme.py / border', kind:'color', options:[
      {id:'A',light:'#C9D9E8',dark:'#344B5A',note:'当前正式值 (default)'},
      {id:'B',light:'#B9CEDF',dark:'#405967',note:'增强边界'}] }
  ],
  accent: [
    { key:'--blue', intent:'theme.py / blue', kind:'color', options:[
      {id:'A',light:'#2878C8',dark:'#4D9DE0',note:'当前正式值 (default)'},
      {id:'B',light:'#1769AA',dark:'#63A9E4',note:'更稳重'}] },
    { key:'--cyan', intent:'theme.py / cyan', kind:'color', options:[
      {id:'A',light:'#0B9EA5',dark:'#20C7C2',note:'当前正式值 (default)'},
      {id:'B',light:'#087E86',dark:'#39D2CC',note:'降低饱和度'}] },
    { key:'--orange', intent:'theme.py / orange', kind:'color', options:[
      {id:'A',light:'#D78900',dark:'#F2A93B',note:'当前正式值 (default)'},
      {id:'B',light:'#B97800',dark:'#F0B429',note:'警示色'}] },
    { key:'--red', intent:'theme.py / red', kind:'color', options:[
      {id:'A',light:'#D24B4B',dark:'#E05252',note:'当前正式值 (default)'},
      {id:'B',light:'#B93D3D',dark:'#ED6A6A',note:'风险强调'}] }
  ],
  typography: [
    {key:'--font-app-title',intent:'theme.py / pageTitle',kind:'slider',min:20,max:28,step:1,defaultValue:24,unit:'px'},
    {key:'--font-panel-title',intent:'theme.py / sectionTitle',kind:'slider',min:13,max:18,step:1,defaultValue:15,unit:'px'},
    {key:'--font-body',intent:'theme.py / base font',kind:'slider',min:12,max:16,step:1,defaultValue:13,unit:'px'}
  ],
  radius: [
    {key:'--radius-default',intent:'theme.py / control radius',kind:'slider',min:0,max:10,step:1,defaultValue:4,unit:'px'},
    {key:'--radius-shell',intent:'theme.py / panel radius',kind:'slider',min:2,max:14,step:1,defaultValue:7,unit:'px'}
  ],
  density: [
    {key:'--row-density',intent:'Qt workbench spacing',kind:'segmented',options:[
      {id:'compact',value:'4px',note:'紧凑'},
      {id:'normal',value:'6px',note:'当前正式值 (default)'},
      {id:'spacious',value:'9px',note:'宽松'}],defaultId:'normal'}
  ]
};"""


def chart(lines: list[tuple[str, str, str]], bands: bool = False) -> str:
    legend = "".join(f'<span><i style="background:{color}"></i>{name}</span>' for name, color, _ in lines)
    paths = "".join(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>' for _, color, points in lines)
    band = '<span class="event-band" style="left:38%;width:12%"></span><span class="event-band" style="left:72%;width:8%"></span>' if bands else ""
    return f'<div class="legend">{legend}</div><div class="chart">{band}<svg viewBox="0 0 500 150" preserveAspectRatio="none">{paths}<line x1="210" y1="0" x2="210" y2="150" stroke="var(--accent-warning)" stroke-dasharray="4 4"/></svg></div>'


PRESSURE = [
    ("观测压力", "var(--accent-primary)", "0,112 35,108 70,105 105,110 140,83 175,65 210,70 245,55 280,61 315,48 350,54 385,47 420,72 455,58 500,64"),
    ("PKN先验", "var(--accent-warning)", "0,116 35,111 70,108 105,109 140,98 175,82 210,75 245,64 280,58 315,50 350,48 385,50 420,52 455,49 500,51"),
    ("EnKF后验", "var(--accent-success)", "0,114 35,109 70,107 105,109 140,94 175,78 210,72 245,60 280,56 315,49 350,50 385,49 420,54 455,50 500,52"),
]
FLOW = [
    ("当前值", "var(--accent-success)", "0,125 60,125 60,110 125,110 125,70 210,70 210,88 290,88 290,48 385,48 385,78 455,78 455,116 500,116"),
    ("建议值", "var(--accent-primary)", "0,118 60,118 60,102 125,102 125,62 210,62 210,80 290,80 290,43 385,43 385,72 455,72 455,108 500,108"),
]
SAND = [
    ("当前值", "var(--accent-warning)", "0,132 80,132 80,98 170,98 170,128 260,128 260,82 350,82 350,48 420,48 420,120 500,120"),
    ("建议值", "var(--accent-danger)", "0,129 80,129 80,95 170,95 170,126 260,126 260,80 350,80 350,45 420,45 420,114 500,114"),
]


def app_markup(theme: str) -> str:
    pane_symbol = "▶" if theme == "light" else "◀"
    theme_toggle_text = "☀ 浅色" if theme == "light" else "☾ 暗色"
    return f'''<div class="preview theme-{theme}" id="preview-{theme}" data-theme="{theme}">
      <button class="pane-toggle-btn" type="button" data-pane="{theme}" title="Expand {theme} pane">{pane_symbol}</button>
      <div class="label" data-fd-id="theme-label">{theme.title()} theme · 正式 APP 工作台</div>
      <div class="app-shell" data-fd-id="application-shell">
        <div class="product-bar"><div><div class="product-name" data-fd-id="product-title" data-fd-editable="text">智能压裂精准调控平台</div><div class="product-meta">当前井段由统一数据选择器控制</div></div><div class="product-actions"><div class="global-dataset-control" data-fd-id="global-dataset-selector"><span>数据井段</span><select data-fd-editable="value"><option>焦页84-Z1 · Stage 08（压力 + DAS）</option><option>FDBH1 · 单井段（无 DAS）</option><option>FDBH16 · 单井段（无 DAS）</option></select></div><button class="theme-toggle" type="button" data-theme-toggle="{theme}">{theme_toggle_text}</button><span class="pill" data-tone="info"><span class="dot"></span>浅色 / 暗色可切换</span></div></div>
        <div class="product-body">
          <nav class="app-nav" data-fd-id="application-navigation">
            <button class="active" data-page="fsl">工况识别与风险预测</button>
            <button data-page="dt">裂缝数字孪生</button>
            <button data-page="hmi">智能决策与安全建议</button>
            <button data-page="integrated">联合动态演示</button>
          </nav>
          <main class="page-stack">
            <section class="app-page active" data-page-panel="fsl" data-fd-id="page-fsl">
              <div class="page-head"><div><div class="page-name" data-fd-editable="text">第一部分 · 工况识别与风险预测</div><div class="page-flow">选择井段 → 分析 → 回放事件 → 导出</div></div></div>
              <div class="inner-tabs" data-fd-id="fsl-subtabs"><button class="inner-tab active" data-subpage="prediction">预测功能</button><button class="inner-tab" data-subpage="knowledge">知识图谱功能</button><button class="inner-tab" data-subpage="data-import">数据导入</button></div>
              <div class="subpage-panel" data-subpage-panel="prediction">
              <div class="toolbar" data-fd-id="fsl-stage-toolbar"><span class="tool-label">当前井段</span><span class="state-note">跟随窗口顶部数据井段选择</span><button class="preview-btn primary">重新读取与分析</button><button class="preview-btn">导出图表</button><button class="preview-btn">导出事件报告</button></div>
              <div class="toolbar"><button class="preview-btn primary">播放</button><button class="preview-btn">重置</button><button class="preview-btn">上一步</button><button class="preview-btn">下一步</button><div class="timeline-bar"></div><span class="tool-label">t=当前帧 / 井段总时长</span></div>
              <div class="state-note" data-fd-id="fsl-condition-model-state">工况实测标签与事件边界已接入；逐点工况预测模型尚未形成可按井段定位的正式产物，因此不绘制假预测。可在下方迁移工作台训练并应用。</div>
              <div class="work-grid"><div class="work-panel" data-fd-id="fsl-timeline"><div class="panel-title">施工压力、排量、砂比与逐点预测</div>{chart(PRESSURE[:1] + [("压力逐点预测", "var(--accent-info)", "0,114 35,110 70,107 105,109 140,86 175,67 210,72 245,58 280,62 315,50 350,56 385,49 420,70 455,60 500,65")], True)}</div><div class="work-panel"><div class="panel-title">事件区间</div><table class="record-table"><tr><th>类型</th><th>边界来源</th><th>操作</th></tr><tr><td>主缝延伸</td><td>所选井段</td><td>查看曲线</td></tr><tr><td>砂堵</td><td>分析结果</td><td>重新标注</td></tr><tr><td>压力异常</td><td>分析结果</td><td>导出</td></tr></table></div></div>
              <div class="work-panel" data-fd-id="cross-well-transfer"><div class="panel-title">跨井迁移实验</div><div class="toolbar"><span class="tool-label">迁移井</span><select><option>焦页109-3HF</option></select><span class="tool-label">目标井</span><select><option>焦页171-1HF</option></select><select><option>重新训练模型</option><option>使用已有模型</option></select><button class="preview-btn primary">开始训练</button><button class="preview-btn">应用到目标井</button></div><div class="state-note">两口井数据可用；训练结果写入独立运行目录，不修改原始数据。</div></div>
              </div>
              <div class="subpage-panel" data-subpage-panel="knowledge" style="display:none">
                <div class="toolbar"><span class="tool-label">知识图谱</span><button class="preview-btn primary">逐步展开</button><button class="preview-btn">重置</button><input placeholder="搜索节点"><button class="preview-btn">定位</button></div>
                <div class="toolbar"><span class="tool-label">API 地址</span><input placeholder="OpenAI 兼容接口地址"><span class="tool-label">模型</span><input placeholder="模型名称"><button class="preview-btn">保存 API 配置</button></div>
                <div class="kg-preview"><div class="kg-canvas"><svg viewBox="0 0 560 300" preserveAspectRatio="none"><g stroke="var(--border-subtle)" stroke-width="2"><line x1="280" y1="145" x2="130" y2="70"/><line x1="280" y1="145" x2="430" y2="70"/><line x1="280" y1="145" x2="130" y2="235"/><line x1="280" y1="145" x2="430" y2="235"/></g><g fill="var(--accent-primary)"><circle cx="280" cy="145" r="22"/><circle cx="130" cy="70" r="16"/><circle cx="430" cy="70" r="16"/><circle cx="130" cy="235" r="16"/><circle cx="430" cy="235" r="16"/></g><g fill="var(--text-primary)" font-size="13" text-anchor="middle"><text x="280" y="180">砂堵风险</text><text x="130" y="100">施工压力</text><text x="430" y="100">排量</text><text x="130" y="265">处置措施</text><text x="430" y="265">施工工况</text></g></svg></div><div class="work-panel"><div class="panel-title">当前节点</div><div class="state-note">点击节点查看实体类型、关联关系和原文证据。</div><div class="panel-title" style="margin-top:10px">导入数据并解析</div><button class="preview-btn primary">选择资料</button><button class="preview-btn">导入并解析</button><div class="state-note" style="margin-top:8px">支持 JSON、CSV、TXT、MD、DOCX；解析结果可用于本地问答。</div></div></div>
                <div class="work-panel" style="margin-top:8px"><div class="panel-title">询问并回答</div><div class="toolbar"><input placeholder="例如：每口井出现了哪些工况？"><button class="preview-btn primary">查询本地知识</button><button class="preview-btn">调用 API 回答</button></div><div class="state-note">未配置 API 时使用已保存的本地知识与导入资料回答。</div></div>
              </div>
              <div class="subpage-panel" data-subpage-panel="data-import" style="display:none">
                <div class="toolbar" data-fd-id="fsl-data-import-toolbar"><button class="preview-btn primary">选择表格</button><button class="preview-btn">移除选中</button><button class="preview-btn">清空列表</button><button class="preview-btn primary">导入可识别文件</button></div>
                <div class="work-panel"><div class="panel-title">待导入文件与字段检查</div><table class="record-table"><tr><th>文件</th><th>格式</th><th>识别结果</th><th>可用场景</th></tr><tr><td>FDBH26.xlsx</td><td>.xlsx</td><td>标准施工表</td><td>工况识别、无 DAS 原始登记</td></tr></table></div>
              </div>
            </section>
            <section class="app-page" data-page-panel="dt" data-fd-id="page-dt">
              <div class="page-head"><div><div class="page-name">第二部分 · 光纤驱动裂缝数字孪生</div><div class="page-flow">选择场景与井段 → 播放 → 参数更新 → 回退</div></div></div>
              <div class="toolbar" data-fd-id="dt-model-toolbar"><span class="tool-label">模型</span><select><option>PyFrac原生模型 · 真实离线推演</option><option>当前在线 PKN + KG-EnKF</option><option disabled>PyFrac代理模型 · 质量门未通过</option></select><span class="tool-label">当前井段</span><span class="state-note">跟随窗口顶部数据井段选择</span></div>
              <div class="toolbar" data-fd-id="pyfrac-parameter-controls"><span class="tool-label">最小水平应力</span><input value="112.50 MPa"><span class="tool-label">目标模拟时间</span><input value="4435 s"><button class="preview-btn primary">开始重新推演</button><button class="preview-btn">停止计算</button><button class="preview-btn">保存参数方案</button><button class="preview-btn">回退参数方案</button></div>
              <div class="state-note">已完成真实原生运行：370 个自适应内部计算点；370 个点保存压力场与裂缝前缘。每次参数变化均从初始状态重算。</div>
              <div class="toolbar"><button class="preview-btn primary">播放</button><button class="preview-btn">上一个内部点</button><button class="preview-btn">下一个内部点</button><div class="timeline-bar"></div><span class="tool-label">内部点 370/370 · t=4435 s</span></div>
              <div class="native-state"><div class="work-panel" data-fd-id="pyfrac-field-view"><div class="panel-title">PyFrac内生压力场与裂缝前缘</div><div class="field-map"></div></div><div><div class="work-panel"><div class="panel-title">真实自适应时间步</div>{chart([("内部Δt","var(--accent-primary)","0,140 30,112 55,135 80,72 110,128 140,46 175,118 205,40 235,105 270,75 305,130 340,32 380,100 420,58 460,120 500,65")])}</div><div class="work-panel" style="margin-top:8px"><div class="panel-title">终点方案对比 · 上次 / 本次</div>{chart([("上次净压力","var(--text-muted)","0,90 80,86 160,76 240,70 320,64 400,58 500,54"),("本次净压力","var(--accent-primary)","0,96 80,88 160,79 240,67 320,61 400,54 500,50")])}</div></div></div>
              <div class="work-grid equal" style="margin-top:8px"><div class="work-panel"><div class="panel-title">半缝长 / m</div>{chart([("本次半缝长","var(--accent-success)","0,140 80,132 160,120 240,104 320,83 400,56 500,22")])}</div><div class="work-panel"><div class="panel-title">最大缝宽 / mm</div>{chart([("本次最大缝宽","var(--accent-warning)","0,140 80,137 160,128 240,116 320,94 400,70 500,45")])}</div></div>
            </section>
            <section class="app-page" data-page-panel="hmi" data-fd-id="page-hmi">
              <div class="page-head"><div><div class="page-name">第三部分 · 智能决策与人机协同</div><div class="page-flow">查看状态 → 审核建议 → 确认、修改或撤销</div></div></div>
              <div class="toolbar" data-fd-id="hmi-data-context"><span class="tool-label">当前井段</span><span class="state-note">跟随窗口顶部数据井段选择</span><button class="preview-btn">实时预测并缓存</button><span class="state-note">完成后写入当前井段缓存</span></div>
              <div class="toolbar" data-fd-id="hmi-model-selector"><span class="tool-label">智能体模型</span><select><option>SAC · 真实离线回放</option><option>TD3 · 真实离线回放</option><option>PPO · 真实离线回放</option></select><span class="state-note">切换后替换第三部分的真实回放数据源</span></div>
              <div class="toolbar"><button class="preview-btn primary">播放</button><button class="preview-btn">上一步</button><button class="preview-btn">下一步</button><div class="timeline-bar"></div><span class="tool-label">t=1s / 240帧</span></div>
              <div class="work-panel" data-fd-id="hmi-risk-timeline"><div class="panel-title">实时风险状态带 · 点击区间查看原因及对建议的影响</div><div class="risk-band"><div class="risk-row">高风险<span style="left:72%;width:8%;background:var(--accent-danger)"></span></div><div class="risk-row">关注<span style="left:38%;width:12%;background:var(--accent-warning)"></span><span style="left:82%;width:9%;background:var(--accent-warning)"></span></div><div class="risk-row">正常<span style="left:8%;width:28%;background:var(--accent-success)"></span><span style="left:52%;width:18%;background:var(--accent-success)"></span></div></div></div>
              <div class="work-panel"><div class="panel-title">压力响应</div>{chart(PRESSURE)}</div>
              <div class="work-grid equal"><div class="work-panel"><div class="panel-title">排量 · 当前 / 建议</div>{chart(FLOW)}</div><div class="work-panel"><div class="panel-title">砂比 · 当前 / 建议</div>{chart(SAND)}</div></div>
              <div class="work-panel" data-fd-id="operator-review"><div class="panel-title">当前建议审核（不下发现场）</div><div class="toolbar"><span class="tool-label">审核排量</span><input value="5.52 m³/min"><span class="tool-label">审核砂比</span><input value="1.76%"><button class="preview-btn primary">确认采用</button><button class="preview-btn">暂不采用</button><button class="preview-btn">修改后采用</button><button class="preview-btn">撤销上次确认</button></div><div class="state-note">当前建议：保持控制量，等待下一观测更新后再次审核。</div></div>
            </section>
            <section class="app-page" data-page-panel="integrated" data-fd-id="page-integrated">
              <div class="page-head"><div><div class="page-name">联合动态演示</div><div class="page-flow">统一时间轴串联施工曲线、压力预测、裂缝演化和人工建议</div></div></div>
              <div class="toolbar"><select><option>有 DAS · 压力 + 分簇观测</option></select><span class="state-note">当前井段跟随窗口顶部数据井段选择</span><button class="preview-btn primary">播放</button><button class="preview-btn">暂停</button><button class="preview-btn">回到初始状态</button><div class="timeline-bar"></div></div>
              <div class="work-grid"><div><div class="work-panel"><div class="panel-title">压力对比</div>{chart(PRESSURE)}</div><div class="work-grid equal" style="margin-top:8px"><div class="work-panel"><div class="panel-title">排量</div>{chart(FLOW)}</div><div class="work-panel"><div class="panel-title">砂比</div>{chart(SAND)}</div></div></div><div class="work-panel"><div class="panel-title">同步裂缝状态</div><div class="fracture-stage">{''.join('<span class="cluster"></span>' for _ in range(6))}</div><div class="state-note" style="margin-top:8px">暂停后可调整视角；相机参数自动保存。</div></div></div>
            </section>
          </main>
        </div>
      </div>
    </div>'''


def build() -> Path:
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("<title>frontend-design preview</title>", "<title>智能压裂正式 APP · 应用导向重构</title>")
    html = html.replace("</style>", EXTRA_CSS + "\n</style>", 1)
    preview_pattern = re.compile(r"\s*<!-- LIGHT PREVIEW -->.*?</div>\s*</div>\s*</section>", re.S)
    replacement = "\n      <!-- REAL APPLICATION PREVIEW -->\n" + app_markup("light") + "\n" + app_markup("dark") + "\n      </div>\n    </section>"
    html, count = preview_pattern.subn(replacement, html, count=1)
    if count != 1:
        raise RuntimeError("preview region was not found in the frontend-design template")
    html, token_count = re.subn(
        r"const TOKENS = \{.*?\n\};\n\nconst SECTION_ORDER",
        REAL_TOKENS + "\n\nconst SECTION_ORDER",
        html,
        count=1,
        flags=re.S,
    )
    if token_count != 1:
        raise RuntimeError("token definition was not found in the frontend-design template")
    source = f".frontend-design/{TOPIC}/{OUTPUT.name}"
    html = html.replace("source: '.frontend-design/__topic__/__date__-__slug__.html'", f"source: '{source}'")
    html = html.replace("topic: '__topic__'", f"topic: '{TOPIC}'")
    html = html.replace("targetFile: '<your stylesheet path>'", "targetFile: 'App/ui/theme.py'")
    html = html.replace("<h1>frontend-design preview</h1>", "<h1>智能压裂正式 APP · 应用导向工作台</h1>")
    html = html.replace("Decision controls on the left · live preview on the right · click <strong>Comment Mode</strong> to annotate any element", "左侧调节蓝白/深色设计参数 · 右侧切换五个真实业务页面 · 可评论、编辑并导出 Markdown")
    html = html.replace("<strong>Hover</strong> a dropdown option to preview live.\n          <strong>Click</strong> to commit. Drag sliders for continuous values.\n          Enable <strong>Comment Mode</strong> in the header to annotate any element in the preview.", "调节颜色、字号、圆角和密度；点击右侧导航检查五个业务工作台。开启 <strong>Comment Mode</strong> 可直接标注意见。")
    html = html.replace("renderControls();\napplyAllDecisions();", """document.querySelectorAll('[data-theme-toggle]').forEach(button => {
  button.addEventListener('click', () => {
    const preview = button.closest('.preview');
    const isLight = preview.classList.contains('theme-light');
    preview.classList.toggle('theme-light', !isLight);
    preview.classList.toggle('theme-dark', isLight);
    preview.dataset.theme = isLight ? 'dark' : 'light';
    button.textContent = isLight ? '☾ 暗色' : '☀ 浅色';
  });
});
document.querySelectorAll('.inner-tabs').forEach(tabBar => {
  tabBar.querySelectorAll('.inner-tab').forEach(button => {
    button.addEventListener('click', () => {
      const page = tabBar.closest('.app-page');
      const subpage = button.dataset.subpage;
      tabBar.querySelectorAll('.inner-tab').forEach(item => item.classList.toggle('active', item === button));
      page.querySelectorAll('[data-subpage-panel]').forEach(panel => panel.style.display = panel.dataset.subpagePanel === subpage ? 'block' : 'none');
    });
  });
});
document.querySelectorAll('.app-nav button').forEach(button => {
  button.addEventListener('click', () => {
    const page = button.dataset.page;
    document.querySelectorAll('.app-nav button').forEach(item => item.classList.toggle('active', item.dataset.page === page));
    document.querySelectorAll('.app-page').forEach(panel => panel.classList.toggle('active', panel.dataset.pagePanel === page));
    setTimeout(renderMarkers, 20);
  });
});
renderControls();
applyAllDecisions();""")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
