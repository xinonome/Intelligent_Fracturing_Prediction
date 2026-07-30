import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ''));
const source = path.join(root, 'source.pptx');
const output = 'C:/Users/xinonome/Downloads/基于多源信息融合的压裂施工参数智能精准调控方法——结题验收汇报_合同二三部分版.pptx';
const renderDir = path.join(root, 'final-render');
await fs.mkdir(renderDir, { recursive: true });

const p = await PresentationFile.importPptx(await FileBlob.load(source));
const original = [...p.slides.items];
const keep = [1,2,3,4,7,8,9,10,11,12,13,15,17,18,19,21,22,23,24,25,26,28,29,30,31,32,33,34,40,41,43];
const refs = new Map(original.map((slide, index) => [index + 1, slide]));

for (let i = original.length; i >= 1; i--) {
  if (!keep.includes(i)) original[i - 1].delete();
}

function shape(slideNo, name) {
  const slide = refs.get(slideNo);
  const found = slide.shapes.items.find((item) => item.name === name);
  if (!found) throw new Error(`Missing shape ${name} on source slide ${slideNo}`);
  return found;
}

function replace(slideNo, name, value) {
  const target = shape(slideNo, name);
  const old = String(target.text ?? '');
  if (target.text?.replace) target.text.replace(old, value);
  else target.text = value;
}

function setText(slideNo, name, value) {
  shape(slideNo, name).text = value;
}

replace(2, 'info2', '项目定位：聚焦合同第二部分裂缝数字孪生与第三部分知识嵌入智能调控，形成“多源观测—物理正演—参数反演—策略决策—安全验证”闭环原型\n交付形态：算法源代码、真实数据适配器、模型参数、验证指标、三维可视化、智能体策略和统一调用接口');
setText(3, 'items', '一、合同第二、第三部分目标与验收指标\n二、合同第二部分：裂缝实时扩展数字孪生成果\n三、合同第三部分：知识嵌入智能调控成果\n四、工程化交付、技术边界与下一步工作');

replace(4, 'subtitle', '合同第二、第三部分形成“观测—反演—决策—验证”闭环');
replace(4, 'part1-title', '合同第二部分：数字孪生');
replace(4, 'part1-body', '· 多源融合与时空同步\n· 裂缝几何正演与反演');
replace(4, 'part2-title', '合同第三部分：智能调控');
replace(4, 'part2-body', '· 物理约束分层强化学习\n· 不确定性与人机协同');
replace(4, 'part3-title', '关键验收指标');
replace(4, 'part3-body', '· 反演误差≤15%，计算≤15秒\n· 5分钟预警、180秒安全验证');
replace(4, 'flow', 'DAS分簇监测、施工压力、井轨迹 → 时空同步与井底/净压力换算 → coupled PKN/BEM正演 → EnKF更新物理参数 → 裂缝状态重计算 → 300秒状态与60秒候选动作 → 分层强化学习 → 安全投影、人工确认与180秒验证');

replace(7, 'subtitle', '成果严格归入合同第二、第三部分八个小项');
replace(7, 'c1-title', '2（1）+ 2（2）');
replace(7, 'c1-body', '多源数据融合与时空同步\n高效裂缝几何正演/反演模型');
replace(7, 'c2-title', '2（3）+ 2（4）');
replace(7, 'c2-body', 'EnKF在线参数修正\n双向数字孪生闭环与3D展示');
replace(7, 'c3-title', '3（1）+ 3（2）');
replace(7, 'c3-body', '物理约束分层强化学习\n不确定性认知与决策引擎');
replace(7, 'c4-title', '3（3）+ 3（4）');
replace(7, 'c4-body', '可信可控人机协同接口\n多工况训练与180秒验证');

const titles = new Map([
  [8, '合同2（1）①：施工压力换算纳入多源数据标准化链'],
  [9, '合同2（1）②：光纤、压力和井轨迹对齐为统一时空记录'],
  [10, '合同2（2）①：PKN快速正演输出分簇裂缝几何状态'],
  [11, '合同2（2）②：Carter滤失与质量守恒修正基础PKN'],
  [12, '合同2（2）③：分簇导流与应力遮挡表达多簇竞争'],
  [13, '合同2（3）①：EnKF状态量是PKN物理参数而非最终缝长'],
  [15, '合同2（3）②：更新参数后重新正演形成连续反演闭环'],
  [17, '合同2（3）③：在线留出验证三类平均误差均低于15%'],
  [18, '合同2（4）①：300成员EnKF单步P95低于0.25秒'],
  [19, '合同2（4）②：统一接口支持PKN、reduced BEM和代理模型切换'],
  [21, '合同2（4）③：三维裂缝与二维缝长联动展示闭环过程'],
  [23, '合同3（1）①：高层规则选模式，低层PPO/SAC优化连续动作'],
  [24, '合同3（1）②：安全投影将物理边界写入动作空间'],
  [22, '合同3（1）③：300秒状态驱动未来60秒排量与砂比建议'],
  [25, '合同3（2）①：真实秒点构造状态—动作—未来响应样本'],
  [26, '合同3（2）②：学习代理与PKN-EnKF共同评估候选动作'],
  [31, '合同3（2）③：操作风险与模型不确定性分离处置'],
  [32, '合同3（3）①：建议动作携带风险、证据链和确认要求'],
  [33, '合同3（3）②：统一CLI/API支撑建议、确认与系统接入'],
  [28, '合同3（4）①：课程训练逐级增加高风险工况难度'],
  [29, '合同3（4）②：策略与历史动作在同一仿真环境公平对比'],
  [30, '合同3（4）③：逐场景执行动作后180秒安全验证'],
  [34, '工程化交付：指标、失败窗口和源码指纹支持复现'],
]);
for (const [slideNo, title] of titles) replace(slideNo, 'subtitle', title);

replace(8, 'h1', '施工数据标准化');
replace(8, 'work', '施工秒点提供累计液量、排量、施工压力和砂比；压力换算结果与光纤分簇记录按时间戳对齐，形成数字孪生观测量。');
replace(8, 'note', '该换算是合同2（1）的数据预处理环节，不单独作为裂缝反演成果；裂缝弯曲摩阻和经验系数仍需现场标定。');

setText(17, 's1-t', '1.95%–2.37%\n液量TVD');
setText(17, 's2-t', '2.61%–3.38%\n砂量TVD');
setText(17, 's3-t', '5.10%–6.30%\n井底压力');
setText(17, 's4-t', '3个种子\n均值<15%');
replace(17, 'warn', '300成员EnKF降低随机种子敏感性；结果属于液砂份额和压力的观测空间验证。');

setText(18, 's1-t', '0.18–0.24s\n单步P95');
setText(18, 's2-t', '<15s\n时效达标');
setText(18, 's3-t', '100%\n时效达标率');
replace(18, 'work', '计时包含控制量生成、300成员PKN集合正演、观测算子、EnKF参数更新和后验重新正演；不包含WebGL渲染。');
replace(18, 'warn', '在线平均误差达标，但冻结参数留出仍存在16.6%–25.1%液砂误差，说明当前模型依赖持续观测同化。');

replace(22, 'subtitle', titles.get(22));
replace(23, 'subtitle', titles.get(23));
replace(24, 'subtitle', titles.get(24));
replace(24, 'h1', '物理与专家安全边界');
replace(24, 'rw', '候选动作先满足最大排量、最大砂比、单步变化和提砂规则，再进入数字孪生计算奖励；风险达到阈值80%时提前切换安全模式。');

replace(30, 'v180', '真实历史场景10万步策略复评安全率由98.71%提升至99.43%；六场景阶段最佳总体安全率90.28%，其中砂堵风险约42%，仍是主要短板。');
replace(30, 'note', '99.43%尚未达到合同要求的调整后180秒全窗口安全；既有异常恢复窗口与新发异常需分别统计。');

replace(33, 'm1-title', '数据与数字孪生接口');
replace(33, 'm1-body', '统一数据记录、PKN/BEM、EnKF和3D结果');
replace(33, 'm2-title', '智能体环境接口');
replace(33, 'm2-body', 'Gymnasium状态、动作、奖励和安全标志');
replace(33, 'm3-title', '人机协同接口');
replace(33, 'm3-body', '建议、风险、证据、权限和确认状态');
replace(33, 'm4-title', '联合演示App');
replace(33, 'm4-body', '数字孪生、策略推理与结果展示');
replace(33, 'cap', 'run_project.py提供dt、hmi和app统一入口；模块只调用公开接口，数据统一从Data读取，输出统一写入outputs。');

replace(34, 'm1-title', 'DT-Crack');
replace(34, 'm1-body', '参数、观测误差、耗时与3D结果');
replace(34, 'm2-title', 'HMI-KE');
replace(34, 'm2-body', '策略、奖励、失败窗口和安全率');
replace(34, 'm3-title', '自动测试');
replace(34, 'm3-body', '18项核心测试通过');
replace(34, 'm4-title', '保密发布');
replace(34, 'm4-body', '排除Data、Excel、密钥和大型历史日志');
replace(34, 't1-t', '300成员\nEnKF');
replace(34, 't2-t', '18项\n测试通过');
replace(34, 't3-t', 'SHA-256\n源码指纹');

replace(40, 'subtitle', '结论：第二部分已达到观测空间阶段指标，第三部分接近但尚未完全达标');
replace(40, 'cc0-t', '合同2（1）：真实光纤、施工压力和井轨迹已形成统一时空数据链');
replace(40, 'cc1-t', '合同2（2）至2（4）：coupled PKN、EnKF参数反演、模型接口和3D闭环已跑通');
replace(40, 'cc2-t', '合同3（1）至3（4）：分层智能体、动作响应代理、人机接口和180秒验证已形成原型');
replace(40, 'cc3-t', '当前在线观测误差与计算时效达到阶段指标；冻结泛化和砂堵场景仍需专项提升');
replace(40, 'cc4-t', '所有结论均以代码、CSV、summary.json和失败窗口为证据，不将观测空间误差表述为真实裂缝几何精度。');

replace(41, 'sg0-t', '补充真实DAS原始振幅、射孔簇位置和独立裂缝几何解释标签');
replace(41, 'sg1-t', '用多井多段数据校准井筒摩阻、孔眼摩阻、地应力和滤失参数');
replace(41, 'sg2-t', '将60秒动作响应扩展到5分钟滚动预警，并校准概率不确定性');
replace(41, 'sg3-t', '针对既有异常恢复与砂堵场景开展专家示范、课程训练和安全强化学习');
replace(41, 'sg4-t', '接入高保真BEM/数值模型并训练快速代理，开展跨模型一致性验证');
replace(41, 'sg5-t', '在甲方软硬件环境复测15秒端到端时延和180秒连续安全率');
replace(41, 'sg6-t', '完善权限审计、人工确认、异常降级、版本管理和现场联调接口');

for (let i = 0; i < p.slides.items.length; i++) {
  const n = String(i + 1).padStart(2, '0');
  const slide = p.slides.items[i];
  const png = await p.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(renderDir, `slide-${n}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(renderDir, `slide-${n}.layout.json`), await layout.text());
}

const deck = await PresentationFile.exportPptx(p);
await deck.save(output);
console.log(JSON.stringify({ output, slides: p.slides.items.length, renderDir }, null, 2));
