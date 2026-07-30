import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ''));
const input = 'C:/Users/xinonome/Downloads/基于多源信息融合的压裂施工参数智能精准调控方法——结题验收汇报_合同二三部分版.pptx';
const output = 'C:/Users/xinonome/Downloads/基于多源信息融合的压裂施工参数智能精准调控方法——结题验收汇报_合同二三部分_公式增强版.pptx';
const renderDir = path.join(root, 'formula-render');
await fs.rm(renderDir, { recursive: true, force: true });
await fs.mkdir(renderDir, { recursive: true });

const p = await PresentationFile.importPptx(await FileBlob.load(input));
const slide = p.slides.items[7];
const byName = (name) => {
  const found = slide.shapes.items.find((item) => item.name === name);
  if (!found) throw new Error(`Missing shape on output slide 8: ${name}`);
  return found;
};
const set = (name, text) => { byName(name).text = text; };

set('h1', '基础PKN输入、解析关系与守恒约束');
set('work', `输入参数：E、ν、μ、H、Q_i、C_L、t\n\n平面应变模量： E′ = E / (1 − ν²)\n\nPKN/Nordgren 无滤失长度：\nL_i(t) = 0.6839 [E′Q_i³/(μH⁴)]^(1/5)t^(4/5)\n\n井筒最大缝宽：\nw_w,i(t) = 2.5 [μQ_i/(E′H)]^(1/4)t^(1/8)`);
set('pkncap', 'PKN解析模型的裂缝长度—宽度—压力关系');
set('out', `剖面与守恒： w_i(x,t)=w_w,i[1−x/L_i]^(1/4)；V_frac=V_inj−V_leak\n分簇耦合： Q_i=Q_total·g_i/Σ_jg_j，保证 Σ_iQ_i=Q_total\n输出：L_i、w_i、A_i、V_i、p_net、p_bh；EnKF 更新参数后重新调用 PKN 正演。`);

for (let i = 0; i < p.slides.items.length; i += 1) {
  const n = String(i + 1).padStart(2, '0');
  const slide = p.slides.items[i];
  const png = await p.export({ slide, format: 'png', scale: 1.5 });
  await fs.writeFile(path.join(renderDir, `slide-${n}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(renderDir, `slide-${n}.layout.json`), await layout.text());
}
const deck = await PresentationFile.exportPptx(p);
await deck.save(output);
console.log(JSON.stringify({ output, renderDir, slides: p.slides.items.length }, null, 2));
