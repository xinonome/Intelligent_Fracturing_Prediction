import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const root = path.dirname(new URL(import.meta.url).pathname.replace(/^\//, ''));
const source = path.join(root, 'source.pptx');
const out = path.join(root, 'template-inspect');
await fs.mkdir(path.join(out, 'source-slides'), { recursive: true });
await fs.mkdir(path.join(out, 'layouts'), { recursive: true });
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const slides = p.slides.items;
for (let i = 0; i < slides.length; i++) {
  const n = String(i + 1).padStart(2, '0');
  const png = await p.export({ slide: slides[i], format: 'png', scale: 1 });
  await fs.writeFile(path.join(out, 'source-slides', `source-slide-${n}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slides[i].export({ format: 'layout' });
  await fs.writeFile(path.join(out, 'layouts', `source-slide-${n}.layout.json`), await layout.text());
}
const inspection = await p.inspect({ kind: 'slide,textbox,shape,image,table,chart,notes,layout', maxChars: 400000 });
await fs.writeFile(path.join(out, 'template-inspect.ndjson'), inspection.ndjson);
const montage = await p.export({ format: 'webp', montage: true, scale: 0.5 });
await fs.writeFile(path.join(out, 'source-montage.webp'), new Uint8Array(await montage.arrayBuffer()));
await fs.writeFile(path.join(out, 'template-manifest.json'), JSON.stringify({ slideCount: slides.length }, null, 2));
console.log(JSON.stringify({ slideCount: slides.length, out }, null, 2));
