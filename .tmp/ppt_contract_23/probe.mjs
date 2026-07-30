import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('./source.pptx'));
console.log('slides methods', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides)));
console.log('slide methods', Object.getOwnPropertyNames(Object.getPrototypeOf(p.slides.items[0])));
