import { promises as fs } from 'node:fs';
import path from 'node:path';

const cwd = process.cwd();
const root = path.resolve(cwd, '..');

const defaultMapping = [
  [path.join(root, 'data/pia/synthetic_driver_states.csv'), path.join(cwd, 'public/data/synthetic_driver_states.csv')],
  [path.join(root, 'data/pia/pia_outcomes_log.csv'), path.join(cwd, 'public/data/pia_outcomes_log.csv')],
  [path.join(root, 'data/hase/pia_outcomes_features.csv'), path.join(cwd, 'public/data/pia_outcomes_features.csv')],
  [path.join(root, 'reports/pia_plan_summary.csv'), path.join(cwd, 'public/data/pia_plan_summary.csv')],
  [path.join(root, 'reports/pia_llm_outbox.jsonl'), path.join(cwd, 'public/data/pia_llm_outbox.jsonl')],
];

async function ensureDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function copyFile(src, dest) {
  try {
    await ensureDir(dest);
    await fs.copyFile(src, dest);
    console.log(`✓ Copiado ${src} -> ${dest}`);
  } catch (error) {
    if (error.code === 'ENOENT') {
      console.warn(`⚠️  No se encontró ${src}`);
    } else {
      console.error(`✗ Error copiando ${src}:`, error);
    }
  }
}

async function main() {
  const pairs = defaultMapping;
  await Promise.all(pairs.map(([src, dest]) => copyFile(src, dest)));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
