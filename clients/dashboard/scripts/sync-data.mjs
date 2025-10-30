import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const dashboardRoot = path.resolve(__dirname, '..');
const projectRoot = path.resolve(dashboardRoot, '..', '..');

const defaultMapping = [
  [path.join(projectRoot, 'data/processed/pia/synthetic_driver_states.csv'), path.join(dashboardRoot, 'public/data/synthetic_driver_states.csv')],
  [path.join(projectRoot, 'data/processed/pia/pia_outcomes_log.csv'), path.join(dashboardRoot, 'public/data/pia_outcomes_log.csv')],
  [path.join(projectRoot, 'data/processed/hase/pia_outcomes_features.csv'), path.join(dashboardRoot, 'public/data/pia_outcomes_features.csv')],
  [path.join(projectRoot, 'data/processed/pia/pia_hotspots.csv'), path.join(dashboardRoot, 'public/data/pia_hotspots.csv')],
  [path.join(projectRoot, 'reports/pia_plan_summary.csv'), path.join(dashboardRoot, 'public/data/pia_plan_summary.csv')],
  [path.join(projectRoot, 'reports/pia_llm_outbox.jsonl'), path.join(dashboardRoot, 'public/data/pia_llm_outbox.jsonl')],
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
