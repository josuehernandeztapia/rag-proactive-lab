#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🚀 RAG Proactive Lab - Custom Build Script');

function run(command, options = {}) {
  console.log(`\n$ ${command}`);
  execSync(command, { stdio: 'inherit', ...options });
}

try {
  const repoRoot = process.cwd();
  const pythonPath = process.env.RAG_PYTHON || path.join(repoRoot, '.venv', 'bin', process.platform === 'win32' ? 'python.exe' : 'python');
  if (!fs.existsSync(pythonPath)) {
    throw new Error(`Python interpreter not found at ${pythonPath}. Set RAG_PYTHON env var or create the venv.`);
  }

  // Build Dashboard
  console.log('📊 Building React Dashboard...');
  run('npm run build', { cwd: path.join(repoRoot, 'clients', 'dashboard') });
  console.log('✅ Dashboard build complete');

  // Validate API deps
  console.log('🐍 Validating Python API...');
  run(`${pythonPath} -c "import fastapi; print('✅ API validation complete')"`, { cwd: repoRoot });

  // Run tests
  console.log('🧪 Running Tests...');
  run(`${pythonPath} -m pytest tests/ -v`, { cwd: repoRoot });
  console.log('✅ Tests complete');

  console.log('\n🎉 BUILD SUCCESS - All components ready!');
} catch (error) {
  console.error('\n❌ Build failed:', error.message);
  process.exit(error.status || 1);
}
