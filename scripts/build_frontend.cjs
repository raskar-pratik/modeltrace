#!/usr/bin/env node
/** Delegate exclusively to installed upstream TypeScript and Vite. No substitute runtime. */
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const frontend = path.join(root, 'frontend');
const suffix = process.platform === 'win32' ? '.cmd' : '';
for (const tool of ['tsc', 'vite']) {
  if (!fs.existsSync(path.join(frontend, 'node_modules', '.bin', tool + suffix))) {
    console.error('Production frontend build requires installed project-local TypeScript and Vite. Run npm --prefix frontend ci.');
    process.exit(1);
  }
}
const npm = 'npm' + suffix;
for (const task of process.argv.includes('--check-only') ? ['typecheck'] : ['build']) {
  const result = spawnSync(npm, ['run', task], { cwd: frontend, stdio: 'inherit', shell: process.platform === 'win32' });
  if (result.error) console.error(result.error.message);
  if (result.status !== 0) process.exit(result.status || 1);
}
