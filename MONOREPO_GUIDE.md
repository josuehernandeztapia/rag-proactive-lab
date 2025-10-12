# 🚀 Monorepo Guide - RAG Proactive Lab

## 📋 Quick Start

```bash
# Install dependencies
npm install

# Start all services in development
npm run dev

# Build all projects
npm run build

# Run all tests
npm run test

# Lint all code
npm run lint
```

## 🏗️ Architecture Overview

This is an **Nx-powered monorepo** that organizes the RAG Proactive Lab ecosystem into logical, scalable modules:

```
rag-proactive-lab/
├── services/           # Backend services
│   └── api/           # FastAPI main service
├── clients/           # Frontend applications
│   ├── dashboard/     # React analytics dashboard
│   └── pwa/          # Angular production PWA
├── libs/              # Shared libraries
│   └── shared/       # Common types & utilities
├── agents/           # AI agent implementations
├── tools/            # Organized scripts & utilities
└── data/             # Datasets with Git LFS
```

## 🎯 Benefits of This Structure

### ✅ **Developer Experience**
- **Unified Commands**: Single `npm run` commands for all projects
- **Shared Dependencies**: No more version conflicts between projects
- **Consistent Tooling**: Same linting, testing, and build processes

### ✅ **Code Quality**
- **Shared Libraries**: Eliminate type duplication between frontend/backend
- **Dependency Graph**: Nx understands project relationships
- **Incremental Builds**: Only rebuild what changed

### ✅ **Scaling**
- **Clear Boundaries**: Each agent/service has its own space
- **Easy Additions**: Add new agents with `nx generate`
- **Parallel Execution**: Run tests/builds in parallel

## 🔧 Common Commands

### Development
```bash
# Start specific services
nx serve dashboard          # React dashboard only
nx serve api               # FastAPI only
nx run-many --target=serve --projects=api,dashboard --parallel

# Build specific project
nx build dashboard
nx build pwa

# Test specific project
nx test shared
nx test dashboard
```

### Code Quality
```bash
# Lint everything
npm run lint

# Format code
npm run format

# Check formatting
npm run format:check
```

### Project Management
```bash
# See project dependencies
nx graph

# Reset workspace
npm run clean

# Check what's affected by changes
nx affected:test
nx affected:build
```

## 📁 Project Structure Details

### `/services/api/`
FastAPI backend service with all webhooks, endpoints, and agent coordination.

### `/clients/dashboard/`
React + Vite dashboard for real-time analytics and monitoring.

### `/clients/pwa/`
Angular PWA - the full production application with comprehensive testing.

### `/libs/shared/`
Shared TypeScript types, utilities, and constants used across projects.

### `/agents/`
Individual agent implementations (PIA, HASE, Guardian, etc.).

### `/tools/`
Organized utility scripts:
- `batch/` - Batch processing scripts
- `prep/` - Data preparation utilities
- `data-ops/` - ETL and data operations

## 🚀 Adding New Projects

### New Agent
```bash
nx generate @nx/node:application new-agent
# Creates services/new-agent/
```

### New Frontend
```bash
nx generate @nx/react:application new-dashboard
# Creates clients/new-dashboard/
```

### New Shared Library
```bash
nx generate @nx/js:library new-lib
# Creates libs/new-lib/
```

## 🔗 Inter-Project Dependencies

Projects can import from each other using workspace references:

```typescript
// In dashboard (React)
import { DriverState, GuardianAlert } from '@rag-proactive-lab/shared';

// In API (Python)
# Use shared constants through build process
```

## 📊 Monitoring & Analytics

```bash
# View dependency graph
nx graph

# See affected projects
nx affected:apps
nx affected:libs

# Build only what changed
nx affected:build

# Test only what changed
nx affected:test
```

## 🛠️ Troubleshooting

### Common Issues

**Dependencies not found:**
```bash
npm run clean
npm install
```

**Build failures:**
```bash
nx reset
npm run build
```

**Nx cache issues:**
```bash
nx reset
rm -rf node_modules/.cache
```

### Performance Tips

- Use `nx affected:*` commands for faster CI/CD
- Leverage Nx caching for repeated builds
- Run commands in parallel with `--parallel` flag

## 📚 Resources

- [Nx Documentation](https://nx.dev)
- [Workspace Generators](https://nx.dev/generators/using-generators)
- [Project Configuration](https://nx.dev/reference/project-configuration)

---

**💡 Pro Tip**: Use `nx affected` commands in CI/CD to only build/test what actually changed, making your pipelines incredibly fast!