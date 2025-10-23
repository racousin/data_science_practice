# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Course Structure

- **Repository**: `data_science_practice_2025` (student submission repository)
- **Website Integration**: Students update their answers directly to the repository with validation through the website
- **Content Location**: `/website/public/repositories/data_science_practice/`
- **Page Components Location**: `/website/src/pages/data-science-practice/`
- **Project Pages Location**: `/website/src/pages/data-science-practice/project-pages/`
- **Workflows**: `.github/workflows/` contains automation for validation and permissions

## File Naming Convention
- **Route paths in SideNavigation.js**: Use kebab-case (e.g., `/git-hub-desktop`)
- **Component files**: Use PascalCase (e.g., `GitHubDesktop.js`)
- **Route mapping**: The route `/git-hub-desktop` maps to file `GitHubDesktop.js`

## Development Guidelines

### UI Framework
- Use Mantine components for all UI elements
- Keep UI minimal and clean: avoid excessive Cards, Alerts, and borders
- Use Alerts sparingly - only for critical information
- Prefer simple text and headings over boxed/bordered content

### Content Structure
- **Course content pages**: Use `<div data-slide>` to create slide-based content
- **Exercise pages**: Do NOT use slides - present content directly
- **Project pages**: Do NOT use slides - present content directly

### Code Examples
- Import CodeBlock: `import CodeBlock from 'components/CodeBlock';`
- Keep code snippets short (3-5 lines maximum per example)
- Add explanations between code blocks

### Mathematical Expressions
- Import math components: `import { InlineMath, BlockMath } from 'react-katex';`
- Use `<InlineMath>` for inline expressions
- Use `<BlockMath>` for standalone equations

### Tone
Maintain neutral and scientific tone throughout all content