# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PRIMARY FOCUS: Data Science Practice Course Review & Improvement

We are reviewing and improving the Data Science Practice course week by week. The course content is located in `website/public/repositories/data_science_practice` and uses a dedicated GitHub repository for student submissions.

### Course Structure

- **Repository**: `data_science_practice_2025` (student submission repository)
- **Website Integration**: Students update their answers directly to the repository with validation through the website
- **Content Location**: `/website/public/repositories/data_science_practice/`
- **Page Components Location**: `/website/src/pages/data-science-practice/`
- **Workflows**: `.github/workflows/` contains automation for validation and permissions

### Important File Naming Convention
- **Route paths in SideNavigation.js**: Use kebab-case (e.g., `/git-hub-desktop`)
- **Component files**: Use PascalCase (e.g., `GitHubDesktop.js`)
- **Route mapping**: The route `/git-hub-desktop` maps to file `GitHubDesktop.js`

### Development Guidelines

1. **UI Framework**: Always use Mantine components and properties
   - Use Mantine's component library for all UI elements
   - **Keep the UI minimal and clean**: Avoid excessive use of Cards, Alerts, and borders
   - Use Alerts sparingly - only for critical information or warnings
   - Prefer simple text and headings over boxed/bordered content
   - Cards should be used only when grouping related information is essential
   - Leverage Mantine's built-in styling and theming system
   - Follow Mantine's patterns for responsive design

2. **Content Structure**:
   - **For course content pages**: Use `<div data-slide>` to create slide-based content
   - **For exercise pages**: Do NOT use slides - present content directly
   - Each slide (in course pages) should contain focused, digestible content sections
   - Use slides to break up complex topics into manageable chunks (course pages only)

3. **Code Examples**:
   - Import CodeBlock: `import CodeBlock from 'components/CodeBlock';`
   - **Keep code snippets short** (3-5 lines maximum per example)
   - Add explanations between code blocks to maintain clarity
   - Break complex code examples into multiple smaller snippets with explanations

4. **Mathematical Expressions**:
   - Import math components: `import { InlineMath, BlockMath } from 'react-katex';`
   - Use `<InlineMath>` for inline mathematical expressions
   - Use `<BlockMath>` for standalone mathematical equations

5. **Visual Design Principles**:
   - Maintain a clean, lightweight interface
   - Use white space effectively instead of borders
   - Minimize visual clutter - focus on content readability
   - Only use colored elements (Alerts, Badges) for truly important information
   - Prefer Typography (Title, Text) components over decorated containers

6. **Tone**: Maintain neutral and scientific tone throughout all content

7. **Exercise Development**:
   - Each exercise should have clear objectives
   - Include automated tests where applicable
   - Ensure compatibility with the website validation system

8. **Student Workflow**:
   - Students push to `data_science_practice_2025` repository
   - Automatic validation through website integration
   - Immediate feedback on submissions

### Technical Requirements

1. **Consistency**: All exercises must follow established patterns
2. **Progressive Difficulty**: Build complexity gradually throughout the course
3. **Automated Validation**: All exercises should include validation tests
4. **Clear Instructions**: Every task should have unambiguous requirements
5. **Website Integration**: Ensure all content works with the validation system
6. **Component Structure**: Use proper React component patterns with hooks when needed

### Code Component Usage Examples

```javascript
// Slide structure (ONLY for course pages, NOT for exercises)
<div data-slide>
  <h2>Topic Title</h2>
  <p>Content explanation...</p>
</div>

// Exercise structure (NO slides, minimal UI)
<Title order={2} mb="md">Section Title</Title>
<Text size="md" mb="md">Content explanation...</Text>
<List spacing="sm">
  <List.Item>Point 1</List.Item>
  <List.Item>Point 2</List.Item>
</List>

// Code block usage
<CodeBlock
  code={`def hello():
    return "Hello World"`}
  language="python"
/>

// Use Alerts sparingly - only for critical information
<Alert icon={<IconAlertCircle />} color="yellow">
  Critical warning or important note only
</Alert>

// Math expressions
<p>The formula is <InlineMath>{'x^2 + y^2 = z^2'}</InlineMath></p>
<BlockMath>{'\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}'}</BlockMath>
```