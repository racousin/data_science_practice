const fs = require('fs');
const path = require('path');

// // Constants for special markers
// const MARKERS = {
//   CODE: '__CODE_BLOCK__',
//   LATEX: '__LATEX_BLOCK__',
//   IMAGE: '__IMAGE_BLOCK__',
//   INLINE_LATEX: '__INLINE_LATEX__'
// };

// // Helper functions for processing different content types
// const processors = {
//   // Process inline LaTeX expressions
//   inlineLatex: (latex) => {
//     return latex
//       .replace(/\\/g, '\\\\')          // Escape backslashes first
//       .replace(/`/g, '\\`')            // Escape backticks
//       .replace(/\$/g, '\\$')           // Escape dollar signs
//       .replace(/'/g, "\\'")            // Escape single quotes
//       .replace(/"/g, '\\"')            // Escape double quotes
//       .replace(/_/g, '\\_')            // Escape underscores
//       .trim();
//   },

//   // Process display/block LaTeX expressions
//   displayLatex: (latex) => {
//     // First, escape all backslashes except those in \text{} commands
//     let processed = latex.replace(/\\text\{([^}]*)\}/g, (match) => {
//       // Temporarily replace \text{} content with a marker
//       return `__TEXT_MARKER_${match.length}__`;
//     });

//     // Now escape all necessary characters
//     processed = processed
//       .replace(/\\/g, '\\\\')          // Escape backslashes
//       .replace(/`/g, '\\`')            // Escape backticks
//       .replace(/\$/g, '\\$')           // Escape dollar signs
//       .replace(/'/g, "\\'")            // Escape single quotes
//       .replace(/"/g, '\\"')            // Escape double quotes
//       .replace(/_/g, '\\_')            // Escape underscores
//       .replace(/\n/g, ' ')             // Replace newlines with spaces
//       .trim();

//     // Restore \text{} commands
//     processed = processed.replace(/__TEXT_MARKER_(\d+)__/g, (match, length) => {
//       const content = latex.match(new RegExp(`\\\\text\\{([^}]{${length - 7}})\\}`))[0];
//       return content.replace(/\\/g, '\\\\');
//     });

//     return processed;
//   },

//   // Process code blocks
//   code: (code) => {
//     return code
//       .replace(/\\/g, '\\\\')          // Escape backslashes
//       .replace(/`/g, '\\`')            // Escape backticks
//       .replace(/\$/g, '\\$')           // Escape dollar signs
//       .replace(/'/g, "\\'");           // Escape single quotes
//   },

//   // Process image paths
//   image: (alt, src) => {
//     const imagePath = src.startsWith('/') ? src : `/images/${src}`;
//     return `![${alt}](${imagePath})`;
//   }
// };

// // Content extractors
// const extractors = {
//   // Extract and store code blocks
//   extractCodeBlocks: (content) => {
//     const blocks = [];
//     const processed = content.replace(/```[\s\S]*?```/g, (match) => {
//       blocks.push(match);
//       return `${MARKERS.CODE}${blocks.length - 1}`;
//     });
//     return { processed, blocks };
//   },

//   // Extract and store LaTeX display blocks
//   extractLatexBlocks: (content) => {
//     const blocks = [];
//     const processed = content.replace(/\\\[([\s\S]*?)\\\]/g, (match, latex) => {
//       blocks.push(latex);
//       return `${MARKERS.LATEX}${blocks.length - 1}`;
//     });
//     return { processed, blocks };
//   },

//   // Extract and store inline LaTeX
//   extractInlineLatex: (content) => {
//     const blocks = [];
//     const processed = content.replace(/\\\(([\s\S]*?)\\\)/g, (match, latex) => {
//       blocks.push(latex);
//       return `${MARKERS.INLINE_LATEX}${blocks.length - 1}`;
//     });
//     return { processed, blocks };
//   },

//   // Extract and store images
//   extractImages: (content) => {
//     const blocks = [];
//     const processed = content.replace(/!\[(.*?)\]\((.*?)\)/g, (match, alt, src) => {
//       blocks.push({ alt, src });
//       return `${MARKERS.IMAGE}${blocks.length - 1}`;
//     });
//     return { processed, blocks };
//   }
// };

// // Content restorers
// const restorers = {
//   // Restore code blocks
//   restoreCodeBlocks: (content, blocks) => {
//     return content.replace(new RegExp(`${MARKERS.CODE}(\\d+)`, 'g'), 
//       (match, index) => processors.code(blocks[index]));
//   },

//   // Restore LaTeX blocks
//   restoreLatexBlocks: (content, blocks) => {
//     return content.replace(new RegExp(`${MARKERS.LATEX}(\\d+)`, 'g'), 
//       (match, index) => `\\[${processors.displayLatex(blocks[index])}\\]`);
//   },

//   // Restore inline LaTeX
//   restoreInlineLatex: (content, blocks) => {
//     return content.replace(new RegExp(`${MARKERS.INLINE_LATEX}(\\d+)`, 'g'), 
//       (match, index) => `\\(${processors.inlineLatex(blocks[index])}\\)`);
//   },

//   // Restore images
//   restoreImages: (content, blocks) => {
//     return content.replace(new RegExp(`${MARKERS.IMAGE}(\\d+)`, 'g'), 
//       (match, index) => processors.image(blocks[index].alt, blocks[index].src));
//   }
// };

// // Main processing function and execution remain the same
// const processMarkdown = (content) => {
//   let processedContent = content;

//   // Step 1: Extract all special content
//   const { processed: withoutCode, blocks: codeBlocks } = 
//     extractors.extractCodeBlocks(processedContent);
//   const { processed: withoutLatex, blocks: latexBlocks } = 
//     extractors.extractLatexBlocks(withoutCode);
//   const { processed: withoutInlineLatex, blocks: inlineLatexBlocks } = 
//     extractors.extractInlineLatex(withoutLatex);
//   const { processed: withoutImages, blocks: imageBlocks } = 
//     extractors.extractImages(withoutInlineLatex);

//   // Step 2: Process the main content
//   processedContent = withoutImages
//     .replace(/([^\\])`/g, '$1\\`')     // Escape unescaped backticks
//     .replace(/([^\\])\$/g, '$1\\$')    // Escape unescaped dollar signs
//     .replace(/([^\\])'/g, "$1\\'")     // Escape unescaped quotes
//     .replace(/\r\n/g, '\n')            // Normalize line endings
//     .replace(/\n{3,}/g, '\n\n');       // Normalize multiple newlines

//   // Step 3: Restore all special content
//   processedContent = restorers.restoreImages(processedContent, imageBlocks);
//   processedContent = restorers.restoreInlineLatex(processedContent, inlineLatexBlocks);
//   processedContent = restorers.restoreLatexBlocks(processedContent, latexBlocks);
//   processedContent = restorers.restoreCodeBlocks(processedContent, codeBlocks);

//   return processedContent;
// };

// Main execution function remains the same...
// Main execution function
const main = async () => {
//   try {
//     const sourcePath = path.join(__dirname, '../src/pages/module9/class.md');
//     const targetPath = path.join(__dirname, '../src/pages/module9/class.js');

//     console.log('Reading markdown file...');
//     const content = fs.readFileSync(sourcePath, 'utf-8');

//     console.log('Processing content...');
//     const processedContent = processMarkdown(content);

//     console.log('Generating JavaScript module...');
//     const output = `// Generated file - do not edit directly
// export const timeSeriesContent = \`${processedContent}\`;
// `;

//     fs.writeFileSync(targetPath, output, 'utf-8');
//     console.log('Successfully generated class.js');

//   } catch (error) {
//     console.error('Error:', error.message);
//     process.exit(1);
//   }
};

main();