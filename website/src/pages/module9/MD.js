import React from 'react';
import ReactMarkdownMath from 'react-markdown-math';
import 'katex/dist/katex.min.css';

const MarkdownViewer = ({ content }) => {
  return (
    <div className="markdown-content">
      <ReactMarkdownMath
        source={content}
        escapeHtml={false}
      />
    </div>
  );
};

export default MarkdownViewer;