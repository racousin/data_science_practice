import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

const TimeSeriesProcessing = () => {
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/class.md')
      .then(response => response.text())
      .then(text => setContent(text))
      .catch(console.error);
  }, []);

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        rehypePlugins={[rehypeKatex]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default TimeSeriesProcessing;