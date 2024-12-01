import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import CodeBlock from "components/CodeBlock";
import 'katex/dist/katex.min.css';
import ModuleFrame from 'components/ModuleFrame';
import { Text } from '@mantine/core';
const convertLatexDelimiters = (content) => {
  content = content.replace(/\\\[([\s\S]*?)\\\]/g, (_, match) => `$$${match}$$`);
  content = content.replace(/\\\(([\s\S]*?)\\\)/g, (_, match) => `$${match}$`);
  return content;
};

const CourseGenerativeModels = () => {
  const [content, setContent] = useState('');

  useEffect(() => {
    fetch('/alessandro/GenAI/class.md')
      .then(response => response.text())
      .then(text => {
        const convertedText = convertLatexDelimiters(text);
        setContent(convertedText);
      })
      .catch(error => {
        console.error('Error fetching markdown:', error);
      });
  }, []);

  const handleLinkClick = (e, href) => {
    if (href?.startsWith('#')) {
      e.preventDefault();
      const targetElement = document.querySelector(href);
      if (targetElement) {
        targetElement.scrollIntoView({ behavior: 'smooth' });
      }
    }
  };

  return (
    <ModuleFrame
      module={11}
      isCourse={true}
      title="Module 11: Generative Models"
    >              <Text mt="md" c="dimmed" size="sm">
    Author: Alessandro Bucci
  </Text>
      {/* Add a container with max-width and center alignment */}
      <div className="max-w-6xl mx-auto px-4">
        {/* Add responsive container for markdown content */}
        <div className="markdown-content prose prose-lg max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkMath, remarkGfm]}
            rehypePlugins={[rehypeKatex]}
            components={{
              img: (props) => (
                <div className="relative w-full overflow-hidden">
                  <img 
                    {...props}
                    className="max-w-full h-auto object-contain my-4 mx-auto rounded-lg shadow-md"
                    style={{
                      maxWidth: '100vh' // Limit maximum height to 80% of viewport height
                    }}
                    loading="lazy"
                    onError={(e) => {
                      console.error(`Failed to load image: ${props.src}`);
                      e.target.alt = 'Failed to load image';
                    }}
                  />
                </div>
              ),
              code: (props) => {
                const match = /language-(\w+)/.exec(props.className || '');
                return !props.inline && match ? (
                  <div className="w-full overflow-x-auto">
                    <CodeBlock
                      code={String(props.children).replace(/\n$/, '')}
                      language={match[1]}
                    />
                  </div>
                ) : (
                  <code className={`px-1 py-0.5 rounded bg-gray-100 font-mono text-sm ${props.className}`} {...props}>
                    {props.children}
                  </code>
                );
              },
              a: (props) => (
                <a
                  href={props.href}
                  onClick={(e) => handleLinkClick(e, props.href)}
                  className="text-blue-600 hover:text-blue-800 hover:underline"
                  {...props}
                >
                  {props.children}
                </a>
              ),
              h1: (props) => (
                <h1 
                  id={String(props.children).toLowerCase().replace(/\s+/g, '-')}
                  className="text-3xl font-bold mt-8 mb-4"
                  {...props} 
                />
              ),
              h2: (props) => (
                <h2 
                  id={String(props.children).toLowerCase().replace(/\s+/g, '-')}
                  className="text-2xl font-bold mt-6 mb-3"
                  {...props} 
                />
              ),
              h3: (props) => (
                <h3 
                  id={String(props.children).toLowerCase().replace(/\s+/g, '-')}
                  className="text-xl font-semibold mt-4 mb-2"
                  {...props} 
                />
              ),
              h4: (props) => (
                <h4 
                  id={String(props.children).toLowerCase().replace(/\s+/g, '-')}
                  className="text-lg font-semibold mt-3 mb-2"
                  {...props} 
                />
              ),
              // Add styling for other markdown elements
              p: (props) => (
                <p className="my-4" {...props}>
                  {props.children}
                </p>
              ),
              ul: (props) => (
                <ul className="list-disc pl-6 my-4" {...props}>
                  {props.children}
                </ul>
              ),
              ol: (props) => (
                <ol className="list-decimal pl-6 my-4" {...props}>
                  {props.children}
                </ol>
              ),
              blockquote: (props) => (
                <blockquote className="border-l-4 border-gray-300 pl-4 my-4 italic" {...props}>
                  {props.children}
                </blockquote>
              ),
              table: (props) => (
                <div className="w-full overflow-x-auto my-4">
                  <table className="min-w-full divide-y divide-gray-200" {...props}>
                    {props.children}
                  </table>
                </div>
              )
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </div>
    </ModuleFrame>
  );
};

export default CourseGenerativeModels;