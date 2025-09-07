import React, { useState, useEffect, useCallback } from 'react';
import { Carousel } from '@mantine/carousel';
import { ActionIcon, Tooltip } from '@mantine/core';
import { IconPresentation, IconX } from '@tabler/icons-react';
import { useSidebar } from '../contexts/SidebarContext';
import '@mantine/carousel/styles.css';

const SlideView = ({ children, enabled = false }) => {
  const { slideMode, setSlideMode } = useSidebar();
  const [slides, setSlides] = useState([]);

  // Helper function to extract slide titles and separate content
  const extractTitleAndContent = (element) => {
    const clonedEl = element.cloneNode(true);
    
    let title = null;
    
    // Look for elements with slide-title class
    const titleEl = clonedEl.querySelector('.slide-title');
    
    if (titleEl) {
      title = titleEl.outerHTML;
      titleEl.remove(); // Remove from content
    }
    
    return {
      title,
      content: clonedEl.outerHTML
    };
  };

  useEffect(() => {
    if (!enabled) return;

    let isDetecting = false; // Prevent recursive calls

    const detectSlides = () => {
      if (isDetecting) return;
      isDetecting = true;
      
      const slideElements = document.querySelectorAll('[data-slide], .slide');
      
      if (slideElements.length > 0) {
        const slidesArray = Array.from(slideElements).map((el, index) => {
          const { title, content } = extractTitleAndContent(el);
          
          return {
            key: index,
            title,
            content
          };
        });
        setSlides(slidesArray);
      }
      
      isDetecting = false;
    };

    // Initial detection with multiple attempts to catch dynamic content
    const timer1 = setTimeout(detectSlides, 100);
    const timer2 = setTimeout(detectSlides, 500);
    const timer3 = setTimeout(detectSlides, 1000);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, [enabled, children]); // Add children as dependency to re-detect when content changes

  const handleKeyPress = useCallback((e) => {
    // Start presentation with 'S' key
    if (!slideMode && (e.key === 's' || e.key === 'S') && slides.length > 0) {
      e.preventDefault();
      enterFullscreen();
      return;
    }
    
    // Exit with Escape
    if (slideMode && e.key === 'Escape') {
      exitFullscreen();
    }
  }, [slideMode, slides.length]);

  const enterFullscreen = async () => {
    setSlideMode(true);
    try {
      await document.documentElement.requestFullscreen();
    } catch (err) {
      console.log('Fullscreen not supported or denied');
    }
  };

  const exitFullscreen = async () => {
    setSlideMode(false);
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.log('Exit fullscreen error:', err);
    }
  };

  // Listen for fullscreen changes to sync state
  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement && slideMode) {
        setSlideMode(false);
      }
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, [slideMode]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);


  if (!enabled || slides.length === 0) {
    return <>{children}</>;
  }

  if (slideMode) {
    return (
      <div className="fixed inset-0 w-full h-screen bg-white dark:bg-gray-900 z-50">
        <style>
          {`
            /* Override Container padding in slide mode */
            .mantine-Container-root {
              padding-inline: 0 !important;
              max-width: 100% !important;
            }
          `}
        </style>
        {/* Exit button - top right corner */}
        <Tooltip label="Exit (Esc)" position="left">
          <ActionIcon
            onClick={exitFullscreen}
            className="absolute top-4 right-4 z-10"
            size="lg"
            variant="subtle"
            style={{ 
              backgroundColor: 'transparent',
              opacity: 0.3,
              transition: 'opacity 0.2s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.3'}
          >
            <IconX size={20} color="white" />
          </ActionIcon>
        </Tooltip>

        {/* Carousel with native controls */}
        <Carousel
          height="100vh"
          slideSize="100%"
          align="center"
          loop
          withIndicators
          controlsOffset="xl"
          controlSize={40}
          className="h-full"
          styles={{
            root: {
              height: '100vh',
              width: '100vw'
            },
            viewport: {
              height: '100vh',
              width: '100vw'
            },
            container: {
              height: '100vh',
              width: '100vw'
            },
            control: {
              backgroundColor: 'rgba(0, 0, 0, 0.6)',
              border: 'none',
              color: 'white',
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.8)'
              }
            },
            indicators: {
              bottom: 20
            },
            indicator: {
              width: 12,
              height: 4,
              transition: 'width 250ms ease',
              '&[data-active]': {
                width: 40,
                backgroundColor: '#228be6'
              }
            }
          }}
        >
          {slides.map((slide) => (
            <Carousel.Slide key={slide.key}>
              <div 
                className="slide-content"
                style={{ 
                  height: '100vh',
                  width: '100vw',
                  display: 'grid',
                  gridTemplateRows: 'auto 1fr',
                  gridTemplateAreas: `
                    "header"
                    "content"
                  `,
                  padding: '2rem',
                  boxSizing: 'border-box',
                  overflow: 'auto'
                }}
              >
                {/* Header section for Title order={3} */}
                {slide.title && (
                  <div 
                    style={{
                      gridArea: 'header',
                      textAlign: 'center',
                      marginBottom: '2rem'
                    }}
                    dangerouslySetInnerHTML={{ __html: slide.title }}
                  />
                )}
                
                {/* Main content - vertically centered */}
                <div 
                  style={{
                    gridArea: 'content',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    width: '100%'
                  }}
                >
                  <div 
                    style={{
                      width: '100%',
                      maxWidth: 'none'
                    }}
                    dangerouslySetInnerHTML={{ __html: slide.content }}
                  />
                </div>
              </div>
            </Carousel.Slide>
          ))}
        </Carousel>
      </div>
    );
  }

  // Normal view with presentation button
  return (
    <div className="relative">
      {slides.length > 0 && (
        <Tooltip label="Start presentation (S)">
          <ActionIcon
            onClick={enterFullscreen}
            className="fixed top-4 right-4 z-40"
            size="lg"
            variant="subtle"
            color="gray"
            style={{ opacity: 0.6 }}
            onMouseEnter={(e) => e.currentTarget.style.opacity = '1'}
            onMouseLeave={(e) => e.currentTarget.style.opacity = '0.6'}
          >
            <IconPresentation size={22} />
          </ActionIcon>
        </Tooltip>
      )}
      {children}
    </div>
  );
};

export default SlideView;