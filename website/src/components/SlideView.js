import React, { useState, useEffect, useCallback } from 'react';
import { Carousel } from '@mantine/carousel';
import { ActionIcon, Tooltip } from '@mantine/core';
import { IconPresentation, IconX } from '@tabler/icons-react';
import '@mantine/carousel/styles.css';

const SlideView = ({ children, enabled = false }) => {
  const [slideMode, setSlideMode] = useState(false);
  const [slides, setSlides] = useState([]);

  useEffect(() => {
    if (!enabled) return;

    const detectSlides = () => {
      const slideElements = document.querySelectorAll('[data-slide], .slide');
      
      if (slideElements.length > 0) {
        const slidesArray = Array.from(slideElements).map((el, index) => {
          return (
            <div key={index} dangerouslySetInnerHTML={{ __html: el.outerHTML }} />
          );
        });
        setSlides(slidesArray);
      }
    };

    // Initial detection
    const timer = setTimeout(detectSlides, 200);

    // Watch for DOM changes
    const observer = new MutationObserver(detectSlides);
    setTimeout(() => {
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    }, 100);

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [children, enabled]);

  const handleKeyPress = useCallback((e) => {
    // Start presentation with 'S' key
    if (!slideMode && (e.key === 's' || e.key === 'S') && slides.length > 0) {
      e.preventDefault();
      setSlideMode(true);
      return;
    }
    
    // Exit with Escape
    if (slideMode && e.key === 'Escape') {
      setSlideMode(false);
      if (document.fullscreenElement) {
        document.exitFullscreen();
      }
    }
  }, [slideMode, slides.length]);

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
        {/* Exit button */}
        <Tooltip label="Exit (Esc)" position="left">
          <ActionIcon
            onClick={() => setSlideMode(false)}
            className="absolute top-4 right-4 z-10"
            size="lg"
            variant="filled"
            color="dark"
          >
            <IconX size={20} />
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
          {slides.map((slide, index) => (
            <Carousel.Slide key={index}>
              <div className="h-full w-full flex items-center justify-center overflow-auto p-8">
                <div className="max-w-6xl w-full">
                  {slide}
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
            onClick={() => setSlideMode(true)}
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