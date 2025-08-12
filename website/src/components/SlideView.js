import React, { useState, useEffect, useCallback } from 'react';
import { Carousel } from '@mantine/carousel';
import { ActionIcon, Tooltip, Text } from '@mantine/core';
import { IconPresentation, IconArrowLeft, IconArrowRight, IconX, IconMaximize, IconMinimize } from '@tabler/icons-react';
import '@mantine/carousel/styles.css';

const SlideView = ({ children, enabled = false }) => {
  const [slideMode, setSlideMode] = useState(false);
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [slides, setSlides] = useState([]);

  useEffect(() => {
    if (!enabled) return;

    const detectSlides = () => {
      // Try to find slides in the DOM directly
      const slideElements = document.querySelectorAll('[data-slide], .slide');
      
      if (slideElements.length > 0) {
        const slidesArray = Array.from(slideElements).map((el, index) => {
          // Clone the element's content for rendering in carousel
          return (
            <div key={index} dangerouslySetInnerHTML={{ __html: el.outerHTML }} />
          );
        });
        setSlides(slidesArray);
        return true;
      }
      return false;
    };

    // Initial detection with delay
    const timer = setTimeout(() => {
      detectSlides();
    }, 200);

    // Watch for DOM changes
    const observer = new MutationObserver(() => {
      detectSlides();
    });

    // Start observing after a delay
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

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  }, []);

  const handleKeyPress = useCallback((e) => {
    // Start presentation with 'S' key when not in slide mode
    if (!slideMode && (e.key === 's' || e.key === 'S') && slides.length > 0) {
      e.preventDefault();
      setSlideMode(true);
      return;
    }
    
    if (!slideMode) return;
    
    switch(e.key) {
      case 'ArrowLeft':
        setCurrentSlide(prev => Math.max(0, prev - 1));
        break;
      case 'ArrowRight':
        setCurrentSlide(prev => Math.min(slides.length - 1, prev + 1));
        break;
      case 'Escape':
        setSlideMode(false);
        if (document.fullscreenElement) {
          document.exitFullscreen();
        }
        break;
      case 'f':
      case 'F':
        toggleFullscreen();
        break;
      default:
        break;
    }
  }, [slideMode, slides.length, toggleFullscreen]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [handleKeyPress]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  if (!enabled || slides.length === 0) {
    return <>{children}</>;
  }

  if (slideMode) {
    return (
      <div className={`relative ${isFullscreen ? 'h-screen' : 'min-h-screen'} w-full bg-white dark:bg-gray-900 overflow-hidden`}>
          {/* Main slide content */}
          <Carousel
            slideSize="100%"
            height="100%"
            align="center"
            slidesToScroll={1}
            withControls={false}
            withIndicators={false}
            slideGap="md"
            loop={false}
            onSlideChange={setCurrentSlide}
            initialSlide={currentSlide}
            className="w-full h-full"
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

          {/* Bottom right: slide counter and navigation */}
          <div 
            className="absolute bottom-8 right-8 z-50 flex items-center gap-2"
            style={{ 
              backgroundColor: 'rgba(0, 0, 0, 0.2)', 
              padding: '8px 12px', 
              borderRadius: '8px',
              backdropFilter: 'blur(10px)'
            }}
          >
            <Tooltip label="Previous (←)">
              <ActionIcon 
                onClick={() => setCurrentSlide(prev => Math.max(0, prev - 1))}
                disabled={currentSlide === 0}
                size="sm"
                variant="subtle"
                color="gray"
              >
                <IconArrowLeft size={18} />
              </ActionIcon>
            </Tooltip>

            <Text size="sm" className="text-gray-200 px-2">
              {currentSlide + 1} / {slides.length}
            </Text>

            <Tooltip label="Next (→)">
              <ActionIcon 
                onClick={() => setCurrentSlide(prev => Math.min(slides.length - 1, prev + 1))}
                disabled={currentSlide === slides.length - 1}
                size="sm"
                variant="subtle"
                color="gray"
              >
                <IconArrowRight size={18} />
              </ActionIcon>
            </Tooltip>
          </div>

          {/* Top right: fullscreen and close */}
          <div 
            className="absolute top-8 right-8 z-50 flex items-center gap-2"
            style={{ 
              backgroundColor: 'rgba(0, 0, 0, 0.2)', 
              padding: '8px', 
              borderRadius: '8px',
              backdropFilter: 'blur(10px)'
            }}
          >
            <Tooltip label={isFullscreen ? "Exit fullscreen" : "Fullscreen (F)"}>
              <ActionIcon 
                onClick={toggleFullscreen}
                size="sm"
                variant="subtle"
                color="gray"
              >
                {isFullscreen ? <IconMinimize size={18} /> : <IconMaximize size={18} />}
              </ActionIcon>
            </Tooltip>

            <Tooltip label="Exit (Esc)">
              <ActionIcon 
                onClick={() => {
                  setSlideMode(false);
                  if (document.fullscreenElement) {
                    document.exitFullscreen();
                  }
                }}
                size="sm"
                variant="subtle"
                color="gray"
              >
                <IconX size={18} />
              </ActionIcon>
            </Tooltip>
          </div>
      </div>
    );
  }

  return (
    <div className="relative">
      {slides.length > 0 && (
        <>
          {/* Discrete slide button in top right */}
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
        </>
      )}
      {children}
    </div>
  );
};

export default SlideView;