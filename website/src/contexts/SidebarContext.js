import React, { createContext, useContext, useState } from 'react';

const SidebarContext = createContext();

export const SidebarProvider = ({ children, sidebarOpened, toggleSidebar }) => {
  const [slideMode, setSlideMode] = useState(false);
  
  return (
    <SidebarContext.Provider value={{ 
      sidebarOpened, 
      toggleSidebar, 
      slideMode, 
      setSlideMode 
    }}>
      {children}
    </SidebarContext.Provider>
  );
};

export const useSidebar = () => {
  const context = useContext(SidebarContext);
  if (context === undefined) {
    return { 
      sidebarOpened: false, 
      toggleSidebar: () => {}, 
      slideMode: false, 
      setSlideMode: () => {} 
    };
  }
  return context;
};