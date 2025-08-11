import React from "react";
import { NavLink, Stack, Anchor } from "@mantine/core";
import { Link, useLocation } from "react-router-dom";

const NavigationMenu = ({ links, prefix }) => {
  const location = useLocation();

  const scrollToElement = (id) => {
    const element = document.getElementById(id);
    if (element) {
      window.scrollTo({
        top: element.offsetTop - 170,
        behavior: "smooth",
      });
    }
  };

  return (
    <Stack gap="xs">
      {links.map((link) => (
        <React.Fragment key={link.to}>
          <NavLink
            component={Link}
            to={`${prefix}${link.to}`}
            active={location.pathname === `${prefix}${link.to}`}
            label={link.label}
          />
          {link.subLinks &&
            location.pathname === `${prefix}${link.to}` &&
            link.subLinks.map((subLink) => (
              <Anchor
                key={subLink.id}
                onClick={() => scrollToElement(subLink.id)}
                size="sm"
                pl="md"
                c="dimmed"
                style={{ cursor: "pointer" }}
              >
                {subLink.label}
              </Anchor>
            ))}
        </React.Fragment>
      ))}
    </Stack>
  );
};

export default NavigationMenu;
