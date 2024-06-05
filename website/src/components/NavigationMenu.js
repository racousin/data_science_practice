import React from "react";
import { Nav, NavLink } from "react-bootstrap";
import { Link, useLocation } from "react-router-dom";

const NavigationMenu = ({ links, prefix }) => {
  const location = useLocation();

  return (
    <Nav variant="pills" className="flex-column">
      {links.map((link) => (
        <NavLink
          key={link.to}
          as={Link} // Correct usage of `as` with `Link`
          to={`${prefix}${link.to}`}
          className={
            location.pathname === `${prefix}${link.to}`
              ? "active nav-link"
              : "nav-link"
          }
        >
          {link.label}
        </NavLink>
      ))}
    </Nav>
  );
};

export default NavigationMenu;
