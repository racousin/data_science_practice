import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from '@mantine/core';
import { IconArrowLeft } from "@tabler/icons-react";

const BackButton = () => {
  const navigate = useNavigate();

  const goBack = () => {
    navigate(-1); // Navigate back in the history stack
  };

  return (
    <Button variant="default" onClick={goBack} leftSection={<IconArrowLeft size={16} />}>
      Back
    </Button>
  );
};

export default BackButton;
