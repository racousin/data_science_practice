import React from 'react';
import { Container, Text, Title, Stack, List } from '@mantine/core';
import { Cpu } from 'lucide-react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise3 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise3.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module1/exercises/exercise3.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module1/exercises/exercise3.ipynb";

  return (

          <DataInteractionPanel
            notebookUrl={notebookUrl}
            notebookHtmlUrl={notebookHtmlUrl}
            notebookColabUrl={notebookColabUrl}
            className="mt-6"
          />
  );
};

export default Exercise3;