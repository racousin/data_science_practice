import React from 'react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise2 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise2.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise2.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module3/exercises/exercise2.ipynb";

  return (
    <DataInteractionPanel
      notebookUrl={notebookUrl}
      notebookHtmlUrl={notebookHtmlUrl}
      notebookColabUrl={notebookColabUrl}
      className="mt-6"
    />
  );
};

export default Exercise2;