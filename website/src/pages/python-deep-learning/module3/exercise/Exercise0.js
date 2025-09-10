import React from 'react';
import DataInteractionPanel from 'components/DataInteractionPanel';

const Exercise1 = () => {
  const notebookUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise0.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/python-deep-learning/module3/exercises/exercise0.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/python-deep-learning/module3/exercises/exercise0.ipynb";

  return (
    <DataInteractionPanel
      notebookUrl={notebookUrl}
      notebookHtmlUrl={notebookHtmlUrl}
      notebookColabUrl={notebookColabUrl}
      className="mt-6"
    />
  );
};

export default Exercise1;