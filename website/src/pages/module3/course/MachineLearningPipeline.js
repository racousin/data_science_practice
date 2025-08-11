import React from "react";
import { Row, Col } from 'react-bootstrap';
import { Container, Grid } from '@mantine/core';
const MachineLearningPipeline = () => {
  return (
    <Container fluid>
      <h1 className="my-4">Machine Learning Pipeline</h1>
      <p>
        The machine learning pipeline is a systematic approach to developing and
        deploying ML models. It consists of several interconnected stages, each
        crucial for creating effective and reliable models.
      </p>
      <Row>
        <Col md={12}>
        <h2 id="problem-definition">0. Problem Definition</h2>
          <p>
            <strong>Objective:</strong> Clearly define the problem and set project goals
          </p>
          <p>
            <strong>Input:</strong> Business challenge or opportunity
          </p>
          <p>
            <strong>Output:</strong> Well-defined problem statement, project objectives, and success criteria
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Identify the core business problem or opportunity</li>
            <li>Determine if machine learning is the appropriate solution</li>
            <li>Define specific, measurable, achievable, relevant, and time-bound (SMART) objectives</li>
            <li>Establish clear success criteria and metrics</li>
            <li>Align ML goals with overall business strategy</li>
          </ul>
          <p>
            <strong>Roles:</strong> Business Analyst, Data Scientist, Domain Expert, Project Manager
          </p>
          <p>
            <strong>Tools:</strong> Project management software, collaboration tools
          </p>
          <h2 id = "data-collection">1. Data Collection</h2>
          <p>
            <strong>Objective:</strong> Gather relevant data to address the defined problem
          </p>
          <p>
            <strong>Input:</strong> Problem statement, data requirements
          </p>
          <p>
            <strong>Output:</strong> Raw dataset(s), data documentation
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Identify appropriate data sources based on the problem definition</li>
            <li>Assess data availability, quality, and relevance</li>
            <li>Ensure data collection adheres to technical, legal, and ethical standards</li>
            <li>Set up data versioning and storage systems</li>
            <li>Document data sources, collection methods, and any known limitations</li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Engineer, Data Scientist, Domain Expert
          </p>
          <p>
            <strong>Tools:</strong> SQL, Hadoop, Apache Kafka, web scraping tools, data cataloging software
          </p>
          <h2 id="data-cleaning">2. Data Preprocessing and Feature Engineering</h2>
          <p>
            <strong>Objective:</strong> Clean data and create informative
            features
          </p>
          <p>
            <strong>Input:</strong> Raw dataset(s)
          </p>
          <p>
            <strong>Output:</strong> Processed dataset with engineered features
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Handle missing data, outliers, and inconsistencies</li>
            <li>Normalize or standardize features as needed</li>
            <li>Create domain-specific features</li>
            <li>Apply dimensionality reduction techniques if necessary</li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Machine Learning Engineer
          </p>
          <p>
            <strong>Tools:</strong> Pandas, NumPy, Scikit-learn, OpenCV, NLTK,
            TensorFlow
          </p>
          <h2 id="model-building">3. Model Selection, Training, and Evaluation</h2>
          <p>
            <strong>Objective:</strong> Select, train, and evaluate appropriate
            ML models
          </p>
          <p>
            <strong>Input:</strong> Processed dataset
          </p>
          <p>
            <strong>Output:</strong> Trained and validated model(s)
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>
              Choose algorithms based on problem type and data characteristics
            </li>
            <li>Split data into training, validation, and test sets</li>
            <li>Implement cross-validation and hyperparameter tuning</li>
            <li>Evaluate model performance using appropriate metrics</li>
            <li>Analyze error patterns and model behavior</li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Machine Learning Engineer,
            Research Scientist
          </p>
          <p>
            <strong>Tools:</strong> Scikit-learn, TensorFlow, PyTorch, Keras,
            XGBoost, Hugging Face
          </p>
          <h2 id="deployment">4. Deployment, Monitoring, and Maintenance</h2>
          <p>
            <strong>Objective:</strong> Deploy model to production and maintain
            performance
          </p>
          <p>
            <strong>Input:</strong> Validated model
          </p>
          <p>
            <strong>Output:</strong> Deployed model, monitoring system
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Prepare model for production environment</li>
            <li>Set up necessary infrastructure and APIs</li>
            <li>Implement monitoring for model performance and data drift</li>
            <li>Establish protocols for model updates and retraining</li>
          </ul>
          <p>
            <strong>Roles:</strong> MLOps Engineer, DevOps Engineer, Data
            Engineer
          </p>
          <p>
            <strong>Tools:</strong> Docker, Kubernetes, MLflow, Kubeflow,
            TensorFlow Serving
          </p>
          <h2 id="monitoring">5. Model Interpretability and Explainability</h2>
          <p>
            <strong>Objective:</strong> Understand and explain model decisions
          </p>
          <p>
            <strong>Input:</strong> Trained model, test data
          </p>
          <p>
            <strong>Output:</strong> Model explanations, feature importance
          </p>
          <p>
            <strong>Key Considerations:</strong>
          </p>
          <ul>
            <li>Implement appropriate explainability techniques</li>
            <li>Ensure compliance with regulatory requirements</li>
            <li>Communicate insights to stakeholders effectively</li>
          </ul>
          <p>
            <strong>Roles:</strong> Data Scientist, Domain Expert
          </p>
          <p>
            <strong>Tools:</strong> SHAP, LIME, Captum, TensorBoard
          </p>
          <h2 id="best-practices">Best Practices: Baseline and Iterate</h2>
      <p>
        <strong>Objective:</strong> Establish simple, effective baseline approaches and iterate throughout the ML pipeline
      </p>
      <p>
        <strong>Key Principles:</strong>
      </p>
      <ol>
        <li>Always start with the easiest solution before moving to more complex approaches</li>
        <li>Continuously iterate and improve upon your baseline</li>
      </ol>
      {/* <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333" />
    </marker>
  </defs>
  <rect x="50" y="50" width="120" height="60" rx="10" ry="10" fill="#4CAF50" />
  <text x="110" y="85" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Problem Definition</text>
  <rect x="200" y="50" width="120" height="60" rx="10" ry="10" fill="#2196F3" />
  <text x="260" y="85" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Data Collection</text>
  <rect x="350" y="50" width="120" height="60" rx="10" ry="10" fill="#FFC107" />
  <text x="410" y="85" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Data Preprocessing</text>
  <rect x="500" y="50" width="120" height="60" rx="10" ry="10" fill="#9C27B0" />
  <text x="560" y="75" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Model Training</text>
  <text x="560" y="95" font-family="Arial" font-size="14" fill="white" text-anchor="middle">and Evaluation</text>
  <rect x="650" y="50" width="120" height="60" rx="10" ry="10" fill="#FF5722" />
  <text x="710" y="75" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Deployment and</text>
  <text x="710" y="95" font-family="Arial" font-size="14" fill="white" text-anchor="middle">Monitoring</text>
  <line x1="170" y1="80" x2="200" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)" />
  <line x1="320" y1="80" x2="350" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)" />
  <line x1="470" y1="80" x2="500" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)" />
  <line x1="620" y1="80" x2="650" y2="80" stroke="#333" stroke-width="2" marker-end="url(#arrowhead)" />
  <path d="M 710 110 Q 710 300, 110 300 Q 60 300, 60 110" fill="none" stroke="#E91E63" stroke-width="3" stroke-dasharray="5,5" marker-end="url(#arrowhead)" />
  <text x="400" y="340" font-family="Arial" font-size="18" fill="#E91E63" text-anchor="middle" font-weight="bold">Iterate and Improve</text>
  <text x="110" y="250" font-family="Arial" font-size="12" fill="#333" text-anchor="middle">Refine Problem</text>
  <text x="260" y="250" font-family="Arial" font-size="12" fill="#333" text-anchor="middle">Expand Dataset</text>
  <text x="410" y="250" font-family="Arial" font-size="12" fill="#333" text-anchor="middle">Enhance Features</text>
  <text x="560" y="250" font-family="Arial" font-size="12" fill="#333" text-anchor="middle">Optimize Model</text>
  <text x="710" y="250" font-family="Arial" font-size="12" fill="#333" text-anchor="middle">Improve Monitoring</text>
</svg> */}
      <p>
        <strong>Application across pipeline stages:</strong>
      </p>
      <ol>
        <li>
          <strong>Problem Definition:</strong>
          <ul>
            <li>Start with a clear, simple problem statement</li>
            <li>Define straightforward, measurable success criteria</li>
            <li>Consider if the problem can be solved without ML first</li>
            <li>Iterate on the problem definition as you gain more insights</li>
          </ul>
        </li>
        <li>
          <strong>Data Collection:</strong>
          <ul>
            <li>Begin with readily available, structured data sources</li>
            <li>Use simple random sampling before complex sampling techniques</li>
            <li>Start with a smaller, manageable dataset before scaling up</li>
            <li>Iteratively expand your dataset as you identify gaps or biases</li>
          </ul>
        </li>
        <li>
          <strong>Data Preprocessing and Feature Engineering:</strong>
          <ul>
            <li>Use basic cleaning techniques (e.g., removing duplicates, handling missing values) before advanced methods</li>
            <li>Start with raw features before creating complex engineered features</li>
            <li>Apply simple scaling (e.g., min-max scaling) before more complex normalizations</li>
            <li>Iteratively refine your preprocessing steps and feature set</li>
          </ul>
        </li>
        <li>
          <strong>Model Selection, Training, and Evaluation:</strong>
          <ul>
            <li>Begin with simple, interpretable models (e.g., linear regression, logistic regression, decision trees)</li>
            <li>Use basic cross-validation before advanced techniques</li>
            <li>Start with default hyperparameters before extensive tuning</li>
            <li>Iteratively experiment with different models and techniques, building upon what works</li>
          </ul>
        </li>
        <li>
          <strong>Deployment, Monitoring, and Maintenance:</strong>
          <ul>
            <li>Deploy models using simple, reliable methods first (e.g., REST API)</li>
            <li>Start with basic monitoring metrics before implementing complex tracking systems</li>
            <li>Use manual retraining processes before automating the entire pipeline</li>
            <li>Iteratively improve your deployment and monitoring processes based on real-world performance</li>
          </ul>
        </li>
      </ol>
      <p>
        <strong>Benefits of the Baseline and Iterate Approach:</strong>
      </p>
      <ul>
        <li>Quickly establishes a performance benchmark</li>
        <li>Provides early insights into the problem and data</li>
        <li>Serves as a sanity check for more complex solutions</li>
        <li>Helps identify when additional complexity is truly necessary</li>
        <li>Ensures resources are used efficiently</li>
        <li>Facilitates easier debugging and maintenance</li>
        <li>Allows for continuous improvement and adaptation to changing conditions</li>
        <li>Builds team confidence and stakeholder trust through incremental progress</li>
      </ul>
      <p>
        <strong>Iteration Process:</strong>
      </p>
      <ol>
        <li>Establish a baseline solution for each pipeline stage</li>
        <li>Measure and document the performance of the baseline</li>
        <li>Identify the most significant bottlenecks or areas for improvement</li>
        <li>Propose and implement small, incremental changes</li>
        <li>Evaluate the impact of each change</li>
        <li>If beneficial, incorporate the change into the new baseline</li>
        <li>Repeat steps 3-6, continuously refining your solution</li>
      </ol>
      <p>
        <strong>When to move beyond the baseline:</strong>
      </p>
      <ul>
        <li>When baseline performance doesn't meet project requirements</li>
        <li>If the problem complexity clearly demands more advanced techniques</li>
        <li>When there's a significant and measurable improvement from more complex approaches</li>
        <li>As you iterate and gain a deeper understanding of the problem and data</li>
      </ul>
      <p>
        <strong>Roles:</strong> All team members (Data Scientists, ML Engineers, Project Managers, Domain Experts)
      </p>
      <p>
        <strong>Tools:</strong> Version control systems (e.g., Git), experiment tracking tools (e.g., MLflow), collaboration platforms, in addition to stage-specific tools
      </p>
        </Col>
      </Row>
    </Container>
  );
};
export default MachineLearningPipeline;
