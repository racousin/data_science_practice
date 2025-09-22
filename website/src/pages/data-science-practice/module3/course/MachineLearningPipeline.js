import React from "react";
import { Container, Title, Text, List, Flex, Image } from '@mantine/core';

const MachineLearningPipeline = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={1} mb="md">Machine Learning Pipeline</Title>
        <Text size="md" mb="md">
          The machine learning pipeline is a systematic approach to developing and
          deploying ML models. It consists of several interconnected stages, each
          crucial for creating effective and reliable models.
        </Text>
                                    <Flex direction="column" align="center">
                              <Image
                                src="/assets/data-science-practice/module3/pipeline.png"
                                alt="Yutong Liu & The Bigger Picture"
                                style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                                fluid
                              />
                            </Flex>
      </div>

      <div data-slide>
        <Title order={2} id="problem-definition" mb="md">0. Problem Definition</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Clearly define the problem and set project goals
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Business challenge or opportunity
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Well-defined problem statement, project objectives, and success criteria
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Identify the core business problem or opportunity</List.Item>
          <List.Item>Determine if machine learning is the appropriate solution</List.Item>
          <List.Item>Define specific, measurable, achievable, relevant, and time-bound objectives</List.Item>
          <List.Item>Establish clear success criteria and metrics</List.Item>
          <List.Item>Align ML goals with overall business strategy</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> Business Analyst, Data Scientist, Domain Expert, Project Manager
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> Project management software, collaboration tools
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="data-collection" mb="md">1. Data Collection</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Gather relevant data to address the defined problem
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Problem statement, data requirements
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Raw dataset(s), data documentation
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Identify appropriate data sources based on the problem definition</List.Item>
          <List.Item>Assess data availability, quality, and relevance</List.Item>
          <List.Item>Ensure data collection adheres to technical, legal, and ethical standards</List.Item>
          <List.Item>Set up data versioning and storage systems</List.Item>
          <List.Item>Document data sources, collection methods, and any known limitations</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> Data Engineer, Data Scientist, Domain Expert
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> SQL, Hadoop, Apache Kafka, web scraping tools, data cataloging software
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="data-cleaning" mb="md">2. Data Preprocessing and Feature Engineering</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Clean data and create informative features
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Raw dataset(s)
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Processed dataset with engineered features
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Handle missing data, outliers, and inconsistencies</List.Item>
          <List.Item>Normalize or standardize features as needed</List.Item>
          <List.Item>Create domain-specific features</List.Item>
          <List.Item>Apply dimensionality reduction techniques if necessary</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> Data Scientist, Machine Learning Engineer
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> Pandas, NumPy, Scikit-learn, OpenCV, NLTK, TensorFlow
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="model-building" mb="md">3. Model Selection, Training, and Evaluation</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Select, train, and evaluate appropriate ML models
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Processed dataset
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Trained and validated model(s)
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Choose algorithms based on problem type and data characteristics</List.Item>
          <List.Item>Split data into training, validation, and test sets</List.Item>
          <List.Item>Implement cross-validation and hyperparameter tuning</List.Item>
          <List.Item>Evaluate model performance using appropriate metrics</List.Item>
          <List.Item>Analyze error patterns and model behavior</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> Data Scientist, Machine Learning Engineer, Research Scientist
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> Scikit-learn, TensorFlow, PyTorch, Keras, XGBoost, Hugging Face
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="deployment" mb="md">4. Deployment, Monitoring, and Maintenance</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Deploy model to production and maintain performance
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Validated model
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Deployed model, monitoring system
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Prepare model for production environment</List.Item>
          <List.Item>Set up necessary infrastructure and APIs</List.Item>
          <List.Item>Implement monitoring for model performance and data drift</List.Item>
          <List.Item>Establish protocols for model updates and retraining</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> MLOps Engineer, DevOps Engineer, Data Engineer
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> Docker, Kubernetes, MLflow, Kubeflow, TensorFlow Serving
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="monitoring" mb="md">5. Model Interpretability and Explainability</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Understand and explain model decisions
        </Text>
        <Text size="md" mb="sm">
          <strong>Input:</strong> Trained model, test data
        </Text>
        <Text size="md" mb="sm">
          <strong>Output:</strong> Model explanations, feature importance
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Considerations:</strong>
        </Text>
        <List spacing="sm" mb="md">
          <List.Item>Implement appropriate explainability techniques</List.Item>
          <List.Item>Ensure compliance with regulatory requirements</List.Item>
          <List.Item>Communicate insights to stakeholders effectively</List.Item>
        </List>
        <Text size="md" mb="sm">
          <strong>Roles:</strong> Data Scientist, Domain Expert
        </Text>
        <Text size="md" mb="md">
          <strong>Tools:</strong> SHAP, LIME, Captum, TensorBoard
        </Text>
      </div>

      <div data-slide>
        <Title order={2} id="best-practices" mb="md">Best Practices: Baseline and Iterate</Title>
        <Text size="md" mb="sm">
          <strong>Objective:</strong> Establish simple, effective baseline approaches and iterate throughout the ML pipeline
        </Text>
        <Text size="md" mb="sm">
          <strong>Key Principles:</strong>
        </Text>
        <List type="ordered" spacing="sm" mb="md">
          <List.Item>Always start with the easiest solution before moving to more complex approaches</List.Item>
          <List.Item>Continuously iterate and improve upon your baseline</List.Item>
        </List>
      </div>

      <div data-slide>
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
        <Text size="md" mb="sm">
          <strong>Application across pipeline stages:</strong>
        </Text>
        <List type="ordered" spacing="md" mb="md">
          <List.Item>
            <strong>Problem Definition:</strong>
            <List spacing="xs" ml="md">
              <List.Item>Start with a clear, simple problem statement</List.Item>
              <List.Item>Define straightforward, measurable success criteria</List.Item>
              <List.Item>Consider if the problem can be solved without ML first</List.Item>
              <List.Item>Iterate on the problem definition as you gain more insights</List.Item>
            </List>
          </List.Item>
          <List.Item>
            <strong>Data Collection:</strong>
            <List spacing="xs" ml="md">
              <List.Item>Begin with readily available, structured data sources</List.Item>
              <List.Item>Use simple random sampling before complex sampling techniques</List.Item>
              <List.Item>Start with a smaller, manageable dataset before scaling up</List.Item>
              <List.Item>Iteratively expand your dataset as you identify gaps or biases</List.Item>
            </List>
          </List.Item>
          <List.Item>
            <strong>Data Preprocessing and Feature Engineering:</strong>
            <List spacing="xs" ml="md">
              <List.Item>Use basic cleaning techniques (e.g., removing duplicates, handling missing values) before advanced methods</List.Item>
              <List.Item>Start with raw features before creating complex engineered features</List.Item>
              <List.Item>Apply simple scaling (e.g., min-max scaling) before more complex normalizations</List.Item>
              <List.Item>Iteratively refine your preprocessing steps and feature set</List.Item>
            </List>
          </List.Item>
          <List.Item>
            <strong>Model Selection, Training, and Evaluation:</strong>
            <List spacing="xs" ml="md">
              <List.Item>Begin with simple, interpretable models (e.g., linear regression, logistic regression, decision trees)</List.Item>
              <List.Item>Use basic cross-validation before advanced techniques</List.Item>
              <List.Item>Start with default hyperparameters before extensive tuning</List.Item>
              <List.Item>Iteratively experiment with different models and techniques, building upon what works</List.Item>
            </List>
          </List.Item>
          <List.Item>
            <strong>Deployment, Monitoring, and Maintenance:</strong>
            <List spacing="xs" ml="md">
              <List.Item>Deploy models using simple, reliable methods first (e.g., REST API)</List.Item>
              <List.Item>Start with basic monitoring metrics before implementing complex tracking systems</List.Item>
              <List.Item>Use manual retraining processes before automating the entire pipeline</List.Item>
              <List.Item>Iteratively improve your deployment and monitoring processes based on real-world performance</List.Item>
            </List>
          </List.Item>
        </List>
      </div>

    </Container>
  );
};
export default MachineLearningPipeline;
