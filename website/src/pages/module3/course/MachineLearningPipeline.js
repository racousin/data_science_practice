import React from "react";
import { Container, Row, Col } from "react-bootstrap";

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


        </Col>
      </Row>
    </Container>
  );
};

export default MachineLearningPipeline;
