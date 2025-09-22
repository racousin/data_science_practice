import React from "react";
import { Container, Title, Text, List, Image, Flex } from '@mantine/core';

const Introduction = () => {
  return (
    <Container fluid>
      <div data-slide>
        <Title order={2} mb="md">Introduction</Title>
        <Text size="md" mb="md">
          Data science is a multidisciplinary field focused on extracting
          knowledge from data, which are typically large and complex. Data science
          involves various techniques from statistics, machine learning, and
          computer science to analyze data to make informed decisions.
          Interestingly, data science as a formal discipline didn't really exist 20 years ago,
          and the job now covers a vast array of topics and continues to evolve rapidly
          as new technologies and methodologies emerge.
        </Text>
                    <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/ds.png"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                        fluid
                      />
                      <Text component="p" ta="center" mt="xs">
                        Source: https://medium.com/analytics-vidhya
                      </Text>
                    </Flex>
      </div>


      <div data-slide>
        <Title order={2} mb="md" id="roles">Roles in Data Science</Title>
        <Text size="md" mb="md">
          Data science teams comprise professionals with diverse skill sets,
          each contributing uniquely to extracting actionable insights from
          data. Understanding these roles and their responsibilities is
          crucial for effective collaboration and project success. As the
          field evolves, these roles continue to adapt and transform.
        </Text>

        <Title order={3} mb="md">Core Data Science Roles</Title>

        <Title order={4} mb="sm">Data Scientist</Title>
        <Text size="md" mb="sm">
          Data Scientists are central to data science projects. They design
          models and algorithms to analyze complex datasets, deriving
          predictive insights and patterns to support decision-making. They
          typically have a strong background in statistics, machine learning,
          and programming.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Statistical analysis, machine learning,
          programming (Python, R), data visualization
        </Text>

        <Title order={4} mb="sm">Data Engineer</Title>
        <Text size="md" mb="sm">
          Data Engineers develop and maintain the architectures (such as
          databases and large-scale processing systems) that data scientists
          use. They ensure data flows seamlessly between servers and
          applications, making it readily accessible for analysis.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Database management, ETL processes, big
          data technologies (Hadoop, Spark), cloud platforms
        </Text>

        <Title order={4} mb="sm">Machine Learning Engineer</Title>
        <Text size="md" mb="sm">
          Machine Learning Engineers specialize in building and deploying
          machine learning models. They work closely with data scientists to
          optimize algorithms and implement them in production environments,
          often requiring expertise in software development and data
          architecture.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Deep learning frameworks, MLOps,
          software engineering, scalable ML systems
        </Text>

        <Title order={4} mb="sm">Data Analyst</Title>
        <Text size="md" mb="sm">
          Data Analysts focus on parsing data using statistical tools to
          create detailed reports and visualizations. Their insights help
          organizations make strategic decisions based on quantitative data
          and trend analysis.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> SQL, data visualization tools (Tableau,
          Power BI), statistical analysis, business intelligence
        </Text>
</div>
<div data-slide>
                              <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/job-type.png"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(900px, 90vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
        </div>
<div data-slide>
        <Title order={3} mb="md">Other Specialized Roles</Title>

        <Title order={4} mb="sm">Research Scientist</Title>
        <Text size="md" mb="sm">
          Research Scientists in data science focus on developing new
          algorithms, methodologies, and approaches to solve complex data
          problems. They often work on cutting-edge projects and contribute to
          the academic community.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Advanced mathematics, research
          methodologies, publishing academic papers
        </Text>

        <Title order={4} mb="sm">MLOps Engineer</Title>
        <Text size="md" mb="sm">
          MLOps Engineers bridge the gap between data science and IT
          operations. They focus on the deployment, monitoring, and
          maintenance of machine learning models in production environments.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> CI/CD for ML, containerization
          (Docker), orchestration (Kubernetes), monitoring tools
        </Text>

        <Title order={4} mb="sm">Data Architect</Title>
        <Text size="md" mb="sm">
          Data Architects design and manage an organization's data
          infrastructure. They create blueprints for data management systems
          to integrate, centralize, protect, and maintain data sources.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Data modeling, system design, data
          governance, cloud architecture
        </Text>
</div>
        

      <div data-slide>
        <Title order={3} mb="md" id="tools">The Data Science Tools</Title>
        <Text size="md" mb="md">
          Data science relies heavily on a suite of powerful tools that help
          professionals manage data, perform analyses, build models, and
          visualize results.
        </Text>
                    <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/landscape.png"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
                    </div>
                    <div data-slide>
        <Title order={4} mb="sm">Programming Languages</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>Python:</strong> Dominant in data science for its
            simplicity and readability, Python boasts a rich ecosystem of
            libraries like NumPy, Pandas, Scikit-learn, and TensorFlow.
          </List.Item>
          <List.Item>
            <strong>R:</strong> Preferred for statistical analysis and
            graphics, R is widely used in academia and industries that require
            rigorous statistical analysis.
          </List.Item>
        </List>
        </div>
<div data-slide>
        <Title order={4} mb="sm">Data Management and Big Data</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>SQL:</strong> Essential for querying and managing database
            systems. Tools like MySQL, PostgreSQL, and Microsoft SQL Server
            are commonly used.
          </List.Item>
          <List.Item>
            <strong>Hadoop:</strong> A framework that allows for the
            distributed processing of large data sets across clusters of
            computers using simple programming models.
          </List.Item>
          <List.Item>
            <strong>Apache Spark:</strong> Known for its speed and ease of
            use, Spark extends the Hadoop model to also support data streaming
            and complex iterative algorithms.
          </List.Item>
        </List>
        </div>
<div data-slide>
        <Title order={4} mb="sm">Machine Learning Library</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>scikit-learn:</strong> An open-source framework Simple
            and efficient tools for predictive data analysis
          </List.Item>
          <List.Item>
            <strong>TensorFlow:</strong> An open-source framework developed by
            Google for deep learning projects.
          </List.Item>
          <List.Item>
            <strong>PyTorch:</strong> Known for its flexibility and ease of
            use in the research community, particularly in academia.
          </List.Item>
        </List>
        </div>
<div data-slide>
        <Title order={4} mb="sm">Data Visualization Tools</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>Tableau:</strong> Widely recognized for making complex
            data visualizations user-friendly and accessible to business
            professionals.
          </List.Item>
          <List.Item>
            <strong>PowerBI:</strong> Microsoft's analytics service provides
            interactive visualizations and business intelligence capabilities
            with an interface simple enough for end users to create their own
            reports and dashboards.
          </List.Item>
          <List.Item>
            <strong>Matplotlib and Seaborn:</strong> Popular Python libraries
            that offer a wide range of static, animated, and interactive
            visualizations.
          </List.Item>
        </List>
        </div>
<div data-slide>

        <Title order={4} mb="sm">Project Management and Version Control</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>Git:</strong> A distributed version control system essential for tracking changes in source code during software development.
          </List.Item>
          <List.Item>
            <strong>Jira:</strong> An agile project management tool used for issue tracking, bug tracking, and project management in data science teams.
          </List.Item>
        </List>
        </div>
<div data-slide>
        <Title order={4} mb="sm">Infrastructure and Cloud Tools</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>Amazon Web Services (AWS):</strong> Offers a wide range of cloud computing services, including EC2 for compute power, S3 for storage, and SageMaker for machine learning.
          </List.Item>
          <List.Item>
            <strong>Google Cloud Platform (GCP):</strong> Provides services like BigQuery for analytics, Cloud Storage for data storage, and Vertex AI for machine learning operations.
          </List.Item>
          <List.Item>
            <strong>Microsoft Azure:</strong> Features services such as Azure Databricks for big data analytics, Azure Machine Learning for building and deploying models, and Azure Data Lake for data storage.
          </List.Item>
        </List>
                            <Flex direction="column" align="center">
                      <Image
                        src="/assets/data-science-practice/module3/cloud.jpeg"
                        alt="Yutong Liu & The Bigger Picture"
                        style={{ maxWidth: 'min(600px, 70vw)', height: 'auto' }}
                        fluid
                      />
                    </Flex>
      </div>
    </Container>
  );
};

export default Introduction;
