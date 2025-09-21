import React from "react";
import { Container, Title, Text, List, Image } from '@mantine/core';

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
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md" id="data">The Data</Title>
        <Text size="md" mb="md">
          Data is the cornerstone of the modern digital economy, powering
          everything from daily business decisions to advanced artificial
          intelligence systems. As of 2024, the data landscape is
          characterized by unprecedented growth, diversity, and strategic
          importance across all industries.
        </Text>

        <Title order={4} mb="sm">Exponential Growth in Data Volume</Title>
        <Text size="md" mb="md">
          Recent studies suggest that the global data sphere is expected to
          grow to over 200 zettabytes by 2025, with a significant portion
          being generated in real time. This growth is fueled by pervasive
          computing devices, including mobiles, sensors, and the increasing
          number of Internet of Things (IoT) deployments.
        </Text>

        <Title order={4} mb="sm">Variety and Complexity of Data</Title>
        <Text size="md" mb="md">
          Data today comes in various forms: structured data in traditional
          databases, semi-structured data from web applications, and a vast
          amount of unstructured data from sources like social media, videos,
          and the ubiquitous sensors. This variety adds layers of complexity
          to data processing and analytics.
        </Text>

        <Title order={4} mb="sm">Key Players in the Data Ecosystem</Title>
        <Text size="md" mb="md">
          Major technology firms play a pivotal role in shaping the data
          landscape. Companies such as Amazon Web Services, Microsoft Azure,
          and Google Cloud are leading in cloud storage and computing,
          providing the backbone for storing and processing this vast amount
          of data. Social media giants like Facebook and TikTok contribute
          significantly to the generation of user-generated content, offering
          rich datasets that are invaluable for insights and marketing.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md" id="applications">"Recent" Breakthroughs in AI Applications</Title>
        <Text size="md" mb="md">
          Recent years have seen remarkable advancements in AI applications
          across various domains, demonstrating the transformative power of
          data science and machine learning. Here are some notable
          breakthroughs:
        </Text>

        <Title order={4} mb="sm">AlphaFold: Revolutionizing Protein Structure Prediction</Title>
        <Text size="md" mb="sm">
          In 2020, DeepMind's AlphaFold achieved a major breakthrough in the
          protein folding problem. It predicted protein structures with
          unprecedented accuracy, reaching a median score of 92.4 GDT across
          all targets in CASP14. This advancement has significant implications
          for drug discovery and understanding diseases at a molecular level.
        </Text>
        <Text size="sm" mb="md" fs="italic">
          Source: Jumper, J., Evans, R., Pritzel, A. et al. Highly accurate
          protein structure prediction with AlphaFold. Nature 596, 583–589
          (2021).
        </Text>

        <Title order={4} mb="sm">GPT-3 and ChatGPT: Advancing Natural Language Processing</Title>
        <Text size="md" mb="sm">
          OpenAI's GPT-3, released in 2020, demonstrated remarkable natural
          language understanding and generation capabilities. Its successor,
          ChatGPT, launched in 2022, showed even more impressive results in
          conversational AI. ChatGPT reached 100 million monthly active users
          just two months after its launch, showcasing unprecedented adoption
          rates for an AI application.
        </Text>
        <Text size="sm" mb="md" fs="italic">
          Source: OpenAI. (2023). ChatGPT: Optimizing Language Models for
          Dialogue.
        </Text>

        <Title order={4} mb="sm">DALL-E and Midjourney: AI in Image Generation</Title>
        <Text size="md" mb="sm">
          AI models like DALL-E 2 (2022) and Midjourney have shown remarkable
          capabilities in generating high-quality images from text
          descriptions. These models have achieved human-level performance in
          certain image generation tasks, with DALL-E 2 scoring 66.4% on the
          CLIP score metric for image-text similarity.
        </Text>
        <Text size="sm" mb="md" fs="italic">
          Source: Ramesh, A., et al. (2022). Hierarchical Text-Conditional
          Image Generation with CLIP Latents. arXiv:2204.06125.
        </Text>

        <Title order={4} mb="sm">AlphaGo and MuZero: Mastering Complex Games</Title>
        <Text size="md" mb="sm">
          DeepMind's AlphaGo defeated the world champion in Go in 2016, a feat
          previously thought to be decades away. Its successor, MuZero,
          demonstrated even more general capabilities, mastering chess, shogi,
          and Atari games without being taught the rules, achieving superhuman
          performance in all of these domains.
        </Text>
        <Text size="sm" mb="md" fs="italic">
          Source: Silver, D., et al. (2020). Mastering Atari, Go, Chess and
          Shogi by Planning with a Learned Model. Nature 588, 604–609.
        </Text>

        <Title order={4} mb="sm">GPT-4: Multimodal AI</Title>
        <Text size="md" mb="sm">
          Released in 2023, GPT-4 showcased impressive multimodal
          capabilities, able to process both text and images. It demonstrated
          human-level performance on various academic and professional tests,
          scoring in the 90th percentile on the Uniform Bar Exam and
          outperforming 99% of human test-takers on the Biology Olympiad.
        </Text>
        <Text size="sm" mb="md" fs="italic">
          Source: OpenAI. (2023). GPT-4 Technical Report.
        </Text>
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

        <div style={{ textAlign: 'center' }}>
          <Image
            src="/assets/data-science-practice/module3/job-type.png"
            alt="Jobs in Data Science"
            style={{ maxWidth: '100%' }}
          />
        </div>
        <Title order={3} mb="md">Specialized and Emerging Roles</Title>

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

        <Title order={3} mb="md">Business and Domain-Specific Roles</Title>

        <Title order={4} mb="sm">Business Intelligence Developer</Title>
        <Text size="md" mb="sm">
          BI Developers create and manage platforms for data visualization and
          reporting. They transform complex data into easily understandable
          dashboards and reports for business stakeholders.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> BI tools (Power BI, Tableau), data
          warehousing, SQL, business analysis
        </Text>

        <Title order={4} mb="sm">Domain Expert</Title>
        <Text size="md" mb="sm">
          Domain Experts bring specific industry or field knowledge to data
          science projects. They help interpret results in the context of the
          business and ensure that data science solutions align with
          industry-specific needs and regulations.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Deep industry knowledge, ability to
          translate between technical and business languages
        </Text>

        <Title order={3} mb="md">Leadership and Management Roles</Title>

        <Title order={4} mb="sm">Chief Data Officer (CDO)</Title>
        <Text size="md" mb="sm">
          The CDO is responsible for enterprise-wide data strategy,
          governance, and utilization. They ensure that the organization
          leverages its data assets effectively and in compliance with
          regulations.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Strategic planning, data governance,
          executive communication, change management
        </Text>

        <Title order={4} mb="sm">Data Science Manager / Team Lead</Title>
        <Text size="md" mb="sm">
          Data Science Managers oversee teams of data professionals, aligning
          data science projects with business objectives. They manage
          resources, timelines, and stakeholder expectations.
        </Text>
        <Text size="md" mb="md">
          <strong>Key Skills:</strong> Project management, team leadership,
          technical expertise, stakeholder management
        </Text>

        <Text size="md" mb="md">
          These roles often have overlapping responsibilities, and their
          specific duties can vary significantly among organizations. The
          common goal remains: to harness the power of data to drive
          decision-making, innovation, and business value. As the field of
          data science continues to evolve, new roles may emerge, and existing
          ones may transform to meet the ever-changing challenges of working
          with data.
        </Text>
      </div>

      <div data-slide>
        <Title order={3} mb="md" id="tools">The Data Science Tools</Title>
        <Text size="md" mb="md">
          Data science relies heavily on a suite of powerful tools that help
          professionals manage data, perform analyses, build models, and
          visualize results.
        </Text>

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

        <Title order={4} mb="sm">AI-Powered Coding Assistants</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>ChatGPT:</strong> A large language model capable of generating code, explaining concepts, and assisting with problem-solving in data science.
          </List.Item>
          <List.Item>
            <strong>GitHub Copilot:</strong> An AI pair programmer that suggests code completions and entire functions in real-time.
          </List.Item>
          <List.Item>
            <strong>Claude:</strong> An AI assistant that can help with code generation, debugging, and explaining data science concepts.
          </List.Item>
        </List>

        <Title order={4} mb="sm">Project Management and Version Control</Title>
        <List spacing="sm" mb="md">
          <List.Item>
            <strong>Git:</strong> A distributed version control system essential for tracking changes in source code during software development.
          </List.Item>
          <List.Item>
            <strong>Jira:</strong> An agile project management tool used for issue tracking, bug tracking, and project management in data science teams.
          </List.Item>
        </List>

        <Title order={4} mb="sm">Cloud Tools</Title>
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
          <List.Item>
            <strong>Databricks:</strong> A unified analytics platform built on top of Apache Spark, offering collaborative notebooks and integrated workflows for big data processing and machine learning.
          </List.Item>
        </List>
      </div>
    </Container>
  );
};

export default Introduction;
