import React from 'react';
import { Container, Title, Text, Stack, Grid, Image, Paper } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

const MLParadigmCard = ({ title, bgColor, description, formula, formulaDesc, characteristics, examples, challenges }) => (
  <Paper className={`p-6 ${bgColor} rounded-lg h-full`}>
    <Title order={3} mb="md">{title}</Title>
    
    <Text className="mb-4">{description}</Text>
    
    <div className="mb-4">
      <BlockMath>{formula}</BlockMath>
      <Text size="sm" className="text-gray-600 mt-2">{formulaDesc}</Text>
    </div>
    
    <div className="mb-4">
      <Text fw="bold" mb="xs">Key Characteristics:</Text>
      <ul className="list-disc pl-6">
        {characteristics.map((item, idx) => (
          <li key={idx} className="mb-1">{item}</li>
        ))}
      </ul>
    </div>
  </Paper>
);

const MLParadigmsComparison = () => {
  const paradigms = [
    {
      title: "Supervised Learning",
      bgColor: "bg-blue-50",
      description: "Learning from labeled data pairs (x, y) to predict outcomes or classify new instances. The model learns a function f_θ that maps inputs x to outputs y by minimizing a loss function ℓ over the training dataset. The learned mapping should generalize to unseen examples by capturing underlying patterns rather than memorizing the training data.",
      formula: "\\min_{\\theta} \\mathcal{L}(\\theta) = \\min_{\\theta} \\sum_{i=1}^n \\ell(f_\\theta(x_i), y_i)",
      formulaDesc: "Minimizing loss function over labeled training data (x, y)",
      characteristics: [
        "Paired input-output training data (x, y) required",
        "Direct feedback through differentiable loss functions",
        "Clear evaluation metrics (accuracy, precision, recall)",
        "Focus on generalization to unseen data",
        "Examples: classification, regression, object detection"
      ]
    },
    {
      title: "Unsupervised Learning",
      bgColor: "bg-green-50",
      description: "Discovering hidden patterns and structure in unlabeled data x. The model learns internal representations or groupings by optimizing objectives like reconstruction error, density estimation, or clustering criteria. Key approaches include dimensionality reduction via learned encodings z = e_θ(x), generative modeling of p(x), and clustering to identify natural groupings.",
      formula: "\\min_{\\theta} \\mathcal{L}(\\theta) = \\min_{\\theta} \\sum_{i=1}^n \\ell(f_\\theta(x_i), x_i)",
      formulaDesc: "Optimizing internal data representation or clustering objective",
      characteristics: [
        "Only unlabeled inputs x available",
        "Internal representation learning",
        "Data distribution modeling p(x)",
        "Natural grouping discovery",
        "Examples: clustering, dimensionality reduction, generative models"
      ]
    },
    {
      title: "Reinforcement Learning",
      bgColor: "bg-purple-50",
      description: "Training agents to make optimal decisions through trial and error. The agent discovers which actions yield the best long-term results by interacting with its environment. When the agent encounters a situation, it chooses an action based on its learned policy. As the agent moves through different states and takes actions, it creates a sequence of experiences called a trajectory. The agent's goal is to maximize the sum of all rewards received along this trajectory (denoted as return G(τ)).",
      formula: "\\pi^* = \\arg \\max_{\\pi} \\mathbb{E}_{\\tau\\sim \\pi}[{G(\\tau)}]",
      formulaDesc: "Finding optimal policy π* that maximizes expected returns G(τ) over trajectories τ",
      characteristics: [
        "Sequential decision making over trajectories",
        "Optimization of cumulative rewards",
        "Exploration vs exploitation tradeoff",
        "Learning from interaction without supervision",
        "Examples: game playing, robotics, resource management"
      ]
    }
  ];

  return (
    <>
      <Title order={2} mb="xl" id="ml-paradigms">
        Machine Learning Paradigms Comparison
      </Title>
      <Grid gutter="lg">
        {paradigms.map((paradigm, idx) => (
          <Grid.Col span={4} key={idx}>
            <MLParadigmCard {...paradigm} />
          </Grid.Col>
        ))}
      </Grid>
    </>
  );
};

const applications = [
  {
    title: "Autonomous Systems",
    description: "Robotics, self-driving vehicles, drone navigation"
  },
  {
    title: "Industrial Control",
    description: "Process optimization, energy management"
  },
  {
    title: "Natural Language",
    description: "Dialogue systems, chatbot policies"
  },
  {
    title: "Financial Markets",
    description: "Trading strategies, risk management"
  }
];

const RLApplications = () => {
  return (
    <>
      <Title order={2} id="applications" className="mb-6">
        Applications of RL
      </Title>
      
      <Text className="mb-6">
        Reinforcement Learning has revolutionized numerous fields with its ability to learn complex decision-making strategies:
      </Text>

      <Grid>
        <Grid.Col span={{ base: 12, md: 5 }}>
          <div className="grid grid-cols-1 gap-4">
            {applications.map((app, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <Title order={4} className="mb-2" style={{ color: '#0096FF' }}>
                  {app.title}
                </Title>
                <Text size="sm">
                  {app.description}
                </Text>
              </div>
            ))}
          </div>
        </Grid.Col>

        <Grid.Col span={{ base: 12, md: 7 }} className="flex items-stretch">
          <div className="bg-gray-100 w-full p-8 rounded-lg flex items-center justify-center min-h-[400px]">
            <div className="w-full relative">
              <Image
                src="/assets/data-science-practice/module9/game.png"
                alt="RL in Gaming"
                className="w-full h-full object-contain"
                fit="contain"
              />
            </div>
          </div>
        </Grid.Col>
      </Grid>
    </>
  );
};

const Introduction = () => {
  return (
    <Container size="xl" py="xl">
      {/* Hero Section with AlphaGo */}
      <Stack spacing="xl">
        <div data-slide className="relative w-full h-64 bg-gray-100 rounded-lg overflow-hidden mb-8">
          <Image
            src="/assets/data-science-practice/module9/alphago.jpg"
            alt="AlphaGo vs Ke Jie match"
            className="w-full h-full object-cover"
          />
          <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-4">
            <Text size="lg">
              In 2017, AlphaGo defeated Ke Jie, the world's top-ranked Go player.
            </Text>
          </div>
        </div>

<div data-slide>
  <MLParadigmsComparison/>
</div>

        {/* RL Framework */}
        
          <Title order={2} mb="xl" id="rl-framework">
            The RL Framework
          </Title>
          <Grid gutter="lg">
            <Grid.Col span={6}>
              <Stack>
                <Text>
                  Reinforcement Learning operates without explicit "correct" actions. Instead:
                </Text>
                <ul className="list-disc pl-6">
                  <li className="mb-2">
                    Agents learn through trial and error in an environment
                  </li>
                  <li className="mb-2">
                    Actions are guided by rewards that represent objectives
                  </li>
                  <li className="mb-2">
                    Learning occurs through the optimization of a reward signal
                  </li>
                </ul>
                <BlockMath>
                  {`R_t = r(s_t, a_t)`}
                </BlockMath>
                <Text size="sm" color="dimmed">
                  Where <InlineMath>{`R_t`}</InlineMath> is the reward at time t,
                  based on state <InlineMath>{`s_t`}</InlineMath> and action <InlineMath>{`a_t`}</InlineMath>
                </Text>
              </Stack>
            </Grid.Col>
            <Grid.Col span={6}>
              <div className="bg-gray-100 p-6 rounded-lg h-full flex items-center justify-center">
                <Image
                  src="/assets/data-science-practice/module9/rl.png"
                  alt="RL Framework Diagram"
                  className="max-w-full h-auto"
                />
              </div>
            </Grid.Col>
          </Grid>
        

<div data-slide>
  <RLApplications/>
</div>

{/* Limitations and Challenges */}
<div data-slide>
  <Title order={2} mb="xl" id="limitations">
    Key Limitations and Practical Challenges
  </Title>
  <Grid gutter="lg">
    <Grid.Col span={12}>
      <Stack spacing="md">
        <Paper className="p-6 bg-orange-50">
          <Stack spacing="md">
            <div>
              <Title order={4} className="mb-2">Computational Demands</Title>
              <Text>
                Training RL agents often requires massive computational resources. For instance, training AlphaGo involved:
                <ul className="list-disc pl-6 mt-2">
                  <li>Multiple TPU clusters</li>
                  <li>Millions of self-play games</li>
                  <li>Weeks to months of training time</li>
                </ul>
              </Text>
            </div>

            <div>
              <Title order={4} className="mb-2">Simulation Requirements</Title>
              <Text>
                Effective RL training typically needs:
                <ul className="list-disc pl-6 mt-2">
                  <li>Fast, accurate simulation environments</li>
                  <li>Realistic physics models for robotics applications</li>
                  <li>Complex multi-agent scenarios for real-world applications</li>
                </ul>
                Building these environments can be extremely challenging and resource-intensive.
              </Text>
            </div>

            <div>
              <Title order={4} className="mb-2">Alternative Approaches</Title>
              <Text>
                In many scenarios, simpler solutions are more effective:
                <ul className="list-disc pl-6 mt-2">
                  <li>Classical optimization for well-defined problems</li>
                  <li>Rule-based systems for clear decision spaces</li>
                  <li>Supervised learning for prediction tasks</li>
                </ul>
              </Text>
            </div>

            <div>
              <Title order={4} className="mb-2">Trust and Safety Concerns</Title>
              <Text>
                Critical questions remain about deploying RL in high-stakes scenarios:
                <ul className="list-disc pl-6 mt-2">
                  <li>How do we verify RL agent reliability?</li>
                  <li>Can we guarantee safe behavior in all scenarios?</li>
                  <li>What level of human oversight is needed?</li>
                </ul>
                By the end of this course, will you feel ready to board a plane piloted by an RL agent</Text>
            </div>
          </Stack>
        </Paper>
      </Stack>
    </Grid.Col>
  </Grid>
</div>
</Stack>
</Container>
);
};

export default Introduction;