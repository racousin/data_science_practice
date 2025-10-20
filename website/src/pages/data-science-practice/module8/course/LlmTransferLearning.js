import React from "react";
import { Text, Title, List, Table, Flex, Image } from '@mantine/core';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import CodeBlock from "components/CodeBlock";
import DataInteractionPanel from 'components/DataInteractionPanel';

const LlmTransferLearning = () => {
  // Notebook URLs for LoRA fine-tuning demonstration
  const notebookUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course_lora_finetuning.ipynb";
  const notebookHtmlUrl = process.env.PUBLIC_URL + "/modules/data-science-practice/module8/course/module8_course_lora_finetuning.html";
  const notebookColabUrl = process.env.PUBLIC_URL + "website/public/modules/data-science-practice/module8/course/module8_course_lora_finetuning.ipynb";

  const metadata = {
    description: "Complete LoRA fine-tuning demonstration: load a pretrained GPT-2 model, test it, then fine-tune it efficiently on TinyStories using LoRA.",
    source: "TinyStories Dataset",
    target: "Children's story generation",
    listData: [
      { name: "Pretrained Model", description: "GPT-2 (124M parameters) from HuggingFace" },
      { name: "LoRA Config", description: "Rank r=8, only ~0.3% parameters trained" },
      { name: "Comparison", description: "Before vs. after fine-tuning analysis" }
    ],
  };

  return (
    <>
      <div data-slide>
        <Title order={1}>LLM and Transfer Learning</Title>

        <Text mt="md">
          Transfer learning has become the dominant paradigm in natural language processing, enabling the use of
          pre-trained language models for a wide variety of downstream tasks with minimal additional training.
        </Text>

        <Text mt="md">
          This approach leverages massive pre-trained models trained on billions of tokens, allowing practitioners
          to achieve state-of-the-art performance without requiring extensive computational resources or large
          domain-specific datasets.
        </Text>
        <Flex>
                <Image
                  src="/assets/data-science-practice/module8/flop.jpeg"
                  alt="CPU vs GPU Architecture"
                  style={{ maxWidth: 'min(800px, 90vw)', height: 'auto' }}
                  fluid
                />
                            <Text component="p" ta="center" mt="xs" size="sm" c="dimmed">
                        Source: https://epoch.ai/blog/tracking-large-scale-ai-models</Text>
                
              </Flex>

        <Text mt="lg" fw={500}>
          Training LLaMA 3.1 405B: <InlineMath math="3.8 \times 10^{25}" /> FLOPs
        </Text>

        <Text mt="sm">
          On a Personal Computer (200 GFLOPS)
        </Text>

        <Text mt="xs">
          Time = Total FLOPs / FLOPS = <InlineMath math="1.9 \times 10^{14}" /> seconds = 6.0 million years
        </Text>

        <Table striped mt="lg">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Parameters</Table.Th>
              <Table.Th>Number GPU</Table.Th>
              <Table.Th>Training Time</Table.Th>
              <Table.Th>Estimated Cost</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>1B</Table.Td>
              <Table.Td>~10-50 GPUs</Table.Td>
              <Table.Td>Days to week</Table.Td>
              <Table.Td>$10K - $100K</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>10B</Table.Td>
              <Table.Td>~100-500 GPUs</Table.Td>
              <Table.Td>Weeks to Month</Table.Td>
              <Table.Td>$100K - $1M</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>100B+</Table.Td>
              <Table.Td>1,000+ GPUs</Table.Td>
              <Table.Td>Months</Table.Td>
              <Table.Td>$1M - $100M+</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="md">
          Training large foundation models requires substantial infrastructure and capital, making it accessible
          primarily to major technology companies and well-funded research labs such as Meta (Llama), OpenAI (GPT),
          Google (Gemini), Anthropic (Claude), and Mistral AI.
        </Text>

        <Table striped mt="lg">
          <Table.Thead>
            <Table.Tr>
              <Table.Th></Table.Th>
              <Table.Th>Training Set (Words)</Table.Th>
              <Table.Th>Training Set (Tokens)</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td fw={500}>Recent LLMs</Table.Td>
              <Table.Td></Table.Td>
              <Table.Td></Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td pl="xl">Llama 3</Table.Td>
              <Table.Td>11 trillion</Table.Td>
              <Table.Td>15T</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td pl="xl">GPT-4</Table.Td>
              <Table.Td>5 trillion</Table.Td>
              <Table.Td>6.5T</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td fw={500}>Humans</Table.Td>
              <Table.Td></Table.Td>
              <Table.Td></Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td pl="xl">Human, age 5</Table.Td>
              <Table.Td>30 million</Table.Td>
              <Table.Td></Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td pl="xl">Human, age 20</Table.Td>
              <Table.Td>150 million</Table.Td>
              <Table.Td></Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="md">
          Despite consuming orders of magnitude more data (15 trillion tokens vs 150 million words), current LLMs
          remain remarkably inefficient compared to human learning. A 20-year-old human achieves sophisticated language
          understanding and reasoning with approximately 10<sup>11</sup> (100 billion) times less training data than models like Llama 3,
          highlighting fundamental differences in learning efficiency and generalization capabilities.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Transfer Learning Overview</Title>

        <Text mt="md">
          Transfer learning in NLP follows a two-stage paradigm:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Pre-training:</strong> Train a large model on massive amounts of unlabeled text data using self-supervised objectives
          </List.Item>
          <List.Item>
            <strong>Fine-tuning:</strong> Adapt the pre-trained model to specific downstream tasks using smaller labeled datasets
          </List.Item>
        </List>

        <Text mt="lg">
          This paradigm can be mathematically expressed as:
        </Text>

        <BlockMath>
          {`\\theta^* = \\arg\\min_{\\theta} \\mathcal{L}_{\\text{pretrain}}(\\theta; \\mathcal{D}_{\\text{pretrain}})`}
        </BlockMath>

        <BlockMath>
          {`\\theta_{\\text{task}} = \\arg\\min_{\\theta} \\mathcal{L}_{\\text{task}}(\\theta; \\mathcal{D}_{\\text{task}}) \\quad \\text{initialized from } \\theta^*`}
        </BlockMath>

        <Text mt="md">
          Where <InlineMath math="\theta^*" /> represents pre-trained parameters and <InlineMath math="\theta_{\text{task}}" />
          represents task-specific parameters.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/transferlearning.png"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Pre-training Objectives</Title>

        <Text mt="md">
          Different pre-training objectives have been developed to teach models language understanding:
        </Text>

        <Title order={3} mt="lg">Causal Language Modeling (CLM)</Title>
        <Text>
          Predict the next token given previous tokens (used by GPT models):
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{\\text{CLM}} = -\\sum_{t=1}^{T} \\log P(w_t | w_1, \\ldots, w_{t-1}; \\theta)`}
        </BlockMath>

        <Title order={3} mt="lg">Masked Language Modeling (MLM)</Title>
        <Text>
          Predict masked tokens given surrounding context (used by BERT):
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{\\text{MLM}} = -\\sum_{i \\in \\mathcal{M}} \\log P(w_i | w_{\\setminus \\mathcal{M}}; \\theta)`}
        </BlockMath>

      </div>

      <div data-slide>
        <Title order={2}>Why Transfer Learning Works</Title>

        <Text mt="md">
          Transfer learning is effective because pre-trained models learn rich, hierarchical representations:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Low-level features:</strong> Syntax, grammar, and basic linguistic patterns
          </List.Item>
          <List.Item>
            <strong>Mid-level features:</strong> Semantic relationships, entity recognition, and compositional understanding
          </List.Item>
          <List.Item>
            <strong>High-level features:</strong> Abstract reasoning, world knowledge, and task-specific patterns
          </List.Item>
        </List>

        <Text mt="lg">
          These learned representations are transferable across tasks because natural language structure is
          consistent across different applications. The model captures:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Distributional semantics (words with similar contexts have similar meanings)</List.Item>
          <List.Item>Syntactic structure and grammatical relationships</List.Item>
          <List.Item>World knowledge and factual information</List.Item>
          <List.Item>Commonsense reasoning patterns</List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Hugging Face Ecosystem Overview</Title>

        <Text mt="md">
          Hugging Face has established itself as the central hub for NLP and machine learning, providing:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>
            <strong>Model Hub:</strong> Repository of over 2000,000 pre-trained models
          </List.Item>
          <List.Item>
            <strong>Datasets:</strong> Collection of over 500,000 datasets for various tasks
          </List.Item>
          <List.Item>
            <strong>Transformers Library:</strong> Unified API for working with transformer models
          </List.Item>
          <List.Item>
            <strong>PEFT Library:</strong> Parameter-efficient fine-tuning methods
          </List.Item>
          <List.Item>
            <strong>Accelerate:</strong> Distributed training and mixed precision support
          </List.Item>
          <List.Item>
            <strong>Tokenizers:</strong> Fast and efficient tokenization implementations
          </List.Item>
        </List>

        <Text mt="lg">
          This ecosystem provides a standardized framework for model development, training, and deployment,
          significantly reducing the barrier to entry for NLP applications.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/huggingface-ecosystem.png"
            alt="Hugging Face ecosystem components and their relationships"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Hugging Face Transformers Library</Title>

        <Text mt="md">
          The Transformers library provides three core abstractions:
        </Text>

        <Title order={3} mt="lg">Models</Title>
        <Text>
          Pre-trained neural networks that can be loaded and used directly:
        </Text>
        <List spacing="xs" mt="xs">
          <List.Item>AutoModel classes for automatic architecture detection</List.Item>
          <List.Item>Task-specific model classes (e.g., ForSequenceClassification)</List.Item>
          <List.Item>Support for multiple frameworks (PyTorch, TensorFlow, JAX)</List.Item>
        </List>

        <Title order={3} mt="lg">Tokenizers</Title>
        <Text>
          Convert text to numerical representations:
        </Text>
        <List spacing="xs" mt="xs">
          <List.Item>Fast Rust-based implementations</List.Item>
          <List.Item>Handle special tokens and padding automatically</List.Item>
          <List.Item>Support for various tokenization strategies (BPE, WordPiece, SentencePiece)</List.Item>
        </List>

        <Title order={3} mt="lg">Pipelines</Title>
        <Text>
          High-level APIs for common NLP tasks with minimal code.
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Loading Pre-trained Models</Title>

        <Text mt="md">
          Hugging Face provides two types of model loading APIs:
        </Text>

        <Title order={3} mt="lg">AutoModel: The "Body" Only</Title>
        <Text>
          Loads just the transformer encoder/decoder without any task-specific head.
          Returns raw contextualized embeddings.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoModel

# Loads the transformer body only
model = AutoModel.from_pretrained("bert-base-uncased")`}
        />

        <Text mt="md" size="sm">
          Use this when you want to build custom task heads or extract embeddings.
        </Text>

        <Title order={3} mt="lg">AutoModelForXXX: The "Body" + "Brain"</Title>
        <Text>
          Loads the transformer with a specialized head pre-configured for specific tasks.
          Ready to use for the target task.
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoModelForSequenceClassification

# Loads transformer + classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)`}
        />

        <Text mt="md" size="sm">
          Use this for common NLP tasks with pre-configured architectures.
        </Text>


        <Title order={3} mt="lg">Common Task-Specific Model Classes</Title>

        <List spacing="xs" mt="xs" size="sm">
          <List.Item>AutoModelForCausalLM - Text generation (GPT-style)</List.Item>
          <List.Item>AutoModelForSeq2SeqLM - Translation, summarization (T5-style)</List.Item>
          <List.Item>AutoModelForTokenClassification - Named entity recognition</List.Item>
          <List.Item>AutoModelForQuestionAnswering - Extractive QA</List.Item>
        </List>

        <Flex justify="center" mt="lg" mb="md">
          <Image
            src="/assets/data-science-practice/module8/hf.png"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Tokenizers in Hugging Face</Title>

        <Text mt="md">
          Tokenizers convert text to input tensors that models can process:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Transfer learning accelerates NLP development."
inputs = tokenizer(text, return_tensors="pt")`}
        />

        <Text mt="md">
          AutoTokenizer automatically loads the correct tokenizer for your model, eliminating the need
          to know which specific tokenizer implementation to use.
        </Text>

        <Text mt="md">
          Without AutoTokenizer, you would need to know the specific tokenizer for each model:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer

# BERT uses WordPiece
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# GPT-2 uses Byte-Pair Encoding (BPE)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# T5 uses SentencePiece
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")`}
        />

        <Text mt="md" size="sm">
          Different models use different tokenization strategies. AutoTokenizer handles these differences
          automatically based on the model configuration.
        </Text>

        <Text mt="md">
          The tokenizer returns a dictionary with:
        </Text>

        <List spacing="xs" mt="xs" size="sm">
          <List.Item><strong>input_ids:</strong> Token IDs</List.Item>
          <List.Item><strong>attention_mask:</strong> Mask indicating real vs padding tokens</List.Item>
          <List.Item><strong>token_type_ids:</strong> Segment IDs (for models like BERT)</List.Item>
        </List>

        <Text mt="lg">
          Encoding and decoding:
        </Text>

        <CodeBlock
          language="python"
          code={`# Encode text to IDs
ids = tokenizer.encode("Hello world")

# Decode IDs back to text
text = tokenizer.decode(ids, skip_special_tokens=True)`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Model Inference Example</Title>

        <Text mt="md">
          Using a pre-trained model for text generation:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)`}
        />

        <Text mt="md">
          Generate text:
        </Text>

        <CodeBlock
          language="python"
          code={`prompt = "The future of artificial intelligence"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    temperature=0.8,
    do_sample=True,
    top_p=0.9
)

generated_text = tokenizer.decode(outputs[0])`}
        />

        <Text mt="md" size="sm">
          Key generation parameters: temperature (controls randomness), top_p (nucleus sampling),
          max_length (maximum tokens), and do_sample (whether to use sampling vs greedy decoding).
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/text-generation-sampling.png"
            alt="Text generation strategies: greedy, sampling, and nucleus sampling"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Zero-Shot Learning</Title>

        <Text mt="md">
          Zero-shot learning allows models to perform tasks without task-specific training by leveraging
          their pre-trained knowledge and appropriate prompting.
        </Text>

        <Text mt="md">
          Definition: The ability to perform a task without seeing any examples during training:
        </Text>

        <BlockMath>
          {`f: \\mathcal{X} \\rightarrow \\mathcal{Y} \\quad \\text{where } \\mathcal{Y} \\text{ contains unseen classes}`}
        </BlockMath>

        <Text mt="lg">
          Zero-shot capabilities emerge from:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Large-scale pre-training on diverse data</List.Item>
          <List.Item>Instruction tuning on multiple task types</List.Item>
          <List.Item>Natural language understanding of task descriptions</List.Item>
          <List.Item>Learned reasoning and generalization patterns</List.Item>
        </List>

        <Text mt="lg">
          Common zero-shot applications include sentiment analysis, topic classification, translation,
          summarization, and question answering.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/zero-shot-learning.png"
            alt="Zero-shot learning framework and application examples"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Zero-Shot Classification Example</Title>

        <Text mt="md">
          Using the Hugging Face pipeline for zero-shot classification:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)`}
        />

        <Text mt="md">
          Classify text without training:
        </Text>

        <CodeBlock
          language="python"
          code={`text = "The stock market reached new highs today."
candidate_labels = ["business", "sports", "politics", "technology"]

result = classifier(text, candidate_labels)
print(result)`}
        />

        <Text mt="md">
          Output structure:
        </Text>

        <CodeBlock
          language="python"
          code={`{
    'sequence': 'The stock market reached new highs today.',
    'labels': ['business', 'politics', 'technology', 'sports'],
    'scores': [0.95, 0.03, 0.01, 0.01]
}`}
        />

        <Text mt="md" size="sm">
          The model converts classification into natural language inference (NLI) by checking if the
          sequence entails a hypothesis like "This text is about business."
        </Text>
      </div>

      <div data-slide>
        <Title order={2}>Few-Shot Learning</Title>

        <Text mt="md">
          Few-shot learning (also called in-context learning) enables models to adapt to tasks using
          only a few examples provided in the prompt.
        </Text>

        <Text mt="md">
          Formally, given k examples <InlineMath math="\{(x_i, y_i)\}_{i=1}^k" /> and a new input x,
          predict y:
        </Text>

        <BlockMath>
          {`P(y | x, \\{(x_1, y_1), \\ldots, (x_k, y_k)\\})`}
        </BlockMath>

        <Text mt="lg">
          Example prompt structure:
        </Text>

        <CodeBlock
          language="python"
          code={`prompt = """Classify sentiment as positive or negative.

Text: The service was excellent.
Sentiment: positive

Text: The food was terrible.
Sentiment: negative

Text: The room was clean and comfortable.
Sentiment:"""`}
        />

        <Text mt="md">
          Few-shot learning works because large language models can recognize patterns and adapt their
          behavior based on in-context examples without parameter updates.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/few-shot-learning.png"
            alt="Few-shot learning and in-context learning mechanism"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Traditional Fine-Tuning</Title>

        <Text mt="md">
          Traditional fine-tuning updates all (or most) model parameters on a downstream task:
        </Text>

        <BlockMath>
          {`\\theta_{\\text{new}} = \\theta_{\\text{pretrained}} - \\eta \\nabla_{\\theta} \\mathcal{L}_{\\text{task}}`}
        </BlockMath>

        <Text mt="md">
          Where all parameters in <InlineMath math="\theta" /> are updated via gradient descent.
        </Text>

        <Title order={3} mt="lg">Advantages</Title>
        <List spacing="xs" mt="xs">
          <List.Item>Highest performance on the target task</List.Item>
          <List.Item>Full model capacity for task adaptation</List.Item>
          <List.Item>Well-established training procedures</List.Item>
        </List>

        <Title order={3} mt="lg">Disadvantages</Title>
        <List spacing="xs" mt="xs">
          <List.Item>High memory requirements (stores gradients for all parameters)</List.Item>
          <List.Item>Computationally expensive</List.Item>
          <List.Item>Risk of catastrophic forgetting</List.Item>
          <List.Item>Requires storing full model copy for each task</List.Item>
        </List>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/fine-tuning-process.png"
            alt="Traditional fine-tuning process and parameter updates"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Fine-Tuning Process</Title>

        <Text mt="md">
          Fine-tuning typically involves careful consideration of which layers to update:
        </Text>

        <Title order={3} mt="lg">Freezing Strategies</Title>

        <Text mt="sm">
          <strong>Full fine-tuning:</strong> Update all parameters
        </Text>
        <BlockMath>
          {`\\theta = \\{\\theta_{\\text{encoder}}, \\theta_{\\text{head}}\\} \\quad \\text{all trainable}`}
        </BlockMath>

        <Text mt="md">
          <strong>Frozen backbone:</strong> Only update task-specific head
        </Text>
        <BlockMath>
          {`\\theta_{\\text{encoder}} \\text{ frozen}, \\quad \\theta_{\\text{head}} \\text{ trainable}`}
        </BlockMath>

        <Text mt="md">
          <strong>Gradual unfreezing:</strong> Progressively unfreeze layers from top to bottom
        </Text>

        <Title order={3} mt="lg">Learning Rate Considerations</Title>
        <Text>
          Use smaller learning rates than pre-training (typically 1e-5 to 5e-5) to avoid catastrophic
          forgetting. Often use discriminative learning rates:
        </Text>

        <List spacing="xs" mt="xs" size="sm">
          <List.Item>Lower layers: smaller learning rate (e.g., 1e-5)</List.Item>
          <List.Item>Higher layers: larger learning rate (e.g., 3e-5)</List.Item>
          <List.Item>Task head: largest learning rate (e.g., 5e-5)</List.Item>
        </List>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/layer-freezing-strategies.png"
            alt="Layer freezing and gradual unfreezing strategies"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Fine-Tuning Implementation</Title>

        <Text mt="md">
          Using the Hugging Face Trainer API:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")`}
        />

        <Text mt="md">
          Tokenize data:
        </Text>

        <CodeBlock
          language="python"
          code={`tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                     truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)`}
        />

        <Text mt="md">
          Configure training:
        </Text>

        <CodeBlock
          language="python"
          code={`training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Parameter Efficient Fine-Tuning (PEFT)</Title>

        <Text mt="md">
          Parameter-efficient fine-tuning methods update only a small subset of parameters while
          keeping the majority of the pre-trained model frozen.
        </Text>

        <Text mt="md">
          Key motivation: For a model with N parameters, full fine-tuning requires:
        </Text>

        <List spacing="xs" mt="xs" size="sm">
          <List.Item>Memory for N parameters</List.Item>
          <List.Item>Memory for N gradients</List.Item>
          <List.Item>Memory for N optimizer states (e.g., momentum, variance for Adam)</List.Item>
          <List.Item>Storage of N parameters per task</List.Item>
        </List>

        <Text mt="lg">
          PEFT methods aim to achieve comparable performance while updating only a small fraction
          of parameters (typically 0.1-1% of total parameters), dramatically reducing memory and
          storage requirements.
        </Text>

        <Text mt="md">
          General PEFT formulation:
        </Text>

        <BlockMath>
          {`\\theta = \\{\\theta_{\\text{frozen}}, \\theta_{\\text{trainable}}\\} \\quad \\text{where } |\\theta_{\\text{trainable}}| \\ll |\\theta_{\\text{frozen}}|`}
        </BlockMath>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/peft-comparison.png"
            alt="Parameter efficient fine-tuning methods comparison"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>LoRA: Low-Rank Adaptation</Title>

        <Text mt="md">
          LoRA (Hu et al., 2021) represents weight updates using low-rank decomposition, significantly
          reducing the number of trainable parameters.
        </Text>

        <Text mt="md">
          Core principle: Weight updates during fine-tuning have low intrinsic rank. Instead of updating
          the full weight matrix W, LoRA introduces trainable low-rank matrices:
        </Text>

        <BlockMath>
          {`W' = W_0 + \\Delta W = W_0 + BA`}
        </BlockMath>

        <Text mt="md">
          Where:
        </Text>

        <List spacing="xs" mt="xs">
          <List.Item>
            <InlineMath math="W_0 \in \mathbb{R}^{d \times k}" /> is the frozen pre-trained weight matrix
          </List.Item>
          <List.Item>
            <InlineMath math="B \in \mathbb{R}^{d \times r}" /> is a trainable matrix
          </List.Item>
          <List.Item>
            <InlineMath math="A \in \mathbb{R}^{r \times k}" /> is a trainable matrix
          </List.Item>
          <List.Item>
            <InlineMath math="r \ll \min(d, k)" /> is the rank (typically 4-64)
          </List.Item>
        </List>

        <Text mt="lg" size="sm" fs="italic">
          Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) - https://arxiv.org/abs/2106.09685
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/lora-architecture-diagram.png"
            alt="LoRA architecture showing low-rank decomposition matrices"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>LoRA Architecture</Title>

        <Text mt="md">
          During forward pass, the computation becomes:
        </Text>

        <BlockMath>
          {`h = W_0 x + BAx`}
        </BlockMath>

        <Text mt="md">
          Where <InlineMath math="W_0" /> remains frozen and only B and A are updated via backpropagation.
        </Text>

        <Title order={3} mt="lg">Initialization</Title>

        <List spacing="xs" mt="xs">
          <List.Item>
            Matrix A is initialized with random Gaussian values: <InlineMath math="A \sim \mathcal{N}(0, \sigma^2)" />
          </List.Item>
          <List.Item>
            Matrix B is initialized to zero: <InlineMath math="B = 0" />
          </List.Item>
          <List.Item>
            This ensures <InlineMath math="\Delta W = BA = 0" /> at initialization
          </List.Item>
        </List>

        <Title order={3} mt="lg">Scaling</Title>
        <Text>
          The update is scaled by a factor to control learning rate:
        </Text>

        <BlockMath>
          {`h = W_0 x + \\frac{\\alpha}{r} BAx`}
        </BlockMath>

        <Text mt="md">
          Where <InlineMath math="\alpha" /> is a hyperparameter (typically set to 16 or 32).
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/lora-forward-pass.png"
            alt="LoRA forward pass computation flow"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>LoRA Parameters and Complexity</Title>

        <Text mt="md">
          Parameter reduction calculation:
        </Text>

        <Title order={3} mt="md">Original Parameters</Title>
        <BlockMath>
          {`\#\\text{params}_{\\text{full}} = d \\times k`}
        </BlockMath>

        <Title order={3} mt="md">LoRA Parameters</Title>
        <BlockMath>
          {`\#\\text{params}_{\\text{LoRA}} = r \\times (d + k)`}
        </BlockMath>

        <Title order={3} mt="md">Reduction Ratio</Title>
        <BlockMath>
          {`\\frac{\#\\text{params}_{\\text{LoRA}}}{\#\\text{params}_{\\text{full}}} = \\frac{r(d + k)}{dk}`}
        </BlockMath>

        <Text mt="lg">
          Example: For a weight matrix of shape 4096 × 4096 with rank r = 8:
        </Text>

        <List spacing="xs" mt="xs" size="sm">
          <List.Item>Full parameters: 4096 × 4096 = 16,777,216</List.Item>
          <List.Item>LoRA parameters: 8 × (4096 + 4096) = 65,536</List.Item>
          <List.Item>Reduction: 0.39% of original parameters</List.Item>
          <List.Item>Memory savings: 256× reduction</List.Item>
        </List>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/lora-parameter-efficiency.png"
            alt="LoRA parameter reduction and memory savings visualization"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>LoRA Implementation</Title>

        <Text mt="md">
          Using the PEFT library from Hugging Face:
        </Text>

        <CodeBlock
          language="python"
          code={`from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)`}
        />

        <Text mt="md">
          Configure LoRA:
        </Text>

        <CodeBlock
          language="python"
          code={`lora_config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=32,                # Scaling factor
    target_modules=["c_attn"],    # Which modules to apply LoRA
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()`}
        />

        <Text mt="md" size="sm">
          Output: "trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.237"
        </Text>

        <Text mt="lg">
          Training proceeds normally with the Trainer API, but only LoRA parameters are updated.
        </Text>
      </div>


      <div data-slide>
        <Title order={2}>Other PEFT Methods</Title>

        <Title order={3} mt="md">Prefix Tuning</Title>
        <Text>
          Prepends trainable prefix vectors to keys and values at each layer:
        </Text>
        <BlockMath>
          {`[P_K; K], [P_V; V] \\quad \\text{where } P_K, P_V \\in \\mathbb{R}^{l \\times d}`}
        </BlockMath>
        <Text size="sm">
          Only prefix parameters are trained; l is the prefix length (typically 10-100 tokens).
        </Text>

        <Title order={3} mt="lg">Adapters</Title>
        <Text>
          Inserts small bottleneck layers between transformer blocks:
        </Text>
        <BlockMath>
          {`h' = h + f(hW_{\\text{down}})W_{\\text{up}}`}
        </BlockMath>
        <Text size="sm">
          Where <InlineMath math="W_{\text{down}} \in \mathbb{R}^{d \times r}" /> and
          <InlineMath math="W_{\text{up}} \in \mathbb{R}^{r \times d}" /> with <InlineMath math="r \ll d" />.
        </Text>

        <Title order={3} mt="lg">Prompt Tuning</Title>
        <Text>
          Adds trainable soft prompts to input embeddings:
        </Text>
        <BlockMath>
          {`[P; E(x)] \\quad \\text{where } P \\in \\mathbb{R}^{l \\times d}`}
        </BlockMath>
        <Text size="sm">
          Only the prompt embedding matrix P is trained; extremely parameter-efficient (as few as 20 tokens).
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/peft-methods-comparison.png"
            alt="Comparison of Prefix Tuning, Adapters, and Prompt Tuning architectures"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Knowledge Distillation</Title>

        <Text mt="md">
          Knowledge distillation transfers knowledge from a large teacher model to a smaller student model,
          creating efficient models that retain much of the teacher's performance.
        </Text>

        <Text mt="md">
          Framework:
        </Text>

        <List spacing="xs" mt="xs">
          <List.Item>
            <strong>Teacher model:</strong> Large, high-performance model <InlineMath math="f_T(\theta_T)" />
          </List.Item>
          <List.Item>
            <strong>Student model:</strong> Smaller, efficient model <InlineMath math="f_S(\theta_S)" />
          </List.Item>
          <List.Item>
            <strong>Goal:</strong> Train student to mimic teacher's behavior
          </List.Item>
        </List>

        <Text mt="lg">
          Distillation works by training the student on soft targets (probability distributions) from
          the teacher rather than hard labels, providing richer learning signals.
        </Text>

        <Text mt="md" size="sm" fs="italic">
          Reference: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015) - https://arxiv.org/abs/1503.02531
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/knowledge-distillation-diagram.png"
            alt="Knowledge distillation from teacher to student model"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Distillation Loss</Title>

        <Text mt="md">
          The student is trained using a combination of two losses:
        </Text>

        <Title order={3} mt="md">Cross-Entropy Loss</Title>
        <Text>
          Standard task loss with ground truth labels:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{\\text{CE}} = -\\sum_{i} y_i \\log P_S(y_i | x)`}
        </BlockMath>

        <Title order={3} mt="lg">Knowledge Distillation Loss</Title>
        <Text>
          KL divergence between teacher and student output distributions:
        </Text>
        <BlockMath>
          {`\\mathcal{L}_{\\text{KD}} = \\text{KL}(P_T^{\\tau} || P_S^{\\tau})`}
        </BlockMath>

        <Text mt="md">
          Where temperature-scaled probabilities are:
        </Text>
        <BlockMath>
          {`P_i^{\\tau} = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}`}
        </BlockMath>

        <Title order={3} mt="lg">Combined Loss</Title>
        <BlockMath>
          {`\\mathcal{L} = \\alpha \\mathcal{L}_{\\text{CE}} + (1-\\alpha) \\tau^2 \\mathcal{L}_{\\text{KD}}`}
        </BlockMath>

        <Text mt="md" size="sm">
          Temperature <InlineMath math="\tau > 1" /> softens probability distributions, and
          <InlineMath math="\alpha" /> balances the two objectives (typically 0.1-0.3).
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/distillation-loss-components.png"
            alt="Components of distillation loss and temperature scaling"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Distillation Implementation</Title>

        <Text mt="md">
          Basic distillation training loop:
        </Text>

        <CodeBlock
          language="python"
          code={`import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=2.0, alpha=0.1):
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss
    kd_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean')
    kd_loss = kd_loss * (temperature ** 2)

    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * ce_loss + (1 - alpha) * kd_loss`}
        />

        <Text mt="md">
          Training step:
        </Text>

        <CodeBlock
          language="python"
          code={`# Get teacher predictions (no gradient)
with torch.no_grad():
    teacher_logits = teacher_model(inputs)

# Get student predictions
student_logits = student_model(inputs)

# Compute distillation loss
loss = distillation_loss(
    student_logits, teacher_logits, labels,
    temperature=2.0, alpha=0.1
)

loss.backward()
optimizer.step()`}
        />
      </div>

      <div data-slide>
        <Title order={2}>Model Compression Comparison</Title>

        <Text mt="md">
          Different compression techniques serve different purposes:
        </Text>

        <Title order={3} mt="md">Pruning</Title>
        <Text>
          Removes unnecessary weights or neurons based on importance metrics. Reduces parameters
          and computation but requires careful selection of what to prune.
        </Text>

        <Title order={3} mt="lg">Quantization</Title>
        <Text>
          Reduces numerical precision (e.g., FP32 to INT8 or INT4). Significantly reduces memory
          and speeds up inference with minimal accuracy loss.
        </Text>

        <Title order={3} mt="lg">Knowledge Distillation</Title>
        <Text>
          Creates a smaller architecture trained to mimic a larger model. Provides the most
          flexibility in architecture design.
        </Text>

        <Title order={3} mt="lg">PEFT (LoRA, etc.)</Title>
        <Text>
          Keeps base model frozen and adds small trainable modules. Optimizes training efficiency
          rather than inference efficiency.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/compression-techniques.png"
            alt="Overview of model compression techniques and their tradeoffs"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Computational Efficiency Comparison</Title>

        <Table striped mt="md">
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Method</Table.Th>
              <Table.Th>Training Memory</Table.Th>
              <Table.Th>Inference Speed</Table.Th>
              <Table.Th>Model Size</Table.Th>
              <Table.Th>Performance</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            <Table.Tr>
              <Table.Td>Full Fine-tuning</Table.Td>
              <Table.Td>Very High</Table.Td>
              <Table.Td>Baseline</Table.Td>
              <Table.Td>Full Size</Table.Td>
              <Table.Td>Best</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>LoRA</Table.Td>
              <Table.Td>Low (0.1-1% params)</Table.Td>
              <Table.Td>Baseline</Table.Td>
              <Table.Td>Full Size + Adapters</Table.Td>
              <Table.Td>Very Good</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>QLoRA</Table.Td>
              <Table.Td>Very Low (4-bit)</Table.Td>
              <Table.Td>Slower (quantized)</Table.Td>
              <Table.Td>25% of Full Size</Table.Td>
              <Table.Td>Very Good</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Distillation</Table.Td>
              <Table.Td>Medium</Table.Td>
              <Table.Td>Much Faster</Table.Td>
              <Table.Td>30-50% of Full</Table.Td>
              <Table.Td>Good</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Quantization Only</Table.Td>
              <Table.Td>N/A (post-training)</Table.Td>
              <Table.Td>2-4× Faster</Table.Td>
              <Table.Td>25-50% of Full</Table.Td>
              <Table.Td>Good to Very Good</Table.Td>
            </Table.Tr>
            <Table.Tr>
              <Table.Td>Pruning</Table.Td>
              <Table.Td>Medium</Table.Td>
              <Table.Td>Faster (sparse)</Table.Td>
              <Table.Td>50-90% of Full</Table.Td>
              <Table.Td>Good</Table.Td>
            </Table.Tr>
          </Table.Tbody>
        </Table>

        <Text mt="md" size="sm">
          Note: Metrics are approximate and depend on specific implementation and model architecture.
        </Text>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/efficiency-comparison-chart.png"
            alt="Visual comparison of computational efficiency metrics across methods"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>Best Practices for Fine-Tuning</Title>

        <Title order={3} mt="md">Data Preparation</Title>
        <List spacing="xs" mt="xs">
          <List.Item>Ensure data quality and relevance to target task</List.Item>
          <List.Item>Balance dataset classes to avoid bias</List.Item>
          <List.Item>Use appropriate train/validation/test splits (typically 80/10/10)</List.Item>
          <List.Item>Apply same preprocessing as used in pre-training</List.Item>
        </List>

        <Title order={3} mt="lg">Hyperparameter Selection</Title>
        <List spacing="xs" mt="xs">
          <List.Item>Start with learning rate 1e-5 to 5e-5 for full fine-tuning</List.Item>
          <List.Item>Use learning rate warmup (typically 3-10% of total steps)</List.Item>
          <List.Item>Apply weight decay (0.01) for regularization</List.Item>
          <List.Item>Use gradient clipping (max_grad_norm=1.0) to prevent instability</List.Item>
        </List>

        <Title order={3} mt="lg">Monitoring and Evaluation</Title>
        <List spacing="xs" mt="xs">
          <List.Item>Track both training and validation metrics</List.Item>
          <List.Item>Use early stopping based on validation performance</List.Item>
          <List.Item>Save checkpoints regularly during training</List.Item>
          <List.Item>Evaluate on held-out test set only once at the end</List.Item>
        </List>

        <Flex justify="center" mt="xl" mb="md">
          <Image
            src="/assets/data-science-practice/module8/fine-tuning-best-practices.png"
            alt="Best practices workflow for fine-tuning LLMs"
            style={{ maxWidth: "100%", height: "auto" }}
          />
        </Flex>
      </div>

      <div data-slide>
        <Title order={2}>References</Title>

        <List spacing="md" mt="lg">
          <List.Item>
            <Text size="sm">
              Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
              <br />
              https://arxiv.org/abs/2106.09685
            </Text>
          </List.Item>
          <List.Item>
            <Text size="sm">
              Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
              <br />
              https://arxiv.org/abs/2305.14314
            </Text>
          </List.Item>
          <List.Item>
            <Text size="sm">
              Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
              <br />
              https://arxiv.org/abs/1503.02531
            </Text>
          </List.Item>
          <List.Item>
            <Text size="sm">
              Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)" (2020)
              <br />
              https://arxiv.org/abs/1910.10683
            </Text>
          </List.Item>
          <List.Item>
            <Text size="sm">
              Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
              <br />
              https://arxiv.org/abs/1810.04805
            </Text>
          </List.Item>
          <List.Item>
            <Text size="sm">
              Brown et al., "Language Models are Few-Shot Learners" (2020)
              <br />
              https://arxiv.org/abs/2005.14165
            </Text>
          </List.Item>
        </List>
      </div>

      <div data-slide>
        <Title order={2}>Hands-On: LoRA and PEFT Methods Demo</Title>

        <Text mt="md">
          This interactive notebook demonstrates the complete PEFT workflow, comparing LoRA, Prefix Tuning, and IA3 on GPT-2.
        </Text>

        <Text mt="md">
          In this hands-on demonstration, you will:
        </Text>

        <List spacing="sm" mt="md">
          <List.Item>Load GPT-2 (124M parameters) and test its pretrained capabilities</List.Item>
          <List.Item>Configure and compare three PEFT methods: LoRA, Prefix Tuning, and IA3</List.Item>
          <List.Item>Analyze parameter efficiency across methods (0.01% to 0.5% trainable)</List.Item>
          <List.Item>Fine-tune GPT-2 on TinyStories using LoRA with only ~300K trainable parameters</List.Item>
          <List.Item>Compare story generation before and after fine-tuning</List.Item>
          <List.Item>Explore storage efficiency and adapter weight sharing</List.Item>
        </List>

        <Text mt="lg">
          This demonstration showcases how PEFT enables efficient adaptation of large pretrained models to specific
          domains with minimal computational resources, making state-of-the-art models accessible for fine-tuning.
        </Text>

        <DataInteractionPanel
          notebookUrl={notebookUrl}
          notebookHtmlUrl={notebookHtmlUrl}
          notebookColabUrl={notebookColabUrl}
          metadata={metadata}
        />
      </div>
    </>
  );
};

export default LlmTransferLearning;
