import React from "react";
import { Container, Row, Col, Card, Alert, ListGroup } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { FaLightbulb, FaCode, FaExclamationTriangle, FaTools, FaRocket } from "react-icons/fa";

const TransferLearning = () => {
  return (
    <Container className="py-4">
      <h1>Transfer Learning in NLP</h1>
      
      <Row className="mb-4">
        <Col>
          <p className="lead">
            Transfer learning has revolutionized NLP by allowing us to leverage pre-trained language models 
            and adapt them to specific tasks with minimal training data. This approach has become the foundation 
            of modern NLP systems.
          </p>
        </Col>
      </Row>

      <section id="transfer-learning-fundamentals" className="mb-5">
        <h2>1. Transfer Learning Fundamentals</h2>
        
        <Row className="mb-3">
          <Col>
            <p>
              Transfer learning involves taking knowledge gained from solving one problem and applying it 
              to a different but related problem. In NLP, this typically means:
            </p>
            
            <ListGroup variant="flush" className="mb-3">
              <ListGroup.Item>
                <strong>Pre-training:</strong> Training a model on a large general corpus to learn language patterns
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Fine-tuning:</strong> Adapting the pre-trained model to a specific downstream task
              </ListGroup.Item>
            </ListGroup>
            
            <Card className="mb-3 border-info">
              <Card.Header className="bg-info text-white">
                <FaLightbulb className="me-2" />Why Transfer Learning Works
              </Card.Header>
              <Card.Body>
                <p>
                  Language models pre-trained on large corpora learn:
                </p>
                <ul>
                  <li>Linguistic patterns and regularities</li>
                  <li>Semantic relationships between words</li>
                  <li>Syntactic structures and grammatical rules</li>
                  <li>Contextual word representations</li>
                </ul>
                <p>
                  This knowledge is transferable to many downstream tasks, requiring fewer 
                  task-specific examples and less training time.
                </p>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        <Row>
          <Col md={6}>
            <Card className="h-100">
              <Card.Header>Traditional ML Approach</Card.Header>
              <Card.Body>
                <ul>
                  <li>Train from scratch for each task</li>
                  <li>Requires large task-specific datasets</li>
                  <li>Long training times</li>
                  <li>Limited generalization</li>
                </ul>
              </Card.Body>
            </Card>
          </Col>
          <Col md={6}>
            <Card className="h-100">
              <Card.Header>Transfer Learning Approach</Card.Header>
              <Card.Body>
                <ul>
                  <li>Start with pre-trained model</li>
                  <li>Fine-tune on smaller task-specific datasets</li>
                  <li>Shorter training times</li>
                  <li>Better generalization</li>
                </ul>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </section>

      <section id="transfer-learning-paradigms" className="mb-5">
        <h2>2. Transfer Learning Paradigms in NLP</h2>
        
        <Row className="mb-4">
          <Col md={6}>
            <Card className="mb-3 h-100">
              <Card.Header>Feature-based Transfer</Card.Header>
              <Card.Body>
                <p>Use representations from pre-trained models as features for task-specific models.</p>
                <p><strong>Examples:</strong> Word2Vec, GloVe, ELMo</p>
                <p><strong>Use case:</strong> When you need fixed representations or have computational constraints</p>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6}>
            <Card className="mb-3 h-100">
              <Card.Header>Fine-tuning</Card.Header>
              <Card.Body>
                <p>Update all or part of a pre-trained model's parameters for a downstream task.</p>
                <p><strong>Examples:</strong> BERT, RoBERTa, GPT models</p>
                <p><strong>Use case:</strong> When you want task-specific contextual representations</p>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        <Alert variant="info">
          <FaLightbulb className="me-2" />
          <strong>Key Insight:</strong> While the feature-based approach treats pre-trained models as fixed feature extractors,
          fine-tuning allows the model to adapt its representations to the target task.
        </Alert>
      </section>

      <section id="huggingface-ecosystem" className="mb-5">
        <h2>3. The Hugging Face Ecosystem</h2>
        
        <p>
          Hugging Face has become the go-to platform for transfer learning in NLP, providing:
        </p>
        
        <Row className="mb-4">
          <Col md={4}>
            <Card className="h-100">
              <Card.Header className="bg-primary text-white">Transformers Library</Card.Header>
              <Card.Body>
                <p>Access to state-of-the-art pre-trained models and architectures</p>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={4}>
            <Card className="h-100">
              <Card.Header className="bg-primary text-white">Datasets Library</Card.Header>
              <Card.Body>
                <p>Standardized access to NLP datasets for benchmarking and training</p>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={4}>
            <Card className="h-100">
              <Card.Header className="bg-primary text-white">Model Hub</Card.Header>
              <Card.Body>
                <p>Community-driven repository of pre-trained models and fine-tuned variants</p>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        <Row>
          <Col>
            <h4>Installing Hugging Face Libraries</h4>
            <CodeBlock language="bash" code={`pip install transformers datasets evaluate`} />
          </Col>
        </Row>
      </section>

      <section id="practical-examples" className="mb-5">
        <h2>4. Practical Examples with Hugging Face</h2>
        
        <section id="example-loading-models" className="mb-4">
          <h3>4.1 Loading Pre-trained Models</h3>
          
          <p>
            Hugging Face makes it incredibly easy to load pre-trained models for use in your applications:
          </p>
          
          <CodeBlock language="python" code={`from transformers import AutoModel, AutoTokenizer

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Example text
text = "Transfer learning has revolutionized NLP applications."

# Tokenize and prepare for the model
inputs = tokenizer(text, return_tensors="pt")

# Get model outputs
outputs = model(**inputs)

# Access the last hidden states
last_hidden_states = outputs.last_hidden_state`} />
          
          <Alert variant="info">
            <FaLightbulb className="me-2" />
            <strong>Tip:</strong> The <code>Auto*</code> classes automatically select the right model type based on the model name,
            making your code more flexible when switching between different architectures.
          </Alert>
        </section>

        <section id="example-sentiment-analysis" className="mb-4">
          <h3>4.2 Fine-tuning for Sentiment Analysis</h3>
          
          <p>
            Let's see how to fine-tune a pre-trained model for sentiment analysis using the SST-2 dataset:
          </p>
          
          <CodeBlock language="python" code={`from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from evaluate import load

# 1. Load dataset
dataset = load_dataset("glue", "sst2")

# 2. Load tokenizer and tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Load pre-trained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2
)

# 4. Define evaluation metric
accuracy = load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# 7. Train the model
trainer.train()`} />

          <Alert variant="success">
            <FaRocket className="me-2" />
            <strong>Time-saving feature:</strong> Hugging Face's <code>Trainer</code> class handles the training loop,
            evaluation, and logging, allowing you to focus on model architecture and hyperparameters.
          </Alert>
        </section>

        <section id="example-text-generation" className="mb-4">
          <h3>4.3 Text Generation with GPT</h3>
          
          <p>
            Fine-tuning a generative model like GPT for custom text generation tasks:
          </p>
          
          <CodeBlock language="python" code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 1. Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token (GPT2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare dataset (assume we have a text file called 'custom_data.txt')
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

train_dataset = load_dataset("custom_data.txt", tokenizer)

# 3. Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're not doing masked language modeling with GPT
)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-custom",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 5. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 6. Train the model
trainer.train()

# 7. Generate text with fine-tuned model
prompt = "Transfer learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(
    input_ids, 
    max_length=50, 
    num_return_sequences=1, 
    temperature=0.7,
    do_sample=True,
    no_repeat_ngram_size=2,
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)`} />
        </section>

        <section id="example-question-answering" className="mb-4">
          <h3>4.4 Fine-tuning for Question Answering</h3>
          
          <p>
            Adapting a pre-trained model for extractive question answering:
          </p>
          
          <CodeBlock language="python" code={`from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# 1. Load SQuAD dataset
datasets = load_dataset("squad")

# 2. Load tokenizer
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 3. Tokenize dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Map from token to chars position in the original context
    offset_mapping = inputs.pop("offset_mapping")
    
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    # Loop through examples
    for i, offset in enumerate(offset_mapping):
        # Get start/end char position of the answer
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        # Find the start and end token positions
        start_token = 0
        end_token = 0
        
        # Find tokens that contain the answer
        for idx, (start, end) in enumerate(offset):
            if start <= start_char < end:
                start_token = idx
            if start <= end_char <= end:
                end_token = idx
                break
        
        start_positions.append(start_token)
        end_positions.append(end_token)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)

# 4. Load model for question answering
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./qa-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 6. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 7. Train the model
trainer.train()`} />

          <Alert variant="warning">
            <FaExclamationTriangle className="me-2" />
            <strong>Note:</strong> Question answering requires careful handling of token positions to map between
            the model's tokenization and the original text positions.
          </Alert>
        </section>
      </section>

      <section id="advanced-techniques" className="mb-5">
        <h2>5. Advanced Transfer Learning Techniques</h2>
        
        <Row className="mb-4">
          <Col md={6}>
            <Card className="h-100">
              <Card.Header>Parameter-Efficient Fine-tuning (PEFT)</Card.Header>
              <Card.Body>
                <p>
                  PEFT methods update only a small subset of model parameters, reducing memory requirements
                  and training time while maintaining performance.
                </p>
                <h5>Popular PEFT Methods:</h5>
                <ul>
                  <li>LoRA (Low-Rank Adaptation)</li>
                  <li>Adapter layers</li>
                  <li>Prompt tuning</li>
                </ul>
                <CodeBlock language="python" code={`from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

# Load base model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

# Create PEFT model
peft_model = get_peft_model(model, peft_config)

# Only LoRA parameters will be updated during training
print(f"Trainable parameters: {peft_model.num_parameters(only_trainable=True)}")
print(f"Total parameters: {peft_model.num_parameters()}")`} />
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6}>
            <Card className="h-100">
              <Card.Header>Domain Adaptation</Card.Header>
              <Card.Body>
                <p>
                  Continue pre-training on domain-specific data before fine-tuning on the target task.
                </p>
                <h5>When to Use:</h5>
                <ul>
                  <li>Target domain differs significantly from pre-training data</li>
                  <li>Domain has specialized vocabulary or structures</li>
                  <li>Examples: legal, medical, scientific texts</li>
                </ul>
                <CodeBlock language="python" code={`from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load domain-specific data
domain_data = load_dataset("text", data_files={"train": "domain_corpus.txt"})

# Tokenize data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

tokenized_dataset = domain_data.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Continue pre-training
training_args = TrainingArguments(
    output_dir="./domain-adapted-bert",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)

trainer.train()`} />
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        <Card className="mb-4">
          <Card.Header>Progressive Freezing Techniques</Card.Header>
          <Card.Body>
            <p>
              Progressive freezing involves gradually unfreezing layers during fine-tuning, starting from the top layers.
              This helps prevent catastrophic forgetting and can improve performance on the target task.
            </p>
            
            <CodeBlock language="python" code={`import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Initially freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classification head
for param in model.classifier.parameters():
    param.requires_grad = True

# Train for a few epochs with just the head unfrozen
# ...

# Now unfreeze the last encoder layer
for param in model.bert.encoder.layer[-1].parameters():
    param.requires_grad = True

# Train for more epochs
# ...

# Continue unfreezing more layers if needed`} />
          </Card.Body>
        </Card>
      </section>

      <section id="best-practices" className="mb-5">
        <h2>6. Best Practices for Transfer Learning in NLP</h2>
        
        <Row>
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header className="bg-success text-white">
                <FaTools className="me-2" />Do's
              </Card.Header>
              <ListGroup variant="flush">
                <ListGroup.Item>Explore multiple pre-trained models for your task</ListGroup.Item>
                <ListGroup.Item>Use appropriate learning rates (usually 2e-5 to 5e-5)</ListGroup.Item>
                <ListGroup.Item>Apply learning rate schedulers (linear or cosine)</ListGroup.Item>
                <ListGroup.Item>Use early stopping to prevent overfitting</ListGroup.Item>
                <ListGroup.Item>Clean and preprocess your data effectively</ListGroup.Item>
                <ListGroup.Item>Perform domain adaptation for specialized domains</ListGroup.Item>
              </ListGroup>
            </Card>
          </Col>
          
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header className="bg-danger text-white">
                <FaExclamationTriangle className="me-2" />Don'ts
              </Card.Header>
              <ListGroup variant="flush">
                <ListGroup.Item>Don't use excessive learning rates that destabilize training</ListGroup.Item>
                <ListGroup.Item>Don't train for too many epochs (3-5 is often sufficient)</ListGroup.Item>
                <ListGroup.Item>Don't ignore sequence length limitations</ListGroup.Item>
                <ListGroup.Item>Don't use huge batch sizes without gradient accumulation</ListGroup.Item>
                <ListGroup.Item>Don't fine-tune on extremely small datasets (use few-shot techniques)</ListGroup.Item>
                <ListGroup.Item>Don't forget to evaluate on a proper validation set</ListGroup.Item>
              </ListGroup>
            </Card>
          </Col>
        </Row>
        
        <Card className="mb-4">
          <Card.Header className="bg-info text-white">
            <FaLightbulb className="me-2" />Hyperparameter Recommendations
          </Card.Header>
          <Card.Body>
            <Row>
              <Col md={6}>
                <h5>For BERT-like models:</h5>
                <ul>
                  <li>Learning rate: 2e-5 to 5e-5</li>
                  <li>Batch size: 16 or 32 (use gradient accumulation if needed)</li>
                  <li>Epochs: 3-5</li>
                  <li>Weight decay: 0.01</li>
                  <li>Warmup steps: 10% of total steps</li>
                </ul>
              </Col>
              <Col md={6}>
                <h5>For GPT-like models:</h5>
                <ul>
                  <li>Learning rate: 1e-5 to 3e-5</li>
                  <li>Batch size: Smaller (4-8) due to memory constraints</li>
                  <li>Epochs: 3-4</li>
                  <li>Weight decay: 0.01</li>
                  <li>Warmup steps: 10% of total steps</li>
                </ul>
              </Col>
            </Row>
          </Card.Body>
        </Card>
      </section>

      <section id="real-world-applications" className="mb-5">
        <h2>7. Real-world Applications</h2>
        
        <Row>
          <Col md={4} className="mb-3">
            <Card className="h-100">
              <Card.Header>Text Classification</Card.Header>
              <Card.Body>
                <ul>
                  <li>Sentiment analysis</li>
                  <li>Topic classification</li>
                  <li>Spam detection</li>
                  <li>Intent recognition</li>
                </ul>
                <CodeBlock language="python" code={`# Quick inference example
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love transfer learning in NLP!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]`} />
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={4} className="mb-3">
            <Card className="h-100">
              <Card.Header>Information Extraction</Card.Header>
              <Card.Body>
                <ul>
                  <li>Named entity recognition</li>
                  <li>Relation extraction</li>
                  <li>Event extraction</li>
                </ul>
                <CodeBlock language="python" code={`# Named Entity Recognition
from transformers import pipeline

ner = pipeline("ner")
text = "Apple is looking at buying U.K. startup for $1 billion"
result = ner(text)
print(result)  # Entities with their positions and labels`} />
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={4} className="mb-3">
            <Card className="h-100">
              <Card.Header>Text Generation</Card.Header>
              <Card.Body>
                <ul>
                  <li>Chatbots</li>
                  <li>Content creation</li>
                  <li>Summarization</li>
                  <li>Translation</li>
                </ul>
                <CodeBlock language="python" code={`# Text summarization
from transformers import pipeline

summarizer = pipeline("summarization")
article = """
Transfer learning has revolutionized NLP...
"""
summary = summarizer(article, max_length=100, min_length=30)
print(summary[0]['summary_text'])`} />
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </section>

      <section id="limitations-challenges" className="mb-4">
        <h2>8. Limitations and Challenges</h2>
        
        <Alert variant="warning">
          <FaExclamationTriangle className="me-2" />
          <strong>Challenges in Transfer Learning for NLP:</strong>
        </Alert>
        
        <Row>
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header>Technical Challenges</Card.Header>
              <Card.Body>
                <ul>
                  <li>Computational resources required for large models</li>
                  <li>Catastrophic forgetting during fine-tuning</li>
                  <li>Domain mismatch between pre-training and target task</li>
                  <li>Handling long sequences beyond model context windows</li>
                </ul>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header>Ethical Considerations</Card.Header>
              <Card.Body>
                <ul>
                  <li>Bias in pre-trained models</li>
                  <li>Environmental impact of training large models</li>
                  <li>Privacy concerns with models trained on web data</li>
                  <li>Attribution and transparency issues</li>
                </ul>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </section>

      <section id="future-directions" className="mb-5">
        <h2>9. Future Directions</h2>
        
        <Row>
          <Col>
            <ListGroup variant="flush">
              <ListGroup.Item>
                <strong>More efficient transfer learning techniques:</strong> Parameter-efficient methods like LoRA, QLoRA, and adapter methods
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Multi-task and continual learning:</strong> Models that can adapt to multiple tasks with minimal forgetting
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Cross-lingual transfer:</strong> Better methods for transferring knowledge across languages
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Few-shot and zero-shot learning:</strong> Reducing the need for labeled data in target tasks
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Retrieval-augmented models:</strong> Combining transfer learning with external knowledge sources
              </ListGroup.Item>
            </ListGroup>
          </Col>
        </Row>
      </section>

      <section id="resources" className="mb-5">
        <h2>10. Additional Resources</h2>
        
        <Row>
          <Col md={6}>
            <Card className="mb-3">
              <Card.Header>Documentation</Card.Header>
              <ListGroup variant="flush">
                <ListGroup.Item>
                  <a href="https://huggingface.co/docs" target="_blank" rel="noopener noreferrer">
                    Hugging Face Documentation
                  </a>
                </ListGroup.Item>
                <ListGroup.Item>
                  <a href="https://huggingface.co/docs/transformers/index" target="_blank" rel="noopener noreferrer">
                    Transformers Documentation
                  </a>
                </ListGroup.Item>
                <ListGroup.Item>
                  <a href="https://huggingface.co/docs/peft/index" target="_blank" rel="noopener noreferrer">
                    PEFT Documentation
                  </a>
                </ListGroup.Item>
              </ListGroup>
            </Card>
          </Col>
          
          <Col md={6}>
                         <Card className="mb-3">
              <Card.Header>Tutorials and Courses</Card.Header>
              <ListGroup variant="flush">
                <ListGroup.Item>
                  <a href="https://huggingface.co/learn/nlp-course/chapter1/1" target="_blank" rel="noopener noreferrer">
                    Hugging Face NLP Course
                  </a>
                </ListGroup.Item>
                <ListGroup.Item>
                  <a href="https://www.coursera.org/learn/natural-language-processing-tensorflow" target="_blank" rel="noopener noreferrer">
                    Natural Language Processing with TensorFlow
                  </a>
                </ListGroup.Item>
                <ListGroup.Item>
                  <a href="https://github.com/huggingface/notebooks" target="_blank" rel="noopener noreferrer">
                    Hugging Face Notebooks
                  </a>
                </ListGroup.Item>
              </ListGroup>
            </Card>
          </Col>
        </Row>
      </section>
    </Container>
  );
};

export default TransferLearning;
