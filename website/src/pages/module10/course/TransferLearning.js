import React from "react";
import { Container, Grid, Card, Alert } from '@mantine/core';
import CodeBlock from "components/CodeBlock";
import { FaLightbulb, FaCode, FaExclamationTriangle, FaTools, FaRocket } from "react-icons/fa";
import { InlineMath, BlockMath } from 'react-katex';
const TransferLearning = () => {
  return (
    <Container className="py-4">
      <h1>Transfer Learning in NLP</h1>
      {/* Section 1: Introduction */}
      <section className="my-5">
        <h2>1. Why Transfer Learning?</h2>
        <Card className="my-4">
          <Card.Body>
            <Card.Title><FaLightbulb className="text-warning me-2" />Transfer Learning in NLP</Card.Title>
            <p>
              Language models like BERT, GPT, and others are trained on billions of data points using millions of dollars worth of computing resources. These models have learned powerful representations of language through this extensive training.
            </p>
            <p>
              Rather than training a model from scratch for your specific NLP task (which would require enormous amounts of data and computing resources), transfer learning allows you to:
            </p>
            <ListGroup variant="flush">
              <ListGroup.Item>
                <strong>Leverage existing knowledge</strong> - Use representations learned from massive datasets
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Reduce training time and cost</strong> - Fine-tune only what's necessary for your specific task
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Achieve better performance with less data</strong> - Pre-trained models already understand language structure
              </ListGroup.Item>
            </ListGroup>
          </Card.Body>
        </Card>
        <Alert variant="info" className="d-flex align-items-start">
          <FaExclamationTriangle className="me-2 mt-1" />
          <div>
            <p className="mb-0">
              <strong>When to use transfer learning:</strong>
            </p>
            <ul>
              <li>For <strong>generic tasks</strong>, using pre-trained models directly (zero-shot or few-shot) may be sufficient</li>
              <li>For <strong>specific domain tasks</strong>, fine-tuning pre-trained models on your domain data typically yields the best performance-to-cost ratio</li>
              <li><strong>Training from scratch</strong> is rarely necessary unless your task is extremely unique or requires specialized vocabulary not present in existing models</li>
            </ul>
          </div>
        </Alert>
      </section>
      {/* Section 2: Using Pre-trained Models */}
      <section className="my-5">
        <h2>2. Using Pre-trained Models</h2>
        <p>
          Let's see how to use popular pre-trained models in practice:
        </p>
        <Card className="my-4">
          <Card.Header as="h3">2.1 Using BERT for Text Classification</Card.Header>
          <Card.Body>
            <p>
              BERT produces contextual embeddings excellent for many NLP tasks. Here's how to use a pre-trained BERT model for classification:
            </p>
            <h4>Tokenization</h4>
            <CodeBlock
              language="python"
              code={`# Import the BERT tokenizer
from transformers import BertTokenizer
# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Example text
text = "Transfer learning helps leverage pre-trained knowledge."
# Tokenize with BERT's special tokens and padding
encoded_input = tokenizer(
    text,
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'  # Return PyTorch tensors
)
print(encoded_input.keys())
# Output: dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])`}
            />
            <h4>Using the Model</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import BertModel
import torch
# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')
# Get embeddings
with torch.no_grad():
    outputs = model(**encoded_input)
# Get the embeddings from the last hidden state
last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
# For classification tasks, typically use the [CLS] token embedding (first token)
cls_embedding = last_hidden_state[:, 0, :]
print(f"CLS embedding shape: {cls_embedding.shape}")
# Output: CLS embedding shape: torch.Size([1, 768])`}
            />
          </Card.Body>
        </Card>
        <Card className="my-4">
          <Card.Header as="h3">2.2 Using GPT for Text Generation</Card.Header>
          <Card.Body>
            <p>
              GPT models excel at text generation tasks. Here's how to use a pre-trained GPT-2 model:
            </p>
            <h4>Tokenization</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import GPT2Tokenizer
# Load pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a padding token by default
# Example prompt
prompt = "Transfer learning is useful because"
# Tokenize the input
encoded_input = tokenizer(prompt, return_tensors='pt')`}
            />
            <h4>Generating Text</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import GPT2LMHeadModel
# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
# Generate text
output = model.generate(
    encoded_input['input_ids'],
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    no_repeat_ngram_size=2,
    do_sample=True
)
# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
# Example output: "Transfer learning is useful because it allows us to leverage pre-trained models 
# that have been trained on massive datasets, saving computational resources and improving performance."`}
            />
          </Card.Body>
        </Card>
        <Card className="my-4">
          <Card.Header as="h3">2.3 Zero-shot and Few-shot Learning with Instruction Models</Card.Header>
          <Card.Body>
            <p>
              Instruction-tuned models like FLAN-T5 or instruction-tuned versions of GPT can perform tasks without fine-tuning:
            </p>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Load pre-trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
# Zero-shot example
task_prompt = "Classify the sentiment of this review as positive or negative: 'The food was delicious and the service was excellent.'"
# Tokenize input and generate response
input_ids = tokenizer(task_prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
# Expected output: "positive"
# Few-shot example
few_shot_prompt = """
Classify the sentiment of reviews:
Review: "The movie was boring and too long."
Sentiment: negative
Review: "I loved the acting and the plot twists."
Sentiment: positive
Review: "The hotel room was dirty and the staff was rude."
Sentiment:
"""
input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
# Expected output: "negative"`}
            />
            <Alert variant="success" className="mt-3">
              <strong><FaLightbulb className="me-2" />Pro Tip:</strong> Instruction-tuned models can perform a wide variety of tasks without fine-tuning, making them extremely versatile. For specialized tasks, fine-tuning usually provides better results.
            </Alert>
          </Card.Body>
        </Card>
      </section>
      {/* Section 3: Fine-tuning Models */}
      <section className="my-5">
        <h2>3. Fine-tuning Pre-trained Models</h2>
        <p>
          While pre-trained models provide strong capabilities out-of-the-box, fine-tuning them on domain-specific data often yields significant performance improvements.
        </p>
        <Card className="my-4">
          <Card.Header as="h3">3.1 Traditional Fine-tuning</Card.Header>
          <Card.Body>
            <p>
              Traditional fine-tuning updates all (or most) of the model's parameters. While effective, this approach requires significant computational resources.
            </p>
            <h4>Fine-tuning BERT for Sentiment Classification</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
# Load dataset (example: IMDB movie reviews)
dataset = load_dataset("imdb")
# Load pre-trained model with classification head
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification
)
# Tokenize the dataset
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
# Fine-tune the model
trainer.train()`}
            />
            <Alert variant="info" className="mt-3">
              <FaExclamationTriangle className="me-2" />
              <div>
                <strong>Challenges with Traditional Fine-tuning:</strong>
                <ul className="mb-0">
                  <li>High memory requirements (especially for larger models)</li>
                  <li>Longer training times</li>
                  <li>Risk of catastrophic forgetting of pre-trained knowledge</li>
                </ul>
              </div>
            </Alert>
          </Card.Body>
        </Card>
        <Card className="my-4">
          <Card.Header as="h3">3.2 Parameter-Efficient Fine-tuning with LoRA</Card.Header>
          <Card.Body>
            <p>
              Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that significantly reduces trainable parameters by adding small, trainable "update matrices" to frozen pre-trained weights.
            </p>
            <h4>How LoRA Works</h4>
            <p>
              LoRA represents weight updates using low-rank decomposition:
            </p>
            <BlockMath math="\Delta W = BA" />
            <p>
              Where:
            </p>
            <ul>
              <li>
                <InlineMath math="\Delta W" /> is the weight update matrix (same shape as the original weight matrix <InlineMath math="W" />)
              </li>
              <li>
                <InlineMath math="B" /> is a matrix of shape <InlineMath math="[d, r]" />
              </li>
              <li>
                <InlineMath math="A" /> is a matrix of shape <InlineMath math="[r, k]" />
              </li>
              <li>
                <InlineMath math="r" /> is the rank, which is typically much smaller than <InlineMath math="d" /> and <InlineMath math="k" />
              </li>
            </ul>
            <p>
              The effective weight during inference becomes:
            </p>
            <BlockMath math="W' = W + \Delta W = W + BA" />
            <h4>LoRA Fine-tuning Example</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
# Load base model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Define LoRA configuration
lora_config = LoraConfig(
    r=8,                   # Rank of the update matrices
    lora_alpha=32,         # Alpha parameter for scaling
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to query and value projection matrices
    lora_dropout=0.05,     # Dropout probability for LoRA layers
    bias="none",           # Don't add bias parameters
    task_type=TaskType.CAUSAL_LM  # Specify the task type
)
# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)
# Print trainable parameters report
print(f"Trainable parameters: {lora_model.print_trainable_parameters()}")
# Example output: "Trainable parameters: 294,912 (~0.37% of total)"
# Fine-tuning is similar to the traditional approach
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
# Load dataset (example: small text dataset)
dataset = load_dataset("text", data_files={"train": "train.txt"})
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_strategy="epoch",
)
# Initialize Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)
# Fine-tune the model
trainer.train()
# Save the LoRA model
lora_model.save_pretrained("./lora_model_saved")`}
            />
            <div className="mt-4">
              <h4>Advantages of LoRA</h4>
              <Grid>
                <Grid.Col span={{ md: 6 }}>
                  <Card className="h-100">
                    <Card.Body>
                      <Card.Title className="d-flex align-items-center">
                        <FaRocket className="text-success me-2" />
                        Efficiency Benefits
                      </Card.Title>
                      <ListGroup variant="flush">
                        <ListGroup.Item>Reduces trainable parameters by 99%+</ListGroup.Item>
                        <ListGroup.Item>Lower memory requirements</ListGroup.Item>
                        <ListGroup.Item>Faster training times</ListGroup.Item>
                        <ListGroup.Item>Smaller storage footprint (only save adapter weights)</ListGroup.Item>
                      </ListGroup>
                    </Card.Body>
                  </Card>
                </Grid.Col>
                <Grid.Col span={{ md: 6 }}>
                  <Card className="h-100">
                    <Card.Body>
                      <Card.Title className="d-flex align-items-center">
                        <FaTools className="text-primary me-2" />
                        Practical Benefits
                      </Card.Title>
                      <ListGroup variant="flush">
                        <ListGroup.Item>Performance comparable to full fine-tuning</ListGroup.Item>
                        <ListGroup.Item>Reduces risk of catastrophic forgetting</ListGroup.Item>
                        <ListGroup.Item>Multiple adapters can be combined or switched</ListGroup.Item>
                        <ListGroup.Item>Enables fine-tuning of models too large for traditional methods</ListGroup.Item>
                      </ListGroup>
                    </Card.Body>
                  </Card>
                </Grid.Col>
              </Grid>
            </div>
          </Card.Body>
        </Card>
        <Card className="my-4">
          <Card.Header as="h3">3.3 Other Parameter-Efficient Fine-tuning Methods</Card.Header>
          <Card.Body>
            <p>
              Beyond LoRA, several other parameter-efficient fine-tuning methods exist:
            </p>
            <ListGroup className="mb-4">
              <ListGroup.Item>
                <strong>Prefix Tuning:</strong> Prepends trainable prefix vectors to the input of transformer layers
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Prompt Tuning:</strong> Adds trainable "soft prompts" to the input embeddings
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>Adapter Tuning:</strong> Inserts small trainable modules between layers of the pre-trained model
              </ListGroup.Item>
              <ListGroup.Item>
                <strong>QLoRA:</strong> Combines LoRA with quantization for even more memory efficiency
              </ListGroup.Item>
            </ListGroup>
            <h4>QLoRA Example</h4>
            <CodeBlock
              language="python"
              code={`# Import necessary modules
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
# Setup quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# Load large model with quantization
model_name = "meta-llama/Llama-2-7b-hf"  # Example of a large model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
# Define LoRA configuration similar to before
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
# Apply LoRA to the quantized model
qlora_model = get_peft_model(model, lora_config)
# Fine-tuning process continues as with regular LoRA`}
            />
            <Alert variant="success" className="mt-3">
              <FaLightbulb className="me-2" />
              <div>
                <strong>Best Practices for Efficient Fine-tuning:</strong>
                <ul className="mb-0">
                  <li>Use gradient accumulation for larger effective batch sizes</li>
                  <li>Apply learning rate warmup and decay</li>
                  <li>Monitor for overfitting on smaller datasets</li>
                  <li>Consider mixed precision training for additional efficiency</li>
                  <li>Try different rank values (r) to balance parameter count and performance</li>
                </ul>
              </div>
            </Alert>
          </Card.Body>
        </Card>
        <Card className="my-4">
          <Card.Header as="h3">3.4 Comparing Fine-tuning Methods</Card.Header>
          <Card.Body>
            <table className="table table-bordered">
              <thead className="table-light">
                <tr>
                  <th>Method</th>
                  <th>Trainable Parameters</th>
                  <th>Memory Usage</th>
                  <th>Training Speed</th>
                  <th>Performance</th>
                  <th>Best For</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Full Fine-tuning</td>
                  <td>100%</td>
                  <td>High</td>
                  <td>Slow</td>
                  <td>Excellent</td>
                  <td>When resources are not limited and maximum performance is required</td>
                </tr>
                <tr>
                  <td>LoRA</td>
                  <td>~0.1-1%</td>
                  <td>Low</td>
                  <td>Fast</td>
                  <td>Very Good</td>
                  <td>Most use cases, especially with limited resources</td>
                </tr>
                <tr>
                  <td>QLoRA</td>
                  <td>~0.1-1%</td>
                  <td>Very Low</td>
                  <td>Medium</td>
                  <td>Very Good</td>
                  <td>Very large models that wouldn't fit in memory otherwise</td>
                </tr>
                <tr>
                  <td>Adapter Tuning</td>
                  <td>~1-5%</td>
                  <td>Medium</td>
                  <td>Medium</td>
                  <td>Good</td>
                  <td>When modular task adaptation is needed</td>
                </tr>
                <tr>
                  <td>Prompt Tuning</td>
                  <td>~0.01-0.1%</td>
                  <td>Very Low</td>
                  <td>Very Fast</td>
                  <td>Good</td>
                  <td>Simple task adaptation with minimal resources</td>
                </tr>
              </tbody>
            </table>
          </Card.Body>
        </Card>
      </section>
    </Container>
  );
};
export default TransferLearning;