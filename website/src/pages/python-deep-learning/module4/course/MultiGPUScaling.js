import React from 'react';
import { Container, Title, Text, Stack, Alert, Flex, Image, Paper, Badge, List, Grid, Table, Code } from '@mantine/core';
import { InlineMath, BlockMath } from 'react-katex';
import { IconAlertCircle, IconBulb } from '@tabler/icons-react';
import CodeBlock from '../../../../components/CodeBlock';

const MultiGPUScaling = () => {
  return (
    <Container fluid>
      <Stack spacing="md">
        <div data-slide>
        <Title order={1}>Multi-GPU Scaling Strategies</Title>
        
        <Text>
          As models grow larger and datasets expand, single GPU training becomes insufficient. 
          This section explores different parallelization strategies to scale training across multiple GPUs and nodes.
        </Text>
        </div>
<div data-slide>
        <Title id="parallelization-overview" order={2} mt="xl">Parallelization Strategies Overview</Title>
        
        <Text>
          There are four main types of parallelism in distributed deep learning, each optimizing different aspects:
        </Text>

        <Grid mt="md">
          <Grid.Col span={6}>
            <Paper p="md" withBorder>
              <Badge color="blue" size="lg" mb="sm">Data Parallelism (DP)</Badge>
              <Text size="sm">
                • Replicate entire model on each GPU<br/>
                • Split batch across GPUs<br/>
                • Synchronize gradients after backward pass<br/>
                • Best for: Models that fit in single GPU memory
              </Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            <Paper p="md" withBorder>
              <Badge color="green" size="lg" mb="sm">Model Parallelism (MP)</Badge>
              <Text size="sm">
                • Split model layers across GPUs<br/>
                • Sequential computation through layers<br/>
                • One GPU active at a time (naive version)<br/>
                • Best for: Models too large for single GPU
              </Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            <Paper p="md" withBorder>
              <Badge color="orange" size="lg" mb="sm">Pipeline Parallelism (PP)</Badge>
              <Text size="sm">
                • Split model into pipeline stages<br/>
                • Process micro-batches concurrently<br/>
                • Reduces GPU idle time vs MP<br/>
                • Best for: Deep models with balanced stages
              </Text>
            </Paper>
          </Grid.Col>
          <Grid.Col span={6}>
            <Paper p="md" withBorder>
              <Badge color="purple" size="lg" mb="sm">Tensor Parallelism (TP)</Badge>
              <Text size="sm">
                • Split individual tensors/layers across GPUs<br/>
                • Parallel matrix operations<br/>
                • Requires high bandwidth between GPUs<br/>
                • Best for: Large transformer layers
              </Text>
            </Paper>
          </Grid.Col>
        </Grid>
</div>
<div data-slide>
  https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview
</div>
{/* 
<div data-slide>
        <Title order={3} mt="lg">Memory and Compute Distribution</Title>
        
        <Text>For a model with P parameters, batch size B, and N GPUs:</Text>
        
        <Table mt="md" withBorder>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Parameters per GPU</th>
              <th>Activations per GPU</th>
              <th>Communication Pattern</th>
              <th>Main Bottleneck</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Data Parallel</strong></td>
              <td><InlineMath>{`P`}</InlineMath></td>
              <td><InlineMath>{`\\frac{B}{N} \\times A`}</InlineMath></td>
              <td>All-reduce gradients</td>
              <td>Communication bandwidth</td>
            </tr>
            <tr>
              <td><strong>Model Parallel</strong></td>
              <td><InlineMath>{`\\frac{P}{N}`}</InlineMath></td>
              <td><InlineMath>{`B \\times A`}</InlineMath></td>
              <td>Point-to-point activations</td>
              <td>GPU utilization</td>
            </tr>
            <tr>
              <td><strong>Pipeline Parallel</strong></td>
              <td><InlineMath>{`\\frac{P}{N}`}</InlineMath></td>
              <td><InlineMath>{`\\frac{B}{M} \\times A`}</InlineMath></td>
              <td>Point-to-point micro-batches</td>
              <td>Bubble overhead</td>
            </tr>
            <tr>
              <td><strong>Tensor Parallel</strong></td>
              <td><InlineMath>{`\\frac{P}{N}`}</InlineMath></td>
              <td><InlineMath>{`B \\times \\frac{A}{N}`}</InlineMath></td>
              <td>All-reduce/all-gather tensors</td>
              <td>Communication latency</td>
            </tr>
          </tbody>
        </Table>
        
        <Text size="sm" c="dimmed" mt="sm">
          A = activations per sample, M = number of micro-batches in pipeline parallelism
        </Text>
</div>
<div data-slide>
        <Title id="data-parallel" order={2} mt="xl">Data Parallelism (DP/DDP)</Title>
        
        <Alert icon={<IconBulb />} color="blue" mt="md">
          Data parallelism is the most common starting point for distributed training. 
          It's simple to implement and works well when the model fits in GPU memory.
        </Alert>

        <Title order={3} mt="lg">How Data Parallelism Works</Title>

        <List spacing="sm">
          <List.Item>Each GPU holds a complete copy of the model</List.Item>
          <List.Item>Training batch is split evenly across GPUs</List.Item>
          <List.Item>Each GPU computes forward and backward pass on its mini-batch</List.Item>
          <List.Item>Gradients are synchronized across all GPUs</List.Item>
          <List.Item>All models are updated with the averaged gradients</List.Item>
        </List>

    
</div>

<div data-slide>
        <Title order={3} mt="lg">Gradient Synchronization</Title>

        <Text>
          DDP uses the all-reduce algorithm to efficiently average gradients across all GPUs:
        </Text>

        <BlockMath>{`\\nabla_{\\text{avg}} = \\frac{1}{N} \\sum_{i=1}^{N} \\nabla_i`}</BlockMath>
        
        <Text mt="md">The communication cost depends on the network topology and algorithm:</Text>
        
        <BlockMath>{`\\text{Time} = \\alpha + \\beta \\times \\frac{2(N-1)}{N} \\times M`}</BlockMath>
        
        <Text size="sm" c="dimmed">
          α = latency, β = inverse bandwidth, M = message size (gradient size), N = number of GPUs
        </Text>

        <Title order={3} mt="lg">Effective Batch Size</Title>

        <Text>
          With data parallelism, the effective batch size scales with the number of GPUs:
        </Text>

        <BlockMath>{`B_{\\text{effective}} = B_{\\text{per\\_gpu}} \\times N_{\\text{gpus}}`}</BlockMath>

        <Alert icon={<IconAlertCircle />} color="red" mt="md">
          Large batch sizes may require learning rate scaling and warmup to maintain convergence quality.
          Common practice: scale learning rate linearly with batch size.
        </Alert>
</div>
<div data-slide>
        <Title id="model-parallel" order={2} mt="xl">Model Parallelism (MP)</Title>

        <Text>
          Model parallelism splits the model across multiple GPUs, with each GPU holding a subset of layers.
          This enables training models that don't fit in a single GPU's memory.
        </Text>

        <Title order={3} mt="lg">How Model Parallelism Works</Title>

        <List spacing="sm">
          <List.Item>Model layers are partitioned across GPUs</List.Item>
          <List.Item>Each GPU processes the full batch through its layers</List.Item>
          <List.Item>Activations are passed between GPUs sequentially</List.Item>
          <List.Item>Gradients flow backward through the same path</List.Item>
        </List>
</div>


        <Title order={3} mt="lg">GPU Utilization in Model Parallelism</Title>

        <Text>
          The main drawback is sequential execution leading to GPU idle time:
        </Text>

        <BlockMath>{`\\text{Utilization} = \\frac{1}{N_{\\text{gpus}}}`}</BlockMath>

        <Text mt="md">
          With N GPUs, each GPU is only active 1/N of the time in naive model parallelism.
        </Text>

        <Title id="pipeline-parallel" order={2} mt="xl">Pipeline Parallelism (PP)</Title>

        <Text>
          Pipeline parallelism improves upon model parallelism by splitting the batch into micro-batches
          and processing them in a pipeline fashion, reducing GPU idle time.
        </Text>

        <Title order={3} mt="lg">How Pipeline Parallelism Works</Title>

        <List spacing="sm">
          <List.Item>Model is divided into sequential stages across GPUs</List.Item>
          <List.Item>Mini-batch is split into smaller micro-batches</List.Item>
          <List.Item>Micro-batches are processed in pipeline fashion</List.Item>
          <List.Item>GPUs work on different micro-batches simultaneously</List.Item>
          <List.Item>Gradients are accumulated across micro-batches</List.Item>
        </List>

        <Title order={3} mt="lg">Pipeline Schedule</Title>

        <Text>
          Common pipeline schedules include GPipe and 1F1B (one-forward-one-backward):
        </Text>


        <Title order={3} mt="lg">Pipeline Efficiency</Title>

        <Text>
          Pipeline efficiency depends on the number of micro-batches (M) and pipeline stages (P):
        </Text>

        <BlockMath>{`\\text{Bubble Rate} = \\frac{P - 1}{M}`}</BlockMath>

        <Text mt="md">
          The bubble (idle) time decreases as the number of micro-batches increases, 
          but this also increases memory usage for storing intermediate activations.
        </Text>

        <Alert icon={<IconBulb />} color="blue" mt="md">
          <strong>Rule of thumb:</strong> Use at least 4× more micro-batches than pipeline stages 
          for reasonable efficiency (e.g., 16 micro-batches for 4 stages).
        </Alert>

        <Title id="tensor-parallel" order={2} mt="xl">Tensor Parallelism (TP)</Title>

        <Text>
          Tensor parallelism splits individual layers (tensors) across multiple GPUs, 
          enabling parallel computation within a single layer. This is particularly effective 
          for large transformer models.
        </Text>

        <Title order={3} mt="lg">How Tensor Parallelism Works</Title>

        <List spacing="sm">
          <List.Item>Individual weight matrices are split across GPUs</List.Item>
          <List.Item>Matrix operations are performed in parallel</List.Item>
          <List.Item>Results are gathered/reduced as needed</List.Item>
          <List.Item>Requires high bandwidth between GPUs (NVLink preferred)</List.Item>
        </List>

        <Title order={3} mt="lg">Linear Layer Tensor Parallelism</Title>

        <Text>
          For a linear layer with weight matrix W ∈ ℝ^(d_out × d_in), tensor parallelism can split the computation:
        </Text>

        <Paper p="md" withBorder mt="md">
          <Title order={4}>Column-wise Parallelism</Title>

          <Text>Split W into [W₁, W₂, ..., Wₙ] along the output dimension:</Text>

          <BlockMath>{`Y = XW = X[W_1, W_2, ..., W_n] = [XW_1, XW_2, ..., XW_n]`}</BlockMath>
          
          <Text mt="md"><strong>How it works:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item>Each GPU stores a vertical slice of the weight matrix</List.Item>
            <List.Item>Input X is replicated across all GPUs</List.Item>
            <List.Item>Each GPU computes its portion of the output independently</List.Item>
            <List.Item>No communication needed during forward pass if input is replicated</List.Item>
            <List.Item>Output can remain distributed or be gathered depending on next layer</List.Item>
          </List>
          
          <Text mt="md"><strong>Advantages:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item>No communication in forward pass with replicated input</List.Item>
            <List.Item>Works well when followed by row-wise parallel layer</List.Item>
            <List.Item>Efficient for large output dimensions</List.Item>
          </List>
        </Paper>

        <Paper p="md" withBorder mt="md">
          <Title order={4}>Row-wise Parallelism</Title>

          <Text>Split W into rows along the input dimension:</Text>

          <BlockMath>{`Y = XW = [X_1, X_2, ..., X_n] \\begin{bmatrix} W_1 \\\\ W_2 \\\\ ... \\\\ W_n \\end{bmatrix} = \\sum_{i=1}^{n} X_i W_i`}</BlockMath>
          
          <Text mt="md"><strong>How it works:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item>Each GPU stores a horizontal slice of the weight matrix</List.Item>
            <List.Item>Input X must be split across GPUs accordingly</List.Item>
            <List.Item>Each GPU computes a partial sum XᵢWᵢ</List.Item>
            <List.Item>All-reduce operation sums partial results across GPUs</List.Item>
            <List.Item>Final output Y is replicated on all GPUs after reduction</List.Item>
          </List>
          
          <Text mt="md"><strong>Advantages:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item>No communication in backward pass for gradient w.r.t. input</List.Item>
            <List.Item>Works well when preceded by column-wise parallel layer</List.Item>
            <List.Item>Efficient for large input dimensions</List.Item>
          </List>
        </Paper>

        <Title order={3} mt="lg">Transformer Tensor Parallelism</Title>

        <Paper p="md" withBorder>
          <Text>
            Transformers are particularly well-suited for tensor parallelism due to their structure:
          </Text>

          <Title order={4} mt="md">1. Multi-Head Attention Parallelism</Title>
          <List size="sm" mt="xs">
            <List.Item>Attention heads are naturally independent and can be split across GPUs</List.Item>
            <List.Item>Each GPU computes attention for H/N heads (H=total heads, N=GPUs)</List.Item>
            <List.Item>Q, K, V projections can be column-parallel</List.Item>
            <List.Item>Output projection is row-parallel to gather results</List.Item>
            <List.Item>Minimal communication overhead due to head independence</List.Item>
          </List>

          <Title order={4} mt="md">2. MLP/FFN Layer Parallelism (Megatron-style)</Title>
          <Text mt="xs">
            The transformer MLP typically expands hidden dimension by 4x, making it memory-intensive:
          </Text>
          <List size="sm" mt="xs">
            <List.Item><strong>First Linear (h → 4h):</strong> Use column-parallel to split the expanded dimension</List.Item>
            <List.Item><strong>Activation:</strong> Applied element-wise, no communication needed</List.Item>
            <List.Item><strong>Second Linear (4h → h):</strong> Use row-parallel to reduce back to hidden size</List.Item>
            <List.Item><strong>Result:</strong> Only one all-reduce needed at the end of MLP block</List.Item>
          </List>
          
          <Alert icon={<IconBulb />} color="blue" mt="md">
            <strong>Efficiency tip:</strong> By carefully arranging column and row parallelism, 
            Megatron-LM achieves only 2 all-reduce operations per transformer layer 
            (one for attention, one for MLP), minimizing communication overhead.
          </Alert>

          <Title order={4} mt="md">3. Embedding Layer Parallelism</Title>
          <List size="sm" mt="xs">
            <List.Item>Vocabulary can be split across GPUs (each GPU handles V/N tokens)</List.Item>
            <List.Item>Input tokens are scattered to appropriate GPUs</List.Item>
            <List.Item>Embedding lookups happen locally on each GPU</List.Item>
            <List.Item>Results are gathered for the transformer blocks</List.Item>
            <List.Item>Particularly useful for large vocabularies (50K+ tokens)</List.Item>
          </List>
        </Paper>

        <Title order={3} mt="lg">Communication Patterns in Tensor Parallelism</Title>

        <Table mt="md" withBorder>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Forward Pass</th>
              <th>Backward Pass</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Column Parallel</td>
              <td>Identity (if input replicated)</td>
              <td>All-reduce gradients</td>
            </tr>
            <tr>
              <td>Row Parallel</td>
              <td>All-reduce outputs</td>
              <td>Identity</td>
            </tr>
            <tr>
              <td>Attention Heads</td>
              <td>All-gather Q,K,V → All-reduce output</td>
              <td>Reverse of forward</td>
            </tr>
          </tbody>
        </Table>

        <Title id="hybrid-parallelism" order={2} mt="xl">Hybrid Parallelism Strategies</Title>

        <Text>
          Modern large-scale training combines multiple parallelism strategies to overcome 
          individual limitations and maximize efficiency.
        </Text>

        <Title order={3} mt="lg">Common Hybrid Approaches</Title>

        <Paper p="md" withBorder mt="md">
          <Title order={4}>3D Parallelism (DP + PP + TP)</Title>
          <Text size="sm" mt="xs">
            Used in models like GPT-3 and Megatron-Turing NLG:
          </Text>
          <List size="sm" mt="xs">
            <List.Item>Tensor parallelism within nodes (high bandwidth)</List.Item>
            <List.Item>Pipeline parallelism across nodes (tolerates latency)</List.Item>
            <List.Item>Data parallelism for remaining scale-out</List.Item>
          </List>
        </Paper>

        <Paper p="md" withBorder mt="md">
          <Title order={4}>ZeRO + Pipeline Parallelism</Title>
          <Text size="sm" mt="xs">
            Combines memory optimization with pipeline efficiency:
          </Text>
          <List size="sm" mt="xs">
            <List.Item>ZeRO shards optimizer states, gradients, and parameters</List.Item>
            <List.Item>Pipeline parallelism for model that still doesn't fit</List.Item>
            <List.Item>Reduces memory redundancy while maintaining throughput</List.Item>
          </List>
        </Paper>

        <Title order={3} mt="lg">Choosing the Right Strategy</Title>

        <Table mt="md" withBorder>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Recommended Strategy</th>
              <th>Reasoning</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Model fits on GPU, large dataset</td>
              <td>Data Parallelism (DDP)</td>
              <td>Simple, efficient, minimal communication</td>
            </tr>
            <tr>
              <td>Model slightly too large</td>
              <td>ZeRO or Gradient Checkpointing</td>
              <td>Memory optimization without model splitting</td>
            </tr>
            <tr>
              <td>Very large transformer</td>
              <td>TP within node + PP across nodes</td>
              <td>TP needs high bandwidth, PP tolerates latency</td>
            </tr>
            <tr>
              <td>Extreme scale (1000+ GPUs)</td>
              <td>3D Parallelism</td>
              <td>Combines all strategies for maximum scale</td>
            </tr>
          </tbody>
        </Table>

        <Title order={3} mt="lg">Performance Considerations</Title>

        <Alert icon={<IconBulb />} color="green" mt="md">
          <strong>Key Performance Metrics:</strong>
          <List size="sm" mt="xs">
            <List.Item><strong>MFU (Model FLOPs Utilization):</strong> Actual vs theoretical FLOPS</List.Item>
            <List.Item><strong>Scaling Efficiency:</strong> Speedup vs number of GPUs</List.Item>
            <List.Item><strong>Communication/Computation Ratio:</strong> Time spent on communication</List.Item>
            <List.Item><strong>Memory Efficiency:</strong> Model size vs available memory</List.Item>
          </List>
        </Alert>

        <Title order={3} mt="lg">Implementing 3D Parallelism</Title>

        <Paper p="md" withBorder>
          <Title order={4}>3D Parallelism Configuration</Title>
          
          <Text>
            3D parallelism combines TP, PP, and DP to maximize hardware utilization. Here's how to organize it:
          </Text>
          
          <Text mt="md"><strong>Typical Setup (Example: 128 GPUs):</strong></Text>
          <List size="sm" mt="xs">
            <List.Item><strong>Tensor Parallel Size = 8:</strong> Within each node (high NVLink bandwidth)</List.Item>
            <List.Item><strong>Pipeline Parallel Size = 4:</strong> Across 4 nodes (tolerates network latency)</List.Item>
            <List.Item><strong>Data Parallel Size = 4:</strong> Remaining dimension (128 ÷ 8 ÷ 4 = 4)</List.Item>
          </List>
          
          <Text mt="md"><strong>Process Group Organization:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item><strong>TP Groups:</strong> GPUs [0-7], [8-15], [16-23], etc. (same node)</List.Item>
            <List.Item><strong>PP Groups:</strong> GPUs [0,8,16,24], [1,9,17,25], etc. (across nodes)</List.Item>
            <List.Item><strong>DP Groups:</strong> GPUs with same TP and PP position across replicas</List.Item>
          </List>
          
          <Text mt="md"><strong>Communication Hierarchy:</strong></Text>
          <List size="sm" mt="xs">
            <List.Item><strong>Intra-node (TP):</strong> Uses NVLink/NVSwitch - very high bandwidth (600 GB/s)</List.Item>
            <List.Item><strong>Inter-node (PP):</strong> Uses InfiniBand/Ethernet - moderate bandwidth (100-200 GB/s)</List.Item>
            <List.Item><strong>Cross-replica (DP):</strong> Gradient all-reduce across data parallel groups</List.Item>
          </List>
          
          <Alert icon={<IconBulb />} color="green" mt="md">
            <strong>Best Practice:</strong> Place communication-heavy operations (tensor parallelism) 
            within nodes where bandwidth is highest, and latency-tolerant operations (pipeline parallelism) 
            across nodes.
          </Alert>
        </Paper>

        <Title order={2} mt="xl">Summary and Best Practices</Title>

        <Paper p="lg" withBorder mt="md">
          <Title order={3}>Quick Decision Guide</Title>
          
          <List spacing="md" mt="md">
            <List.Item>
              <strong>Start with Data Parallelism</strong> - It's simple and often sufficient
            </List.Item>
            <List.Item>
              <strong>Add ZeRO optimization</strong> - When approaching memory limits
            </List.Item>
            <List.Item>
              <strong>Use Tensor Parallelism</strong> - For very wide layers (transformers)
            </List.Item>
            <List.Item>
              <strong>Apply Pipeline Parallelism</strong> - For very deep models
            </List.Item>
            <List.Item>
              <strong>Combine strategies</strong> - For extreme scale (billions of parameters)
            </List.Item>
          </List>

          <Alert icon={<IconAlertCircle />} color="blue" mt="lg">
            <strong>Remember:</strong> The optimal strategy depends on your specific model architecture, 
            hardware setup (especially interconnect bandwidth), and scaling requirements. 
            Profile and experiment to find the best configuration.
          </Alert>
        </Paper> */}

      </Stack>
    </Container>
  );
};

export default MultiGPUScaling;