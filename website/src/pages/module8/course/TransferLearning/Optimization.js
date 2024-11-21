import React from 'react';
import { Container, Stack, Title, Text } from '@mantine/core';
import CodeBlock from 'components/CodeBlock';
import 'katex/dist/katex.min.css';
import { BlockMath } from 'react-katex';
import { Book, GitBranch, Settings, Brain } from 'lucide-react';

const Optimization = () => {
    return (
      <Stack spacing="md">
        <Text>
          Effective optimization requires careful tuning of learning rates and training schedules.
        </Text>
  
        <CodeBlock
          language="python"
          code={`
  def configure_optimizer(model, base_lr=1e-4, head_lr=1e-3):
      """Configure optimizer with different learning rates"""
      optimizer = torch.optim.AdamW([
          {'params': model.fc.parameters(), 'lr': head_lr},
          {'params': (p for n, p in model.named_parameters() 
                     if 'fc' not in n), 'lr': base_lr}
      ])
      
      scheduler = torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=[head_lr, base_lr],
          steps_per_epoch=steps_per_epoch,
          epochs=num_epochs
      )
      
      return optimizer, scheduler`}
        />
      </Stack>
    );
  };

export default Optimization