# src/dataset.py

import torch
import random
import numpy as np
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequences, maxlen):
        self.sequences = sequences
        self.maxlen = maxlen
        self.user_ids = list(sequences.keys())

    def __len__(self):
        return len(self.user_ids)

    # def __getitem__(self, idx):
    #     user_id = self.user_ids[idx]
    #     seq = self.sequences[user_id]

    #     # Обрезаем или дополняем последовательность до maxlen
    #     seq = seq[-self.maxlen:]
    #     seq = [0] * (self.maxlen - len(seq)) + seq  # Дополняем нулями спереди

    #     input_seq = torch.tensor(seq[:-1], dtype=torch.long)  # Входная последовательность
    #     target = torch.tensor(seq[1:], dtype=torch.long)      # Целевые значения

    #     return input_seq, target, torch.tensor(user_id, dtype=torch.long)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]

        # Обрезаем или дополняем последовательность до maxlen
        seq = seq[-self.maxlen:]
        seq = [0] * (self.maxlen - len(seq)) + seq  # Дополняем нулями спереди

        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target = torch.tensor(seq[1:], dtype=torch.long)

        # Проверка на NaN или Inf
        if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
            print(f"NaN or Inf detected in input_seq for user {user_id}")
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"NaN or Inf detected in target for user {user_id}")


        return input_seq, target, torch.tensor(user_id, dtype=torch.long)


class BERT4RecDataset(Dataset):
    def __init__(self, sequences, maxlen, mask_token, num_items, 
                 mask_prob=0.15, mode='train', seed=42):
        """
        Dataset for BERT4Rec model.
        
        Args:
            sequences: Dictionary of user sequences {user_id: [item1, item2, ...]}
            maxlen: Maximum sequence length
            mask_token: Token ID used for masking (usually num_items + 1)
            num_items: Number of items in the dataset
            mask_prob: Probability of masking a token during training
            mode: 'train', 'valid', or 'test'
            seed: Random seed for reproducibility
        """
        self.sequences = sequences
        self.maxlen = maxlen
        self.mask_token = mask_token
        self.num_items = num_items
        self.mask_prob = mask_prob
        self.mode = mode
        self.user_ids = list(sequences.keys())
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = self.sequences[user_id]
        
        # Trim sequence to maxlen
        if len(seq) > self.maxlen:
            # Keep the most recent items
            seq = seq[-self.maxlen:]
        
        # Create attention mask (1 for real items, 0 for padding)
        attention_mask = [0] * (self.maxlen - len(seq)) + [1] * len(seq)
        
        # Pad sequence
        seq = [0] * (self.maxlen - len(seq)) + seq
        
        if self.mode == 'train':
            # Create inputs and targets
            input_seq = seq.copy()
            target_seq = seq.copy()
            
            # Randomly mask items for training
            for i in range(len(seq)):
                if seq[i] == 0:  # Skip padding
                    continue
                    
                if random.random() < self.mask_prob:
                    input_seq[i] = self.mask_token  # Replace with mask token
            
            # For items that are not masked, set target to -100 (ignore in loss)
            for i in range(len(seq)):
                if input_seq[i] != self.mask_token:
                    target_seq[i] = -100
                    
        else:  # valid or test
            # For inference, we append a mask token at the end to predict the next item
            input_seq = seq[:-1] + [self.mask_token]  # Replace last token with mask
            target_seq = [-100] * (len(seq) - 1) + [seq[-1]]  # Only predict the last item
            
            # Update attention mask for the mask token
            attention_mask = attention_mask[:-1] + [1]
        
        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(user_id, dtype=torch.long)
        )
