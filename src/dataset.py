# src/dataset.py

import torch
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
