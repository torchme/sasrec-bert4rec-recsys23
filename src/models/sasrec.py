# # src/models/sasrec.py

# import torch
# import torch.nn as nn

# class SASRec(nn.Module):
#     def __init__(self, item_num, maxlen, hidden_units, num_blocks, num_heads, dropout_rate):
#         super(SASRec, self).__init__()
#         self.item_num = item_num
#         self.maxlen = maxlen
#         self.hidden_units = hidden_units
#         self.num_blocks = num_blocks
#         self.num_heads = num_heads
#         self.dropout_rate = dropout_rate

#         self.item_embedding = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
#         self.position_embedding = nn.Embedding(maxlen, hidden_units)
#         self.dropout = nn.Dropout(dropout_rate)

#         self.attention_layers = nn.ModuleList([
#             nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate)
#             for _ in range(num_blocks)
#         ])
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(hidden_units)
#             for _ in range(num_blocks + 1)  # Добавляем один дополнительный слой нормализации
#         ])

#         self.fc = nn.Linear(hidden_units, item_num + 1)

#     def forward(self, seq, user_profile_emb=None):
#         # seq: [batch_size, seq_len]
#         seq_len = seq.size(1)
#         batch_size = seq.size(0)

#         # Эмбеддинги предметов и позиций
#         item_emb = self.item_embedding(seq)  # [batch_size, seq_len, hidden_units]
#         position_ids = torch.arange(seq_len, dtype=torch.long, device=seq.device).unsqueeze(0).expand(batch_size, -1)
#         pos_emb = self.position_embedding(position_ids)
#         seq_emb = item_emb + pos_emb
                
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"1. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
#         # Добавление эмбеддингов пользователя, если есть
#         if user_profile_emb is not None:
#             seq_emb += user_profile_emb.unsqueeze(1)
        
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"2. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
        
#         seq_emb = self.dropout(seq_emb)

#         variance = seq_emb.var(-1, keepdim=True)
#         if (variance == 0).any():
#             print(f"Zero variance detected in seq_emb before LayerNorm at layer {i+1}")
                
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"3. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
        
#         seq_emb = self.layer_norms[0](seq_emb)  # Применяем первый слой нормализации
                
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"4. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
#         # Транспонируем для соответствия входам MultiheadAttention
#         seq_emb = seq_emb.transpose(0, 1)  # [seq_len, batch_size, hidden_units]
                
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"5. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
#         # Создание масок
#         key_padding_mask = (seq == 0)  # [batch_size, seq_len]
#         subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=seq.device), diagonal=1).bool()  # [seq_len, seq_len]
#         if torch.isnan(subsequent_mask).any() or torch.isinf(subsequent_mask).any():
#             print("6. NaN or Inf detected in subsequent_mask")

#         for i in range(self.num_blocks):
#             residual = seq_emb
#             seq_emb2, attn_weights = self.attention_layers[i](
#                 seq_emb, seq_emb, seq_emb,
#                 attn_mask=subsequent_mask,
#                 key_padding_mask=key_padding_mask
#             )

#             seq_emb2 = self.dropout(seq_emb2)
                    
#             if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#                 print(f"7. NaN or Inf detected in seq_emb after layer {i+1}")
#                 return None
#             seq_emb = residual + seq_emb2
                    
#             if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#                 print(f"8. NaN or Inf detected in seq_emb after layer {i+1}")
#                 return None
#             seq_emb = self.layer_norms[i + 1](seq_emb)  # Применяем соответствующий слой нормализации
                
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"9. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None
#         # Транспонируем обратно
#         seq_emb = seq_emb.transpose(0, 1)  # [batch_size, seq_len, hidden_units]
        
#         if torch.isnan(seq_emb).any() or torch.isinf(seq_emb).any():
#             print(f"10. NaN or Inf detected in seq_emb after layer {i+1}")
#             return None

#         logits = self.fc(seq_emb)  # [batch_size, seq_len, item_num + 1]

#         return logits


import torch
import torch.nn as nn
import numpy as np

from src.models.utils import mean_weightening


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(nn.Module):
    def __init__(self, item_num, maxlen=50, hidden_units=64,
                 num_blocks=2, num_heads=2, dropout_rate=0.2,
                 initializer_range=0.02, add_head=True):
        super(SASRec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.add_head = add_head

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.attention_layers.append(
                nn.MultiheadAttention(hidden_units, num_heads, dropout=dropout_rate)
            )
            self.forward_layernorms.append(
                nn.LayerNorm(hidden_units, eps=1e-8)
            )
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

        # Инициализация параметров
        self.apply(self._init_weights)
        self.profile_transform = nn.Linear(128, self.hidden_units)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight.data[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, user_profile_emb=None):
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, dtype=torch.long,
                                 device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)

        if user_profile_emb is not None:
            seqs += user_profile_emb.unsqueeze(1)

        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device)
        )

        for i in range(len(self.attention_layers)):
            seqs = seqs.transpose(0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            seqs = Q + mha_outputs
            seqs = seqs.transpose(0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        outputs = self.last_layernorm(seqs)
        reconstruction_input = mean_weightening(outputs)

        if self.add_head:
            outputs = torch.matmul(
                outputs, self.item_emb.weight.transpose(0, 1)
            )

        return outputs, reconstruction_input

    def aggregate_profile(self, user_profile_emb):
        """
        user_profile_emb: [batch_size, emb_dim]  или  [batch_size, K, emb_dim]
        Возвращает: [batch_size, hidden_units] (если use_down_scale=True) либо [batch_size, emb_dim].
        """
        if user_profile_emb is None:
            return None

        if user_profile_emb.dim() == 2:
            return self.profile_transform(user_profile_emb)
        raise Exception('aggregate_profile: Not Implemented error')