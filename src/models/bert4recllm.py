# src/models/bert4rec_mod.py

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class BERT4RecLLM(nn.Module):
    """Модифицированная реализация модели BERT4Rec с интеграцией эмбеддингов пользователей."""
    
    def __init__(self, item_num, maxlen, hidden_units, num_heads, num_layers, dropout_rate, user_emb_dim):
        super(BERT4RecLLM, self).__init__()
        
        self.item_num = item_num
        self.maxlen = maxlen
        self.user_emb_dim = user_emb_dim
        
        # Конфигурация BERT
        bert_config = BertConfig(
            vocab_size=item_num + 1,
            hidden_size=hidden_units,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_units * 4,
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
            max_position_embeddings=maxlen,
            type_vocab_size=1,
            pad_token_id=0
        )
        
        self.bert = BertModel(bert_config)
        
        # Проекция эмбеддингов пользователей
        self.user_emb_projection = nn.Linear(user_emb_dim, hidden_units)
        
        # Предсказание товаров
        self.out = nn.Linear(hidden_units, item_num + 1)
        
        # Инициализация весов
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Инициализация весов."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if isinstance(module, nn.Embedding):
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_seq, user_profile_emb):
        """
        input_seq: Tensor с индексами товаров [batch_size, seq_len]
        user_profile_emb: Эмбеддинги пользователей [batch_size, user_emb_dim]
        """
        attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]
        
        # Проекция и добавление эмбеддингов пользователей
        user_emb = self.user_emb_projection(user_profile_emb)  # [batch_size, hidden_units]
        user_emb = user_emb.unsqueeze(1)  # [batch_size, 1, hidden_units]
        
        # Получаем эмбеддинги товаров
        inputs_embeds = self.bert.embeddings(input_ids=input_seq)
        
        # Добавляем эмбеддинги пользователей к эмбеддингам товаров
        inputs_embeds = inputs_embeds + user_emb  # [batch_size, seq_len, hidden_units]
        
        # Получаем выходы из BERT
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_units]
        
        # Предсказание следующего товара
        logits = self.out(sequence_output)  # [batch_size, seq_len, item_num + 1]
        return logits
