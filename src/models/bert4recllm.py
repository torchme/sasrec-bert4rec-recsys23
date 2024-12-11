# src/models/bert4recllm.py

import torch
import torch.nn as nn
from src.models.bert4rec import BERT4Rec
from transformers import BertConfig, BertModel

class BERT4RecLLM(BERT4Rec):
    def __init__(self, item_num, maxlen, hidden_units, num_heads, num_layers, dropout_rate,
                 user_emb_dim, reconstruction_layer=-1, add_head=True):
        """
        BERT4Rec с поддержкой пользовательских эмбеддингов и реконструкции профиля (LLM часть).

        Args:
            item_num (int): Количество предметов.
            maxlen (int): Максимальная длина последовательности.
            hidden_units (int): Размер скрытого представления.
            num_heads (int): Количество голов в Multi-Head Attention.
            num_layers (int): Количество слоев трансформера.
            dropout_rate (float): Доля дропаутов.
            user_emb_dim (int): Размерность эмбеддингов профиля пользователя.
            reconstruction_layer (int): Индекс слоя для получения скрытого состояния для реконструкции профиля.
                                        -1 означает использовать финальный слой.
            add_head (bool): Применять ли выходную проекцию на словарь предметов.
        """
        super(BERT4RecLLM, self).__init__(
            item_num=item_num,
            maxlen=maxlen,
            hidden_units=hidden_units,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            add_head=add_head
        )
        
        self.user_emb_dim = user_emb_dim
        self.reconstruction_layer = reconstruction_layer

        # Преобразование пользовательского профиля к размерности hidden_units
        self.profile_transform = nn.Linear(user_emb_dim, hidden_units)
        # Декодер для реконструкции профиля
        self.profile_decoder = nn.Linear(hidden_units, user_emb_dim)

    def forward(self, input_seq, user_profile_emb=None, return_hidden_states=False):
        """
        Args:
            input_seq (torch.Tensor): Индексы предметов [batch_size, seq_len].
            user_profile_emb (torch.Tensor or None): Эмбеддинги профиля пользователя [batch_size, user_emb_dim].
            return_hidden_states (bool): Возвращать ли скрытые состояния всех слоёв.

        Returns:
            torch.Tensor: Логиты предсказаний [batch_size, seq_len, item_num + 1] или скрытые состояния [batch_size, seq_len, hidden_units], если add_head=False.
            torch.Tensor or None: Реконструированный профиль [batch_size, user_emb_dim] или None, если user_profile_emb отсутствует.
            torch.Tensor or None: Скрытое состояние для реконструкции профиля [batch_size, hidden_units] или None.
        """
        attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]
        
        if user_profile_emb is not None:
            # Преобразуем user_profile_emb
            transformed_profile = self.profile_transform(user_profile_emb)  # [batch_size, hidden_units]
            # Расширяем до [batch_size, seq_len, hidden_units]
            transformed_profile = transformed_profile.unsqueeze(1).expand(-1, input_seq.size(1), -1)
            # Получаем входные эмбеддинги и добавляем к ним трансформированный профиль
            inputs_embeds = self.bert.embeddings(input_ids=input_seq) + transformed_profile
            # Передаём эмбеддинги напрямую через inputs_embeds
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        else:
            # Стандартный проход без добавления профиля
            outputs = self.bert(input_ids=input_seq, attention_mask=attention_mask, output_hidden_states=True)
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_units]
        
        if self.add_head:
            logits = self.out(sequence_output)  # [batch_size, seq_len, item_num + 1]
        else:
            logits = sequence_output  # [batch_size, seq_len, hidden_units]
        
        hidden_states = outputs.hidden_states  # Tuple of hidden states

        # Выбор слоя для реконструкции
        if self.reconstruction_layer == -1:
            selected_hidden_state = hidden_states[-1]  # Финальный слой
        else:
            selected_hidden_state = hidden_states[self.reconstruction_layer]  # Выбранный слой

        # Среднее по последовательности
        representation = selected_hidden_state.mean(dim=1)  # [batch_size, hidden_units]

        # Реконструкция профиля
        if user_profile_emb is not None:
            reconstructed_profile = self.profile_decoder(representation)  # [batch_size, user_emb_dim]
        else:
            reconstructed_profile = None

        if return_hidden_states:
            return logits, reconstructed_profile, representation
        else:
            return logits, reconstructed_profile
