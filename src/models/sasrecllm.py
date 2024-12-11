# src/models/sasrec_llm.py

import torch
import torch.nn as nn
from src.models.sasrec import SASRec

class SASRecLLM(SASRec):
    def __init__(self, item_num, profile_emb_dim, *args, **kwargs):
        """
        Инициализирует модель SASRecLLM.

        Args:
            item_num (int): Количество предметов.
            profile_emb_dim (int): Размерность эмбеддингов профиля пользователя.
            *args: Дополнительные позиционные аргументы для SASRec.
            **kwargs: Дополнительные именованные аргументы для SASRec.
        """
        # Извлекаем 'reconstruction_layer' из kwargs, если он есть
        self.reconstruction_layer = kwargs.pop('reconstruction_layer', -1)
        
        super().__init__(item_num, *args, **kwargs)

        # Трансформация эмбеддингов профилей пользователей
        self.profile_transform = nn.Linear(profile_emb_dim, self.hidden_units)

        # Слой для реконструкции профилей пользователей
        self.profile_decoder = nn.Linear(self.hidden_units, profile_emb_dim)

    def forward(self, input_ids, user_profile_emb=None, return_hidden_states=False):
        """
        Прямой проход модели.

        Args:
            input_ids (torch.Tensor): Входные последовательности [batch_size, seq_len].
            user_profile_emb (torch.Tensor, optional): Эмбеддинги профиля пользователя [batch_size, profile_emb_dim].
            return_hidden_states (bool, optional): Возвращать ли скрытые состояния.

        Returns:
            torch.Tensor: Логиты предсказаний [batch_size, seq_len, item_num + 1].
            torch.Tensor or None: Реконструированные эмбеддинги профиля [batch_size, profile_emb_dim] или None.
            torch.Tensor or None: Вход для реконструкции профиля [batch_size, hidden_units] или None.
        """
        device = input_ids.device

        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, dtype=torch.long,
                                 device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)

        
        if user_profile_emb is not None:
            # Преобразуем эмбеддинги профиля и добавляем к последовательности
            user_profile_emb = user_profile_emb.to(device)
            profile_transformed = self.profile_transform(user_profile_emb)  # [batch_size, hidden_units]
            profile_transformed = profile_transformed.unsqueeze(1).expand(-1, seq_len, -1)
            seqs += profile_transformed

        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=device)
        )

        hidden_states = []

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

            hidden_states.append(seqs.clone())

        outputs = self.last_layernorm(seqs)

        # Получаем представление для реконструкции профиля из outputs до применения add_head
        if self.reconstruction_layer == -1:
            reconstruction_input = outputs.mean(dim=1)  # [batch_size, hidden_units]
        else:
            reconstruction_input = hidden_states[self.reconstruction_layer].mean(dim=1)  # [batch_size, hidden_units]

        # Применяем add_head после сохранения reconstruction_input
        if self.add_head:
            outputs = torch.matmul(
                outputs, self.item_emb.weight.transpose(0, 1)
            )

        reconstructed_profile = None
        reconstruction_input_final = None
        
        if user_profile_emb is not None:
            # Реконструкция профиля пользователя
            reconstructed_profile = self.profile_decoder(reconstruction_input)  # [batch_size, profile_emb_dim]
            reconstruction_input_final = reconstruction_input

        if return_hidden_states:
            return outputs, reconstructed_profile, reconstruction_input_final
        else:
            return outputs, reconstructed_profile
