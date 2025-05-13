import torch
import torch.nn as nn
from src.models.bert4rec import BERT4Rec
from transformers import BertConfig, BertModel

from src.models.utils import mean_weightening, exponential_weightening, SimpleAttentionAggregator


class BERT4RecLLM(BERT4Rec):
    def __init__(self, profile_emb_dim, weighting_scheme: str, use_down_scale, use_upscale, weight_scale:float,
                 item_num, maxlen, hidden_units, num_heads, num_layers, dropout_rate,
                 reconstruction_layer=-1, add_head=True, mask_token=None):
        """
        BERT4Rec с поддержкой пользовательских эмбеддингов и реконструкции профиля (LLM часть).

        Args:
            profile_emb_dim (int): Размерность эмбеддингов профиля пользователя.
            weighting_scheme (str): Схема взвешивания для агрегации скрытых состояний.
            use_down_scale (bool): Использовать ли проекцию профиля пользователя.
            use_upscale (bool): Использовать ли проекцию скрытого состояния.
            weight_scale (float): Масштаб для экспоненциального взвешивания.
            item_num (int): Количество предметов.
            maxlen (int): Максимальная длина последовательности.
            hidden_units (int): Размер скрытого представления.
            num_heads (int): Количество голов в Multi-Head Attention.
            num_layers (int): Количество слоев трансформера.
            dropout_rate (float): Доля дропаутов.
            reconstruction_layer (int): Индекс слоя для получения скрытого состояния для реконструкции профиля.
                                        -1 означает использовать финальный слой.
            add_head (bool): Применять ли выходную проекцию на словарь предметов.
            mask_token (int): Токен для маскирования элементов.
        """
        super(BERT4RecLLM, self).__init__(
            item_num=item_num,
            maxlen=maxlen,
            hidden_units=hidden_units,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            mask_token=mask_token,
            add_head=add_head
        )

        self.reconstruction_layer = reconstruction_layer
        if weighting_scheme == 'mean':
            self.weighting_fn = mean_weightening
            self.weighting_kwargs = {}
        elif weighting_scheme == 'exponential':
            self.weighting_fn = exponential_weightening
            self.weighting_kwargs = {'weight_scale': weight_scale}
        elif weighting_scheme == 'attention':
            self.weighting_fn = SimpleAttentionAggregator(self.hidden_units)
            self.weighting_kwargs = {}

        self.use_down_scale = use_down_scale
        self.use_upscale = use_upscale

        if use_down_scale:
            self.profile_transform = nn.Linear(profile_emb_dim, self.hidden_units)
        if use_upscale:
            self.hidden_layer_transform = nn.Linear(self.hidden_units, profile_emb_dim)

    def forward(self, input_seq, attention_mask=None, user_profile_emb=None, return_hidden_states=False):
        """
        Args:
            input_seq (torch.Tensor): Индексы предметов [batch_size, seq_len].
            attention_mask (torch.Tensor): Маска внимания [batch_size, seq_len].
            user_profile_emb (torch.Tensor or None): Эмбеддинги профиля пользователя [batch_size, user_emb_dim].
            return_hidden_states (bool): Возвращать ли скрытые состояния всех слоёв.

        Returns:
            torch.Tensor: Логиты предсказаний [batch_size, seq_len, vocab_size] или скрытые состояния [batch_size, seq_len, hidden_units], если add_head=False.
            torch.Tensor or None: Реконструированный профиль [batch_size, hidden_units] для использования в реконструкции.
        """
        # Если маска внимания не предоставлена, создаем её
        if attention_mask is None:
            attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]
        
        outputs = self.bert(input_ids=input_seq, attention_mask=attention_mask, output_hidden_states=True)

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_units]

        if self.add_head:
            logits = self.out(sequence_output)  # [batch_size, seq_len, vocab_size]
        else:
            logits = sequence_output  # [batch_size, seq_len, hidden_units]

        hidden_states = outputs.hidden_states  # Tuple of hidden states

        # Выбор слоя для реконструкции
        if self.reconstruction_layer == -1:
            selected_hidden_state = hidden_states[-1]  # Финальный слой
        else:
            selected_hidden_state = hidden_states[self.reconstruction_layer]  # Выбранный слой

        # агрегация по последовательности
        reconstruction_input = self.weighting_fn(selected_hidden_state, **self.weighting_kwargs)  # [batch_size, hidden_units]
        return logits, reconstruction_input
