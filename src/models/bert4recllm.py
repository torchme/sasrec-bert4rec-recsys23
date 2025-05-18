import torch
import torch.nn as nn
from src.models.bert4rec import BERT4Rec
from transformers import BertConfig, BertModel

from src.models.utils import mean_weightening, exponential_weightening, SimpleAttentionAggregator


class BERT4RecLLM(BERT4Rec):
    def __init__(
        self,
        profile_emb_dim,
        weighting_scheme: str,
        use_down_scale: bool,
        use_upscale: bool,
        weight_scale: float,
        item_num,
        maxlen,
        hidden_units,
        num_heads,
        num_layers,
        dropout_rate,
        reconstruction_layer=-1,
        mask_token=None,
        add_head=True,
        multi_profile=False,  # <-- добавим флаг для много-профильного сценария
        multi_profile_aggr_scheme = 'mean',
    ):
        """
        BERT4Rec с поддержкой пользовательских эмбеддингов и реконструкции профиля (LLM часть).
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
        self.multi_profile = multi_profile

        # Агрегация последовательности взаимодействий
        if weighting_scheme == 'mean':
            self.weighting_fn = mean_weightening
            self.weighting_kwargs = {}
        elif weighting_scheme == 'exponential':
            self.weighting_fn = exponential_weightening
            self.weighting_kwargs = {'weight_scale': weight_scale}
        elif weighting_scheme == 'attention':
            self.weighting_fn = SimpleAttentionAggregator(self.hidden_units)
            self.weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such weighting_scheme {weighting_scheme} exists')

        # Агрегация нескольких профилей
        if multi_profile_aggr_scheme == 'mean':
            self.profile_aggregator = mean_weightening
            self.multi_profile_weighting_kwargs = {}
        elif multi_profile_aggr_scheme == 'attention':
            self.profile_aggregator = SimpleAttentionAggregator(profile_emb_dim if not use_down_scale
                                                                else self.hidden_units)
            self.multi_profile_weighting_kwargs = {}
        else:
            raise NotImplementedError(f'No such multi_profile_aggr_scheme {multi_profile_aggr_scheme} exists')

        self.use_down_scale = use_down_scale
        self.use_upscale = use_upscale

        if use_down_scale:
            self.profile_transform = nn.Linear(profile_emb_dim, self.hidden_units)
        if use_upscale:
            self.hidden_layer_transform = nn.Linear(self.hidden_units, profile_emb_dim)

    def aggregate_profile(self, user_profile_emb):
        """
        user_profile_emb: [batch_size, emb_dim]  или  [batch_size, K, emb_dim]
        Возвращает: [batch_size, hidden_units] (если use_down_scale=True) либо [batch_size, emb_dim].
        """
        if user_profile_emb is None:
            return None

        if user_profile_emb.dim() == 2:
            # Случай single-profile (batch_size, emb_dim)
            if self.use_down_scale:
                return self.profile_transform(user_profile_emb)  # => [batch_size, hidden_units]
            else:
                return user_profile_emb.detach().clone()

        # Иначе multi-profile => [batch_size, K, emb_dim]
        bsz, K, edim = user_profile_emb.shape

        # Сначала down_scale (если нужно)
        if self.use_down_scale:
            # Применим линейно к каждому из K профилей
            user_profile_emb = user_profile_emb.view(bsz * K, edim)
            user_profile_emb = self.profile_transform(user_profile_emb)  # => [bsz*K, hidden_units]
            user_profile_emb = user_profile_emb.view(bsz, K, self.hidden_units)
            emb_dim_now = self.hidden_units
        else:
            emb_dim_now = edim

        # Теперь агрегируем
        aggregated = self.profile_aggregator(user_profile_emb,
                                             *self.multi_profile_weighting_kwargs)  # => [bsz, emb_dim_now])
        return aggregated

    def forward(self, input_seq, user_profile_emb=None, attention_mask=None, return_hidden_states=False):
        """
        Args:
            input_seq: [batch_size, seq_len]
            user_profile_emb: [batch_size, emb_dim] или [batch_size, K, emb_dim]
            attention_mask (torch.Tensor): Маска внимания [batch_size, seq_len].
        """
        # 1) Сначала можем агрегировать профили, если нужно.
        # user_profile_emb_agg = self.aggregate_profile(user_profile_emb)
        # user_profile_emb_agg => [B, hidden_units] (если use_down_scale=True)
        #                       или [B, emb_dim]    (если False)
        # Далее вы решаете, как использовать user_profile_emb_agg в BERT4Rec:
        #   - Можете сложить с sequence_output
        #   - Можете вернуть просто как reconstruction_input
        #   - Или вообще использовать в лоссе отдельно.

        # 2) Прямой проход BERT
        # Если маска внимания не предоставлена, создаем её
        if attention_mask is None:
            attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]

        outputs = self.bert(
            input_ids=input_seq,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_units]

        # 3) Логиты или фичи
        if self.add_head:
            logits = self.out(sequence_output)  # [batch_size, seq_len, vocab_size]
        else:
            logits = sequence_output  # [batch_size, seq_len, hidden_units]

        hidden_states = outputs.hidden_states  # tuple всех слоёв [batch_size, seq_len, hidden_units]

        # 4) Выбор слоя для reconstruction
        if self.reconstruction_layer == -1:
            selected_hidden_state = hidden_states[-1]  # финальный
        else:
            selected_hidden_state = hidden_states[self.reconstruction_layer]

        # 5) Агрегация (mean/attention) item-секвенции => reconstruction_input
        #    (не путайте с user_profile_emb)
        reconstruction_input = self.weighting_fn(selected_hidden_state, **self.weighting_kwargs) # => [batch_size, hidden_units]

        return logits, reconstruction_input
