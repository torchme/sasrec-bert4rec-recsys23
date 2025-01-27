# src/models/sasrecllm.py

import torch
import torch.nn as nn
from src.models.sasrec import SASRec
from src.models.utils import mean_weightening, exponential_weightening, SimpleAttentionAggregator

class SASRecLLM(SASRec):
    def __init__(
        self,
        item_num,
        profile_emb_dim,
        weighting_scheme='mean',
        use_down_scale=True,
        use_upscale=False,
        weight_scale=None,
        multi_profile=False,
        multi_profile_aggr_scheme='mean',
        *args,
        **kwargs
    ):
        # Извлекаем reconstruction_layer
        self.reconstruction_layer = kwargs.pop('reconstruction_layer', -1)
        super().__init__(item_num, *args, **kwargs)

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
        self.multi_profile = multi_profile

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
            user_profile_emb = user_profile_emb.view(bsz*K, edim)
            user_profile_emb = self.profile_transform(user_profile_emb)  # => [bsz*K, hidden_units]
            user_profile_emb = user_profile_emb.view(bsz, K, self.hidden_units)
            emb_dim_now = self.hidden_units
        else:
            emb_dim_now = edim

        # Теперь агрегируем
        aggregated = self.profile_aggregator(user_profile_emb, *self.multi_profile_weighting_kwargs)  # => [bsz, emb_dim_now])
        return aggregated

    def forward(self, input_ids, return_hidden_states=False):
        # 1) агрегируем профиль
        # user_profile_emb_agg = self.aggregate_profile(user_profile_emb)

        # 2) Обычный SASRec forward
        seqs = self.item_emb(input_ids)
        seqs *= self.hidden_units ** 0.5

        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, dtype=torch.long,
                                 device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (input_ids == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=seqs.device)
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

        # 3) Reconstruction layer: Получаем представление для реконструкции профиля из outputs до применения add_head
        if self.reconstruction_layer == -1:
            reconstruction_input = self.weighting_fn(outputs, **self.weighting_kwargs)  # [batch_size, hidden_units]
        else:
            reconstruction_input = self.weighting_fn(hidden_states[self.reconstruction_layer % 2], **self.weighting_kwargs)  # [batch_size, hidden_units]

        # 4) add_head => logits
        if self.add_head:
            outputs = torch.matmul(outputs, self.item_emb.weight.transpose(0, 1))

        return outputs, reconstruction_input
