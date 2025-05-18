import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BERT4Rec(nn.Module):
    """Классическая реализация модели BERT4Rec."""

    def __init__(self, item_num, maxlen, hidden_units, num_heads, num_layers, dropout_rate, mask_token=None, add_head=True):
        super(BERT4Rec, self).__init__()

        self.item_num = item_num
        self.maxlen = maxlen
        self.add_head = add_head

        # Set masking token
        self.mask_token = mask_token if mask_token is not None else item_num + 1
        # Vocab size: items + padding token (0) + masking token
        vocab_size = item_num + 2  # +1 for padding, +1 for mask token

        # Конфигурация BERT
        bert_config = BertConfig(
            vocab_size=vocab_size,
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

        if self.add_head:
            # Предсказание товаров
            self.out = nn.Linear(hidden_units, vocab_size)
        self.hidden_units = hidden_units

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if isinstance(module, nn.Embedding):
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_seq, attention_mask=None, user_profile_emb=None):
        """
        input_seq: Tensor с индексами товаров [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] или None
        user_profile_emb: Не используется в классической модели BERT4Rec
        """
        attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]
        if attention_mask is None:
            attention_mask = (input_seq != 0).long()  # [batch_size, seq_len]

        # Получаем выходы из BERT
        outputs = self.bert(input_ids=input_seq, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_units]

        if self.add_head:
            # Предсказание следующего товара
            logits = self.out(sequence_output)  # [batch_size, seq_len, item_num + 1]
        else:
            logits = sequence_output  # [batch_size, seq_len, hidden_units]

        # Возвращаем формат: (outputs, reconstructed_profile)
        return logits, None