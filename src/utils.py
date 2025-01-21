# src/utils.py

import random
import numpy as np
import torch
import json

from torch import nn


def set_seed(seed):
    """Устанавливает зерно для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_user_profile_embeddings(file_path, user_id_mapping):
    """Загружает эмбеддинги профилей пользователей и сопоставляет их с индексами пользователей."""
    with open(file_path, 'r') as f:
        user_profiles_data = json.load(f)
    # Создаём словарь: оригинальный ID пользователя -> эмбеддинг
    user_profiles_dict = {str(user): user_profiles_data[user] for user in user_profiles_data}

    embedding_dim = len(next(iter(user_profiles_dict.values())))
    num_users = len(user_id_mapping)
    user_profiles_list = [None for _ in range(num_users)]
    null_profile_binary_mask = [False for _ in range(num_users)]

    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_dict.get(str(original_id))
        if embedding is not None:
            user_profiles_list[idx] = embedding
        else:
            # Если эмбеддинг не найден, инициализируем нулями
            user_profiles_list[idx] = [0.0] * embedding_dim
            null_profile_binary_mask[idx] = True

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float)
    null_profile_binary_mask_tensor = torch.BoolTensor(null_profile_binary_mask)
    return user_profiles_tensor, null_profile_binary_mask_tensor


def init_criterion_reconstruct(criterion_name):
    if criterion_name == 'MSE':
        return lambda x,y: nn.MSELoss()(x,y)
    if criterion_name == 'RMSE':
        return lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
    if criterion_name == 'CosSim':
        return lambda x,y: 1 - torch.mean(nn.CosineSimilarity(dim=1, eps=1e-6)(x,y))
    raise Exception('Not existing reconstruction loss')


def calculate_recsys_loss(target_seq, outputs, criterion):
    # Проверяем, если outputs является кортежем (на всякий случай)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    logits = outputs.view(-1, outputs.size(-1))
    targets = target_seq.view(-1)

    loss = criterion(logits, targets)
    return loss


def calculate_guide_loss(model, user_profile_emb, hidden_for_reconstruction,
                                 null_profile_binary_mask_batch, criterion_reconstruct_fn, device):
    if model.use_down_scale:
        user_profile_emb_transformed = model.profile_transform(user_profile_emb)
    else:
        user_profile_emb_transformed = user_profile_emb.detach().clone().to(device)
    if model.use_upscale:
        hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
    user_profile_emb_transformed[null_profile_binary_mask_batch] = hidden_for_reconstruction[
        null_profile_binary_mask_batch]

    loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
    return loss_guide