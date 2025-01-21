# src/utils.py
import random
import numpy as np
import torch
import json

from torch import nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_user_profile_embeddings(file_path, user_id_mapping):
    """
    Старый метод: загружает эмбеддинги профилей пользователей из ОДНОГО JSON-файла,
    результат: [num_users, emb_dim], [num_users]
    """
    with open(file_path, 'r') as f:
        user_profiles_data = json.load(f)

    embedding_dim = len(next(iter(user_profiles_data.values())))
    num_users = len(user_id_mapping)
    user_profiles_list = [None for _ in range(num_users)]
    null_profile_binary_mask = [False for _ in range(num_users)]

    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_data.get(str(original_id))
        if embedding is not None:
            user_profiles_list[idx] = embedding
        else:
            user_profiles_list[idx] = [0.0] * embedding_dim
            null_profile_binary_mask[idx] = True

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float)
    null_profile_binary_mask_tensor = torch.BoolTensor(null_profile_binary_mask)
    return user_profiles_tensor, null_profile_binary_mask_tensor


def load_user_profiles_multi(files_list, user_id_mapping):
    """
    Новый метод: загружает несколько JSON-файлов => [num_users, K, emb_dim], [num_users, K]
    """
    all_tensors = []
    all_masks = []

    for file_path in files_list:
        with open(file_path, 'r') as f:
            user_profiles_data = json.load(f)

        embedding_dim = len(next(iter(user_profiles_data.values())))
        num_users = len(user_id_mapping)

        user_profiles_list = [None] * num_users
        null_profile_binary_mask = [False] * num_users

        for original_id, idx in user_id_mapping.items():
            emb = user_profiles_data.get(str(original_id))
            if emb is not None:
                user_profiles_list[idx] = emb
            else:
                user_profiles_list[idx] = [0.0] * embedding_dim
                null_profile_binary_mask[idx] = True

        user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float32)
        null_mask_tensor = torch.tensor(null_profile_binary_mask, dtype=torch.bool)

        all_tensors.append(user_profiles_tensor)
        all_masks.append(null_mask_tensor)

    # stack => [num_users, K, emb_dim], [num_users, K]
    user_profiles_tensor_3d = torch.stack(all_tensors, dim=1)
    null_profile_binary_mask_2d = torch.stack(all_masks, dim=1)

    return user_profiles_tensor_3d, null_profile_binary_mask_2d


def load_user_profile_embeddings_any(config, user_id_mapping):
    """
    Универсальная функция:
    Если config['model']['multi_profile'] == True => грузим список файлов
    Иначе => грузим одиночный файл.
    """
    multi_profile = config['model'].get('multi_profile', False)

    if multi_profile:
        files_list = config['data']['user_profile_embeddings_files']
        return load_user_profiles_multi(files_list, user_id_mapping)
    else:
        path = config['data']['user_profile_embeddings_path']
        return load_user_profile_embeddings(path, user_id_mapping)


def init_criterion_reconstruct(criterion_name):
    if criterion_name == 'MSE':
        return lambda x,y: nn.MSELoss()(x,y)
    if criterion_name == 'RMSE':
        return lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
    if criterion_name == 'CosSim':
        return lambda x,y: torch.mean(nn.CosineSimilarity(dim=1, eps=1e-6)(x,y))
    raise Exception('Not existing reconstruction loss')
