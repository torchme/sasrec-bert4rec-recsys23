# src/utils.py

import random
import numpy as np
import torch
import json

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
    user_profiles_dict = {str(user['id']): user['embedding'] for user in user_profiles_data}

    embedding_dim = len(next(iter(user_profiles_dict.values())))
    num_users = len(user_id_mapping)
    user_profiles_list = []

    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_dict.get(str(original_id))
        if embedding is not None:
            user_profiles_list.append(embedding)
        else:
            # Если эмбеддинг не найден, инициализируем нулями
            user_profiles_list.append([0.0] * embedding_dim)

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return user_profiles_tensor