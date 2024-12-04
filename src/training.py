# src/training.py

import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.sasrec import SASRec
from src.models.sasrecllm import SASRecLLM
from src.utils import set_seed, load_user_profile_embeddings
from src.dataset import SequenceDataset
from src.evaluation import evaluate_model
import mlflow
from tqdm import tqdm

# Активируем обнаружение аномалий в PyTorch
torch.autograd.set_detect_anomaly(True)

def train_model(config):
    # Установка зерна для воспроизводимости
    set_seed(config['seed'])

    # Загрузка обработанных данных
    with open(config['data']['train_sequences'], 'rb') as f:
        train_sequences = pickle.load(f)
    with open(config['data']['valid_sequences'], 'rb') as f:
        valid_sequences = pickle.load(f)
    with open(config['data']['test_sequences'], 'rb') as f:
        test_sequences = pickle.load(f)
    with open(config['data']['mappings'], 'rb') as f:
        mappings = pickle.load(f)
    with open(config['data']['counts'], 'rb') as f:
        counts = pickle.load(f)

    user_id_mapping, item_id_mapping = mappings
    num_users, num_items = counts
    # Обновляем параметры модели
    config['model']['item_num'] = num_items
    config['model']['user_num'] = num_users

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE:', device)

    # Получение имени модели из конфигурации
    model_name = config['model']['model_name']

    if model_name == 'SASRecLLM':
        user_profile_embeddings, null_profile_binary_mask = load_user_profile_embeddings(
            config['data']['user_profile_embeddings_path'],
            user_id_mapping
        )  # Tensor размерности [num_users, profile_emb_dim]
        profile_emb_dim = user_profile_embeddings.size(1)

        # Перемещаем эмбеддинги профилей пользователей на устройство
        user_profile_embeddings = user_profile_embeddings.to(device)
        null_profile_binary_mask = null_profile_binary_mask.to(device)
    else:
        user_profile_embeddings = None
        null_profile_binary_mask = None
        profile_emb_dim = None

    # Создание датасетов
    train_dataset = SequenceDataset(train_sequences, config['model']['maxlen'])
    valid_dataset = SequenceDataset(valid_sequences, config['model']['maxlen'])
    test_dataset = SequenceDataset(test_sequences, config['model']['maxlen'])

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Инициализация модели
    model = get_model(
        model_name, config, device,
        profile_emb_dim=profile_emb_dim
    )

    # Оптимизатор и функция потерь
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем паддинги

    # Создание директории для сохранения модели
    model_dir = config['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Начало логирования с помощью MLflow
    mlflow.start_run(run_name=config['experiment_name'])
    mlflow.log_params(config['model'])
    mlflow.log_params(config['training'])

    # Параметры для комбинированной функции потерь
    alpha = config['training'].get('alpha', 0.5)
    fine_tune_epoch = config['training'].get('fine_tune_epoch', config['training']['epochs'] // 2)

    # Цикл обучения
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        # c = 0
        for batch in (train_loader):
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            user_ids = user_ids.to(device)

            if model_name == 'SASRecLLM':
                # Получаем эмбеддинги профиля пользователя, если они существуют
                if user_profile_embeddings is not None:
                    user_profile_emb = user_profile_embeddings[user_ids]
                    null_profile_binary_mask_batch = null_profile_binary_mask[user_ids]
                else:
                    user_profile_emb = None

                outputs, hidden_for_reconstruction = model(input_seq, user_profile_emb=user_profile_emb)

                logits = outputs.view(-1, outputs.size(-1))
                targets = target_seq.view(-1)

                # лосс модели
                loss_model = criterion(logits, targets)

                # pass
                user_profile_emb_transformed = model.profile_transform(user_profile_emb)
                user_profile_emb_transformed[null_profile_binary_mask_batch] = hidden_for_reconstruction[null_profile_binary_mask_batch]

                loss_guide = nn.MSELoss()(hidden_for_reconstruction, user_profile_emb_transformed)
                if epoch < fine_tune_epoch:
                    loss = alpha * loss_guide + (1 - alpha) * loss_model
                else:
                    loss = loss_model
            else:
                # Для SASRec получаем только outputs
                outputs = model(input_seq)

                # Проверяем, если outputs является кортежем (на всякий случай)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                logits = outputs.view(-1, outputs.size(-1))
                targets = target_seq.view(-1)

                loss = criterion(logits, targets)

            # Шаги оптимизации
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

            # c += 1
            # if c == 1:
            #     break

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{config['training']['epochs']}, Loss: {avg_loss:.4f}")

        # Логирование метрик в MLflow
        mlflow.log_metric('train_loss', avg_loss, step=epoch)

        # Оценка на валидационном наборе
        if epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate_model(model, valid_loader, device, mode='validation')
            print(f"Validation Metrics: {val_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in val_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'val_{sanitized_metric_name}', metric_value, step=epoch)

            test_metrics = evaluate_model(model, test_loader, device, mode='test')
            print(f"Test Metrics: {test_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in test_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'test_{sanitized_metric_name}', metric_value, step=epoch)

    # Оценка на тестовом наборе данных с использованием нового метода
    test_metrics = evaluate_model(model, test_loader, device, mode='test')
    print(f"Test Metrics: {test_metrics}")
    # Логирование метрик с заменой недопустимых символов
    for metric_name, metric_value in test_metrics.items():
        sanitized_metric_name = metric_name.replace('@', '_')
        mlflow.log_metric(f'test_{sanitized_metric_name}', metric_value)

    # Сохранение модели
    model_save_path = os.path.join(model_dir, f'{model_name}_model.pt')
    torch.save(model.state_dict(), model_save_path)
    # mlflow.log_artifact(model_save_path, artifact_path=model_dir)

    mlflow.end_run()

def get_model(model_name, config, device, profile_emb_dim=None):
    if model_name == 'SASRecLLM':
        model = SASRecLLM(
            item_num=config['model']['item_num'],
            profile_emb_dim=profile_emb_dim,
            maxlen=config['model']['maxlen'],
            hidden_units=config['model']['hidden_units'],
            num_blocks=config['model']['num_blocks'],
            num_heads=config['model']['num_heads'],
            dropout_rate=config['model']['dropout_rate'],
            initializer_range=config['model'].get('initializer_range', 0.02),
            add_head=config['model'].get('add_head', True),
            reconstruction_layer=config['model'].get('reconstruction_layer', -1)
        ).to(device)
    elif model_name == 'SASRec':
        model = SASRec(
            item_num=config['model']['item_num'],
            maxlen=config['model']['maxlen'],
            hidden_units=config['model']['hidden_units'],
            num_blocks=config['model']['num_blocks'],
            num_heads=config['model']['num_heads'],
            dropout_rate=config['model']['dropout_rate'],
            initializer_range=config['model'].get('initializer_range', 0.02),
            add_head=config['model'].get('add_head', True)
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_model(config)
