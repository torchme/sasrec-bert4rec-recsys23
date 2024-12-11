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
from src.models.bert4rec import BERT4Rec
from src.models.bert4recllm import BERT4RecLLM
from src.utils import set_seed, load_user_profile_embeddings
from src.dataset import SequenceDataset
from src.evaluation import evaluate_model
import mlflow

# Активируем обнаружение аномалий в PyTorch
torch.autograd.set_detect_anomaly(True)

def validate_config(config):
    # Проверка типов параметров
    assert isinstance(config['training']['learning_rate'], (float, int)), "learning_rate должен быть числом"
    assert isinstance(config['model']['num_blocks'], int), "num_blocks должен быть целым числом"
    assert isinstance(config['model']['num_heads'], int), "num_heads должен быть целым числом"
    # Добавьте другие проверки при необходимости

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

    # Валидация конфигурации
    validate_config(config)

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Получение имени модели из конфигурации
    model_name = config['model']['model_name']

    if model_name in ['SASRecLLM', 'BERT4RecLLM']:
        user_profile_embeddings = load_user_profile_embeddings(
            config['data']['user_profile_embeddings_path'],
            user_id_mapping
        )  # Tensor размерности [num_users, profile_emb_dim]
        profile_emb_dim = user_profile_embeddings.size(1)

        # Перемещаем эмбеддинги профилей пользователей на устройство
        user_profile_embeddings = user_profile_embeddings.to(device)
    else:
        user_profile_embeddings = None
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

    # Проверка на NaN в весах модели
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf detected in model parameter: {name}")

    # Оптимизатор и функция потерь
    print(f"Learning rate type: {type(config['training']['learning_rate'])}, value: {config['training']['learning_rate']}")
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
        total_loss_model = 0
        total_loss_guide = 0

        for batch_idx, batch in enumerate(train_loader, 1):
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            user_ids = user_ids.to(device)

            if model_name in ['SASRecLLM', 'BERT4RecLLM']:
                # Получаем эмбеддинги профиля пользователя, если они существуют
                if user_profile_embeddings is not None:
                    user_profile_emb = user_profile_embeddings[user_ids]
                else:
                    user_profile_emb = None

                outputs, reconstructed_profile = model(input_seq, user_profile_emb=user_profile_emb)

                logits = outputs.view(-1, outputs.size(-1))
                targets = target_seq.view(-1)

                # Проверка на NaN и Inf в логитах
                # if torch.isnan(logits).any() or torch.isinf(logits).any():
                #     print(f"NaN or Inf detected in logits at epoch {epoch}, batch {batch_idx}")
                #     continue  # Пропустить этот батч или обработать по-другому

                loss_model = criterion(logits, targets)

                if reconstructed_profile is not None:
                    loss_guide = nn.MSELoss()(reconstructed_profile, user_profile_emb)
                    #cosine_loss = nn.CosineEmbeddingLoss()
                    #target = torch.ones(user_profile_emb.size(0)).to(device)
                    #loss_guide = cosine_loss(reconstructed_profile, user_profile_emb, target)

                    if epoch < fine_tune_epoch:
                        loss = alpha * loss_guide + (1 - alpha) * loss_model
                    else:
                        loss = loss_model
                else:
                    loss = loss_model
                    loss_guide = torch.tensor(0.0, device=device)  # Для логирования

            else:
                # Для SASRec или BERT4Rec получаем только outputs
                outputs, _ = model(input_seq)
                logits = outputs.view(-1, outputs.size(-1))
                targets = target_seq.view(-1)

                # Проверка на NaN и Inf в логитах
                # if torch.isnan(logits).any() or torch.isinf(logits).any():
                #     print(f"NaN or Inf detected in logits at epoch {epoch}, batch {batch_idx}")
                #     continue  # Пропустить этот батч или обработать по-другому

                loss = criterion(logits, targets)
                loss_model = loss
                loss_guide = torch.tensor(0.0, device=device)  # Для логирования

            # Шаги оптимизации
            optimizer.zero_grad()
            loss.backward()

            # Проверка градиентов на NaN и Inf
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            #             print(f"NaN or Inf detected in gradients of {name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            total_loss_model += loss_model.item()
            total_loss_guide += loss_guide.item()

        avg_loss = total_loss / len(train_loader)
        avg_loss_model = total_loss_model / len(train_loader)
        avg_loss_guide = total_loss_guide / len(train_loader)

        print(f"Epoch {epoch}/{config['training']['epochs']}, Loss: {avg_loss:.4f}, "
              f"Loss_Model: {avg_loss_model:.4f}, Loss_Guide: {avg_loss_guide:.4f}")

        # Логирование метрик в MLflow
        mlflow.log_metric('train_loss', avg_loss, step=epoch)
        mlflow.log_metric('train_loss_model', avg_loss_model, step=epoch)
        mlflow.log_metric('train_loss_guide', avg_loss_guide, step=epoch)

        # Оценка на валидационном наборе
        if epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate_model(model, valid_loader, device, mode='validation')
            print(f"Validation Metrics: {val_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in val_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'val_{sanitized_metric_name}', metric_value, step=epoch)

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
    mlflow.log_artifact(model_save_path)

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
    elif model_name == 'BERT4Rec':
        model = BERT4Rec(
            item_num=config['model']['item_num'],
            maxlen=config['model']['maxlen'],
            hidden_units=config['model']['hidden_units'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_blocks'],  # Используем num_blocks как num_layers
            dropout_rate=config['model']['dropout_rate']
        ).to(device)
    elif model_name == 'BERT4RecLLM':
        model = BERT4RecLLM(
            item_num=config['model']['item_num'],
            maxlen=config['model']['maxlen'],
            hidden_units=config['model']['hidden_units'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_blocks'],  # Используем num_blocks как num_layers
            dropout_rate=config['model']['dropout_rate'],
            user_emb_dim=profile_emb_dim,
            reconstruction_layer=config['model'].get('reconstruction_layer', -1),
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

    # # Проверка типов конфигурации
    # print("Configuration Types:")
    # for key, value in config.items():
    #     print(f"{key}: {type(value)}")
    #     if isinstance(value, dict):
    #         for subkey, subvalue in value.items():
    #             print(f"  {subkey}: {type(subvalue)}")

    validate_config(config)

    train_model(config)
