# src/training.py

import os
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # Добавляем поддержку смешанной точности

from src.models.bert4rec import BERT4Rec
from src.models.bert4recllm import BERT4RecLLM
from src.models.sasrec import SASRec
from src.models.sasrecllm import SASRecLLM
from src.utils import set_seed, load_user_profile_embeddings, init_criterion_reconstruct
from src.dataset import SequenceDataset, BERT4RecDataset
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

    # Инициализация GradScaler для смешанной точности
    use_mixed_precision = config['training'].get('use_mixed_precision', False)
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    if use_mixed_precision and device.type == 'cuda':
        print("Using mixed precision training")
    
    # Получение имени модели из конфигурации
    model_name = config['model']['model_name']

    if model_name in ['SASRecLLM', 'BERT4RecLLM']:
        user_profile_embeddings, null_profile_binary_mask = load_user_profile_embeddings(
            config['data']['user_profile_embeddings_path'],
            user_id_mapping
        )  # Tensor размерности [num_users, profile_emb_dim]
        profile_emb_dim = user_profile_embeddings.size(1)
        assert profile_emb_dim != 2

        # Перемещаем эмбеддинги профилей пользователей на устройство
        user_profile_embeddings = user_profile_embeddings.to(device)
        null_profile_binary_mask = null_profile_binary_mask.to(device)
    else:
        user_profile_embeddings = None
        null_profile_binary_mask = None
        profile_emb_dim = None

    # Создание датасетов в зависимости от модели
    if model_name in ['BERT4Rec', 'BERT4RecLLM']:
        # Для BERT4Rec используем специальный датасет с маскированием
        mask_token = num_items + 1  # Маска будет иметь ID, равный num_items + 1
        
        train_dataset = BERT4RecDataset(
            train_sequences, 
            config['model']['maxlen'], 
            mask_token, 
            num_items, 
            mask_prob=config['training'].get('mask_prob', 0.15),
            mode='train'
        )
        
        valid_dataset = BERT4RecDataset(
            valid_sequences, 
            config['model']['maxlen'], 
            mask_token, 
            num_items, 
            mode='valid'
        )
        
        test_dataset = BERT4RecDataset(
            test_sequences, 
            config['model']['maxlen'], 
            mask_token, 
            num_items, 
            mode='test'
        )
        
        # Обновляем конфигурацию модели, чтобы учесть маскирующий токен
        config['model']['mask_token'] = mask_token
        config['model']['vocab_size'] = num_items + 2  # +1 для padding, +1 для mask
    else:
        # Для SASRec используем стандартный датасет
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
    
    # Выбор функции потерь в зависимости от модели
    if model_name in ['BERT4Rec', 'BERT4RecLLM']:
        # Для BERT4Rec игнорируем токены с target -100
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        # Для SASRec игнорируем паддинги (0)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
    if model_name in ['SASRecLLM', 'BERT4RecLLM']:
        if 'reconstruct_loss' in config['training']:
            criterion_reconstruct_fn = init_criterion_reconstruct(config['training']['reconstruct_loss'])
        else:
            criterion_reconstruct_fn = lambda x,y: nn.MSELoss()(x,y)

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
    
    # Параметры для градиентного накопления
    grad_accum_steps = config['training'].get('grad_accum_steps', 4)  # По умолчанию 4 шага аккумуляции
    effective_batch_size = config['training']['batch_size'] * grad_accum_steps
    print(f"Using gradient accumulation: {grad_accum_steps} steps (effective batch size: {effective_batch_size})")

    # Цикл обучения
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Обнуляем градиенты в начале эпохи
        
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            # Обработка входных данных в зависимости от типа модели
            if model_name in ['BERT4Rec', 'BERT4RecLLM']:
                input_seq, target_seq, attention_mask, user_ids = batch
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                attention_mask = attention_mask.to(device)
                user_ids = user_ids.to(device)
            else:
                input_seq, target_seq, user_ids = batch
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                user_ids = user_ids.to(device)
                attention_mask = None

            # Используем autocast для смешанной точности при прямом проходе
            with autocast(enabled=use_mixed_precision and device.type == 'cuda'):
                if model_name in ['SASRecLLM', 'BERT4RecLLM']:
                    # Получаем эмбеддинги профиля пользователя, если они существуют
                    if user_profile_embeddings is not None:
                        user_profile_emb = user_profile_embeddings[user_ids]
                        null_profile_binary_mask_batch = null_profile_binary_mask[user_ids]
                    else:
                        user_profile_emb = None

                    # Прямой проход с учетом типа модели
                    if model_name == 'BERT4RecLLM':
                        outputs, hidden_for_reconstruction = model(
                            input_seq, 
                            attention_mask=attention_mask, 
                            user_profile_emb=user_profile_emb
                        )
                    else:
                        outputs, hidden_for_reconstruction = model(
                            input_seq, 
                            user_profile_emb=user_profile_emb
                        )

                    # Форматирование выходных данных и целей
                    if model_name == 'BERT4RecLLM':
                        # Для BERT4Rec обрабатываем все токены
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_seq.view(-1)
                    else:
                        # Для SASRec обрабатываем только следующий токен
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_seq.view(-1)

                    # Лосс модели
                    loss_model = criterion(logits, targets)

                    # Reconstruction loss
                    if model.use_down_scale:
                        user_profile_emb_transformed = model.profile_transform(user_profile_emb)
                    else:
                        user_profile_emb_transformed = user_profile_emb.detach().clone().to(device)
                    if model.use_upscale:
                        hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
                    
                    user_profile_emb_transformed[null_profile_binary_mask_batch] = hidden_for_reconstruction[null_profile_binary_mask_batch]

                    loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)
                    if epoch < fine_tune_epoch:
                        loss = alpha * loss_guide + (1 - alpha) * loss_model
                    else:
                        loss = loss_model
                else:
                    # Прямой проход для базовых моделей
                    if model_name == 'BERT4Rec':
                        outputs = model(input_seq, attention_mask=attention_mask)
                    else:
                        outputs = model(input_seq)

                    # Проверяем, если outputs является кортежем (на всякий случай)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    # Форматирование выходных данных и целей
                    if model_name == 'BERT4Rec':
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_seq.view(-1)
                    else:
                        logits = outputs.view(-1, outputs.size(-1))
                        targets = target_seq.view(-1)

                    loss = criterion(logits, targets)

            # Масштабируем потери по количеству шагов накопления
            scaled_loss = loss / grad_accum_steps
            
            # Используем scaler для обратного распространения при смешанной точности
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Обновляем веса только после накопления градиентов
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps  # Умножаем обратно для корректного подсчета потерь

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{config['training']['epochs']}, Loss: {avg_loss:.4f}")

        # Логирование метрик в MLflow
        mlflow.log_metric('train_loss', avg_loss, step=epoch)

        # Оценка на валидационном наборе
        if epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate_bert4rec_model(model, valid_loader, device, mode='validation') if model_name in ['BERT4Rec', 'BERT4RecLLM'] else evaluate_model(model, valid_loader, device, mode='validation')
            print(f"Validation Metrics: {val_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in val_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'val_{sanitized_metric_name}', metric_value, step=epoch)

            test_metrics = evaluate_bert4rec_model(model, test_loader, device, mode='test') if model_name in ['BERT4Rec', 'BERT4RecLLM'] else evaluate_model(model, test_loader, device, mode='test')
            print(f"Test Metrics: {test_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in test_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'test_{sanitized_metric_name}', metric_value, step=epoch)

    # Оценка на тестовом наборе данных с использованием нового метода
    test_metrics = evaluate_bert4rec_model(model, test_loader, device, mode='test') if model_name in ['BERT4Rec', 'BERT4RecLLM'] else evaluate_model(model, test_loader, device, mode='test')
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

def evaluate_bert4rec_model(model, data_loader, device, mode='validation', k_list=[5, 10, 20]):
    """
    Оценивает BERT4Rec модель на заданном наборе данных.
    В этой функции мы фокусируемся на предсказании маскированного токена.
    """
    model.eval()
    recalls = {k: 0 for k in k_list}
    ndcgs = {k: 0 for k in k_list}

    with torch.no_grad():
        for batch in data_loader:
            input_seq, target_seq, attention_mask, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            attention_mask = attention_mask.to(device)
            
            # Прямой проход
            outputs = model(input_seq, attention_mask=attention_mask)
            
            # Если модель возвращает кортеж, извлекаем первый элемент
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # В BERT4Rec для инференса мы предсказываем только последний токен
            # Находим индекс маскирующего токена для каждой последовательности
            # Обычно это последний токен в последовательности
            mask_idx = (input_seq == model.mask_token).nonzero(as_tuple=True)
            batch_size = input_seq.size(0)
            
            # Извлекаем логиты для маскирующего токена
            logits = torch.zeros((batch_size, outputs.size(-1)), device=device)
            
            for i in range(batch_size):
                # Находим индексы маскирующих токенов для текущего примера
                indices = (mask_idx[0] == i).nonzero(as_tuple=True)[0]
                if indices.size(0) > 0:
                    # Берем последний маскирующий токен
                    idx = mask_idx[1][indices[-1]]
                    logits[i] = outputs[i, idx]
            
            # Получаем целевые значения (последний токен в target_seq)
            targets = target_seq[:, -1]
            
            # Маскируем паддинг с индексом 0
            logits[:, 0] = -float('inf')
            
            # Получаем топ-K предсказаний
            _, indices = torch.topk(logits, max(k_list), dim=1)
            
            for k in k_list:
                preds_k = indices[:, :k]
                
                # Проверяем, есть ли целевой элемент среди топ-K предсказаний
                correct = preds_k.eq(targets.view(-1, 1))
                
                # Вычисляем Recall@k
                correct_any = correct.any(dim=1).float()
                recalls[k] += correct_any.sum().item()
                
                # Вычисляем NDCG@k
                ranks = torch.where(correct)[1] + 1
                ndcg_k = (1 / torch.log2(ranks.float() + 1)).sum().item() if ranks.numel() > 0 else 0.0
                ndcgs[k] += ndcg_k
    
    # Вычисляем средние значения метрик
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = recalls[k] / len(data_loader.dataset)
        metrics[f'NDCG@{k}'] = ndcgs[k] / len(data_loader.dataset)
    
    return metrics

def get_model(model_name, config, device, profile_emb_dim=None):
    if model_name == 'SASRecLLM':
        model = SASRecLLM(
            item_num=config['model']['item_num'],
            profile_emb_dim=profile_emb_dim,
            weighting_scheme = config['model']['weighting_scheme'] if 'weighting_scheme' in config['model'] else 'mean',
            weight_scale=config['model']['weight_scale'] if 'weight_scale' in config['model'] else None,
            use_down_scale=config['model']['use_down_scale'] if 'use_down_scale' in config['model'] else True,
            use_upscale=config['model']['use_upscale'] if 'use_upscale' in config['model'] else False,
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
            dropout_rate=config['model']['dropout_rate'],
            mask_token=config['model']['mask_token']
        ).to(device)
    elif model_name == 'BERT4RecLLM':
        model = BERT4RecLLM(
            profile_emb_dim=profile_emb_dim,
            weighting_scheme=config['model']['weighting_scheme'] if 'weighting_scheme' in config['model'] else 'mean',
            weight_scale=config['model']['weight_scale'] if 'weight_scale' in config['model'] else None,
            use_down_scale=config['model']['use_down_scale'] if 'use_down_scale' in config['model'] else True,
            use_upscale=config['model']['use_upscale'] if 'use_upscale' in config['model'] else False,
            item_num=config['model']['item_num'],
            maxlen=config['model']['maxlen'],
            hidden_units=config['model']['hidden_units'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_blocks'],  # Используем num_blocks как num_layers
            dropout_rate=config['model']['dropout_rate'],
            reconstruction_layer=config['model'].get('reconstruction_layer', -1),
            add_head=config['model'].get('add_head', True),
            mask_token=config['model']['mask_token']
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def process_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train_model(config)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train recommendation model")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    process_config(args.config)
