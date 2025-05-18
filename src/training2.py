# src/training.py

import os
import time

import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.bert4rec import BERT4Rec
from src.models.bert4recllm import BERT4RecLLM
from src.models.sasrec import SASRec
from src.models.sasrecllm import SASRecLLM
from src.utils import set_seed, load_user_profile_embeddings, init_criterion_reconstruct, \
    load_user_profile_embeddings_any, calculate_recsys_loss, calculate_guide_loss
from src.dataset import SequenceDataset, BERT4RecDataset
from src.evaluation import evaluate_model, evaluate_bert4rec_model
import mlflow
from tqdm import tqdm

# Активируем обнаружение аномалий в PyTorch
torch.autograd.set_detect_anomaly(True)

def train_model(config):
    # Установка зерна для воспроизводимости
    set_seed(config['seed'])

    # Загрузка обработанных данных
    with open(config['data']['profile_train_sequences'], 'rb') as f:
        profile_train_sequences = pickle.load(f)
    with open(config['data']['valid_sequences'], 'rb') as f:
        valid_sequences = pickle.load(f)
    with open(config['data']['test_sequences'], 'rb') as f:
        test_sequences = pickle.load(f)
    with open(config['data']['mappings'], 'rb') as f:
        mappings = pickle.load(f)
    with open(config['data']['counts'], 'rb') as f:
        counts = pickle.load(f)

    test_sequences = get_test_seqs_with_first_elem(profile_train_sequences, test_sequences)

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
    multi_profile = config['model'].get('multi_profile', False)

    if model_name in ['SASRecLLM', 'BERT4RecLLM']:
        # Загружаем эмбеддинги профилей
        # Получаем Tensor [num_users, profile_emb_dim] ИЛИ [num_users, K, profile_emb_dim]
        user_profile_embeddings, null_profile_binary_mask = load_user_profile_embeddings_any(config, user_id_mapping)

        profile_emb_dim = user_profile_embeddings.size(-1)
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

        profile_train_dataset = BERT4RecDataset(
            profile_train_sequences,
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
        profile_train_dataset = SequenceDataset(profile_train_sequences, config['model']['maxlen'])
        valid_dataset = SequenceDataset(valid_sequences, config['model']['maxlen'])
        test_dataset = SequenceDataset(test_sequences, config['model']['maxlen'])

    # Создание DataLoader
    profile_train_loader = DataLoader(profile_train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    if config['data']['finetune_train_sequences'] == config['data']['profile_train_sequences']:
        finetune_train_dataset = profile_train_dataset
        finetune_train_loader = profile_train_loader
    else:
        with open(config['data']['finetune_train_sequences'], 'rb') as f:
            finetune_train_sequences = pickle.load(f)
        finetune_train_dataset = SequenceDataset(finetune_train_sequences, config['model']['maxlen'])
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
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
    else:
        criterion_reconstruct_fn = None

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
    scale_guide_loss = config['training'].get('scale_guide_loss', False)

    save_checkpoints = config['training'].get('save_checkpoints', False)
    eval_every = config['training'].get('eval_every', 1)
    epochs = config['training']['epochs']

    # делаем тренировочный - датасет с профилями
    train_loader = profile_train_loader

    evaluate_function = evaluate_bert4rec_model if model_name in ['Bert4Rec', 'Bert4RecLLM'] else evaluate_model
    # Цикл обучения
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_guide_loss = 0
        total_recsys_loss = 0
        # c = 0

        print('Len of train loader:', len(train_loader))
        for batch in tqdm(train_loader):
            if model_name in ['BERT4Rec', 'BERT4RecLLM']:
                input_seq, target_seq, attention_mask, user_ids = batch
                attention_mask = attention_mask.to(device)
            else:
                input_seq, target_seq, user_ids = batch
                attention_mask = None
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            user_ids = user_ids.to(device)

            if model_name in ['SASRecLLM', 'BERT4RecLLM']:
                # Получаем эмбеддинги профиля пользователя, если они существуют
                if user_profile_embeddings is not None:
                    user_profile_emb = user_profile_embeddings[user_ids].to(device)
                    null_profile_binary_mask_batch = null_profile_binary_mask[user_ids].flatten().to(device)
                else:
                    user_profile_emb = None
                    null_profile_binary_mask_batch = None

                # outputs, hidden_for_reconstruction = model(input_seq)
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

                # # Форматирование выходных данных и целей
                # if model_name == 'BERT4RecLLM':
                #     # Для BERT4Rec обрабатываем все токены
                #     logits = outputs.view(-1, outputs.size(-1))
                #     targets = target_seq.view(-1)
                # else:
                #     # Для SASRec обрабатываем только следующий токен
                #     logits = outputs.view(-1, outputs.size(-1))
                #     targets = target_seq.view(-1)


                # лосс модели
                loss_model = calculate_recsys_loss(target_seq, outputs, criterion)
                loss_guide = calculate_guide_loss(model=model,
                                                 user_profile_emb=user_profile_emb,
                                                 hidden_for_reconstruction=hidden_for_reconstruction,
                                                 null_profile_binary_mask_batch=null_profile_binary_mask_batch,
                                                 criterion_reconstruct_fn=criterion_reconstruct_fn,)
                if scale_guide_loss:
                    loss_model_val = loss_model.item()
                    loss_guide_val = loss_guide.item()
                    eps = 1e-8

                    scale_for_guide = loss_model_val / (loss_guide_val + eps)
                    scaled_loss_guide = loss_guide * scale_for_guide
                    # total_guide_loss += scaled_loss_guide.item()
                    if epoch < fine_tune_epoch:
                        loss = alpha * scaled_loss_guide + (1 - alpha) * loss_model
                    else:
                        loss = loss_model
                else:
                    # Если scale_guide_loss=False, логика остаётся исходной
                    if epoch < fine_tune_epoch:
                        loss = alpha * loss_guide + (1 - alpha) * loss_model
                    else:
                        loss = loss_model

                total_guide_loss += loss_guide.item()

            else:
                # Для SASRec получаем только outputs
                outputs = model(input_seq)
                loss_model = calculate_recsys_loss(target_seq, outputs, criterion)
                loss = loss_model

            # Шаги оптимизации
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            total_recsys_loss += loss_model.item()

            # c += 1
            # if c == 1:
            #     break

        avg_loss = total_loss / len(train_loader)
        avg_guide_loss = total_guide_loss / len(train_loader)
        avg_recsys_loss = total_recsys_loss / len(train_loader)
        print(f"Epoch: {epoch}/{config['training']['epochs']} | Train Total Loss: {avg_loss:.4f} "
              f" | Recsys Total Loss | {avg_recsys_loss:.4f} | Guide Total Loss | {avg_guide_loss:.4f}")
        print('Train Time taken (s):', time.time() - start_time)

        # Сохраняем чекпоинт
        if save_checkpoints:
            checkpoint_path = os.path.join(model_dir, f'{model_name}_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Логирование метрик в MLflow
        mlflow.log_metric('train_loss', avg_loss, step=epoch)
        mlflow.log_metric('train_guide_loss', avg_guide_loss, step=epoch)
        mlflow.log_metric('train_recsys_loss', avg_recsys_loss, step=epoch)

        # Оценка на валидационном наборе
        if epoch % config['training']['eval_every'] == 0:
            val_metrics = evaluate_function(model, valid_loader, device, mode='validation',
                                         model_criterion=criterion, criterion_reconstruct_fn=criterion_reconstruct_fn,
                                         user_profile_embeddings=user_profile_embeddings, null_profile_binary_mask=null_profile_binary_mask)
            print(f"Validation Metrics: {val_metrics}")
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in val_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'val_{sanitized_metric_name}', metric_value, step=epoch)

            start_time = time.time()
            test_metrics = evaluate_function(model, test_loader, device, mode='test')
            print(f"Test Metrics: {test_metrics}")
            print('Test Time taken (s):', time.time() - start_time)
            # Логирование метрик с заменой недопустимых символов
            for metric_name, metric_value in test_metrics.items():
                sanitized_metric_name = metric_name.replace('@', '_')
                mlflow.log_metric(f'test_{sanitized_metric_name}', metric_value, step=epoch)

        # If it is the last epoch with profiles, we need to update the loader
        if (epoch == fine_tune_epoch - 1) and (model_name in ['SASRecLLM', 'BERT4RecLLM']):
            print('Changing loaders')
            train_loader = finetune_train_loader

    # Оценка на тестовом наборе данных с использованием нового метода
    test_metrics = evaluate_function(model, test_loader, device, mode='test')
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
            reconstruction_layer=config['model'].get('reconstruction_layer', -1),
            multi_profile=config['model'].get('multi_profile', False),  # наш дополнительный флаг
            multi_profile_aggr_scheme=config['model']['multi_profile_aggr_scheme'],
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
            add_head=config['model'].get('add_head', True),
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
            multi_profile=config['model'].get('multi_profile', False),  # наш дополнительный флаг
            multi_profile_aggr_scheme=config['model']['multi_profile_aggr_scheme'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def process_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train_model(config)


def get_test_seqs_with_first_elem(train_sequences, test_sequences):
    result = {}
    for user_id, sequence in test_sequences.items():
        if user_id not in train_sequences:
            continue
        result[user_id] = sequence[:len(train_sequences[user_id])+1]
    return result

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description="Train recommendation model")
    # parser.add_argument('--config', type=str, required=True, help="Path to config file")
    # args = parser.parse_args()

    configs = [
        'experiments-2_0/help_distill_sasrec/beauty_sasrec.yaml',
        'experiments-2_0/help_distill_sasrec/kion_sasrec.yaml',
        'experiments-2_0/help_distill_sasrec/ml20m_sasrec.yaml',
        'experiments-2_0/help_distill_sasrec/m2_sasrec.yaml',
    ]

    for config_path in configs:
        print('------------')
        print('START NEW CONFIG:', config_path)
        process_config(config_path)
