import os
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
from src.utils import set_seed, load_user_profile_embeddings, init_criterion_reconstruct, load_user_profile_embeddings_any
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
    multi_profile = config['model'].get('multi_profile', False)

    if model_name in ['SASRecLLM', 'BERT4RecLLM']:
        user_profile_embeddings, null_profile_binary_mask = load_user_profile_embeddings_any(config, user_id_mapping)
        if user_profile_embeddings is not None:
            if user_profile_embeddings.dim() == 3:
                # shape = [num_users, K, emb_dim]
                profile_emb_dim = user_profile_embeddings.shape[-1]
            else:
                # shape = [num_users, emb_dim]
                profile_emb_dim = user_profile_embeddings.shape[1]

            user_profile_embeddings = user_profile_embeddings.to(device)
            null_profile_binary_mask = null_profile_binary_mask.to(device)
        else:
            user_profile_embeddings = None
            null_profile_binary_mask = None
            profile_emb_dim = 0
    else:
        user_profile_embeddings = None
        null_profile_binary_mask = None
        profile_emb_dim = 0


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

    # Цикл обучения
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        # c = 0
        for batch in tqdm(train_loader):
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            user_ids = user_ids.to(device)

            if model_name in ['SASRecLLM', 'BERT4RecLLM']:
                if user_profile_embeddings is not None:
                    user_profile_emb = user_profile_embeddings[user_ids]
                    null_profile_binary_mask_batch = null_profile_binary_mask[user_ids]
                else:
                    user_profile_emb = None
                    null_profile_binary_mask_batch = None

                outputs, hidden_for_reconstruction = model(input_seq, user_profile_emb=user_profile_emb)

                logits = outputs.view(-1, outputs.size(-1))
                targets = target_seq.view(-1)

                # лосс модели
                loss_model = criterion(logits, targets)

                # pass
                if model.use_down_scale:
                    user_profile_emb_transformed = model.profile_transform(user_profile_emb)
                else:
                    user_profile_emb_transformed = user_profile_emb.detach().clone().to(device)
                if model.use_upscale:
                    hidden_for_reconstruction = model.hidden_layer_transform(hidden_for_reconstruction)
                # print(user_profile_emb_transformed.shape, hidden_for_reconstruction.shape, null_profile_binary_mask_batch.shape)
                if (
                    user_profile_emb_transformed is not None
                    and user_profile_emb_transformed.dim() == 3
                    and hidden_for_reconstruction is not None
                    and hidden_for_reconstruction.dim() == 2
                ):
                    # user_profile_emb_transformed: [B, K, H]
                    # hidden_for_reconstruction: [B, H]
                    B, K, H = user_profile_emb_transformed.shape
                    # "Раздуваем" hidden_for_reconstruction, чтобы стало [B, K, H]
                    hidden_for_reconstruction = hidden_for_reconstruction.unsqueeze(1)  # => [B, 1, H]
                    hidden_for_reconstruction = hidden_for_reconstruction.expand(-1, K, -1)  
                    # теперь => [B, K, H]                

                user_profile_emb_transformed[null_profile_binary_mask_batch] = \
                hidden_for_reconstruction[null_profile_binary_mask_batch]

                loss_guide = 0.0
                loss_guide = criterion_reconstruct_fn(hidden_for_reconstruction, user_profile_emb_transformed)

                if scale_guide_loss:

                    loss_model_val = loss_model.item()
                    loss_guide_val = loss_guide.item()
                    eps = 1e-8

                    scale_for_guide = loss_model_val / (loss_guide_val + eps)
                    scaled_loss_guide = loss_guide * scale_for_guide
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
        
        # Сохраняем чекпоинт
        if save_checkpoints:
            checkpoint_path = os.path.join(model_dir, f'{model_name}_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

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
    multi_profile = config['model'].get('multi_profile', False)

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
            multi_profile=multi_profile,  # наш дополнительный флаг
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
            multi_profile=multi_profile,  # наш дополнительный флаг
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

    # with open(args.config, 'r', encoding='utf-8') as f:
    #     config = yaml.safe_load(f)
    #
    # train_model(config)