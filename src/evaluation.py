# src/evaluation.py

import torch
import numpy as np

from src.utils import calculate_guide_loss, calculate_recsys_loss


def evaluate_model(model, data_loader, device, mode='validation',
                   model_criterion=None, criterion_reconstruct_fn=None, user_profile_embeddings=None,
                   null_profile_binary_mask=None, k_list=[5, 10, 20]):
    """
    Оценивает модель на заданном наборе данных.

    Args:
        model (nn.Module): Обученная модель.
        data_loader (DataLoader): DataLoader для набора данных.
        device (torch.device): Устройство для вычислений.
        mode (str): Режим оценки - 'validation' или 'test'.
        k_list (list): Список значений k для метрик.

    Returns:
        dict: Словарь со средними значениями метрик для каждого k.
    """
    model.eval()
    recalls = {k: 0 for k in k_list}
    ndcgs = {k: 0 for k in k_list}
    losses = {'loss_recsys': [], 'loss_guide': []}

    with torch.no_grad():
        # c = 0
        for batch in data_loader:
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            # Получаем эмбеддинги профиля пользователя, если они существуют
            if user_profile_embeddings is not None:
                user_profile_emb = user_profile_embeddings[user_ids].to(device)
                null_profile_binary_mask_batch = null_profile_binary_mask[user_ids].to(device)
            else:
                user_profile_emb = None

            # Прямой проход
            outputs = model(input_seq)
            hidden_for_reconstruction = None

            # Если модель возвращает кортеж, извлекаем первый элемент
            if isinstance(outputs, tuple):
                outputs, hidden_for_reconstruction = outputs

            if model_criterion is not None:
                loss_model = calculate_recsys_loss(target_seq, outputs, model_criterion)
                losses['loss_recsys'].append(loss_model.item())
            if criterion_reconstruct_fn is not None:
                loss_guide = calculate_guide_loss(model, user_profile_emb, hidden_for_reconstruction,
                                 null_profile_binary_mask_batch, criterion_reconstruct_fn)
                losses['loss_guide'].append(loss_guide.item())
                # print(loss_guide.item())

            # Получаем предсказания для последнего элемента в последовательности
            logits = outputs[:, -1, :]  # [batch_size, item_num + 1]

            if mode in ['test', 'validation']:
                # В режиме тестирования рассматриваем только один позитивный элемент
                targets = target_seq[:, -1]  # [batch_size]
                # Маскируем паддинги, предполагая, что индекс 0 используется для паддинга
                logits[:, 0] = -np.inf
            else:
                # В режиме валидации можно использовать другой подход, например, несколько позитивных элементов
                # Здесь предполагается, что target_seq содержит только один позитивный элемент
                targets = target_seq[:, -1]
                logits[:, 0] = -np.inf

            # Получаем топ-K предсказаний
            _, indices = torch.topk(logits, max(k_list), dim=1)  # [batch_size, max(k_list)]

            for k in k_list:
                preds_k = indices[:, :k]  # [batch_size, k]
                
                # Проверяем, есть ли целевой элемент среди топ-K предсказаний
                correct = preds_k.eq(targets.view(-1, 1))  # [batch_size, k]

                # Вычисляем Recall@k
                correct_any = correct.any(dim=1).float()
                recalls[k] += correct_any.sum().item()

                # Вычисляем NDCG@k
                # Находим позиции (ранги) целевого элемента в топ-K
                ranks = torch.where(correct)[1] + 1  # +1 для перехода от индекса к рангу

                # Если целевой элемент найден в топ-K, рассчитываем NDCG, иначе 0
                ndcg_k = (1 / torch.log2(ranks.float() + 1)).sum().item() if ranks.numel() > 0 else 0.0

                # Добавляем метрики для текущего k
                # recalls[k] += (recall_k)
                ndcgs[k] += ndcg_k
            # c += 1
            # if c == 3:
            #     break
    # Вычисляем средние значения метрик по всем батчам
    metrics = {}
    for k in k_list:
        # metrics[f'Recall@{k}'] = np.mean(recalls[k])
        # metrics[f'NDCG@{k}'] = np.mean(ndcgs[k])
        # metrics[f'fIsIn{k}'] = np.mean(is_in_k[k])
        metrics[f'Recall@{k}'] = (recalls[k] / len(data_loader.dataset))
        metrics[f'NDCG@{k}'] = (ndcgs[k] / len(data_loader.dataset))
        # metrics[f'fIsIn{k}'] = (is_in_k[k]/len(data_loader))
    for loss_name in losses:
        if len(losses[loss_name]) > 0:
            metrics[loss_name] = np.mean(losses[loss_name])
    return metrics


def evaluate_bert4rec_model(model, data_loader, device, mode='validation',
                            model_criterion=None, criterion_reconstruct_fn=None, user_profile_embeddings=None,
                            null_profile_binary_mask=None, k_list=[5, 10, 20]):
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
                preds_k = indices[:, :k]  # [batch_size, k]

                # Проверяем, есть ли целевой элемент среди топ-K предсказаний
                correct = preds_k.eq(targets.view(-1, 1))  # [batch_size, k]

                # Вычисляем Recall@k
                correct_any = correct.any(dim=1).float()
                recalls[k] += correct_any.sum().item()

                # Вычисляем NDCG@k
                # Находим позиции (ранги) целевого элемента в топ-K
                ranks = torch.where(correct)[1] + 1  # +1 для перехода от индекса к рангу

                # Если целевой элемент найден в топ-K, рассчитываем NDCG, иначе 0
                ndcg_k = (1 / torch.log2(ranks.float() + 1)).sum().item() if ranks.numel() > 0 else 0.0

                # Добавляем метрики для текущего k
                # recalls[k] += (recall_k)
                ndcgs[k] += ndcg_k

    # Вычисляем средние значения метрик
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = recalls[k] / len(data_loader.dataset)
        metrics[f'NDCG@{k}'] = ndcgs[k] / len(data_loader.dataset)

    return metrics