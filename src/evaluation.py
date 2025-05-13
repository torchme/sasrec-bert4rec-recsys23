# src/evaluation.py

import torch
import numpy as np

def evaluate_model(model, data_loader, device, mode='validation', k_list=[5, 10, 20]):
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

    with torch.no_grad():
        for batch in data_loader:
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Прямой проход
            outputs = model(input_seq)

            # Если модель возвращает кортеж, извлекаем первый элемент
            if isinstance(outputs, tuple):
                outputs = outputs[0]

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
                ndcgs[k] += ndcg_k

    # Вычисляем средние значения метрик по всем батчам
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = (recalls[k] / len(data_loader.dataset))
        metrics[f'NDCG@{k}'] = (ndcgs[k] / len(data_loader.dataset))

    return metrics
