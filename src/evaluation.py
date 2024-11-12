# src/evaluation.py

import torch
import numpy as np

def evaluate_model(model, data_loader, device, k_list=[5, 10, 20]):
    model.eval()
    recalls = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}

    with torch.no_grad():
        for batch in data_loader:
            input_seq, target_seq, user_ids = batch
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # Прямой проход
            outputs = model(input_seq)

            # Проверяем, если модель возвращает кортеж, извлекаем первый элемент
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Получаем предсказания для последнего элемента в последовательности
            logits = outputs[:, -1, :]  # [batch_size, item_num + 1]
            targets = target_seq[:, -1]  # [batch_size]

            # Маскируем паддинги
            logits[:, 0] = -np.inf  # Предполагаем, что индекс 0 используется для паддинга

            # Вычисляем топ-K предсказания
            _, indices = torch.topk(logits, max(k_list))  # [batch_size, max(k_list)]

            for k in k_list:
                preds_k = indices[:, :k]  # [batch_size, k]
                correct = preds_k.eq(targets.view(-1, 1))  # [batch_size, k]

                recall_k = correct.any(dim=1).float().mean().item()

                # Вычисление NDCG
                ranks = torch.where(correct)[1] + 1  # +1 для перехода от индекса к рангу
                ndcg_k = (1 / torch.log2(ranks.float() + 1)).mean().item() if ranks.numel() > 0 else 0.0

                recalls[k].append(recall_k)
                ndcgs[k].append(ndcg_k)

    # Вычисляем средние значения метрик
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = np.mean(recalls[k])
        metrics[f'NDCG@{k}'] = np.mean(ndcgs[k])

    return metrics
