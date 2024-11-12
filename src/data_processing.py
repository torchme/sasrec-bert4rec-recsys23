# src/data_processing.py

import os
import pandas as pd
import pickle

def load_raw_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, 'train.inter'), sep='\t')
    valid = pd.read_csv(os.path.join(data_dir, 'valid.inter'), sep='\t')
    test = pd.read_csv(os.path.join(data_dir, 'test.inter'), sep='\t')
    return train, valid, test

def process_data(train, valid, test):
    # Приведение типов данных
    for df in [train, valid, test]:
        df['user_id'] = df['user_id:token'].astype(int)
        df['item_id'] = df['item_id:token'].astype(int)
        df['rating'] = df['rating:float'].astype(float)
        df['timestamp'] = df['timestamp:float'].astype(int)
    
        # Удаляем исходные столбцы
        df.drop(columns=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'], inplace=True)

    # Объединяем все данные для построения словарей пользователей и товаров
    all_data = pd.concat([train, valid, test])

    # Создаем словари для преобразования user_id и item_id в последовательные индексы
    user_id_mapping = {id: idx for idx, id in enumerate(all_data['user_id'].unique())}
    item_id_mapping = {id: idx for idx, id in enumerate(all_data['item_id'].unique())}

    num_users = len(user_id_mapping)
    num_items = len(item_id_mapping)

    # Применяем маппинг к данным
    for df in [train, valid, test]:
        df['user_id'] = df['user_id'].map(user_id_mapping)
        df['item_id'] = df['item_id'].map(item_id_mapping)

    # Создаем последовательности взаимодействий для каждого пользователя
    def create_sequences(df):
        df = df.sort_values(['user_id', 'timestamp'])
        user_sequences = df.groupby('user_id')['item_id'].apply(list).to_dict()
        return user_sequences

    train_sequences = create_sequences(train)
    valid_sequences = create_sequences(valid)
    test_sequences = create_sequences(test)

    # Возвращаем обработанные данные и маппинги
    return (train_sequences, valid_sequences, test_sequences), (user_id_mapping, item_id_mapping), (num_users, num_items)

def save_processed_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    data_dir = 'data/raw'
    processed_dir = 'data/processed'

    train, valid, test = load_raw_data(data_dir)
    (train_sequences, valid_sequences, test_sequences), mappings, counts = process_data(train, valid, test)

    # Сохраняем последовательности и маппинги
    save_processed_data(train_sequences, os.path.join(processed_dir, 'train_sequences.pkl'))
    save_processed_data(valid_sequences, os.path.join(processed_dir, 'valid_sequences.pkl'))
    save_processed_data(test_sequences, os.path.join(processed_dir, 'test_sequences.pkl'))
    save_processed_data(mappings, os.path.join(processed_dir, 'mappings.pkl'))
    save_processed_data(counts, os.path.join(processed_dir, 'counts.pkl'))

if __name__ == '__main__':
    main()
