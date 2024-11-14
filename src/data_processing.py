import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

class DatasetSettings:
    def __init__(self, data_dir, items_features, users_features, interactions_features, items_filename, users_filename,
                 interactions_filename, separator, user_id='user_id', item_id='item_id', timestamp='timestamp',
                 interaction='rating', interaction_scale=(0, 1), encoding="utf-8", drop_title_row=True,
                 timestamp_format=None,
                 stratify_users=False, stratify_user_column_names=("gender", "age")):
        """
        Класс для указания конфига (настроек) датасета для предобработки. Подходит для датасетов под задачи
        рекомендательных систем.

        Parameters:
        data_dir (str): Путь к директории с данными.
        items_features (list): Названия признаков для документов.
        users_features (list): Названия признаков для пользователей.
        interactions_features (list): Названия признаков для взаимодействий пользователей с документами.
        items_filename (str): Имя файла с фильмами.
        users_filename (str): Имя файла с пользователями.
        interactions_filename (str): Имя файла со взаимодействиями.
        separator (str): Символ разделителя в файле.
        user_id (str): Название колонки с идентификаторами пользователей.
        item_id (str): Название колонки с идентификаторами документов.
        timestamp (str): Название колонки с датой и временем взаимодействий.
        interaction (str): Название колонки с оценками или иным целевым видом взаимодействий.
        interaction_scale (list): Масштабирование оценок взаимодействий
            (коллекция из двух элементов: начало и конец диапазона, включительно). По умолчанию: (0, 1).
        encoding (str): Кодировка файла. По умолчанию: utf-8.
        drop_title_row (bool): Удалять ли заголовок в файле. По умолчанию: True.
        timestamp_format (str): Формат даты и времени в файле для преобразования их в численный временной штамп.
            Если не указан, то в датасете по умолчанию численный временной штамп.
        stratify_users (bool): Делить ли пользователей с сохранением баланса какого-то множества признаков?
            По умолчанию: False
        stratify_user_column_names (list): Список названий столбцов, с сохранением баланса в которых необходимо
            разделить пользователей, если stratify_users указан как True. По умолчанию: ["gender", "age"]
        """

        self.data_dir = data_dir
        if self.item_id not in self.items_features or self.user_id not in self.users_features:
            raise ValueError(f"Invalid item_id or user_id in the dataset settings. "
                             f"item_id: {self.item_id}, user_id: {self.user_id}, "
                             f"items_features: {self.items_features}, users_features: {self.users_features}")
        if item_id not in self.interactions_features or user_id not in interactions_features or (self.timestamp not in
            self.interactions_features):
            raise ValueError(f"Invalid item_id or user_id or timestamp in the dataset settings. "
                             f"Not found in interaction features. "
                             f"item_id: {self.item_id}, user_id: {self.user_id}, timestamp: {self.timestamp}, "
                             f"interactions_features: {self.interactions_features}")
        self.items_features = items_features
        self.users_features = users_features
        self.interactions_features = interactions_features
        self.items_filename = items_filename
        self.users_filename = users_filename
        self.interactions_filename = interactions_filename
        self.separator = separator
        self.encoding = encoding
        self.drop_title_row = drop_title_row
        self.user_id = user_id
        self.item_id = item_id
        self.timestamp = timestamp
        self.interaction = interaction
        self.interaction_scale = interaction_scale
        self.timestamp_format = timestamp_format
        self.stratify_users = stratify_users
        self.stratify_user_column_names = stratify_user_column_names


class DatasetLoaderML:
    def __init__(self, setting: DatasetSettings):
        self.data_dir = setting.data_dir
        self.items_features = setting.items_features
        self.users_features = setting.users_features
        self.interactions_features = setting.interactions_features
        self.items_filename = setting.items_filename
        self.users_filename = setting.users_filename
        self.interactions_filename = setting.interactions_filename
        self.separator = setting.separator
        self.encoding = setting.encoding
        self.drop_title_row = setting.drop_title_row
        self.users = None
        self.movies = None
        self.ratings = None
        self.user_id_column = setting.user_id
        self.item_id_column = setting.item_id
        self.timestamp_column = setting.timestamp
        self.interaction_column = setting.interaction
        self.interaction_scale = setting.interaction_scale
        self.stratify_users = setting.stratify_users
        self.stratify_user_column_names = setting.stratify_user_column_names

    # Загрузка сырых данных
    def load_users(self, print_on_success=True):
        user_columns = self.users_features
        user_path = os.path.join(self.data_dir, self.users_filename)
        self.users = pd.read_csv(user_path, sep=self.separator, engine='python', names=user_columns)
        if self.drop_title_row:
            self.users = self.users.iloc[1:]
        if print_on_success:
            print(f'Users data loaded: {self.users.shape[0]} records')

    def load_documents(self, print_on_success=True):
        movie_columns = self.items_features
        movie_path = os.path.join(self.data_dir, self.items_filename)
        self.movies = pd.read_csv(movie_path, sep=self.separator, engine='python', names=movie_columns, encoding=self.encoding)
        if self.drop_title_row:
            self.movies = self.movies.iloc[1:]
        if print_on_success:
            print(f'Movies data loaded: {self.movies.shape[0]} records')

    def load_ratings(self, print_on_success=True):
        ratings_columns = self.interactions_features
        ratings_path = os.path.join(self.data_dir, self.interactions_filename)
        self.ratings = pd.read_csv(ratings_path, sep=self.separator, engine='python', names=ratings_columns)
        if self.drop_title_row:
            self.ratings = self.ratings.iloc[1:]
        if print_on_success:
            print(f'Ratings data loaded: {self.ratings.shape[0]} records')

    def load_altogether(self, print_on_success=True):
        self.load_users(print_on_success)
        self.load_documents(print_on_success)
        self.load_ratings(print_on_success)

    # Сохранение данных в формат .inter
    def save_inter_files(self, train, validation, test, path_to_save):
        def preprocess_to_inter(df):
            # Создаем копию DataFrame для безопасности
            df_copy = df.copy()

            # Преобразуем user_id и item_id обратно в строковые колонки
            df_copy[f'{self.user_id_column}:token'] = df_copy[self.user_id_column].astype(str)
            df_copy[f'{self.item_id_column}:token'] = df_copy[self.item_id_column].astype(str)

            # Преобразуем interaction и timestamp обратно в float
            df_copy[f'{self.interaction_column}:float'] = df_copy[self.interaction_column].astype(float)
            df_copy[f'{self.timestamp_column}timestamp:float'] = df_copy[self.timestamp_column].astype(float)

            # Добавляем оригинальные колонки обратно в DataFrame
            df_copy[f'{self.user_id_column}:token'] = df_copy[f'{self.user_id_column}:token']
            df_copy[f'{self.item_id_column}:token'] = df_copy[f'{self.item_id_column}:token']
            df_copy[f'{self.interaction_column}:float'] = df_copy[f'{self.interaction_column}:float']
            df_copy[f'{self.timestamp_column}timestamp:float'] = df_copy[f'{self.timestamp_column}timestamp:float']

            return df_copy
        files_pool = []
        for dataframe in [train, validation, test]:
            files_pool.append(preprocess_to_inter(dataframe))
        files_pool[0].to_csv(os.path.join(path_to_save, f'train.inter'), sep='\t', index=False)
        files_pool[1].to_csv(os.path.join(path_to_save, f'valid.inter'), sep='\t', index=False)
        files_pool[2].to_csv(os.path.join(path_to_save, f'test.inter'), sep='\t', index=False)

    # Загрузка данных из .inter файлов
    def load_inter_data(self):
        train = pd.read_csv(os.path.join(self.data_dir, 'train.inter'), sep='\t')
        valid = pd.read_csv(os.path.join(self.data_dir, 'valid.inter'), sep='\t')
        test = pd.read_csv(os.path.join(self.data_dir, 'test.inter'), sep='\t')
        return train, valid, test

    # Основной метод разбиения на train/test/validation
    def session_split(self, boundary=0.8, validation_size=None,
                      path_to_save=None, save_intermediate=True):
        data = self.ratings
        data = data[pd.to_datetime(data[self.timestamp_column], errors='coerce').notna()]
        train, test = self.session_splitter(data, boundary)

        if validation_size is not None:
            train, validation, test = self.make_validation(train, test, validation_size)
        else:
            validation = pd.DataFrame()

        if save_intermediate and path_to_save:
            self.save_inter_files(train, validation, test, path_to_save)

        return train, validation, test

    def make_validation(self, train, test, validation_size):
        """
        Разбивает данные на train, validation
        :param train: DataFrame с данными для train
        :param test: DataFrame с данными для test
        :param validation_size: размер validation-датасета в виде доли единицы
        :return: train, validation, test DataFrames
        """
        if validation_size <= 0 or validation_size >= 1:
            raise ValueError("Validation size should be between 0 and 1 not including")
        validation_users_count = len(train[self.user_id_column].unique()) * validation_size
        validation_users = np.random.choice(train[self.user_id_column].unique(), size=validation_users_count, replace=False)
        validation = train[train[self.user_id_column].isin(validation_users)]
        train = train[~train[self.user_id_column].isin(validation_users)]
        return train, validation, test

    def session_splitter(self, data, boundary):
        data.sort_values([self.user_id_column, self.timestamp_column], inplace=True)
        quant = (pd.to_datetime(data[self.timestamp_column],
                                errors='coerce').astype("int64") // 10**9).quantile(boundary)
        train = data[(pd.to_datetime(data[self.timestamp_column], errors='coerce').astype("int64") // 10**9) <= quant]
        test = data[(pd.to_datetime(data[self.timestamp_column], errors='coerce').astype("int64") // 10**9) > quant]
        return train, test

    def process_split_data(self, train, validation, test, print_on_success=True):
        all_data = pd.concat([train, validation, test])
        user_id_mapping = {id_: idx for idx, id_ in enumerate(all_data[self.user_id_column].unique())}
        item_id_mapping = {id_: idx for idx, id_ in enumerate(all_data[self.item_id_column].unique())}
        num_users, num_items = len(user_id_mapping), len(item_id_mapping)

        def apply_mapping(df):
            df[self.user_id_column] = df[self.user_id_column].map(user_id_mapping)
            df[self.item_id_column] = df[self.item_id_column].map(item_id_mapping)

        for dataframe in [train, validation, test]:
            apply_mapping(dataframe)

        def create_sequences(df):
            return df.sort_values([self.user_id_column, self.timestamp_column]).groupby(self.user_id_column)[
                self.item_id_column].apply(list).to_dict()

        train_sequences = create_sequences(train)
        valid_sequences = create_sequences(validation)
        test_sequences = create_sequences(test)

        if print_on_success:
            print(f'Sequences created, numbers of: \n\tusers: {num_users}\n\titems: {num_items}')

        return ((train_sequences, valid_sequences, test_sequences), (user_id_mapping, item_id_mapping),
                (num_users, num_items))

    def preprocess_data_from_inter(self, print_on_success=True):
        train, validation, test = self.load_inter_data()

        def preprocess_from_inter(df):
            df[self.user_id_column] = df[f'{self.user_id_column}:token'].astype(int)
            df[self.item_id_column] = df[f'{self.item_id_column}:token'].astype(int)
            df[self.interaction_column] = df[f'{self.interaction_column}:float'].astype(float)
            df[self.timestamp_column] = df[f'{self.timestamp_column}:float'].astype(int)
            df.drop(columns=[f'{self.user_id_column}:token', f'{self.item_id_column}:token',
                             f'{self.interaction_column}:float', f'{self.timestamp_column}timestamp:float'],
                    inplace=True)

        for dataframe in [train, validation, test]:
            preprocess_from_inter(dataframe)

        return self.process_split_data(train, validation, test, print_on_success)


    def preprocess_data_from_raw_dataset(self, save_intermediate=True, path_to_save=None,
                                         boundary=0.8, print_on_success=True):
        self.load_altogether(print_on_success)
        train, validation, test = self.session_split(boundary=boundary, save_intermediate=save_intermediate,
                                                     path_to_save=path_to_save)
        return self.process_split_data(train, validation, test, print_on_success)

    # Метод для сохранения обработанных данных
    def save_processed_data(self, data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


def load_settings_from_file(settings_filepath):
    with open(settings_filepath, 'rb') as f:
        return DatasetSettings(*f.readlines())


def load_raw_dataset_and_save_as_pickle(settings, path_to_save, inter_path_to_save=None):
    loader = DatasetLoaderML(settings)
    save = inter_path_to_save is not None
    (train_sequences,
     valid_sequences,
     test_sequences), mappings, counts = loader.preprocess_data_from_raw_dataset(save_intermediate=save,
                                                                                 path_to_save=inter_path_to_save)
    loader.save_processed_data(train_sequences, os.path.join(path_to_save, 'train_sequences.pkl'))
    loader.save_processed_data(valid_sequences, os.path.join(path_to_save, 'valid_sequences.pkl'))
    loader.save_processed_data(test_sequences, os.path.join(path_to_save, 'test_sequences.pkl'))
    loader.save_processed_data(mappings, os.path.join(path_to_save, 'mappings.pkl'))
    loader.save_processed_data(counts, os.path.join(path_to_save, 'counts.pkl'))