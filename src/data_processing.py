import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

from src.data_split_from_raw_file import DataFromRawFileGetter


class DatasetSettings:
    def __init__(self, data_dir, one_datafile_name, split_into_files, items_features, users_features,
                 interactions_features, items_filename, users_filename,
                 interactions_filename, separator, user_id='user_id', item_id='item_id', timestamp='timestamp',
                 interaction='rating', interaction_scale=(0, 1), encoding="utf-8", drop_title_row=True,
                 timestamp_format=None,
                 stratify_users=False, stratify_user_column_names=("gender", "age"), users_only_from_file=None,
                 users_from_file_id_column=None, items_only_from_file=None, items_from_file_id_column=None):
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
        self.one_datafile_name = one_datafile_name
        self.split_into_files = split_into_files
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
        self.timestamp_format = timestamp_format
        self.interaction = interaction
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
        self.interaction_scale = interaction_scale
        self.stratify_users = stratify_users
        self.stratify_user_column_names = stratify_user_column_names
        self.users_only_from_file = users_only_from_file
        self.users_from_file_id_column = users_from_file_id_column
        self.items_only_from_file = items_only_from_file
        self.items_from_file_id_column = items_from_file_id_column

    def __str__(self):
        return (f"self.data_dir {self.data_dir},\n"
                f"self.split_into_files {self.split_into_files},\n"
                f"self.one_datafile_name {self.one_datafile_name},\n"
                f"self.items_features {self.items_features},\n"
                f"self.users_features {self.users_features},\n"
                f"self.interactions_features {self.interactions_features},\n"
                f"self.items_filename {self.items_filename},\n"
                f"self.users_filename {self.users_filename},\n"
                f"self.interactions_filename {self.interactions_filename},\n"
                f"self.separator {self.separator},\n"
                f"self.encoding {self.encoding},\n"
                f"self.drop_title_row {self.drop_title_row},\n"
                f"self.user_id {self.user_id},\n"
                f"self.item_id {self.item_id},\n"
                f"self.timestamp {self.timestamp},\n"
                f"self.interaction {self.interaction},\n"
                f"self.interaction_scale {self.interaction_scale},\n"
                f"self.timestamp_format {self.timestamp_format},\n"
                f"self.stratify_users {self.stratify_users},\n"
                f"self.stratify_user_column_names {self.stratify_user_column_names},\n"
                f"self.users_only_from_file {self.users_only_from_file},\n"
                f"self.users_from_file_id_column {self.users_from_file_id_column},\n"
                f"self.items_only_from_file {self.items_only_from_file},\n"
                f"self.items_from_file_id_column {self.items_from_file_id_column}")


class DatasetLoaderML:
    """
    Класс для загрузки, обработки и сохранения данных о взаимодействиях пользователей с элементами (items)
    в контексте рекомендательных систем.

    Атрибуты:
        data_dir (str): Путь к директории с данными.
        items_features (list): Список названий колонок с данными о документах (items).
        users_features (list): Список названий колонок с данными о пользователях.
        interactions_features (list): Список названий колонок с данными о взаимодействиях.
        items_filename (str): Имя файла с данными о товарах.
        users_filename (str): Имя файла с данными о пользователях.
        interactions_filename (str): Имя файла с данными о взаимодействиях.
        separator (str): Разделитель, используемый в файлах данных.
        encoding (str): Кодировка файлов данных.
        drop_title_row (bool): Флаг для удаления строки с заголовками.
        users (pd.DataFrame): Данные о пользователях.
        items (pd.DataFrame): Данные о товарах.
        ratings (pd.DataFrame): Данные о взаимодействиях.
        user_id_column (str): Название колонки с ID пользователей.
        item_id_column (str): Название колонки с ID товаров.
        timestamp_column (str): Название колонки с временными метками.
        timestamp_format (str): Формат колонки со временем (None если numerical timestamp, иначе - формат).
        interaction_column (str): Название колонки с данными о взаимодействиях.
        interaction_scale (str): Range значений взаимодействий: от-до (включительно).
        stratify_users (bool): Флаг для стратифицированного разбиения данных пользователей.
        stratify_user_column_names (list): Колонки, используемые для стратификации пользователей.

        (остальные настройки не загружаются)
    """

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
        self.items = None
        self.ratings = None
        self.user_id_column = setting.user_id
        self.item_id_column = setting.item_id
        self.timestamp_column = setting.timestamp
        self.timestamp_format = setting.timestamp_format
        self.interaction_column = setting.interaction
        self.interaction_scale = setting.interaction_scale
        self.stratify_users = setting.stratify_users
        self.stratify_user_column_names = setting.stratify_user_column_names

    # Загрузка сырых данных
    def load_users(self, print_on_success=True):
        """
        Загружает данные о пользователях из файла.

        Args:
            print_on_success (bool): Если True, выводит сообщение об успешной загрузке данных.

        Возвращает:
            None
        """
        user_columns = self.users_features
        user_path = os.path.join(self.data_dir, self.users_filename)
        self.users = pd.read_csv(user_path, sep=self.separator, engine='python', names=user_columns)
        if self.drop_title_row:
            self.users = self.users.iloc[1:]
        if print_on_success:
            print(f'Users data loaded: {self.users.shape[0]} records')

    def load_documents(self, print_on_success=True):
        """
        Загружает данные о товарах из файла.

        Args:
            print_on_success (bool): Если True, выводит сообщение об успешной загрузке данных.

        Возвращает:
            None
        """
        movie_columns = self.items_features
        movie_path = os.path.join(self.data_dir, self.items_filename)
        self.items = pd.read_csv(movie_path, sep=self.separator, engine='python', names=movie_columns,
                                 encoding=self.encoding)
        if self.drop_title_row:
            self.items = self.items.iloc[1:]
        if print_on_success:
            print(f'Items data loaded: {self.items.shape[0]} records')

    def load_ratings(self, filter_user_actions_count=3, filter_item_actions_count=3, print_on_success=True):
        """
        Загружает данные о взаимодействиях из файла и применяет фильтры.

        Args:
            filter_user_actions_count (int): Минимальное количество взаимодействий пользователя для фильтрации.
            filter_item_actions_count (int): Минимальное количество взаимодействий товара для фильтрации.
            print_on_success (bool): Если True, выводит сообщение об успешной загрузке данных.

        Возвращает:
            None
        """
        ratings_columns = self.interactions_features
        ratings_path = os.path.join(self.data_dir, self.interactions_filename)
        self.ratings = pd.read_csv(ratings_path, sep=self.separator, engine='python', names=ratings_columns)
        if self.drop_title_row:
            self.ratings = self.ratings.iloc[1:]
        if self.timestamp_format is not None:
            self.ratings[self.timestamp_column] = pd.to_datetime(self.ratings[self.timestamp_column],
                                                                 errors='coerce').astype("int64") // 10 ** 9
        if print_on_success:
            print(f'Interactions data loaded: {self.ratings.shape[0]} records before filtering rules')
        self.filter_ratings_by_interaction_threshold(filter_user_actions_count, filter_item_actions_count)

    def load_altogether(self, filter_user_actions_count=3, filter_item_actions_count=3, print_on_success=True):
        """
        Загружает данные о пользователях, товарах и взаимодействиях одним вызовом.

        Args:
            filter_user_actions_count (int): Минимальное количество взаимодействий пользователя для фильтрации.
            filter_item_actions_count (int): Минимальное количество взаимодействий товара для фильтрации.
            print_on_success (bool): Если True, выводит сообщения об успешной загрузке данных.

        Возвращает:
            None
        """
        self.load_users(print_on_success)
        self.load_documents(print_on_success)
        self.load_ratings(filter_user_actions_count, filter_item_actions_count, print_on_success)

    def filter_ratings_by_interaction_threshold(self, threshold_users, threshold_items, print_on_success=True):
        """
        Применяет фильтрацию данных о взаимодействиях по пороговым значениям.

        Args:
            threshold_users (int): Минимальное количество взаимодействий пользователя для фильтрации.
            threshold_items (int): Минимальное количество взаимодействий товара для фильтрации.
            print_on_success (bool): Если True, выводит сообщение об успешной фильтрации данных.

        Возвращает:
            None
        """
        if threshold_items is not None and threshold_users is not None:
            user_interactions = self.ratings[self.user_id_column].value_counts()
            filtered_users = user_interactions[user_interactions > threshold_users].index
            self.users = self.users[self.users[self.user_id_column].isin(filtered_users)]
            item_interactions = self.ratings[self.item_id_column].value_counts()
            filtered_items = item_interactions[item_interactions > threshold_items].index
            self.items = self.items[self.items[self.item_id_column].isin(filtered_items)]
            self.ratings = self.ratings[self.ratings[self.user_id_column].isin(self.users[self.user_id_column])]
            self.ratings = self.ratings[self.ratings[self.item_id_column].isin(self.items[self.item_id_column])]

            if print_on_success:
                print(f'Users left after applying threshold_users of {threshold_users}: {len(self.users)}')
                print(f'Items left after applying threshold_items of {threshold_items}: {len(self.items)}')
                print(f'Interactions left after applying threshold_users '
                      f'of {threshold_users} actions per user and {threshold_items} '
                      f'actions per item: {len(self.ratings)}')

    # Сохранение данных в формат .inter
    def save_inter_files(self, train, validation, test, path_to_save, save_validation=True):
        """
        Сохраняет данные о взаимодействиях в формате `.inter`.

        Args:
            train (pd.DataFrame): Данные для обучения.
            validation (pd.DataFrame): Данные для валидации.
            test (pd.DataFrame): Данные для тестирования.
            path_to_save (str): Путь для сохранения файлов.
            save_validation (bool): Если True, сохраняет файл с валидацией.

        Возвращает:
            None
        """

        def preprocess_to_inter(df):
            df_copy = df.copy()

            df_copy[f'{self.user_id_column}:token'] = df_copy[self.user_id_column].astype(str)
            df_copy[f'{self.item_id_column}:token'] = df_copy[self.item_id_column].astype(str)

            df_copy[f'{self.interaction_column}:float'] = df_copy[self.interaction_column].astype(float)
            df_copy[f'{self.timestamp_column}:float'] = df_copy[self.timestamp_column].astype(float)
            df_copy.drop(columns=[f'{self.user_id_column}', f'{self.item_id_column}',
                                  f'{self.interaction_column}', f'{self.timestamp_column}'],
                         inplace=True)

            return df_copy

        files_pool = []
        for dataframe in ([train, validation, test] if save_validation else [train, test]):
            files_pool.append(preprocess_to_inter(dataframe))
        files_pool[0].to_csv(os.path.join(path_to_save, f'train.inter'), sep='\t', index=False)
        if save_validation:
            files_pool[1].to_csv(os.path.join(path_to_save, f'valid.inter'), sep='\t', index=False)
        files_pool[2 if save_validation else 1].to_csv(os.path.join(path_to_save, f'test.inter'), sep='\t', index=False)

    # Загрузка данных из .inter файлов
    def load_inter_data(self):
        """
        Загружает данные из файлов `.inter`.

        Returns:
            tuple: Кортеж с данными train, validation и test.
        """
        train = pd.read_csv(os.path.join(self.data_dir, 'train.inter'), sep='\t')
        valid = pd.read_csv(os.path.join(self.data_dir, 'valid.inter'), sep='\t')
        test = pd.read_csv(os.path.join(self.data_dir, 'test.inter'), sep='\t')
        return train, valid, test

    def session_split(self, boundary=0.8, validation_size=0.2,
                      path_to_save=None, save_intermediate=True):
        """
        Разбивает данные на train, validation и test подходом, рекомендованным в статье "Does it look sequential?".
        Разбиение проводится относительно

        Args:
            boundary (float): Доля данных для обучения (train).
            validation_size (float): Доля данных для валидации (validation).
            path_to_save (str): Путь для сохранения промежуточных файлов.
            save_intermediate (bool): Если True, сохраняет промежуточные файлы.

        Returns:
            tuple: Кортеж с DataFrames для train, validation и test.
        """
        data = self.ratings
        if self.timestamp_format is not None:
            data = data[pd.to_datetime(data[self.timestamp_column], errors='coerce').notna()]
        else:
            data[self.timestamp_column] = data[self.timestamp_column].astype(int)
        train, test = self.session_splitter(data, boundary)

        if validation_size is not None:
            train, validation, test = self.make_validation(train, test, validation_size)
        else:
            validation = pd.DataFrame()

        if save_intermediate and path_to_save:
            self.save_inter_files(train, validation, test, path_to_save,
                                  save_validation=(validation_size is not None and validation_size > 0))

        return train, validation, test

    def make_validation(self, train, test, validation_size):
        """
        Разбивает данные на train и validation.

        Args:
            train (pd.DataFrame): Данные для обучения.
            test (pd.DataFrame): Данные для тестирования.
            validation_size (float): Доля данных для валидации.

        Returns:
            tuple: Кортеж с DataFrames для train, validation и test.
        """
        if validation_size <= 0 or validation_size >= 1:
            raise ValueError("Validation size should be between 0 and 1 (exclusive).")
        validation_users_count = int(len(train[self.user_id_column].unique()) * validation_size)

        if self.stratify_users and self.stratify_user_column_names:
            stratify_columns = [col for col in self.stratify_user_column_names if col in train.columns]

            if stratify_columns:
                user_groups = train.groupby(self.user_id_column)[stratify_columns].first().reset_index()

                user_groups['stratify_key'] = user_groups[stratify_columns].astype(str).agg('_'.join, axis=1)

                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=validation_size, random_state=42
                )

                train_idx, validation_idx = next(splitter.split(
                    user_groups, user_groups['stratify_key']
                ))

                validation_users = user_groups.iloc[validation_idx][self.user_id_column].values
            else:
                validation_users = np.random.choice(
                    train[self.user_id_column].unique(),
                    size=validation_users_count,
                    replace=False
                )
        else:
            validation_users = np.random.choice(
                train[self.user_id_column].unique(),
                size=validation_users_count,
                replace=False
            )
        validation = train[train[self.user_id_column].isin(validation_users)]
        train = train[~train[self.user_id_column].isin(validation_users)]
        return train, validation, test

    def session_splitter(self, data, boundary):
        data.sort_values([self.user_id_column, self.timestamp_column], inplace=True)
        data.sort_values([self.user_id_column, self.timestamp_column], inplace=True)
        quant = data[self.timestamp_column].quantile(boundary)
        users_time = data.groupby(self.user_id_column)[self.timestamp_column].agg(list).apply(
            lambda x: x[0] <= quant).reset_index()
        users_time_test = data.groupby(self.user_id_column)[self.timestamp_column].agg(list).apply(
            lambda x: x[-1] > quant).reset_index()

        train_user = list(users_time[users_time[self.timestamp_column]][self.user_id_column])
        test_user = list(users_time_test[users_time_test[self.timestamp_column]][self.user_id_column])

        train = data[data[self.user_id_column].isin(train_user)]
        train = train[train[self.timestamp_column] <= quant]
        test = data[data[self.user_id_column].isin(test_user)]
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
                             f'{self.interaction_column}:float', f'{self.timestamp_column}:float'],
                    inplace=True)

        for dataframe in [train, validation, test]:
            preprocess_from_inter(dataframe)

        return self.process_split_data(train, validation, test, print_on_success)

    def preprocess_data_from_raw_dataset(self, save_intermediate=True, path_to_save=None,
                                         boundary=0.8, filter_user_actions_count=3, filter_item_actions_count=3,
                                         print_on_success=True):
        self.load_altogether(filter_user_actions_count, filter_item_actions_count, print_on_success)
        train, validation, test = self.session_split(boundary=boundary, save_intermediate=save_intermediate,
                                                     path_to_save=path_to_save)
        return self.process_split_data(train, validation, test, print_on_success)

    # Метод для сохранения обработанных данных
    def save_processed_data(self, data, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)


def load_settings_from_file(settings_filepath):
    """
    Загружает настройки из указанного файла.

    Args:
        settings_filepath (str): Путь к файлу с настройками.

    Returns:
        DatasetSettings: Настройки в виде экземпляра класса DatasetSettings.
    """
    with open(settings_filepath, 'r') as f:
        data_dir = f.readline().strip()
        split_into_files = f.readline().strip().lower() == "true"
        one_datafile_name = f.readline().strip()
        one_datafile_name = None if one_datafile_name.lower() == "none" else one_datafile_name
        items_features = f.readline().strip().split(', ')
        users_features = f.readline().strip().split(', ')
        interactions_features = f.readline().strip().split(', ')
        items_filename = f.readline().strip()
        users_filename = f.readline().strip()
        interactions_filename = f.readline().strip()
        separator = f.readline().strip()
        encoding = f.readline().strip()
        drop_title_row = f.readline().strip().lower() == "true"
        item_id = f.readline().strip()
        user_id = f.readline().strip()
        timestamp = f.readline().strip()
        timestamp_format = f.readline().strip()
        timestamp_format = None if timestamp_format.lower() == "none" else timestamp_format
        interaction = f.readline().strip()
        interaction_scale = tuple(map(float, f.readline().strip().split(', ')))
        stratify_users = f.readline().strip().lower() == "true"
        stratify_user_column_names = f.readline().strip()
        users_only_from_file = f.readline().strip()
        users_only_from_file = None if users_only_from_file.lower() == "none" else users_only_from_file
        users_from_file_id_column = f.readline().strip()
        users_from_file_id_column = None if users_from_file_id_column.lower() == "none" else users_from_file_id_column
        items_only_from_file = f.readline().strip()
        items_only_from_file = None if items_only_from_file.lower() == "none" else items_only_from_file
        items_from_file_id_column = f.readline().strip()
        items_from_file_id_column = None if items_from_file_id_column.lower() == "none" else items_from_file_id_column
        return DatasetSettings(data_dir=data_dir,
                               split_into_files=split_into_files,
                               one_datafile_name=one_datafile_name,
                               items_features=items_features,
                               users_features=users_features,
                               interactions_features=interactions_features,
                               items_filename=items_filename,
                               users_filename=users_filename,
                               interactions_filename=interactions_filename,
                               separator=separator,
                               encoding=encoding,
                               drop_title_row=drop_title_row,
                               item_id=item_id,
                               user_id=user_id,
                               timestamp=timestamp,
                               timestamp_format=timestamp_format,
                               interaction=interaction,
                               interaction_scale=interaction_scale,
                               stratify_users=stratify_users,
                               stratify_user_column_names=stratify_user_column_names,
                               users_only_from_file=users_only_from_file,
                               users_from_file_id_column=users_from_file_id_column,
                               items_only_from_file=items_only_from_file,
                               items_from_file_id_column=items_from_file_id_column
                               )


def _load_raw_dataset_and_save_as_pickle(settings, path_to_save, filter_user_actions_count=None,
                                         filter_item_actions_count=None, inter_path_to_save=None):
    """
    Загружает данные из сырых файлов данных (users, items, interactions без разделения)
    и сохраняет их в память в виде pickle-файлов.

    Args:
        settings (DatasetSettings): Настройки загрузки и обработки данных.
        path_to_save (str): Путь к директории, куда будут сохранены pickle-файлы.
        filter_user_actions_count (int): Минимальное число взаимодействий пользователя для его сохранения.
        filter_item_actions_count (int): Минимальное число взаимодействий товара для его сохранения.
        inter_path_to_save (str): Путь к директории, куда будут сохранены промежуточные inter-файлы
            (для последующей загрузки разделения из inter-файлов).
    """
    loader = DatasetLoaderML(settings)
    save = inter_path_to_save is not None
    ((train_sequences,
      valid_sequences,
      test_sequences),
     mappings, counts) = loader.preprocess_data_from_raw_dataset(save_intermediate=save,
                                                                 path_to_save=inter_path_to_save,
                                                                 filter_user_actions_count=filter_user_actions_count,
                                                                 filter_item_actions_count=filter_item_actions_count)
    loader.save_processed_data(train_sequences, os.path.join(path_to_save, 'train_sequences.pkl'))
    loader.save_processed_data(valid_sequences, os.path.join(path_to_save, 'valid_sequences.pkl'))
    loader.save_processed_data(test_sequences, os.path.join(path_to_save, 'test_sequences.pkl'))
    loader.save_processed_data(mappings, os.path.join(path_to_save, 'mappings.pkl'))
    loader.save_processed_data(counts, os.path.join(path_to_save, 'counts.pkl'))


def _load_inter_files_dataset_and_save_as_pickle(settings, path_to_save):
    """
    Загружает данные из интерфайлов и сохраняет их в память в виде pickle-файлов.

    Args:
        settings (DatasetSettings): Настройки загрузки и обработки данных.
        path_to_save (str): Путь для сохранения обработанных данных.
    """
    loader = DatasetLoaderML(settings)
    (train_sequences,
     valid_sequences,
     test_sequences), mappings, counts = loader.preprocess_data_from_inter()
    loader.save_processed_data(train_sequences, os.path.join(path_to_save, 'train_sequences.pkl'))
    loader.save_processed_data(valid_sequences, os.path.join(path_to_save, 'valid_sequences.pkl'))
    loader.save_processed_data(test_sequences, os.path.join(path_to_save, 'test_sequences.pkl'))
    loader.save_processed_data(mappings, os.path.join(path_to_save, 'mappings.pkl'))
    loader.save_processed_data(counts, os.path.join(path_to_save, 'counts.pkl'))


def load_raw_dataset_with_settings_file(settings_filepath, path_to_save,
                                        filter_user_actions_count_min=None,
                                        filter_user_actions_count_max=None,
                                        filter_item_actions_count_min=None,
                                        filter_item_actions_count_max=None,
                                        inter_path_to_save=None):
    """
    Загружает данные из файла настроек и загружает их в память. Подходит для запуска в случае, если настройки хранятся
    в виде файла (в той же папке, где и данные).
    Проводит разделение данных, если они находятся в одном файле, и фильтрует их по границам взаимодействий, чтобы
    оставить только полезные сущности и уменьшить размер данных.

    Args:
        settings_filepath (str): название файла, в котором хранятся настройки датасета
        path_to_save (str): путь куда сохранять данные
        filter_user_actions_count_min (int, optional): минимальное число взаимодействий для пользователя.
            None, если не требуется использовать данную границу
        filter_user_actions_count_max (int, optional): максимальное число взаимодействий для пользователя.
            None, если не требуется использовать данную границу
        filter_item_actions_count_min (int, optional): минимальное число взаимодействий для предмета.
            None, если не требуется использовать данную границу
        filter_item_actions_count_max (int, optional): максимальное число взаимодействий для предмета.
            None, если не требуется использовать данную границу
        inter_path_to_save (str, optional): путь куда сохранять внутренние данные для повторного использования.
    """
    settings = load_settings_from_file(settings_filepath)
    raw_loader = DataFromRawFileGetter(settings)
    raw_loader.load(True, filter_user_actions_count_min, filter_user_actions_count_max,
                    filter_item_actions_count_min, filter_item_actions_count_max)
    _load_raw_dataset_and_save_as_pickle(settings, path_to_save, inter_path_to_save=inter_path_to_save)


def load_raw_dataset(settings, path_to_save,
                     filter_user_actions_count_min=None,
                     filter_user_actions_count_max=None,
                     filter_item_actions_count_min=None,
                     filter_item_actions_count_max=None,
                     inter_path_to_save=None):
    """
    Загружает данные из уже загруженных в память настроек. Подходит для запуска сразу после ручного ввода настроек.
    Проводит разделение данных, если они находятся в одном файле, и фильтрует их по границам взаимодействий, чтобы
    оставить только полезные сущности и уменьшить размер данных.

    Args:
        settings (DatasetSettings): настройки датасета, уже в виде экземпляра класса
        path_to_save (str): путь куда сохранять данные
        filter_user_actions_count_min (int, optional): минимальное число взаимодействий для пользователя.
            None, если не требуется использовать данную границу
        filter_user_actions_count_max (int, optional): максимальное число взаимодействий для пользователя.
            None, если не требуется использовать данную границу
        filter_item_actions_count_min (int, optional): минимальное число взаимодействий для предмета.
            None, если не требуется использовать данную границу
        filter_item_actions_count_max (int, optional): максимальное число взаимодействий для предмета.
            None, если не требуется использовать данную границу
        inter_path_to_save (str, optional): путь куда сохранять внутренние данные для повторного использования.
    """
    raw_loader = DataFromRawFileGetter(settings)
    raw_loader.load(True, filter_user_actions_count_min, filter_user_actions_count_max,
                    filter_item_actions_count_min, filter_item_actions_count_max)
    _load_raw_dataset_and_save_as_pickle(settings, path_to_save, inter_path_to_save=inter_path_to_save)


def load_inter_files_dataset(settings_filepath, path_to_save):
    """
    Загружает данные из .inter файлов. Использовалась в 1-м MVP загрузки данных для модели - с ручным разделением

    Args:
        settings_filepath (str): путь к файлу настроек
        path_to_save (str): путь куда сохранять данные
    """
    settings = load_settings_from_file(settings_filepath)
    _load_inter_files_dataset_and_save_as_pickle(settings, path_to_save)


def old_load_dataset(path_to_save="../data/processed"):
    """
    Код загрузки данных для первого MVP - создаёт искусственные настройки датасета, в которых указаны только нужные
    колонки и значения. Функция для воспроизведения загрузки данных, разделённых вручную

    Args:
        path_to_save (str): путь куда сохранять данные
    """
    settings = DatasetSettings(data_dir="../data/raw",
                               split_into_files=False,
                               one_datafile_name=None,
                               items_features=['item_id'],
                               users_features=['user_id'],
                               interactions_features=['item_id', 'user_id', 'timestamp', 'rating'],
                               items_filename=None,
                               users_filename=None,
                               interactions_filename=None,
                               separator='',
                               encoding='',
                               drop_title_row=False,
                               item_id='item_id',
                               user_id='user_id',
                               timestamp='timestamp',
                               interaction='rating',
                               interaction_scale=(1, 5),
                               timestamp_format=None,
                               stratify_users=False,
                               stratify_user_column_names=[],
                               users_only_from_file=None,
                               users_from_file_id_column=None,
                               items_only_from_file=None,
                               items_from_file_id_column=None)
    _load_inter_files_dataset_and_save_as_pickle(settings, path_to_save)


if __name__ == "__main__":
    old_load_dataset()