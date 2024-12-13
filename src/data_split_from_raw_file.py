import ast
import os

import polars as pl
import json

import pandas as pd


class RawFileSettings:
    pass
    # TODO change all setting files for all the datasets+
    # TODO manual of how settings file is constructed and how to use CLI


class DataFromRawFileGetter:
    """
    Класс для обработки и фильтрации данных из файла, их разбиения на отдельные части
    (пользователи, товары, взаимодействия) и сохранения.

    Атрибуты:
        setting: объект с настройками для обработки данных.
    """
    def __init__(self, setting):
        """
        Инициализация экземпляра класса.

        Аргументы:
            setting: объект с настройками, содержащий пути к файлам, форматы,
                     параметры фильтрации и другие опции.
        """
        self.data_dir = setting.data_dir
        self.split_into_files = setting.split_into_files  # flag if we need to split
        self.one_datafile_name = setting.one_datafile_name  # filename if we need to split
        self.items_features = setting.items_features
        self.users_features = setting.users_features
        self.interactions_features = setting.interactions_features
        self.items_filename = setting.items_filename
        self.users_filename = setting.users_filename
        self.interactions_filename = setting.interactions_filename
        self.separator = setting.separator
        self.encoding = setting.encoding
        self.drop_title_row = setting.drop_title_row
        self.user_id_column = setting.user_id
        self.item_id_column = setting.item_id
        self.timestamp_column = setting.timestamp
        self.timestamp_format = setting.timestamp_format
        self.interaction_column = setting.interaction
        self.interaction_scale = setting.interaction_scale
        self.stratify_users = setting.stratify_users
        self.stratify_user_column_names = setting.stratify_user_column_names
        ################################################################
        self.users_only_from_file = setting.users_only_from_file
        self.users_from_file_id_column = setting.users_from_file_id_column
        self.items_only_from_file = setting.items_only_from_file
        self.items_from_file_id_column = setting.items_from_file_id_column
        ################################################################
        self.data = None
        self.users = None
        self.items = None
        self.interactions = None

    def filter_by_interactions_count(self, items_interactions_min, items_interactions_max, user_interactions_min,
                                     user_interactions_max):
        """
        Фильтрация данных по количеству взаимодействий для пользователей и товаров.

        Аргументы:
            items_interactions_min: минимальное количество взаимодействий для товаров.
            items_interactions_max: максимальное количество взаимодействий для товаров.
            user_interactions_min: минимальное количество взаимодействий для пользователей.
            user_interactions_max: максимальное количество взаимодействий для пользователей.

        Исключения:
            ValueError: если после фильтрации данных не осталось.
        """
        print("Starting filter_by_interactions_count...")
        print(f"Initial data size: {self.data.shape}")

        user_counts = self.data.group_by(self.user_id_column).agg(
            pl.count().alias('user_interaction_count')
        )
        print(f"User counts size: {user_counts.shape}")

        if user_interactions_max is not None and \
                user_interactions_min is not None and user_interactions_max >= user_interactions_min:
            users_to_keep = user_counts.filter(
                (pl.col('user_interaction_count') <= user_interactions_max) &
                (pl.col('user_interaction_count') >= user_interactions_min)
            )
        elif user_interactions_min is not None:
            users_to_keep = user_counts.filter(
                pl.col('user_interaction_count') >= user_interactions_min
            )
        elif user_interactions_max is not None:
            users_to_keep = user_counts.filter(
                user_interactions_max >= pl.col('user_interaction_count')
            )
        else:
            users_to_keep = user_counts
        print(f"Users to keep: {users_to_keep.shape}")

        item_counts = self.data.group_by(self.item_id_column).agg(
            pl.count().alias('item_interaction_count')
        )
        print(f"Item counts size: {item_counts.shape}")

        if items_interactions_max is not None and \
                items_interactions_min is not None and items_interactions_max >= items_interactions_min:
            items_to_keep = item_counts.filter(
                (items_interactions_max >= pl.col('item_interaction_count')) &
                (pl.col('item_interaction_count') >= items_interactions_min)
            )
        elif items_interactions_min is not None:
            items_to_keep = item_counts.filter(
                pl.col('item_interaction_count') >= items_interactions_min
            )
        elif items_interactions_max is not None:
            items_to_keep = item_counts.filter(
                items_interactions_max >= pl.col('item_interaction_count')
            )
        else:
            items_to_keep = item_counts
        print(f"Items to keep: {items_to_keep.shape}")

        self.data = (
            self.data.join(users_to_keep, on=self.user_id_column, how='inner')
            .join(items_to_keep, on=self.item_id_column, how='inner')
        )
        print(f"Filtered data size: {self.data.shape}")

        if self.data.is_empty():
            raise ValueError("Filtered data is empty. Please check filtering thresholds.")

    def load_file(self):
        """
        Загрузка данных из файла в формате CSV, TSV, Parquet или DAT.

        Исключения:
            ValueError: если формат файла не поддерживается.
        """
        print("Starting load_file...")
        file_ext = os.path.splitext(self.one_datafile_name)[1].lower()
        filepath = os.path.join(self.data_dir, self.one_datafile_name)
        print(f"Loading file: {filepath}")

        if file_ext == ".csv":
            self.data = pl.read_csv(filepath, separator=self.separator, encoding=self.encoding)
        elif file_ext == ".tsv":
            self.data = pl.read_csv(filepath, separator="\t", encoding=self.encoding)
        elif file_ext == ".parquet":
            self.data = pl.read_parquet(filepath)
        elif file_ext == ".dat":
            self.data = pl.read_csv(filepath, separator=self.separator, encoding=self.encoding)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}. Supported: .csv, .tsv, .parquet, .dat")

        if self.drop_title_row:
            self.data = self.data[1:]

        print(f"Data loaded successfully with shape: {self.data.shape}")

    def split_and_save(self):
        """
        Разделение данных на users, items, interactions. Далее идёт сохранение их в раздельные файлы по имени и пути,
        указанным в настройках датасета
        """
        if self.data is None:
            raise ValueError("Data is not loaded. Please load the data first.")

        self.interactions = self.data.select(self.interactions_features)
        print(f"Interactions size: {self.interactions.shape}")
        if self.interactions.is_empty():
            raise ValueError("No interactions data to save.")

        self.interactions.write_csv(os.path.join(self.data_dir, self.interactions_filename))
        print(f"Interactions saved to {os.path.join(self.data_dir, self.interactions_filename)}")

        self.users = self.data.select(self.users_features).unique()
        print(f"Users size: {self.users.shape}")
        if self.users.is_empty():
            raise ValueError("No users data to save.")

        self.users.write_csv(os.path.join(self.data_dir, self.users_filename))
        print(f"Users saved to {os.path.join(self.data_dir, self.users_filename)}")

        self.items = self.data.select(self.items_features).unique()
        print(f"Items size: {self.items.shape}")
        if self.items.is_empty():
            raise ValueError("No items data to save.")

        self.items.write_csv(os.path.join(self.data_dir, self.items_filename))
        print(f"Items saved to {os.path.join(self.data_dir, self.items_filename)}")

    def users_filter_by_file_content(self):
        """
        Если указано сохранение только пользователей из файла, то данный метод фильтрует данные
        пользователей и сохраняет только тех, которые есть в указанном файле.

        Возвращает множество ASINов, которые были в файле.

        Исключения:
            FileNotFoundError: если указанного файла не существует.
        """
        if self.users_only_from_file and self.users_from_file_id_column is not None:
            as_in_set = set()
            with open(f'{self.users_only_from_file}', 'r') as f:
                for line in f.readlines():
                    data = ast.literal_eval(line.strip())
                    as_in_set.add(data[self.users_from_file_id_column])

            print(f"Total ASINs in {self.users_only_from_file}: {len(as_in_set)}")
            self.users = self.users[self.users[self.user_id_column].isin(as_in_set)]
            self.interactions = self.interactions[self.interactions[self.user_id_column].isin(as_in_set)]
            return as_in_set
        else:
            print("No users filter by file content provided.")
            return False

    def items_filter_by_file_content(self):
        if self.items_only_from_file and self.items_from_file_id_column is not None:
            asin_set = set()
            with open(f'{self.items_only_from_file}', 'r') as f:
                for line in f.readlines():
                    data = ast.literal_eval(line.strip())
                    asin_set.add(data[self.items_from_file_id_column])

            print(f"Total ASINs in {self.items_only_from_file}: {len(asin_set)}")
            self.items = self.items[self.items[self.item_id_column].isin(asin_set)]
            self.interactions = self.interactions[self.interactions[self.item_id_column].isin(asin_set)]
            return asin_set
        else:
            print("No items filter by file content provided.")
            return False

    def load(self, filter_data=None, items_interactions_min=None, items_interactions_max=None,
             user_interactions_min=None, user_interactions_max=None):
        print("Starting load...")
        if self.split_into_files:
            self.load_file()

            print(f"Loaded data size: {self.data.shape}")
            if filter_data:
                print('Proceeding to filtering data')
                self.filter_by_interactions_count(items_interactions_min, items_interactions_max, user_interactions_min,
                                                  user_interactions_max)

            print('Split and save')
            self.split_and_save()
        else:
            print('Luckily, data is already split into three files!')
        if self.users_only_from_file:
            self.users_filter_by_file_content()
        if self.items_only_from_file:
            self.items_filter_by_file_content()