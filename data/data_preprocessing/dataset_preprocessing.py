import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os
from datetime import datetime


class DatasetSettings:
    def __init__(self, data_dir, items_features, users_features, interactions_features, items_filename, users_filename, interactions_filename, separator, encoding="utf-8", drop_title_row=True):
        self.data_dir = data_dir
        self.items_features = items_features
        self.users_features = users_features
        self.interactions_features = interactions_features
        self.items_filename = items_filename
        self.users_filename = users_filename
        self.interactions_filename = interactions_filename
        self.separator = separator
        self.encoding = encoding
        self.drop_title_row = drop_title_row


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

    def load_users(self):
        user_columns = self.users_features
        user_path = os.path.join(self.data_dir, self.users_filename)
        self.users = pd.read_csv(user_path, sep=self.separator, engine='python', names=user_columns)
        if self.drop_title_row:
            self.users = self.users.iloc[1:]
        print(f'Users data loaded: {self.users.shape[0]} records')

    def load_movies(self):
        movie_columns = self.items_features
        movie_path = os.path.join(self.data_dir, self.items_filename)
        self.movies = pd.read_csv(movie_path, sep=self.separator, engine='python', names=movie_columns, encoding=self.encoding)
        if self.drop_title_row:
            self.movies = self.movies.iloc[1:]
        print(f'Movies data loaded: {self.movies.shape[0]} records')

    def load_ratings(self):
        ratings_columns = self.interactions_features
        ratings_path = os.path.join(self.data_dir, self.interactions_filename)
        self.ratings = pd.read_csv(ratings_path, sep=self.separator, engine='python', names=ratings_columns)
        if self.drop_title_row:
            self.ratings = self.ratings.iloc[1:]
        print(f'Ratings data loaded: {self.ratings.shape[0]} records')

    """
    Common splits
    """

    def split_users(self, proportions, stratify_columns=['sex', 'age'], random_state=42):
        if self.users is None:
            raise ValueError("Users data is not loaded.")
        train_users, test_users = train_test_split(
            self.users,
            test_size=(1 - proportions[0]),
            stratify=self.users[stratify_columns],
            random_state=random_state
        )

        if len(proportions) > 2:
            remaining_proportions = proportions[1] / sum(proportions[1:])
            val_users, test_users = train_test_split(
                test_users,
                test_size=(1 - remaining_proportions),
                stratify=test_users[stratify_columns],
                random_state=random_state
            )
            print(f'Users split into train: {len(train_users)}, val: {len(val_users)}, test: {len(test_users)}')
            return train_users, val_users, test_users

        print(f'Users split into train: {len(train_users)}, test: {len(test_users)}')
        return train_users, test_users

    def split_ratings(self, user_ids, proportions, user_id='user_id', item_id='item_id', timestamp='last_watch_dt'):
        if self.ratings is None:
            raise ValueError("Ratings data is not loaded.")
        user_ratings = self.ratings[self.ratings[user_id].isin(user_ids)]

        def split_user_ratings(user_data):
            user_data = user_data.sort_values(by=timestamp)
            num_ratings = len(user_data)
            train_size = round(proportions[0] * num_ratings)
            if len(proportions) > 2:
                val_size = round(proportions[1] * num_ratings)
                return (user_data[:train_size],
                        user_data[train_size:train_size + val_size],
                        user_data[train_size + val_size:])
            return user_data[:train_size], user_data[train_size:]

        train_ratings, val_ratings, test_ratings = [], [], []

        grouped = user_ratings.groupby(user_id)
        for user_id, group in grouped:
            if len(proportions) > 2:
                train, val, test = split_user_ratings(group)
                train_ratings.append(train)
                val_ratings.append(val)
                test_ratings.append(test)
            else:
                train, test = split_user_ratings(group)
                train_ratings.append(train)
                test_ratings.append(test)

        train_ratings = pd.concat(train_ratings)
        if len(proportions) > 2:
            val_ratings = pd.concat(val_ratings)
        test_ratings = pd.concat(test_ratings)

        if len(proportions) > 2:
            print(f'Ratings split into train: {len(train_ratings)}, val: {len(val_ratings)}, test: {len(test_ratings)}')
            return train_ratings, val_ratings, test_ratings

        print(f'Ratings split into train: {len(train_ratings)}, test: {len(test_ratings)}')
        return train_ratings, test_ratings

    def get_negative_samples(self, num_neg_samples=1000, user_id='user_id', item_id='item_id', timestamp='last_watch_dt'):
        """
        Method to generate negative samples for each user
        """
        if self.ratings is None or self.movies is None:
            raise ValueError("Ratings or movies data is not loaded.")

        user_positive_movies = self.ratings.groupby(user_id)[item_id].apply(set).to_dict()

        all_movies = set(self.movies[item_id].unique())
        negative_samples_per_user = {}

        for user_, positive_movies in user_positive_movies.items():
            negative_movies = list(all_movies - positive_movies)
            negative_samples = np.random.choice(negative_movies, size=num_neg_samples, replace=False)
            negative_samples_per_user[user_] = negative_samples

        return negative_samples_per_user

    def get_pytorch_dataloader(self, ratings, batch_size=32, shuffle=True):

        class DatasetCreator(Dataset):
            def __init__(self, ratings):
                self.ratings = ratings

            def __len__(self):
                return len(self.ratings)

            def __getitem__(self, idx):
                row = self.ratings.iloc[idx]
                user_id = torch.tensor(row['UserID'], dtype=torch.long)
                movie_id = torch.tensor(row['MovieID'], dtype=torch.long)
                rating = torch.tensor(row['Rating'], dtype=torch.float)
                return user_id, movie_id, rating

        dataset = DatasetCreator(ratings)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    """
    Split from 'Does it look sequential' article
    """
    def session_split(self, boundary=0.8, validation_size=None,
                    user_id='user_id', item_id='item_id', timestamp='last_watch_dt',
                    path_to_save=None, dataset_name=None):
        """Session-based split.

        Args:
            data (pd.DataFrame): Events data.
            boundary (float): Quantile for splitting into train and test part.
            validation_size (int): Number of users in validation set. No validation set if None.
            user_id (str): Defaults to 'user_id'.
            item_id (str): Defaults to 'item_id'.
            timestamp (str): Defaults to 'last_watch_dt'.
            path_to_save (str): Path to save resulted data. Defaults to None.
            dataset_name (str): Name of the dataset. Defaults to None.

        Returns:
            Train, validation (optional), test datasets.
        """
        data = self.ratings
        data = data[pd.to_datetime(data[timestamp], errors='coerce').notna()]
        train, test = self.session_splitter(data, boundary, user_id, timestamp)

        if validation_size is not None:
            train, validation, test = self.make_validation(
                train, test, validation_size, user_id, item_id, timestamp)

            if path_to_save is not None:
                train.to_csv(path_to_save + 'train_' + dataset_name + '.csv')
                test.to_csv(path_to_save + 'test_' + dataset_name + '.csv')
                validation.to_csv(path_to_save + 'validation_' + dataset_name + '.csv')

            return train, validation, test

        if path_to_save is not None:
            train.to_csv(path_to_save + 'train_' + dataset_name + '.csv')
            test.to_csv(path_to_save + 'test_' + dataset_name + '.csv')

        else:
            train = train[[user_id, item_id, timestamp]]
            test = test[[user_id, item_id, timestamp]]

        return train, test

    def make_validation(self, train, test, validation_size,
                        user_id='user_id', item_id='item_id', timestamp='last_watch_dt'):
        """Add validation dataset."""

        validation_users = np.random.choice(train[user_id].unique(),
                                            size=validation_size, replace=False)
        validation = train[train[user_id].isin(validation_users)]
        train = train[~train[user_id].isin(validation_users)]

        train = train[[user_id, item_id, timestamp]].astype(int)
        test = test[[user_id, item_id, timestamp]].astype(int)
        validation = validation[[user_id, item_id, timestamp]].astype(int)

        return train, validation, test


    def session_splitter(self, data, boundary, user_id='user_id', timestamp='last_watch_dt'):
        """Make session split."""

        data.sort_values([user_id, timestamp], inplace=True)
        problematic_dates = data[data[timestamp] == "20"]

        quant = (pd.to_datetime(data[timestamp], format='mixed').astype("int64") // 10**9).quantile(boundary)
        users_time = data.groupby(user_id)[timestamp].agg(list).apply(
            lambda x: int(datetime.strptime(x[0], "%Y-%m-%d").timestamp()) <= quant).reset_index()
        users_time_test = data.groupby(user_id)[timestamp].agg(list).apply(
            lambda x: int(datetime.strptime(x[-1], "%Y-%m-%d").timestamp()) > quant).reset_index()

        train_user = list(users_time[users_time[timestamp]][user_id])
        test_user = list(users_time_test[users_time_test[timestamp]][user_id])

        train = data[data[user_id].isin(train_user)]
        train = train[pd.to_datetime(train[timestamp], format='mixed').astype("int64") // 10**9 <= quant]
        test = data[data[user_id].isin(test_user)]

        return train, test
