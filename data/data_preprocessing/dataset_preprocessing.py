import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import os


class DatasetLoaderML:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.users = None
        self.movies = None
        self.ratings = None

    def load_users(self):
        user_columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        user_path = os.path.join(self.data_dir, 'users.dat')
        self.users = pd.read_csv(user_path, sep='::', engine='python', names=user_columns)
        print(f'Users data loaded: {self.users.shape[0]} records')

    def load_movies(self):
        movie_columns = ['MovieID', 'Title', 'Genres']
        movie_path = os.path.join(self.data_dir, 'movies.dat')
        self.movies = pd.read_csv(movie_path, sep='::', engine='python', names=movie_columns, encoding="iso-8859-1")
        print(f'Movies data loaded: {self.movies.shape[0]} records')

    def load_ratings(self):
        ratings_columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        ratings_path = os.path.join(self.data_dir, 'ratings.dat')
        self.ratings = pd.read_csv(ratings_path, sep='::', engine='python', names=ratings_columns)
        print(f'Ratings data loaded: {self.ratings.shape[0]} records')

    def split_users(self, proportions, random_state=42):
        if self.users is None:
            raise ValueError("Users data is not loaded.")

        stratify_columns = ['Gender', 'Age']
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

    def split_ratings(self, user_ids, proportions):
        if self.ratings is None:
            raise ValueError("Ratings data is not loaded.")

        user_ratings = self.ratings[self.ratings['UserID'].isin(user_ids)]

        def split_user_ratings(user_data):
            user_data = user_data.sort_values(by='Timestamp')
            num_ratings = len(user_data)
            train_size = round(proportions[0] * num_ratings)
            if len(proportions) > 2:
                val_size = round(proportions[1] * num_ratings)
                return (user_data[:train_size],
                        user_data[train_size:train_size + val_size],
                        user_data[train_size + val_size:])
            return user_data[:train_size], user_data[train_size:]

        train_ratings, val_ratings, test_ratings = [], [], []

        grouped = user_ratings.groupby('UserID')
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

    def get_negative_samples(self, num_neg_samples=1000):
        """
        Method to generate negative samples for each user
        """
        if self.ratings is None or self.movies is None:
            raise ValueError("Ratings or movies data is not loaded.")

        user_positive_movies = self.ratings.groupby('UserID')['MovieID'].apply(set).to_dict()

        all_movies = set(self.movies['MovieID'].unique())
        negative_samples_per_user = {}

        for user_id, positive_movies in user_positive_movies.items():
            negative_movies = list(all_movies - positive_movies)
            negative_samples = np.random.choice(negative_movies, size=num_neg_samples, replace=False)
            negative_samples_per_user[user_id] = negative_samples

        return negative_samples_per_user

    def get_pytorch_dataloader(self, ratings, batch_size=32, shuffle=True):

        class MovieLensDataset(Dataset):
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

        dataset = MovieLensDataset(ratings)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

