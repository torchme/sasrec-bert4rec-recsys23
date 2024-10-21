from dataset_preprocessing import *

data_dir = 'ML-1M'
dataset_loader = DatasetLoaderML(data_dir)

dataset_loader.load_users()
dataset_loader.load_movies()
dataset_loader.load_ratings()

train_ratings, val_ratings, test_ratings = dataset_loader.split_ratings([2], [0.8, 0.1, 0.1])
print(train_ratings, '\n', val_ratings, '\n', test_ratings)

negative_samples = dataset_loader.get_negative_samples(num_neg_samples=100)
# print(negative_samples)

train_loader = dataset_loader.get_pytorch_dataloader(train_ratings)
print(train_loader)