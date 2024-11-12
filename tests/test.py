from dataset_preprocessing import *

data_dir = 'kion-ru'
kion_ru_settings = DatasetSettings(data_dir,
                                   ["item_id", "content_type", "title", "title_orig", "release_year", "genres", "countries", "for_kids", "age_rating", "studios", "directors", "actors", "description", "keywords"],
                                   ["user_id", "age", "income", "sex", "kids_flg"],
                                   ["user_id", "item_id", "last_watch_dt", "total_dur", "watched_pct"],
                                   "items.csv", "users.csv", "interactions.csv",
                                   ",")
dataset_loader = DatasetLoaderML(kion_ru_settings)

dataset_loader.load_users()
dataset_loader.load_movies()
dataset_loader.load_ratings()

train_ratings, val_ratings, test_ratings = dataset_loader.split_ratings(["962099", "656683"], [0.8, 0.1, 0.1])
print(train_ratings, '\n', val_ratings, '\n', test_ratings)

# negative_samples = dataset_loader.get_negative_samples(num_neg_samples=100)
# print(negative_samples)

train_loader = dataset_loader.get_pytorch_dataloader(train_ratings)
print(train_loader)

print(dataset_loader.session_split())
