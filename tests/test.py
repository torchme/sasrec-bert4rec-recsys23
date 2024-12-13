from src.data_processing import load_settings_from_file, load_raw_dataset, load_inter_files_dataset


# print(load_settings_from_file("../data/raw/kion/setting.txt"))
# load_raw_dataset("../data/raw/kion/setting.txt", "../data/processed/kion/")

# print(load_settings_from_file("../data/raw/kion_en/setting.txt"))
load_raw_dataset("../data/raw/kion_en/setting.txt", "../data/processed/kion_en/")

# print(load_settings_from_file("../data/raw/amazon_beauty/setting.txt"))
# load_raw_dataset("../data/raw/amazon_beauty/setting.txt", "../data/processed/amazon_beauty/")