from src.data_processing import load_settings_from_file, load_raw_dataset, load_inter_files_dataset


print(load_settings_from_file("../data/raw/rees46-2019-Nov/setting.txt"))
load_raw_dataset("../data/raw/rees46-2019-Nov/setting.txt", "../data/processed/rees46-2019-Nov/")
