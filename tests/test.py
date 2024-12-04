from src.data_processing import load_settings_from_file, load_raw_dataset, load_inter_files_dataset
from src.cl_interface import get_settings_from_cli
import warnings
warnings.filterwarnings("ignore")


"""
Файл с тестовыми запусками. Отсюда можно запускать загрузку датасетов, имея относительный путь "../data/raw/<>"
"""
# print(load_settings_from_file("../data/raw/zvuk/setting.txt"))
# load_raw_dataset_with_settings_file("../data/raw/zvuk/setting.txt", "../data/processed/zvuk/",
#                                     50, 1024, 32, 1024)

# print(load_settings_from_file("../data/raw/megamarket/setting.txt"))
# load_raw_dataset_with_settings_file("../data/raw/megamarket/setting.txt", "../data/processed/megamarket/",
#                                     50, 1024, 32, 1024)
