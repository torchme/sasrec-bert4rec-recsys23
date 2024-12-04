import os
import re

from src.data_processing import DatasetSettings, load_raw_dataset, load_settings_from_file


def ask_until_valid(prompt, validate_fn, error_message):
    while True:
        value = input(prompt).strip()
        if validate_fn(value):
            return value
        print(error_message)


def parse_bool(prompt):
    return ask_until_valid(
        prompt + " (true/false): ",
        lambda x: x.lower() in ["true", "false"],
        "Please enter 'true' or 'false'."
    ).lower() == "true"

def parse_int(prompt, bigger_than=0):
    value = ask_until_valid(
        prompt + " (positive integer or none): ",
        lambda x: x.isdigit() and int(x) > bigger_than or x.lower() == "none",
        f"Please enter an integer bigger than {bigger_than} or None.")
    return int(value) if value.lower() != "none" else None

def get_settings_from_cli():
    """
    При помощи интерфейса командной строки, получает в форме вопрос-ответ все необходимые данные для составления
    настроек датасета (с возможностью последующего сохранения)

    """

    raw_inputs = []  # To store raw user-entered strings

    def parse_list(prompt, sep=", "):
        while True:
            value = input(prompt).strip()
            raw_inputs.append(f"{value}")
            items = value.split(sep)
            if all(items) and len(items) == len(set(items)) and all(" " not in item for item in items):
                return items
            print(f"Invalid input. Use '{sep}' as a separator without spaces in item names.")

    def parse_optional_file(prompt):
        filename = input(prompt).strip()
        raw_inputs.append(f"{filename}")
        return None if filename.lower() in ["none", "false", "don't know"] else filename

    def parse_columns(prompt):
        return parse_list(prompt + " (comma-space-separated): ")

    def parse_scale(prompt):
        while True:
            value = input(prompt).strip()
            raw_inputs.append(f"{value}")
            if value.lower() == "none":
                return None
            try:
                scale = tuple(map(float, value.split(", ")))
                if len(scale) == 2:
                    return scale
            except ValueError:
                pass
            print("Invalid scale. Enter two numbers separated by ', ', or 'none'.")

    def parse_column_name(prompt, features):
        return ask_until_valid(
            prompt,
            lambda x: " " not in x and x in features and x != "_NULL_TXT_FILE",
            f"Column name must not contain spaces and must be in the features: {', '.join(features)} " +
            "and must not be used as auxiliary column name in internal processes."
        )

    def parse_filename(prompt, extensions=None):
        extensions = extensions or []
        return ask_until_valid(
            prompt,
            lambda x: any(x.endswith(f".{ext}") for ext in extensions) if extensions else re.search(r'\.[a-zA-Z]+$', x),
            f"Filename must end with {', '.join(f'.{ext}' for ext in extensions)}."
        )

    data_dir = ask_until_valid(
        "Enter the data directory: ",
        lambda x: x.lower() not in ["none", "false", "don't know"],
        "Data directory cannot be 'none', 'false', or 'don't know'."
    )

    split_into_files = parse_bool("Is all the data in one file (requires splitting)?")
    one_datafile_name = (
        parse_filename("Enter the name of the single data file to split: ")
        if split_into_files else None
    )
    if not split_into_files:
        raw_inputs.append(f"none")

    items_features = parse_columns("Enter the item features ")
    users_features = parse_columns("Enter the user features ")
    interactions_features = parse_columns("Enter the interaction features (comma-space-separated): ")

    items_filename = (
        parse_filename("Enter the items file name (e.g., 'items.csv'): ", extensions=["csv"])
        if split_into_files else parse_filename("Enter the items file name (with any extension): ")
    )
    users_filename = (
        parse_filename("Enter the users file name (e.g., 'users.csv'): ", extensions=["csv"])
        if split_into_files else parse_filename("Enter the users file name (with any extension): ")
    )
    interactions_filename = (
        parse_filename("Enter the interactions file name (e.g., 'interactions.csv'): ", extensions=["csv"])
        if split_into_files else parse_filename("Enter the interactions file name (with any extension): ")
    )

    separator = input("Enter the separator (e.g., ',' or '\\t'): ").strip()
    raw_inputs.append(f"{separator}")
    encoding = input("Enter the encoding (e.g., 'utf-8'): ").strip()
    raw_inputs.append(f"{encoding}")
    drop_title_row = parse_bool("Drop the first row as a title row?")

    item_id = parse_column_name("Enter the item ID column name: ", items_features)
    user_id = parse_column_name("Enter the user ID column name: ", users_features)
    timestamp = parse_column_name("Enter the timestamp column name: ", interactions_features)

    timestamp_format = input("Enter the timestamp format (or 'none' for standard numeric timestamp): ").strip()
    raw_inputs.append(f"{timestamp_format}")
    timestamp_format = None if timestamp_format.lower() == "none" else timestamp_format

    interaction = parse_column_name("Enter the interaction column name (e.g. 'rating'): ", interactions_features)
    interaction_scale = parse_scale("Enter the interaction scale (two numbers separated by ', ', or 'none'): ")

    stratify_users = parse_bool("Do you want to stratify users when splitting train and validation?")
    stratify_user_column_names = (
        parse_columns("Enter the columns for stratifying users (comma-space-separated): ")
        if stratify_users else None
    )
    if not stratify_users:
        raw_inputs.append(f"_, _")

    users_only_from_file = parse_optional_file(
        "Enter the filename to filter users (or 'none' if no filtering is needed): "
    )
    if users_only_from_file and not users_only_from_file.endswith(".txt"):
        users_from_file_id_column = parse_column_name("Enter the column name for filtering users: ", users_features)
    else:
        users_from_file_id_column = "_NULL_TXT_FILE" if users_only_from_file else None
    if not users_only_from_file:
        raw_inputs.append("none")

    items_only_from_file = parse_optional_file(
        "Enter the filename to filter items (or 'none' if no filtering is needed): "
    )
    if items_only_from_file and not items_only_from_file.endswith(".txt"):
        items_from_file_id_column = parse_column_name("Enter the column name for filtering items: ", items_features)
    else:
        items_from_file_id_column = "_NULL_TXT_FILE" if items_only_from_file else None
    if not items_only_from_file:
        raw_inputs.append("none")

    settings = DatasetSettings(
        data_dir=data_dir,
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

    save_settings = parse_bool("Do you want to save entered settings as a file?")
    if save_settings:
        while True:
            save_path = input("Enter the directory to save the settings file: ").strip()
            if os.path.isdir(save_path):
                with open(os.path.join(save_path, "setting.txt"), "w") as f:
                    f.writelines(f"{line}\n" for line in raw_inputs[:-1])
                print(f"Settings saved at {os.path.join(save_path, 'setting.txt')}")
                break
            else:
                print("Invalid directory. Please enter a valid path.")

    return settings


def run_preprocessing_pipeline(settings=None, settings_filepath=None):
    if settings is None and settings_filepath is None:
        raise ValueError(f"Impossible to run dataset preprocessing: no settings nor settings_filepath provided.")

    print("Starting dataset preprocessing: loading settings")
    if settings_filepath and not settings:
        settings = load_settings_from_file(settings_filepath)
    print("Proceeding to loading raw dataset")
    path_to_save = settings.data_dir.replace('raw', 'processed')

    filtering = parse_bool("Do you need to filter your dataset in terms of interactions count per user/item?")
    if filtering:
        min_interactions_user = parse_int("Please enter a minimum number of interactions per user (or none to skip)")
        if settings.split_into_files:
            max_interactions_user = parse_int("Please enter a maximum number of interactions per user (or none to skip)",
                                              min_interactions_user - 1)
        else:
            max_interactions_user = None
        min_interactions_item = parse_int("Please enter a minimum number of interactions per item (or none to skip)")
        if settings.split_into_files:
            max_interactions_item = parse_int("Please enter a maximum number of interactions per item (or none to skip)",
                                          min_interactions_item - 1)
        else:
            max_interactions_item = None

    else:
        min_interactions_user = None
        max_interactions_user = None
        min_interactions_item = None
        max_interactions_item = None
    load_raw_dataset(settings, path_to_save=path_to_save,
                     filter_user_actions_count_min=min_interactions_user,
                     filter_user_actions_count_max=max_interactions_user,
                     filter_item_actions_count_min=min_interactions_item,
                     filter_item_actions_count_max=max_interactions_item)
    print(f"Done loading! Please see the output files located in {path_to_save}")


def client_start():

    print("CLI for dataset loading is starting")
    print("You will be asked several questions to know details about how to preprocess your dataset.")
    print("Dataset can be loaded from either one file with preprocessing and splitting, or from three (already split"
          "into users-items-interactions)")
    print("As the result, program will save pickle (.pkl) files to processed data folder, which then can be used to"
          "work with our models")
    print("--------------------------------")
    settings_from_file = parse_bool("Do you have settings file for the dataset you want to load? If not, we will "
                                    "constuct it quickly via same CLI. If yes, we will stick to settings from file. "
                                    "So, any file in mind?")

    if settings_from_file:
        prompt = ("Please enter the absolute or relative path to your settings file (relative from where you run this "
                  "script): ")
        extensions = ["txt", "setting"]
        settings_filepath = ask_until_valid(
            prompt,
            lambda x: any(x.endswith(f".{ext}") for ext in extensions) if extensions else re.search(r'\.[a-zA-Z]+$', x),
            f"Filepath must end with filename with extension in: {', '.join(f'.{ext}' for ext in extensions)}."
        )
        run_preprocessing_pipeline(settings_filepath=settings_filepath)
    else:
        print("We will now construct settings via CLI")
        settings = get_settings_from_cli()
        run_preprocessing_pipeline(settings=settings)


if __name__ == "__main__":
    client_start()
