import pandas as pd
import argparse


def main(old_train_path, old_mapping_path, new_mapping_path, new_train_path):
    user_id2item_id = pd.read_pickle(old_train_path)
    print('Sequences', len(user_id2item_id))
    real_id2old_id = pd.read_pickle(old_mapping_path)
    print(real_id2old_id)
    old_id2real_id = {v: k for k, v in real_id2old_id.items()}
    real_id2new_id = pd.read_pickle(new_mapping_path)

    for user_id in user_id2item_id:
        user_id2item_id[user_id] = [real_id2new_id[old_id2real_id[x]] for x in user_id2item_id[user_id]]

    pd.to_pickle(user_id2item_id, new_train_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and map user-item data.')
    parser.add_argument('old_train_path', type=str, help='Path to the old train data pickle file')
    parser.add_argument('old_mapping_path', type=str, help='Path to the old mapping pickle file')
    parser.add_argument('new_mapping_path', type=str, help='Path to the new mapping pickle file')
    parser.add_argument('new_train_path', type=str, help='Path to save the new train data pickle file')

    args = parser.parse_args()

    main(args.old_train_path, args.old_mapping_path, args.new_mapping_path, args.new_train_path)