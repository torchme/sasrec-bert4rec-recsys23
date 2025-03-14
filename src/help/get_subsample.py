import json

FPATH_ID = '/home/nseverin/generate_user_profiles/recsys-user-profiles/data/amazon-m2/gigamax-short-ru-e5-embs-1.json'
FPATH_DATA = '/home/nseverin/generate_user_profiles/recsys-user-profiles/data/amazon-m2/gemma-short-descriptions/gemma2-short-e5-embs-all.json'
FPATH_SAVE = '/home/nseverin/generate_user_profiles/recsys-user-profiles/data/amazon-m2/gemma-short-descriptions/gemma2-short-e5-embs-all-subsample.json'

with open(FPATH_DATA, 'r') as f:
    user_profiles_data = json.load(f)

with open(FPATH_ID, 'r') as f:
    user_keys = json.load(f).keys()

result = {k:v for k,v in user_profiles_data.items()}

with open(FPATH_ID, 'w') as f:
    json.dump(result, f)