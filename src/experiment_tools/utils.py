import json

def read_text(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def save_text(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)