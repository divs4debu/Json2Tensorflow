import json


def read_json_file(file_path):
    with open(file_path) as f:
        config = json.loads(f.read())
    return config


print(read_json_file("sample_input.json"))