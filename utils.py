import json

def json_load(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def json_dump(data, file_path):
    """Dump data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

