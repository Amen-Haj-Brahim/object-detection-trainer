import json
def get_categories(path):
  with open(path, 'r') as f:
    json_data=json.load(f)
    categories = [c["name"] for c in json_data["categories"]]
    return categories