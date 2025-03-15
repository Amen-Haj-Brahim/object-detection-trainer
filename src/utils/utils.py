import json
def get_categories():
  with open("../../data/train/_annotations.coco.json", 'r') as f:
    json_data=json.load(f)
    categories = [c["name"] for c in json_data["categories"]]
    return categories