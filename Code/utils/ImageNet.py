import json
import os
base_path = os.path.abspath(__file__)
folder = os.path.dirname(base_path)
with open(os.path.join(folder, 'imagenet-simple-labels.json')) as f:
    LABELS = json.load(f)

def imagenet_class_id_to_label(i):
    return LABELS[i]