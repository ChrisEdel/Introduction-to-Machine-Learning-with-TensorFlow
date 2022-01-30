import argparse

import json

import utilities

parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('saved_model', help='Model of a (pretrained) network')
parser.add_argument('--top_k', help='Return top k most likely classes')
parser.add_argument('--category_names', help='Use a mapping of categories to real names')


args = parser.parse_args()

top_k = 1 if args.top_k is None else int(args.top_k)
category_names = "label_map.json" if args.category_names is None else args.category_names

model = utilities.load_model(args.saved_model)
print(model)

probs, predict_classes = utilities.predict(args.image_path, model, top_k)

with open(category_names, 'r') as f:
    label_map = json.load(f)

classes = []
    
for predict_class in predict_classes:
    classes.append(label_map[str(predict_class + 1)])

print(probs)
print(classes)