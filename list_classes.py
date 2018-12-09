import pickle

with open("data/gtsrb.pickle", "rb") as f:
	gtsrb = pickle.load(f)

# Create label map and class map
class_map, label_map = {}, {}

# Sorting must be applied to gtsrb, because the
# mapping needs to be stable through restarts
for filename, classification in sorted(gtsrb.items()):
	for c in classification:
		key = c["class"]
		if key not in class_map:
			class_id = len(class_map)
			class_map[key] = class_id
			label_map[class_id] = key

for k, v in label_map.items():
	print(k, v)