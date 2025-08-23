"""
This file actually was a cell in Google Colab (I paste it here),
and include all the preprocessing which this dataset needed.
"""
import os
import shutil
import pandas as pd
import json


root_dir = './cars196/'

img_counter = 1


# read the 'names.csv' file and create a dict to map classes with index
label_file = 'names.csv'
label_file_path = os.path.join(root_dir, label_file)
df = pd.read_csv(label_file_path, header=None)
class_names = df[0].tolist()
class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

# rename all of the train images
train_img_path = os.path.join(root_dir, 'car_data/car_data/train/')
for subfolder in sorted(os.listdir(train_img_path)):
    subfolder_path = os.path.join(train_img_path, subfolder)
    if os.path.isdir(subfolder_path):
        class_idx = class_to_idx[subfolder]
        # iterate through files in the current subfolder
        for filename in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path):
                # process file
                _, extension = os.path.splitext(file_path)
                new_name = f"{img_counter}_{class_idx}{extension}"
                # rename the image
                os.rename(file_path, os.path.join(subfolder_path, new_name))
                img_counter += 1

# rename all of the test images
test_img_path = os.path.join(root_dir, 'car_data/car_data/test/')
for subfolder in sorted(os.listdir(test_img_path)):
    subfolder_path = os.path.join(test_img_path, subfolder)
    if os.path.isdir(subfolder_path):
        class_idx = class_to_idx[subfolder]
        for filename in sorted(os.listdir(subfolder_path)):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path):
                _, extension = os.path.splitext(file_path)
                new_name = f"{img_counter}_{class_idx}{extension}"
                os.rename(file_path, os.path.join(subfolder_path, new_name))
                img_counter += 1

# moving all the test images into train folder (into their corresponding folder)
for class_name in os.listdir(test_img_path):
    test_class_path = os.path.join(test_img_path, class_name)
    train_class_path = os.path.join(train_img_path, class_name)

    # skip if it is not a folder
    if not os.path.isdir(test_class_path):
        print('it\'s not a folder')
        continue
    # ensure target train folder exists
    if not os.path.exists(train_class_path):
        print('train path for this class doesn\'t exist')
        os.makedirs(train_class_path)

    # move each file
    for filename in os.listdir(test_class_path):
        src_file = os.path.join(test_class_path, filename)
        dst_file = os.path.join(train_class_path, filename)

        shutil.move(src_file, dst_file)

# create train.txt and test.txt files
train_file = root_dir + 'train.txt'
test_file = root_dir + 'test.txt'
class_counts_file = root_dir + 'class_counts.json'
train_data = []
test_data = []
class_counts = {}
for subfolder in os.listdir(train_img_path):
    subfolder_path = os.path.join(train_img_path, subfolder)

    # check if it belongs to train set or test
    for img_file in os.listdir(subfolder_path):
        filename = os.path.basename(img_file)
        img_abs_path = os.path.join(subfolder_path, img_file)
        img_class_idx = class_to_idx[subfolder]
        if img_class_idx <= 97:
            train_data.append((img_abs_path, img_class_idx))
            class_counts[img_class_idx] = class_counts.get(img_class_idx, 0) + 1
        else:
            test_data.append((img_abs_path, img_class_idx))

# save train and test data
for f, v in [(train_file, train_data), (test_file, test_data)]:
    with open(f, 'w') as tf:
        for fname, label in v:
            # TODO: this absolute path of images might need some changes
            print("{},{}".format(fname, label), file=tf)

# save the class counts for training
with open(class_counts_file, 'w') as counts_file:
    json.dump(class_counts, counts_file, indent=2)