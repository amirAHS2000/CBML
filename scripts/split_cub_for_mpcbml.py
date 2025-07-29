import json


cub_root = 'resource/datasets/CUB_200_2011/'
images_file = cub_root + 'images.txt'
train_file = cub_root + 'train.txt'
test_file = cub_root + 'test.txt'
class_counts_file = cub_root + 'class_counts.json'


def main():
    train = []
    test = []
    # dictionary to count the occurrences of each training class
    class_counts = {}

    with open(images_file) as f_img:
        for l_img in f_img:
            i, fname = l_img.split()
            label = int(fname.split('.', 1)[0])
            if label <= 100:
                new_label = label - 1
                train.append((fname, label - 1)) # labels 0 ... 99 (0-based labels for margin_loss)
                # count the training samples per class
                if new_label in class_counts:
                    class_counts[new_label] += 1
                else:
                    class_counts[new_label] = 1
            else:
                test.append((fname, label - 1))  # labels 100 ... 199

    for f, v in [(train_file, train), (test_file, test)]:
        with open(f, 'w') as tf:
            for fname, label in v:
                print("images/{},{}".format(fname, label), file=tf)


    # save the class counts for training
    with open(class_counts_file, 'w') as count_file:
        json.dump(class_counts, count_file, indent=2)


if __name__ == '__main__':
    main()
