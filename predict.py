import gzip
import numpy as np

from decision_tree import C45
from digits_visualisation import data, get_digits_labels, labels_f, images, images_file_offset, image_size, set_count, \
    labels_file_offset


def export_data(dataset, attr_num, set_cardinality, classes):
    result = [[0 for attr in range(0, attr_num + 1)] for example in range(set_cardinality)]  # +1 for label
    i = 0
    j = 0
    for sample in dataset:
        for row in sample:
            for pixel in row:
                result[i][j] = pixel.item(0)
                j += 1
        result[i][j] = classes[i].item(0)  # append label
        j = 0
        i += 1
    j = 0
    i = 0
    return result

test_attr_num = 784
offset = set_count
test_count = 40

if images.closed:
    images = gzip.open('train-images-idx3-ubyte.gz', 'r')
    images.read(images_file_offset)
    buf = images.read(image_size * image_size * test_count)
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(test_count, image_size, image_size, 1)
else:
    buf = images.read(image_size * image_size * test_count)
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(test_count, image_size, image_size, 1)

labels_test_set = gzip.open('train-labels-idx1-ubyte.gz', 'r')
labels_test_set.read(labels_file_offset + offset)
labels_test = []
for j in range(0, test_count):
    tmp = labels_test_set.read(1)
    labels_test.append(np.frombuffer(tmp, dtype=np.uint8).astype(np.int64))
labels_test_set.close()

test_dataset = export_data(test_data, test_attr_num, test_count, labels_test)

tree = C45(data, get_digits_labels(labels_f))
<<<<<<< Updated upstream
tree.predict(data)
=======
tree.predict(test_dataset)
>>>>>>> Stashed changes
