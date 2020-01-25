import matplotlib.pyplot as plt
import numpy as np
import gzip

images = gzip.open('train-images-idx3-ubyte.gz', 'r')
labels = gzip.open('train-labels-idx1-ubyte.gz', 'r')
image_size = 28
set_count = 5
images_file_offset = 16  # non image information
labels_file_offset = 8

images.read(images_file_offset)
buf = images.read(image_size * image_size * set_count)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(set_count, image_size, image_size, 1)

for i in range(0, set_count):
    image = np.asarray(data[i]).squeeze()
    plt.imshow(image)
    plt.show()

labels.read(labels_file_offset)

#  supposing file with labels is opened, function returns list of labels
def get_numbers_labels(labels_set, count = set_count):
    if count <= 0:
        return None
    labels_result = []
    for j in range(0, count):
        tmp = labels_set.read(1)
        labels_result.append(np.frombuffer(tmp, dtype=np.uint8).astype(np.int64))
    return labels_result
