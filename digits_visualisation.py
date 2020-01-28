import matplotlib.pyplot as plt
import numpy as np
import gzip
from RandomForestClassifier import *

images = gzip.open('train-images-idx3-ubyte.gz', 'r')
labels_f = gzip.open('train-labels-idx1-ubyte.gz', 'r')
image_size = 28
set_count = 10
images_file_offset = 16  # non image information
labels_file_offset = 8

images.read(images_file_offset)
buf = images.read(image_size * image_size * set_count)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(set_count, image_size, image_size, 1)

#for i in range(0, set_count):
 #   image = np.asarray(data[i]).squeeze()
 #   plt.imshow(image)
 #   plt.show()
    

labels_f.read(labels_file_offset)

#  function returns list of labels
def get_digits_labels(labels_set, count = set_count):
    if count <= 0:
        return None
    if labels_set.closed:
        labels_set = gzip.open('train-labels-idx1-ubyte.gz', 'r')
        labels_set.read(labels_file_offset)
    labels_result = []
    for j in range(0, count):
        tmp = labels_set.read(1)
        labels_result.append(np.frombuffer(tmp, dtype=np.uint8).astype(np.int64))
    labels_set.close()
    return labels_result




classifier = RandomForestClassifier(10, 10 , 10)

# labels = get_digits_labels(labels_f)
# classifier.train(data, labels)

