import matplotlib.pyplot as plt
import numpy as np
# from RandomForestClassifier import *


def visualize_digits(image_data):
    for image in image_data:
        current_img = np.asarray(image).squeeze()
        plt.imshow(current_img)
        plt.show()


# classifier = RandomForestClassifier(10, 10, 10)

# labels = get_digits_labels(labels_f)
# classifier.train(data, labels)

