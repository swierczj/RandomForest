import matplotlib.pyplot as plt
import numpy as np


def visualize_digits(image_data):
    for image in image_data:
        current_img = np.asarray(image).squeeze()
        plt.imshow(current_img)
        plt.show()
