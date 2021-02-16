from images_files_handler import ImagesFilesHandler
from decision_tree import C45
# from digits import visualize_digits

if __name__ == "__main__":
    handler = ImagesFilesHandler()
    test_count = 20
    test_labels = handler.get_labels_set(test_count)
    test_imgs = handler.get_images_set(test_count)
    tree = C45(test_imgs, test_labels)
    # visualize_digits(test_imgs)
