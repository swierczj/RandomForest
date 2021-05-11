from images_files_handler import ImagesFilesHandler
from decision_tree import C45
from predict import predict
import time

if __name__ == "__main__":
    handler = ImagesFilesHandler()
    train_count = 80
    train_labels = handler.get_labels_set(train_count)
    train_imgs = handler.get_images_set(train_count)
    test_count = int(0.25 * train_count)
    test_labels = handler.get_labels_set(test_count, start_from=train_count)
    test_imgs = handler.get_images_set(train_count, start_from=train_count)

    tree_creation_start = time.perf_counter()
    tree = C45(train_imgs, train_labels)
    tree_creation_end = time.perf_counter()
    print(f'tree created\nTraining time: {tree_creation_end - tree_creation_start}')

    correct_predictions = 0
    for img_to_predict, label in zip(test_imgs, test_labels):
        predicted_label = predict(img_to_predict, tree)
        print(f'Predicted: {predicted_label}, correct label: {label}')
        if predicted_label == label:
            correct_predictions += 1
    accuracy = float(correct_predictions / test_count)
    print(f'ID3 tree accuracy: {accuracy * 100: .2f}%')
