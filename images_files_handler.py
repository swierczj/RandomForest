import gzip
import numpy as np


class ImagesFilesHandler:
    def __init__(self):
        self._training_images_filename = 'train-images-idx3-ubyte.gz'
        self._training_labels_filename = 'train-labels-idx1-ubyte.gz'
        self._test_images_filename = 't10k-images-idx3-ubyte.gz'
        self._test_labels_filename = 't10k-labels-idx1-ubyte.gz'
        self._images_file_offset = 16
        self._labels_file_offset = 8
        self._max_train_images_count = 60000
        self._max_test_images_count = 10000
        self._image_side_len = 28
        self._label_size = 1

    def get_images_set(self, set_count, start_from=0, from_training_set=True):
        set_fname, set_count = self._get_fname_and_set_count(set_count, start_from, from_training_set, True)
        result_set = np.zeros((set_count, self._image_side_len, self._image_side_len), dtype=np.float32)
        with gzip.open(set_fname, 'r') as images:
            images.seek(self._images_file_offset + start_from*(self._image_side_len**2))  # move file handle position to start from desired image
            for i in range(set_count):
                buf = images.read(self._image_side_len**2)  # read one image at once
                img_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape((self._image_side_len,
                                                                                          self._image_side_len))
                result_set[i] = img_data
        return result_set

    def get_labels_set(self, set_count, start_from=0, from_training_set=True):
        set_fname, set_count = self._get_fname_and_set_count(set_count, start_from, from_training_set, False)
        result_set = np.zeros(set_count, dtype=np.uint8)
        with gzip.open(set_fname, 'r') as labels:
            labels.seek(self._labels_file_offset + start_from)  # move file handle position to start from desired label
            for i in range(set_count):
                buf = labels.read(self._label_size)  # read one label at once
                label = np.frombuffer(buf, dtype=np.uint8)  # check type in dbg
                result_set[i] = label
        return result_set

    def _get_fname_and_set_count(self, set_count, starting_pos, from_training_set, getting_images):
        if from_training_set:
            if getting_images:
                set_fname = self._training_images_filename
            else:
                set_fname = self._training_labels_filename
            max_count = self._max_train_images_count
        else:
            if getting_images:
                set_fname = self._test_images_filename
            else:
                set_fname = self._test_labels_filename
            max_count = self._max_test_images_count
        # check if images/labels can be read
        diff = max_count - starting_pos
        # if there's less remaining images/labels in set than desired number, then then read only the remaining img/lbl
        if diff < set_count:
            set_count = diff
        return set_fname, set_count
