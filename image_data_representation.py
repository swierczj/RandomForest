# class representing single image data referencing to original dataset and label of particular image data
class ImageDataRep:
    def __init__(self, index, label):
        self._og_index = index
        self._label = label

    def get_og_index(self):
        return self._og_index

    def get_label(self):
        return self._label
