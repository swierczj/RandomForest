from image_data_representation import ImageDataRep


class Subset:
    def __init__(self):
        self._subset_data = []

    def __getitem__(self, key):
        return self._subset_data[key]

    def __len__(self):
        return len(self._subset_data)

    def __iter__(self):
        yield from self._subset_data

    def add_element(self, og_data: ImageDataRep):
        self._subset_data.append(og_data)
