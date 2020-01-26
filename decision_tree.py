import math

from digits_visualisation import data, set_count, get_digits_labels, labels_f  # , labels_values

'''
C4.5 tree implementation
'''

# test
print(len(data[0][0]))  # [example number 0:set_count-1][row number 0:size-1][column number 0:size-1]
#test = get_digits_labels(labels_f)
# print(test[1][1])
#classes_t = set(test)
# print(set(get_digits_labels(labels)))

class C45:
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.examples_num = len(dataset)
        self.attributes_num = len(dataset[0]) * len(dataset[0][0])
        self.classes = self.get_classes()
        self.attributes_values = self.get_attributes_values()

    # improve to detect single class in subset, useful in splitting in c4.5
    def get_classes(self):
        result = []
        for label in self.labels:
            if label not in result:
                result.append(label[0])  # label is np.array type
        return result

    # check if dataset contains only one class data
    def is_single_class(self):
        return len(self.classes) == 1

    # improve to get atrr values in subset
    def get_attributes_values(self):
        result = []
        for instance in self.dataset:
            for row in instance:
                for pixel in row:
                    if pixel not in result:
                        result.append(pixel[0])  # pixel is np.array type
        result.sort()  # debug purposes
        return result



tree = C45(data, get_digits_labels(labels_f))