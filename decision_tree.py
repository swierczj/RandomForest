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
        self.classes = self.get_classes(self.labels)
        self.attributes_values = self.get_attributes_values(self.dataset)

    # improve to detect single class in subset, useful in splitting in c4.5
    def get_classes(self, labels_subset):
        result = []
        for label in labels_subset:
            if label not in result:
                result.append(label[0])  # label is np.array type
        return result

    # check if dataset contains only one class data
    def is_single_class(self, dataset):
        classes = self.get_classes(dataset)
        if len(classes) == 1:
            return classes, True
        return classes, False

    # improve to get atrr values in subset
    def get_attributes_values(self, subset):
        result = []
        for instance in subset:
            for row in instance:
                for pixel in row:
                    if pixel not in result:
                        result.append(pixel[0])  # pixel is np.array type
        result.sort()  # debug purposes
        return result

    def generate_subtree(self, current_data, current_attributes):
        if(len(current_data)) <= 0:
            return Node(True, "no data")
        current_classes, is_single = self.is_single_class(current_data)
        if is_single:
            return Node(True, current_classes)
        if len(current_attributes) == 0:  # if no attributes left, leaf
            return Node(True, self.get_dominant_class(current_data))
        # if considered dataset needs to be splitted
        best_attribute, splitted_sets = self.split_on_attribute(current_data, current_attributes)
        remaining_attributes = current_attributes[:]
        remaining_attributes.remove(best_attribute)
        node = Node(False, best_attribute)
        node.children = [self.generate_subtree(subset, remaining_attributes) for subset in splitted_sets]


    def get_dominant_class(self, subdata):
        classes_frequency = [0] * len(self.classes)

    # improvement needed, available_attributes needs to be list of indexes of particular attributes and example in for
    # loop is considered as list of attributes values, it won't work now, get_attribute_value() also to be written
    def split_on_attribute(self, dataset, available_attributes):
        result_subsets = []
        max_entropy = float('-inf')
        best_attr = None
        for attribute in available_attributes:
            attribute_index = self.dataset[0].index(attribute)
            attribute_values = get_attribute_values(attribute)
            subsets = [[] for attr_v in attribute_values]
            for example in dataset:
                # iterate through possible values of attribute, append example to particular subset if attr values match
                for i in range(0, len(attribute_values)):
                    if example[attribute] == attribute_values[i]:
                        subsets[i].append(example)
                        break
            entropy = information_gain(dataset, subsets)  # TODO
            if entropy > max_entropy:
                max_entropy = entropy
                result_subsets = subsets
                best_attr = attribute
        return best_attr, result_subsets

    # TODO:
    #def get_label_from_data_sample(self, sample):
    #    data_index = self.dataset.index(sample)



    def entropy(self, labels):
        num_of_classes = len(self.classes)
        classes_count = [0 for n in range(num_of_classes)]
        for sample in labels:
            classes_count[sample] += 1
        ent = 0
        classes_count = [x / num_of_classes for x in classes_count]
        for c in classes_count:
            ent += c* math.log(c)
        return -1 * ent
        
    
    def gain(self, union_set, sets):
        gain = 0
        for small_set in sets:
            gain += self.entropy(small_set) * len(small_set) / len(union_set)
        return self.entropy(union_set) - gain
        
class Node:
    def __init__(self, is_leaf, label):
        self.is_leaf = is_leaf
        self.label = label
        #self.threshold = threshold
        self.children = []

tree = C45(data, get_digits_labels(labels_f))