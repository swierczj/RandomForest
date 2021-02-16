import math
import numpy as np

# from digits import data, set_count, get_digits_labels, labels_f  # , labels_values
from collections import defaultdict

'''
C4.5 tree implementation
'''

# test
# print(len(data[0][0]))  # [example number 0:set_count-1][row number 0:size-1][column number 0:size-1]


class C45:
    def __init__(self, dataset, labels):
        # cleanup here
        self._dataset = dataset
        self._labels = labels
        self._examples_num = len(dataset)
        self._attributes_num = 784  # number of pixels in one image
        self._classes = np.unique(self._labels) # todo: cleanup from here and remove get_classes
        self._min_attribute_val = 0.0
        self._max_attribute_val = 255.0
        self._attributes_values = self.get_attributes_values(self._dataset)
        # todo:
        # self.available_attributes = [i for i in range(0, self.attributes_num)]
        # self.root = self.generate_subtree(self.dataset, self.available_attributes)

    # # improve to detect single class in subset, useful in splitting in c4.5
    # def get_classes(self, subset):
    #     result = []
    #     for record in subset:
    #         if record[-1] not in result:
    #             result.append(record[-1])
    #     return result
    #
    # # check if dataset contains only one class data
    # def is_single_class(self, dataset):
    #     classes = self.get_classes(dataset)
    #     if len(classes) == 1:
    #         return classes, True
    #     return classes, False

    def _is_single_class(self, labels: np.array):
        return len(np.unique(labels)) == 1

    # improve to get atrr values in subset
    def get_attributes_values(self, subset):
        max_count_values = int(self._max_attribute_val + 1)
        result = np.full(max_count_values, -1.0)
        attr_index = 0
        for instance in subset:
            distinct_values = np.unique(instance)
            sets_diff = np.setdiff1d(distinct_values, result)
            ending_index = attr_index + len(sets_diff)
            result[attr_index:ending_index] = sets_diff
            # if there's no -1.0 then all slots has been filled with pixels
            if -1.0 not in result:
                attr_index = max_count_values
                break
            attr_index = ending_index
        result = result[:attr_index]  # trim zeros from back <- change it to trim -1s, get -1 index earlier and return slice
        result.sort()
        return result

    def generate_subtree(self, current_data, current_attributes, parent=None):
        if (len(current_data)) <= 0:
            return Node(True, "no data", -1, parent)
        current_classes, is_single = self.is_single_class(current_data)
        if is_single:
            if parent is not None:
                prev_attr_val = self.get_attribute_values(current_data, parent.label)
                return Node(True, current_classes, prev_attr_val, parent)
            return Node(True, current_classes, -1, parent)
        if len(current_attributes) == 0:  # if no attributes left then generate leaf
            return Node(True, self.get_dominant_class(current_data), -1, parent)
        # if considered dataset needs to be splitted
        best_attribute, splitted_sets = self.split_on_attribute(current_data, current_attributes)
        remaining_attributes = current_attributes[:]
        remaining_attributes.remove(best_attribute)
        if parent is not None:  # marking node with value of attribute which splitted dataset
            prev_attribute_val = self.get_attribute_values(current_data, parent.label)
            node = Node(False, best_attribute, prev_attribute_val, parent)
        else:
            node = Node(False, best_attribute, -1, parent)
        node.children = [self.generate_subtree(subset, remaining_attributes, node) for subset in splitted_sets]
        return node

    def get_dominant_class(self, subdata):
        classes_frequency = [0] * len(self.classes)
        for example in subdata:
            label_index = self.classes.index(example[-1])
            classes_frequency[label_index] += 1
        max_index = classes_frequency.index(max(classes_frequency))
        return self.classes[max_index]

    # improvement needed, available_attributes needs to be list of indexes of particular attributes and example in for
    # loop is considered as list of attributes values, it won't work now, get_attribute_value() also to be written
    def split_on_attribute(self, dataset, available_attributes):
        result_subsets = []  # elements of subsets as elements of list
        max_entropy = float('-inf')
        best_attr = None
        for attribute in available_attributes:
            subsets = self.generate_subsets(dataset, attribute)  # 2, to avoid traversing through dataset too many times
            # subsets working
            # attribute_index = available_attributes.index(attribute)  # 1 it's variable called attribute
            # attribute_values = self.get_attribute_values(dataset, attribute_index) # keys in subsets dict
            # if len(attribute_values) > 5:
            #    print('debug')
            # subsets = [[] for attr_v in attribute_values]
            for example in dataset:
                # iterate through possible values of attribute, append example to particular subset if attr values match
                # if only one attr value then append whole data
                for i in range(0, len(subsets)):  # for every different attribute value
                    if example[attribute] == attribute_values[i]:  # if example attr's value matches one of attr_values
                        subsets[i].append(example)  # then append this example to corresponding subset
                        break
            entropy = self.gain(dataset, subsets)  # calculate InfGain based on this set partition
            if entropy > max_entropy:
                max_entropy = entropy
                result_subsets = subsets
                best_attr = attribute
        print('best attribute: ', best_attr, 'entropy: ', max_entropy)
        return best_attr, result_subsets

    def generate_subsets(self, curr_data, attr):
        subsets = defaultdict(list)  # key will be attribute value and value will be index of example in subset
        for example in curr_data:
            example_index = curr_data.index(example)
            attr_value = example[attr]
            subsets[attr_value].append(example_index)
        return subsets

    def get_attribute_values(self, current_dataset, attribute_index):
        attr_values_result = []
        for example in current_dataset:
            if example[attribute_index] not in attr_values_result:
                attr_values_result.append(example[attribute_index])
        return attr_values_result

    # # [for column in preprocessed append(costam)]
    # def export_data_to_list(self):
    #     #  result = self.preprocessed_data.tolist()
    #     result = [[0 for attr in range(0, self.attributes_num + 1)] for example in range(set_count)]  # +1 for label column
    #     i = 0
    #     j = 0
    #     for example in self.preprocessed_data:
    #         for row in example:
    #             for pixel in row:
    #                 result[i][j] = pixel.item(0)
    #                 j += 1
    #         result[i][j] = self.labels[i].item(0)  # append label
    #         j = 0
    #         i += 1
    #     return result

    def entropy(self, dataset):
        set_quantity = len(dataset)
        if set_quantity == 0:
            return 0
        classes_count = [0 for n in self.classes]
        for sample in dataset:
            class_index = self.classes.index(sample[-1])
            classes_count[class_index] += 1
        entropy = 0
        classes_count = [x / set_quantity for x in
                         classes_count]  # ratio of particular class occurances to set quant, probality of meeting given class in dataset
        for c in classes_count:
            # print(entropy)
            if c > 0:
                entropy += c * math.log(c)
        return -1 * entropy

    def gain(self, union_set, sets):
        gain = 0
        for small_set in sets:
            gain += self.entropy(small_set) * len(small_set) / len(union_set)
        return self.entropy(union_set) - gain

    def predict(self, dataset):
        node = self.root
        mismatches = 0
        error_rate = 0
        for sample in dataset:
            node = self.root
            while not node.is_leaf:
                error = float('inf')
                attr_index = node.label
                current_best = -1
                for child in node.children:
                    if sample[attr_index] == child.prev_attr_value[0]:
                        current_best = node.children.index(child)
                        break
                    else:
                        # current_best = node.children.index(child)
                        current_attr_val = sample[attr_index]
                        prev_val = child.prev_attr_value[0]
                        current_error = abs(sample[attr_index] - child.prev_attr_value[0])
                        if current_error < error:
                            current_best = node.children.index(child)
                            error = current_error
                node = node.children[current_best]
            # node is leaf now
            prediction_value = node.label[0]
            print('sample ', dataset.index(sample) + 1, ' predicted value: ', prediction_value, '; actual value: ',
                  sample[-1])
            if prediction_value != sample[-1]:
                mismatches += 1
        error_rate = mismatches / len(dataset)
        print('error rate: ', error_rate * 100, '%')
        return error_rate

    # def fit(self):
    #     root = self.generate_subtree(self.dataset, self.available_attributes)
    #     print('trained')

    def prune_tree(root, node, dataset, best_score):
        if node.is_leaf():
            label = node.label
            node.parent.is_leaf = True
            node.parent.label = node.label
            if node.height < 20:
                new_score = self.predict(root, dataset)
            else:
                new_score = 0

            if new_score <= best_score:
                return new_score
            else:
                node.parent.is_leaf = False
                node.parent.label = None
                return best_score
        else:
            new_score = best_score
            for child in node.children:
                new_score = prune_tree(root, child, dataset, new_score)
                if node.is_leaf:
                    return new_score
            return new_score


class Node:
    def __init__(self, is_leaf, label, prev_attr_value, parent=None):
        self.label = label
        self.prev_attr_value = prev_attr_value
        self.children = []
        self.is_leaf = is_leaf
        self.parent = parent
    # def is_leaf(self):
    #   return len(self.children) == 0

# tree = C45(data, get_digits_labels(labels_f))
# print(tree.root.label)
