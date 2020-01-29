import math

from digits_visualisation import data, set_count, get_digits_labels, labels_f  # , labels_values

'''
C4.5 tree implementation
'''

# test
print(len(data[0][0]))  # [example number 0:set_count-1][row number 0:size-1][column number 0:size-1]


class C45:
    def __init__(self, dataset, labels):
        self.preprocessed_data = dataset
        self.labels = labels
        self.examples_num = len(dataset)
        self.attributes_num = len(dataset[0]) * len(dataset[0][0])  # no of pixels, multiplying columns with rows
        self.classes = self.get_classes(labels)
        self.attributes_values = self.get_attributes_values(self.preprocessed_data)
        self.dataset = self.export_data_to_list()
        self.available_attributes = [i for i in range(0, self.attributes_num)]
        self.root = self.generate_subtree(self.dataset, self.available_attributes)

    # improve to detect single class in subset, useful in splitting in c4.5
    def get_classes(self, subset):
        result = []
        for record in subset:
            if record[-1] not in result:
                result.append(record[-1])
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

    def generate_subtree(self, current_data, current_attributes, parent=None):
        if(len(current_data)) <= 0:
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
        result_subsets = []
        max_entropy = float('-inf')
        best_attr = None
        for attribute in available_attributes:
            attribute_index = available_attributes.index(attribute)
            attribute_values = self.get_attribute_values(dataset, attribute_index)
            # if len(attribute_values) > 5:
            #    print('debug')
            subsets = [[] for attr_v in attribute_values]
            for example in dataset:
                # iterate through possible values of attribute, append example to particular subset if attr values match
                for i in range(0, len(attribute_values)):
                    if example[attribute] == attribute_values[i]:
                        subsets[i].append(example)
                        break
            entropy = self.gain(dataset, subsets)
            if entropy > max_entropy:
                max_entropy = entropy
                result_subsets = subsets
                best_attr = attribute
        print('best attribute: ', best_attr, 'entropy: ', max_entropy)
        return best_attr, result_subsets

    def get_attribute_values(self, current_dataset, attribute_index):
        attr_values_result = []
        for example in current_dataset:
            if example[attribute_index] not in attr_values_result:
                attr_values_result.append(example[attribute_index])
        return attr_values_result

    def export_data_to_list(self):
        #  result = self.preprocessed_data.tolist()
        result = [[0 for attr in range(0, self.attributes_num + 1)] for example in range(set_count)]  # +1 for label
        i = 0
        j = 0
        for example in self.preprocessed_data:
            for row in example:
                for pixel in row:
                    result[i][j] = pixel.item(0)
                    j += 1
            result[i][j] = self.labels[i].item(0)  # append label
            j = 0
            i += 1
        j = 0
        i = 0
        return result

    def entropy(self, dataset):
        set_quantity = len(dataset)
        if set_quantity == 0:
            return 0
        classes_count = [0 for n in self.classes]
        for sample in dataset:
            class_index = self.classes.index(sample[-1])
            classes_count[class_index] += 1
        entropy = 0
        classes_count = [x / set_quantity for x in classes_count]  # ratio of particular class occurances to set quant, probality of meeting given class in dataset
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
            while not node.is_leaf:
                error = float('inf')
                attr_index = node.label
                current_best = -1
                for child in node.children:
                    if sample[attr_index] == child.prev_attr_value:
                        node = child
                        break
                    else:
                        # current_best = node.children.index(child)
                        current_error = abs(sample[attr_index] - child.prev_attr_value)
                        if current_error < error:
                            current_best = node.children.index(child)
                            error = current_error
                node = node.children[current_best]
            # node is leaf now
            prediction_value = node.label
            print('sample ', dataset.index(sample) + 1, ' predicted value: ', prediction_value, '; actual value: ', sample[-1])
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
    #def is_leaf(self):
    #   return len(self.children) == 0

# tree = C45(data, get_digits_labels(labels_f))
# print(tree.root.label)
