import numpy as np
from subset import Subset
from image_data_representation import ImageDataRep
from collections import defaultdict

'''
C4.5 tree implementation
'''


class C45:
    def __init__(self, dataset, labels):
        self._dataset = dataset
        self._labels = labels
        self._examples_num = len(dataset)
        self._attributes_num = 784  # number of pixels in one image
        self._classes = np.unique(self._labels)
        self._imgs_to_labels = self._init_imgs_data()  # images data representation subset, here it's whole set
        self._min_attribute_val = 0.0
        self._max_attribute_val = 255.0
        self._attributes_values = self._get_attributes_values(self._imgs_to_labels)
        self._available_attributes = np.arange(self._attributes_num)
        self.root = self._generate_subtree(self._imgs_to_labels, self._available_attributes)
        # todo: c45 - prune id3 tree represented by root

    def _get_image_data(self, img_idx):
        return self._dataset[img_idx]

    def _init_imgs_data(self):
        result = Subset()
        dataset_count = len(self._dataset)
        for img_idx in range(dataset_count):
            current_data = ImageDataRep(img_idx, self._labels[img_idx])
            result.add_element(current_data)
        return result

    def _get_attributes_values(self, subset):
        unique_vals = set()
        max_set_size = int(self._max_attribute_val + 1)
        for sample in subset:
            # extract specific image data
            img_data = self._get_image_data(sample.get_og_index())
            distinct_attributes_vals = np.unique(np.ravel(img_data))
            unique_vals.update(distinct_attributes_vals)
            # if set cannot be extended anymore then exit the loop
            if len(unique_vals) == max_set_size:
                break
        result = np.fromiter(unique_vals, dtype=float)
        result.sort()
        return result

    # todo: test every if else
    def _generate_subtree(self, current_data: Subset, current_attributes, parent=None):
        # if no data has been passed
        if len(current_data) == 0:
            return Node(label="no-data", attr_idx=None, parent=parent)
        dominant_class, dominant_class_occurrences = self._get_dominant_class(current_data)
        # if there's only one class in all instances or no attributes left then generate leaf
        if dominant_class_occurrences == len(current_data) or len(current_attributes) == 0:
            subtree_root = Node(label=dominant_class, attr_idx=None, parent=parent)
        # if considered dataset needs to be splitted
        else:
            best_attribute, splitted_sets = self._find_best_split(current_attributes, current_data)
            remaining_attributes = current_attributes[current_attributes != best_attribute]  # remove best attribute, without copying existing array
            node_label = None
            if parent is None:
                node_label = "root"
            subtree_root = Node(label=node_label, attr_idx=best_attribute, parent=parent)
            subtree_root.children = {attr_val: self._generate_subtree(splitted_sets[attr_val], remaining_attributes, subtree_root) for attr_val in splitted_sets}
        return subtree_root

    def _get_dominant_class(self, subdata: Subset):
        classes_frequency = {label: 0 for label in self._classes}
        for example in subdata:
            label = example.get_label()
            classes_frequency[label] += 1
        dominant_class = max(classes_frequency, key=classes_frequency.get)
        return dominant_class, classes_frequency[dominant_class]

    # function that provides values of particular attribute within the whole subset
    def _get_attribute_values(self, current_dataset: Subset, attribute_index):
        unique_vals = set()
        for sample in current_dataset:
            img_data = self._get_image_data(sample.get_og_index())
            attribute_val = np.ravel(img_data)[attribute_index]
            unique_vals.add(attribute_val)
        attr_existing_values = np.fromiter(unique_vals, dtype=float)
        return attr_existing_values

    def _find_best_split(self, current_attributes, current_subset: Subset):
        current_max_gain = np.NINF
        best_attr = None
        result_subsets = defaultdict(Subset)
        presplit_set_size = len(current_subset)
        presplit_set_entropy = self._entropy(current_subset)
        for attribute in current_attributes:
            attribute_subsets = self._generate_subsets(current_subset, attribute)
            inf_gain = self._gain(presplit_set_entropy, presplit_set_size, attribute_subsets)
            if inf_gain > current_max_gain:
                current_max_gain = inf_gain
                result_subsets = attribute_subsets
                best_attr = attribute
        return best_attr, result_subsets

    def _generate_subsets(self, curr_data: Subset, attr):
        attr_value_subsets = defaultdict(Subset)  # mapping subset indices to specific attribute value
        for example in curr_data:
            img_data = self._get_image_data(example.get_og_index())
            attr_value = np.ravel(img_data)[attr]
            attr_value_subsets[attr_value].add_element(example)
        return attr_value_subsets

    def _entropy(self, dataset):
        dataset_entropy = 0
        set_size = len(dataset)
        classes_count = defaultdict(int)
        # populate classes_count
        for sample in dataset:
            sample_label = sample.get_label()
            classes_count[sample_label] += 1
        classes_probability = {label: class_count / set_size for label, class_count in classes_count.items()}
        for label in classes_probability:
            class_prob = classes_probability[label]
            dataset_entropy += class_prob * np.log(class_prob)
        return -dataset_entropy

    def _gain(self, union_set_entropy, union_set_size, sets: defaultdict):
        subsets_entropy = 0
        for attr_val, subset in sets.items():
            subsets_entropy += self._entropy(subset) * len(subset)
        splitted_set_entropy = subsets_entropy / union_set_size
        return union_set_entropy - splitted_set_entropy  # inf_gain is the difference of entropy of set without partition and splitted set


class Node:
    def __init__(self, label, attr_idx, parent=None):
        self.label = label
        self._splitting_attribute = attr_idx
        self.parent = parent
        self.children = defaultdict(Node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_attribute(self):
        return self._splitting_attribute
