from collections import defaultdict
import numpy as np
from decision_tree import C45


def predict(img_data, tree: C45):
    node = tree.root
    # traverse down the tree to leaves
    while not node.is_leaf():
        attr_index = node.get_attribute()
        img_attr_value = np.ravel(img_data)[attr_index]
        node = find_best_fit(img_attr_value, node.children)
    return node.label


def find_best_fit(attr_value, children_nodes: defaultdict):
    if attr_value in children_nodes:
        return children_nodes[attr_value]
    # else if attr_value not found in keys
    min_diff = np.inf
    best_fit = None
    for value in children_nodes:
        diff = abs(attr_value - value)
        if diff < min_diff:
            min_diff = diff
            best_fit = children_nodes[value]
    return best_fit
