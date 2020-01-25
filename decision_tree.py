import math

from digits_visualisation import data, set_count #, labels_values

'''
C4.5 tree implementation
'''

# test
print(data[0])  # [example number 0:set_count-1][row number 0:size-1][column number 0:size-1]

class C45:
    def _init_(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
