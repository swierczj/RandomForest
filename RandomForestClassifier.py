from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier(object):
    def __init__(self, nb_trees: int, nb_samples: int, max_depth: int):
        self.trees = []
        self.nb_trees = nb_trees
        self.nb_samples = nb_samples
        self.max_depth = max_depth
    
    def train(self, data):
        random_samples = map(lambda x: random_samples(data, self.nb_samples),
         range(self.nb_trees))
        self.trees = map(self.train_tree, random_samples)
    
    def train_tree(self, data):
        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(data)
        return tree

    def predict(self, data):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(data))
        return max(predictions, key=predictions.count)