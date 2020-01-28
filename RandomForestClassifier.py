from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

class RandomForestClassifier(object):
    def __init__(self, n_estimators: int, n_samples: int, max_depth: int):
        self.estimators = []
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.max_depth = max_depth
    
    def train(self, train_data, target_data):
        X, y = train_data, target_data
        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            X, y = shuffle(X, y, random_state=0)
            tree.fit(X[:self.n_samples], y[:self.n_samples])
            self.estimators.append(tree)
    
    def predict(self, data):
        predictions = []
        for tree in self.estimators:
            predictions.append(tree.predict(data))
        return max(set(predictions), key=predictions.count)

