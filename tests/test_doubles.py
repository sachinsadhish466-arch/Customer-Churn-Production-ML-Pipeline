import numpy as np


class FakeChurnModel:

    def predict_proba(self, X):

        probability = np.full(len(X), 0.75)

        return np.column_stack([
            1 - probability,
            probability
        ])