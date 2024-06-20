import numpy as np
from retriever import GenericRetriever

class Retriever(GenericRetriever):
    def __init__(self):
        GenericRetriever.__init__(self)

    def retrieve(retrieval_labels, curated_labels, k=10):
        idx = np.random.choice(retrieval_labels.shape[0], k, replace=False)
        output = np.ones(retrieval_labels.shape[0])
        output[idx] = 1
        return output