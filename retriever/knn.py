import numpy as np
from retriever import GenericRetriever
from utils import getMPR

class Retriever(GenericRetriever):
    def __init__(self, **kwargs):
        GenericRetriever.__init__(self, **kwargs)

    def retrieve(self, retrieval_labels, curated_labels, k=10, s=None):

        m = retrieval_labels.shape[0]

        top_indices = np.zeros(m)
        top_indices[np.argsort(s.squeeze())[::-1][:k]] = 1
        score = s.T@top_indices

        MPR, _ = getMPR(retrieval_labels, k, curated_labels, modelname=self.args.functionclass, indices=top_indices)
        return top_indices, [MPR], [score]

