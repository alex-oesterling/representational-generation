import numpy as np
from retriever import GenericRetriever
from utils import getMPR

class Retriever(GenericRetriever):
    def __init__(self, **kwargs):
        GenericRetriever.__init__(self, **kwargs)

    def retrieve(self, retrieval_labels, curated_labels, k=10, s=None):
        MPR_total = 0
        norMPR_total = 0
        score_total = 0
        n_iter = 1
        for i in range(n_iter):
            idx = np.random.choice(retrieval_labels.shape[0], k, replace=False)
            output = np.zeros(retrieval_labels.shape[0])
            output[idx] = 1
            MPR, _ = getMPR(retrieval_labels, k, curated_labels, modelname=self.args.functionclass, indices=output)
            norMPR, _ = MPR, _ = getMPR(retrieval_labels[idx], k, curated_labels, modelname=self.args.functionclass)
            MPR_total += MPR
            norMPR_total += norMPR
            score_total += np.sum(s[idx])
        return idx, [MPR_total/n_iter],[norMPR_total/n_iter], [score_total/n_iter]