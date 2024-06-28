import numpy as np
from retriever import GenericRetriever
from utils import getMPR

class Retriever(GenericRetriever):
    def __init__(self, **kwargs):
        GenericRetriever.__init__(self, **kwargs)
        self.ratio = self.args.ratio

    def retrieve(self, retrieval_labels, curated_labels, k=10, s=None):
        MPR_total = 0
        score_total = 0
        
        n_samples_per_adj = round(self.ratio * 1000)
        idx = np.random.choice(1000, n_samples_per_adj, replace=False)
        output = np.zeros(15*1000)
        for i in range(15):
            output[idx] = 1
            idx += 1000
        # output -= 1000
        MPR, _ = getMPR(retrieval_labels, k, curated_labels, modelname=self.args.functionclass, indices=output)
        MPR_total += MPR
        score_total += np.sum(s[output==1])
    
        return idx, [MPR_total], [score_total]