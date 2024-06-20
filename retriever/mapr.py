from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from retriever.solver import GurobiIP, GurobiLP
import numpy as np
from retriever import GenericRetriever
from utils import getMPR

class Retriever(GenericRetriever):
    def __init__(self):
        GenericRetriever.__init__(self)

    def retrieve(retrieval_labels, curated_labels, k=10):
        cutting_planes = 50
        m = retrieval_labels.shape[0]

        # compute similarities
        s = np.ones(m)

        top_indices = np.zeros(m)
        top_indices[np.argsort(s.squeeze())[::-1][:k]] = 1
        sim_upper_bound = s.T@top_indices
        print("Similarity Upper Bound", sim_upper_bound, flush=True)
        
        oracle = LinearRegression()
        solver2 = GurobiLP(s, retrieval_labels, curation_set=curated_labels, model = oracle)
        # solver = GurobiIP(s, retrieval_labels, curation_set=curated_labels, model = oracle)

        reps_relaxed = []
        sims_relaxed = []
        sparsities = []
        rounded_reps_final = []
        rounded_sims_final = []
        relaxed_indices_list = []
        rounded_indices_list = []
        rhos = np.linspace(0.005, 2.505, 50)
        indices = top_indices
        for rho in rhos[::-1]:
            print('rho : ', rho)
            indices = solver2.fit(k, cutting_planes, rho, indices)
            if indices is None:
                break
            sparsity = sum(indices>1e-4)
            indices_rounded = indices.copy()
            indices_rounded[np.argsort(indices_rounded)[::-1][k:]] = 0
            indices_rounded[indices_rounded>1e-5] = 1.0 

            rep = solver2.get_representation(indices, k)
            sim = solver2.get_similarity(indices)

            rounded_rep = solver2.get_representation(indices_rounded, k)
            rounded_sim = solver2.get_similarity(indices_rounded)
            rounded_rep,_ = getMPR(retrieval_labels, k, curated_labels, oracle, indices_rounded)

            reps_relaxed.append(rep)
            sims_relaxed.append(sim)
            rounded_reps_final.append(rounded_rep)
            rounded_sims_final.append(rounded_sim)
            sparsities.append(sparsity)
            rounded_indices_list.append(indices_rounded)
            relaxed_indices_list.append(indices)

            # indices_gurobi = solver.fit(k, cutting_planes, rho)
            # if indices_gurobi is None:
            #     break
            # rep = solver.get_representation(indices_gurobi, k)
            # sim = solver.get_similarity(indices_gurobi)

            # gurobi_indices_list.append(indices_gurobi)

            # reps_gurobi.append(rep)
            # sims_gurobi.append(sim)
            # rhoslist.append(rho)

        print("final mprs", rounded_reps_final)
        print("final sims", sims_relaxed)

        return relaxed_indices_list[-1], sims_relaxed