import numpy as np
from sklearn.linear_model import LinearRegression
from utils import fon, getMPR
import random
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm
    
class GurobiIP():
    def __init__(self, similarity_scores, dataset, curation_set=None, model=None):
        print("using Gurobi IP...")
        self.n = dataset.shape[0]
        self.d = dataset.shape[1]
        self.dataset = dataset

        if curation_set is None: ## If no curation set is provided, compute MPR over the retrieval set
            self.curation_set = self.dataset
        else:
            self.curation_set = curation_set
        self.m = self.curation_set.shape[0]

        self.expanded_dataset = np.concatenate((self.dataset, self.curation_set), axis=0)

        self.similarity_scores = similarity_scores.squeeze()

        if model is None:
            self.model = LinearRegression()            
        else:
            self.model = model

    def fit(self, k, num_iter, rho):
        self.problem = gp.Model("mixed_integer_optimization")
        self.a = self.problem.addVars(self.m, vtype=GRB.BINARY, name="a")
        obj = gp.quicksum(self.similarity_scores[i]*self.a[i] for i in range(self.m))
        self.problem.setObjective(obj, sense=GRB.MAXIMIZE)
        self.problem.addConstr(sum([self.a[i] for i in range(self.m)]) == k, "constraint_sum_a")
        self.problem.optimize()
       
        for index in tqdm(range(num_iter)):
            gurobi_solution = np.array([self.a[i].x for i in range(len(self.a))])
            # if self.model == "linearrkhs":
            #     term1 = 1/(k**2) * np.sum(np.outer(self.dataset, self.dataset.T))
            #     term2 = 1/(k*self.m) * np.sum(np.outer(self.dataset, self.curation_set.T))
            #     term3 = 1/(self.m**2) * np.sum(np.outer(self.curation_set, self.curation_set.T))
            #     mpr = np.sqrt(term1+term2+term3)
            # else:
            self.sup_function(gurobi_solution, k)
            c = self.model.predict(self.expanded_dataset)
            c /= np.linalg.norm(c)
            c *= c.shape[0]
            mpr = np.abs(np.sum((gurobi_solution/k)*c[:self.n])-np.sum((1/self.m)*c[self.n:]))
            if mpr < rho:
                print("constraints satisfied, exiting early")
                print("\t", np.abs(np.sum((gurobi_solution/k)*c[self.n:])-np.sum((1/self.m)*c[self.n:])))
                print("\t", rho)
                break
            
            self.max_similarity(c, k, rho, index)

            if self.problem.status == 3:
                print("Constraints infeasible, rho = {}".format(rho))
                print(self.problem.NumConstrs)
                return None
            else:
                print(self.problem.ObjVal)
        return gurobi_solution

    def max_similarity(self, c, k, rho, linear_constraint_index):
        sum_a_c = gp.quicksum([self.a[i] * c[:self.n][i] for i in range(self.n)])
        sum_c = gp.quicksum(c[self.n:])
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) <= rho, name="linear_constraint_{}".format(linear_constraint_index))
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) >= -rho, name="neg_linear_constraint_{}".format(linear_constraint_index))
        self.problem.optimize()
        self.problem.update()

    def sup_function(self, a, k):
        curation_indicator = np.concatenate((np.zeros(a.shape[0]), np.ones(self.curation_set.shape[0])))
        a_expanded = np.concatenate((a, np.zeros(self.curation_set.shape[0])))
        alpha = (a_expanded/k - curation_indicator/self.m)
        self.model.fit(self.expanded_dataset, alpha)
    
    def get_representation(self, indices, k):
        self.sup_function(indices, k)
        c = self.model.predict(self.dataset)
        print("norm", np.linalg.norm(c), flush=True)
        c /= np.linalg.norm(c)
        c *= c.shape[0]
        rep = np.abs(np.sum((1/k)*indices*c-(1/self.m)*c))
        return rep

    def get_similarity(self, indices):
        sim = indices.T@self.similarity_scores
        return sim
    
class GurobiLP():
    def __init__(self, similarity_scores, dataset, curation_set=None, model=None):
        print("using Gurobi LP...")
        self.n = dataset.shape[0]
        self.d = dataset.shape[1]
        self.dataset = dataset
        self.problem = None

        # if curation_set is None: ## If no curation set is provided, compute MPR over the retrieval set
        #     self.curation_set = self.dataset
        # else:
        #     self.curation_set = curation_set
        self.curation_set = curation_set
        if curation_set is None:
            self.m = self.dataset.shape[0]
            self.expanded_dataset = np.concatenate((self.dataset, self.dataset), axis=0)
        else:
            self.m =  self.curation_set.shape[0]
            self.expanded_dataset = np.concatenate((self.dataset, self.curation_set), axis=0)

        self.similarity_scores = similarity_scores.squeeze()

        if model is None:
            # self.model = LinearRegression()
            self.model = 'linear'
        else:
            self.model = model

    def fit(self, k, num_iter, rho, KNN_indices):
        # check if rho is satisfied, just return KNN_indices
        mpr, _ = getMPR(self.dataset, k, self.curation_set, self.model, KNN_indices)
        
        print("mpr in solver", mpr)
        if mpr <= rho:
            print("no solve is required, returning KNN indices")
            return KNN_indices
        
        self.problem = gp.Model("mixed_integer_optimization")
        self.problem.params.SoftMemLimit = 16
        self.problem.Params.LogToConsole = 0
        self.a = self.problem.addVars(self.n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="a")
        obj = gp.quicksum(self.similarity_scores[i]*self.a[i] for i in range(self.n))
        self.problem.setObjective(obj, sense=GRB.MAXIMIZE)
        self.problem.addConstr(sum([self.a[i] for i in range(self.n)]) == k, "constraint_sum_a")

        self.problem.optimize()
       
        for index in tqdm(range(num_iter)):
            gurobi_solution = np.array([self.a[i].x for i in range(len(self.a))])

            mpr, c = getMPR(self.dataset, k, self.curation_set, self.model, gurobi_solution)
            
            # if self.model == "linearrkhs":
            #     term1 = 1/(k**2) * np.sum(np.outer(self.dataset, self.dataset.T))
            #     term2 = 1/(k*self.m) * np.sum(np.outer(self.dataset, self.curation_set.T))
            #     term3 = 1/(self.m**2) * np.sum(np.outer(self.curation_set, self.curation_set.T))
            #     mpr = np.sqrt(term1+term2+term3)
            # else:
            #     self.sup_function(gurobi_solution, k)
            #     c = self.model.predict(self.expanded_dataset)
            #     c /= np.linalg.norm(c)
            #     c *= c.shape[0]
            #     mpr = np.abs(np.sum((gurobi_solution/k)*c[:self.n])-np.sum((1/self.m)*c[self.n:]))
            
            if mpr < rho:
                print(f"constraints satisfied, exiting early : mpr{mpr}, rho<{rho}>")
                #print("\t", np.abs(np.sum((gurobi_solution/k)*c[self.n:])-np.sum((1/self.m)*c[self.n:])))
                break
            
            self.max_similarity(c, k, rho, index)

            if self.problem.status == 3:
                print("Constraints infeasible, rho = {}".format(rho))
                print(self.problem.NumConstrs)
                return None
            # else:
                # print(self.problem.ObjVal)
        return gurobi_solution


    def max_similarity(self, c, k, rho, linear_constraint_index):
        sum_a_c = gp.quicksum([self.a[i] * c[:self.n][i] for i in range(self.n)])
        sum_c = gp.quicksum(c[self.n:])
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) <= rho, name="linear_constraint_{}".format(linear_constraint_index))
        self.problem.addConstr(((1/k)*sum_a_c - (1/self.m)*sum_c) >= -rho, name="neg_linear_constraint_{}".format(linear_constraint_index))
        self.problem.optimize()
        self.problem.update()

    def sup_function(self, a, k):
        curation_indicator = np.concatenate((np.zeros(a.shape[0]), np.ones(self.curation_set.shape[0])))
        a_expanded = np.concatenate((a, np.zeros(self.curation_set.shape[0])))
        alpha = (a_expanded/k - curation_indicator/self.m)
        self.model.fit(self.expanded_dataset, alpha)
    
    def get_representation(self, indices, k):
        self.sup_function(indices, k)
        c = self.model.predict(self.dataset)
        # print("norm", np.linalg.norm(c), flush=True)
        c /= np.linalg.norm(c)
        c *= c.shape[0]
        rep = np.abs(np.sum((1/k)*indices*c-(1/self.m)*c))
        return rep

    def get_similarity(self, indices):
        sim = indices.T@self.similarity_scores
        return sim
       
class ClipClip():
    # As defined in the paper "Are Gender-Neutral Queries Really Gender-Neutral? Mitigating Gender Bias in Image Search" (Wang et. al. 2021)
    def __init__(self, features, orderings=None, device='cuda'):
        self.features = features
        self.device = device
        self.m = features.shape[0]
        self.orderings = orderings

    def fit(self, k, num_cols_to_drop, query_embedding):
        indices = self.orderings[num_cols_to_drop:]
        indices = np.sort(indices)
        clip_features = self.features[:, indices]
        clip_query = query_embedding[:, indices]
        # clip_features = torch.index_select(torch.tensor(self.features), 1, torch.tensor(self.orderings[num_cols_to_drop:].copy()).to(self.device))
        # clip_query = torch.index_select(torch.tensor(query_embedding), 1, torch.tensor(self.orderings[num_cols_to_drop:].copy()).to(self.device))

        similarities = (clip_features @ clip_query.T).flatten()
        selections = np.argsort(similarities.squeeze())[::-1][:k]
        #selections = similarities.argsort(descending=True).cpu().flatten()[:k]
        indices = np.zeros(self.m)
        indices[selections] = 1    
        AssertionError(np.sum(indices)==k)
        return indices, selections

class PBM():
    ## As defined in the paper "Mitigating Test-Time Bias for Fair Image Retrieval" (Kong et. al. 2023)
    def __init__(self, features, similarities, pbm_labels, pbm_classes):
        self.features = features
        self.similarities = similarities
        self.m = features.shape[0]
        self.pbm_label = pbm_labels # predicted sensitive group label
        self.pbm_classes = pbm_classes

    def fit(self, k=10, eps=0):
        similarities_sorted = np.argsort(self.similarities.squeeze())[::-1]
        selections = []

        neutrals = [x for x in similarities_sorted if self.pbm_label[x] == 0]
        classes = [[x for x in similarities_sorted if self.pbm_label[x]== i] for i in range(1, len(self.pbm_classes))]

    
        while len(selections) < k:
            if random.random() < eps:
                try:
                    neutral_sim = self.similarities[neutrals[0]]
                except:
                    neutral_sim = -1
                
                max_class, idx = 0, 0
                for i, c in enumerate(classes):
                    try:
                        class_sim = self.similarities[c[0]]
                    except:
                        class_sim = -1
                    if class_sim > max_class:
                        max_class = class_sim
                        idx = i
                if max_class > neutral_sim:
                    selections.append(classes[idx][0])
                    classes[idx].pop(0)
                else:
                    selections.append(neutrals[0])
                    neutrals.pop(0)
                        
            else:
                best_neutral = neutrals[0]
                best_for_classes = [fon(c) for c in classes]
                best_for_classes_vals = [c for c in best_for_classes if c is not None]

                similarities_for_classes = [self.similarities[x] for x in best_for_classes_vals]
                avg_sim = np.mean(similarities_for_classes)
                neutral_sim = self.similarities[best_neutral]

                if avg_sim > neutral_sim:
                    if len(selections) + len(best_for_classes_vals) > k:
                        best_for_classes_vals = random.choices(best_for_classes_vals, k=k-len(selections))
                    selections += best_for_classes_vals

                    for i, x in enumerate(best_for_classes):
                        if x is not None:
                            classes[i].pop(0)
                else:
                    selections.append(best_neutral)
                    neutrals.pop(0)

        indices = np.zeros(self.m)
        indices[selections] = 1    
        AssertionError(np.sum(indices)==k)
        return indices, selections

# MMR algorithm (greedy) with MPR as the diversity metric
class MMR_MPR(): ##FIXME add curation capabilities
    def __init__(self, similarity_scores, dataset, curation_set=None, model=None):
        self.n = dataset.shape[0]
        self.similarity_scores = similarity_scores
        self.dataset = dataset

        if curation_set is None: ## If no curation set is provided, compute MPR over the retrieval set
            self.curation_set = self.dataset
        else:
            self.curation_set = curation_set
        self.m = self.curation_set.shape[0]
        # self.expanded_dataset = np.concatenate((self.dataset, self.curation_set), axis=0)

        if model is None:
            self.model = LinearRegression()
        else:
            self.model = model

    def fit(self, k, lambda_):
        # if self.mean_embedding is None or self.std_embedding is None:
        #     self.mean_embedding, self.std_embedding =  statEmbedding(self.embeddings)

        indices = np.zeros(self.m)
        selection = []
        for i in range(k):
            MMR_temp = np.full(self.n, -np.inf)
            if i==0:
                idx = np.argmax(self.similarity_scores)
                selection.append(idx)
                indices[idx] = 1
                continue
            for j in range(self.n):
                if indices[j] == 1:
                    continue
                # temporary select the jth element
                indices[j] = 1
                rep, _ = getMPR(indices, self.dataset, k, curation_set = self.curation_set, model=self.model)
                MMR_temp[j] = (1-lambda_)* self.similarity_scores.T @ indices + lambda_ * rep
                indices[j] = 0
            # select the element with the highest MMR 
            idx = np.argmax(MMR_temp)
            selection.append(idx)
            indices[np.argmax(MMR_temp)] = 1
        AssertionError(np.sum(indices)==k)
        return indices, selection