
import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import random 
import itertools
import pickle
from tqdm import tqdm
import clip


def make_result_path(args):
    # filename = f'{args.query_dataset}_{args.refer_dataset}_{args.vision_encoder}_{args.target_model}_{args.functionclass}'
    # if args.retriever != 'random_ratio':
    #     filename += f'_{args.retriever}_{args.k}' if args.retrieve else ''
    # else:
    #     filename += f'_{args.retriever}_{args.ratio}' if args.retrieve else ''
    save_dir = os.path.join(args.save_dir, args.date)
    check_log_dir(save_dir)
    return save_dir

def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            return 1
        else:
            return 0
    except OSError:
        print("Failed to create directory!!")

def set_seed(seed): 
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def compute_similarity(visual_features, target_concept, vision_encoder, vision_encoder_name='clip'):
    if vision_encoder_name != 'CLIP':
        raise ValueError('Only CLIP is supported for now')
    similarity = np.zeros(visual_features.shape[0])

    visual_features = visual_features / np.linalg.norm(visual_features, axis=-1, keepdims=True) 
    vision_encoder.eval()
    with torch.no_grad():        
        prompt = f'photo portrait of {target_concept}'
        text = clip.tokenize(prompt)
        if torch.cuda.is_available():
            text = text.cuda()
        text_embedding = vision_encoder.encode_text(text).float()
        text_embedding = text_embedding.cpu().numpy().squeeze()
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        similarity = visual_features @ text_embedding
    return similarity

# for PBM
def fon(l):
    try:
        return l[0]
    except:
        return None

def statEmbedding(embeddings):
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            # print(embeddings[i], embeddings[j])
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    distances = np.array(distances)
    mean_embedding = np.mean(distances)
    std_embedding = np.std(distances)
    return mean_embedding, std_embedding

# def getMPR(indices, labels, oracle, k, m):


def gpu_mem_usage():
    """Computes the GPU memory usage for the current device (GB)."""
    if not torch.cuda.is_available():
        return 0
    # Number of bytes in a megabyte
    _B_IN_GB = 1024 * 1024 * 1024

    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / _B_IN_GB


def print_intersectional_probs(dataset):
    # Combine dataset and curation_set
    probs = np.zeros((2,2,7))
    # Count occurrences of each unique intersectional group
    # unique, counts = np.unique(dataset, axis=0, return_counts=True)
    for i, gender in enumerate(group_dic['gender']):
        for j, age in enumerate(group_dic['age']):
            for k, race in enumerate(group_dic['race']):
                prob = np.mean((dataset[:,i] == 1) & (dataset[:,j+2] == 1) & (dataset[:,k+4] == 1))
                probs[i,j,k] = prob
    refer_prob = np.load('group_ratio.npy')
    np.save('fairdiffusion_ratio.npy', probs)   
    print(probs)
    print(refer_prob)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, weight=1.0, n_attr=2, fmt=':f'):
        self.name = name
        self.n_attr = n_attr
        self.fmt = fmt
        self.reset()
        self.weight = weight
        self.updated_step = -1

    def reset(self):
        self.val = torch.Tensor([0] * self.n_attr)
        self.avg = torch.Tensor([0] * self.n_attr)
        self.sum = torch.Tensor([0] * self.n_attr)
        self.count = 0
        self.cur_count = 0
        self.updated_step = -1

    def update(self, val):
        self.val += val
        self.sum += val
        self.count += val.sum()
        self.cur_count += val.sum()
        self.avg = self.sum / self.count

    def step(self):
        self.val = torch.Tensor([0] * self.n_attr)
        self.sum *= self.weight
        self.count *= self.weight
        self.cur_count = 0
        self.avg = self.sum / self.count

    def get_val(self):
        return self.val / (self.val.sum() + 1e-8)

    def get_discrepancy(self, target_ratio=None):
        if target_ratio is None:
            target_ratio = 1 / self.n_attr
        return max(self.get_val()-target_ratio) - min(self.get_val()-target_ratio)

    def get_mse(self, target_ratio=None):
        if target_ratio is None:
            target_ratio = 1 / self.n_attr
        return torch.mean((self.get_val() - target_ratio)**2)

    def __str__(self):
        val = ' '.join(['{:.3f}'.format(v) for v in self.val / self.val.sum()])
        avg = ' '.join(['{:.3f}'.format(a) for a in self.avg])
        log_str = "{} r: {} (# of current images: {}) (weighted r: {})"\
            .format(self.name, val, self.cur_count, avg)
        return log_str

    def copy(self, meter):
        self.val = meter.val
        self.avg = meter.avg
        self.sum = meter.sum
        self.count = meter.count
