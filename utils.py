import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from transformers import BlipProcessor
import random 

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


def oracle_function(indices, dataset, curation_set=None, modelname='linear'):
    if modelname == 'linear':
        model = LinearRegression()
    elif 'dt' in modelname:
        if len(modelname) > 2:
            max_depth = int(modelname[2:])
        else: 
            max_depth = 2
        model = RandomForestRegressor(max_depth=max_depth, random_state=0)
    elif modelname == 'l2':
        return alpha
    else:
        raise ValueError('Only linear and dt are supported for now')

    k = int(np.sum(indices))
    if curation_set is not None:
        m = curation_set.shape[0]
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
        a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
        m = curation_set.shape[0]
        alpha = (a_expanded/k - curation_indicator/m)
        reg = model.fit(expanded_dataset, alpha) 
    else:
        m = dataset.shape[0]
        alpha = (indices/k - 1/m)
        reg = model.fit(dataset, alpha)
    return reg

def getMPR(dataset, k=0, curation_set=None, modelname=None, indices=None):
    if indices is None:
        indices = np.ones(dataset.shape[0])
    k = int(np.sum(indices))

    reg = oracle_function(indices, dataset, curation_set=curation_set, modelname=modelname)
    if curation_set is not None:
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        m = curation_set.shape[0]
        c = reg.predict(expanded_dataset)
        c /= np.linalg.norm(c)
        # c *= np.sqrt(c.shape[0]) ## sqrt(n+m) = 141   
        # print(np.sqrt(c.shape[0]), np.sqrt(m*k/(m+k)))
        c *= np.sqrt(m*k/(m+k))
        mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))
    else:
        m = dataset.shape[0]
        c = reg.predict(dataset)
        c /= np.linalg.norm(c)
        c *= np.sqrt(c.shape[0]) ## sqrt(n) = 100
        mpr = np.abs(np.sum((indices/k)*c) - np.sum((1/m)*c))
    
    return mpr, c

def feature_extraction(encoder, dataloader, args, query=True):
    encoder.eval()
    encoder = encoder.cuda() if torch.cuda.is_available() else encoder

    if args.vision_encoder == 'BLIP':
        return _blip_extraction(encoder, dataloader, args, query)
    
    elif args.vision_encoder == 'CLIP':
        return _clip_extraction(encoder, dataloader, args, query)

# old version
def _blip_extraction(encoder, dataloader, args, query=True):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    raw_text=["What word best describes the person's appearance?"]
    text = processor(text=raw_text, return_tensors='pt')
    input_ids=text['input_ids']
    attention_mask=text['attention_mask']
    
    raw_dummy_label = ['0']
    dummy_label = processor(text=raw_dummy_label, return_tensors='pt')['input_ids']

    outputs = []
    professions = []
        
    # professions = []
    for batch in dataloader:
        with torch.no_grad():
            image = batch[0]
            batch_size = image.shape[0]
            
            _input_ids = input_ids.repeat(batch_size, 1)
            _attention_mask = attention_mask.repeat(batch_size, 1)
            _dummy_label = dummy_label.repeat(batch_size, 1)

            if torch.cuda.is_available():
                image = image.cuda()
                _input_ids = _input_ids.cuda()
                _attention_mask = _attention_mask.cuda()
                _dummy_label = _dummy_label.cuda()
            
            output = encoder.forward(pixel_values=image, input_ids=_input_ids, attention_mask=_attention_mask, labels=_dummy_label, return_dict=True)
            outputs.append(output['image_embeds'])
            
            if query:
                professions.append(batch[1])

    outputs = torch.cat(outputs)
    outputs = outputs.mean(axis=1)
    return outputs if not query else (outputs, torch.cat(professions))

def group_estimation(features, vision_encoder_name, group=['gender','age','race']):
    path = '/n/holyscratch01/calmon_lab/Lab/datasets/mpr_stuffs/'
    estimated_group_list = []
    for g in group:
        with open(os.path.join(path,'clfs',f'fairface_{vision_encoder_name}_clf_{g}.pkl'), 'rb') as f:
            clf = pickle.load(f)
            estimated_group = clf.predict_proba(features)
            # one_hot_indices = np.argmax(estimated_group, axis=1)
            # estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]            
            estimated_group_list.append(estimated_group)

    # with open(os.path.join(path,'clfs',f'fairface_{vision_encoder_name}_clf_gender.pkl'), 'rb') as f:
        # clf_gender = pickle.load(f)
    # with open(os.path.join(path,'clfs',f'fairface_{vision_encoder_name}_clf_race.pkl'), 'rb') as f:
        # clf_race = pickle.load(f)
        
    # ages = clf_age.predict_proba(features)
    # genders = clf_gender.predict_proba(features)
    # races = clf_race.predict_proba(features)
    if len(estimated_group_list) > 1:
        outputs = np.concatenate(estimated_group_list, axis=1)
    else:
        outputs = np.array(estimated_group_list[0])
    # for i in range(outputs.shape[0]):
        # outputs[i, :] = outputs[i, :] / np.linalg.norm(outputs[i, :])
    return outputs

def compute_similarity(visual_features, profession_labels, profession_set, vision_encoder, vision_encoder_name='clip'):
    if vision_encoder_name != 'CLIP':
        raise ValueError('Only CLIP is supported for now')
    similarity = np.zeros(visual_features.shape[0])

    visual_features = visual_features / np.linalg.norm(visual_features, axis=-1, keepdims=True) 
    vision_encoder.eval()
    for i, profession in enumerate(profession_set):
        with torch.no_grad():        
            profession_idx = profession_labels == i
            prompt = f'photo portrait of {profession}'
            text = clip.tokenize(prompt)
            if torch.cuda.is_available():
                text = text.cuda()
            text_embedding = vision_encoder.encode_text(text).float()
            text_embedding = text_embedding.cpu().numpy().squeeze()
            
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            similarity[profession_idx] = visual_features[profession_idx] @ text_embedding
    return similarity
    
def _clip_extraction(encoder, dataloader, args, query=True):
    dataset_name = args.refer_dataset if not query else args.query_dataset
    # dataset_name += f'_{args.target_profession}' if args.target_profession != 'all' and query else ''
    # path = '/n/holyscratch01/calmon_lab/Lab/datasets/mpr_stuffs/'
    path = dataloader.dataset.datapath
    filename = f'{args.vision_encoder}_embedding.pkl'
    # if query:
    #     pre_name = f'query_embedding/{args.target_model}_' 
    #     filename = pre_name + filename
    # else:
    #     filename = f'refer_embedding/' + filename
    print(os.path.join(path,filename))

    if os.path.exists(os.path.join(path,filename)):
        with open(os.path.join(path,filename), 'rb') as f:
            outputs =  pickle.load(f)
        print(f'embedding vectors of {dataset_name} are successfully loaded in {path}')
        return outputs

    professions = []
    features = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            image = batch[0]
            
            if torch.cuda.is_available():
                image = image.cuda()
            feature = encoder.encode_image(image)
            features.append(feature.cpu())
            
            if query:
                professions.append(batch[1])
    
    features = torch.cat(features)
    outputs = features.numpy()

    with open(os.path.join(path, filename), 'wb') as f:
        if query:
            pickle.dump((outputs, torch.cat(professions)), f)
        else:
            pickle.dump(outputs, f)

    return outputs if not query else (outputs, torch.cat(professions))

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
