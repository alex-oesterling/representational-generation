import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from transformers import BlipProcessor
import random 

import pickle
from tqdm import tqdm

def make_result_path(args):
    filename = f'{args.query_dataset}_{args.refer_dataset}_{args.vision_encoder}_{args.target_model}_{args.target_profession}'
    filename += f'_{args.retriever}_{args.k}' if args.retrieve else ''
    save_dir = os.path.join(args.save_dir, args.date)
    check_log_dir(save_dir)
    
    return os.path.join(save_dir, filename)

def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")

def set_seed(seed): 
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def oracle_function(indices, dataset, curation_set=None, model=None):
    if model is None:
        model = LinearRegression()

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

def getMPR(dataset, k=0, curation_set=None, model=None, indices=None):
    if indices is None:
        indices = np.ones(dataset.shape[0])
    k = int(np.sum(indices))

    reg = oracle_function(indices, dataset, curation_set=curation_set, model=model)
    if curation_set is not None:
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        m = curation_set.shape[0]
        c = reg.predict(expanded_dataset)
        c /= np.linalg.norm(c)
        c *= np.sqrt(c.shape[0]) ## sqrt(n+m) = 141
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
    
def _clip_extraction(encoder, dataloader, args, query=True):
    dataset_name = args.refer_dataset if not query else args.query_dataset
    filename = f'./stuffs/{args.vision_encoder}_{dataset_name}_embedding.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            outputs =  pickle.load(f)
        print(f'embedding vectors of {dataset_name} are successfully loaded')
        return outputs

    genders = []
    ages = []
    races = []
    professions = []

    with open('stuffs/clf_age.pkl', 'rb') as f:
        clf_age = pickle.load(f)
    with open('stuffs/clf_gender.pkl', 'rb') as f:
        clf_gender = pickle.load(f)
    with open('stuffs/clf_race.pkl', 'rb') as f:
        clf_race = pickle.load(f)

    for batch in dataloader:
        with torch.no_grad():
            image = batch[0]
            
            if torch.cuda.is_available():
                image = image.cuda()
            features = encoder.encode_image(image)            
            features = features.cpu().numpy()
            
            agelabels = clf_age.predict_proba(features)
            genderlabels = clf_gender.predict_proba(features)
            racelabels = clf_race.predict_proba(features)

            ages.append(agelabels)
            genders.append(genderlabels)
            races.append(racelabels)
            if query:
                professions.append(batch[1])
    
    ages = np.concatenate(ages)
    genders = np.concatenate(genders)
    races = np.concatenate(races)
    outputs = np.concatenate((genders, ages, races), axis=1)

    for i in range(outputs.shape[0]):
        outputs[i, :] = outputs[i, :] / np.linalg.norm(outputs[i, :])

    with open(filename, 'wb') as f:
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
