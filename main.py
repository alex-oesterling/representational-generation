import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import os
os.environ["WANDB_MODE"] = "offline"

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, getMPR, feature_extraction, make_result_path, group_estimation, compute_similarity

def eval(args):
    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)
    profession_set = query_loader.dataset.profession_set if hasattr(query_loader.dataset, 'profession_set') else ['firefighter']

    # Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    
    # Make embedding vectors
    print('Extract embeddings from the reference distribution')
    refer_embedding = feature_extraction(vision_encoder, refer_loader, args, query=False)
    print('Extract embeddings from the query distribution')
    query_embedding, profession_labels = feature_extraction(vision_encoder, query_loader, args, query=True)
    print(query_embedding.shape, profession_labels.shape)

    # Estimate group labels
    refer_g_embedding = group_estimation(refer_embedding, args.vision_encoder, args.group)
    query_g_embedding = group_estimation(query_embedding, args.vision_encoder, args.group)
    print('Complete estimating group labels')
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
        s = compute_similarity(query_embedding, profession_labels, profession_set, vision_encoder, args.vision_encoder)

    if args.pool_size != 1.0:
        total = query_embedding.shape[0]
        if args.pool_size < 1:
            n_samples = int(query_embedding.shape[0] * args.pool_size)
        else:
            n_samples = int(args.pool_size)
        idx = np.random.choice(total, n_samples, replace=False)
        query_g_embedding = query_g_embedding[idx]
        if args.retrieve:
            s = s[idx]
        profession_labels = profession_labels[idx]
        print('Pool size is reduced to ', args.pool_size)
    
    MPR_dic = {}
    score_dic = {}
    idx_dic = {}
    for i, profession in enumerate(profession_set):
        if profession not in ['firefighter', 'CEO']:  # Use the correct labels for your custom dataset
            continue

        profession_idx = (profession_labels[:, 0] == i)  # Assuming first column is profession index

        if args.retrieve:
            # Ensure profession_idx is used correctly to index 1D arrays
            query_g_embedding_profession = query_g_embedding[profession_idx]
            s_profession = s[profession_idx]

            idx, MPR_list, score_list = _retriever.retrieve(query_g_embedding_profession, refer_g_embedding, k=args.k, s=s_profession)
            idx_dic[profession] = idx
            MPR_dic[profession] = MPR_list
            score_dic[profession] = score_list
            print(f'Profession: {profession}, MPR: {MPR_list[-1]}, score: {score_list[-1]}')
        else:
            idx = None                
            MPR, c = getMPR(query_g_embedding[profession_idx], curation_set=refer_g_embedding, indices=idx, modelname=args.functionclass)
            MPR_dic[profession] = MPR
            print(f'Profession: {profession}, final MPR: {MPR}')

    log_path = make_result_path(args)
    filename = args.time + '_' + wandb.run.id
    with open(os.path.join(log_path, filename + '_MPR.pkl'), 'wb') as f:
        pickle.dump(MPR_dic, f)
    
    if args.retrieve:
        with open(os.path.join(log_path, filename + '_score.pkl'), 'wb') as f:
            pickle.dump(score_dic, f)
        with open(os.path.join(log_path, filename + '_idx.pkl'), 'wb') as f:
            pickle.dump(idx_dic, f)

def train(args):
    train_loader, val_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    dm = networks.ModelFactory.get_model(modelname=args.model)

if __name__ == '__main__':
    args = get_args()    
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    set_seed(args.seed)
    now = datetime.datetime.now()
    formatted_time = now.strftime('%H%M')
    if args.date != 'test':
        args.date = now.strftime('%m%d%y')
    args.time = formatted_time

    run = wandb.init(
            project='mpr_generative',
            entity='sangwonjung-Harvard University',
            name=args.date + '_' + formatted_time,
            settings=wandb.Settings(start_method="fork")
    )
    print('wandb mode : ', run.settings.mode)
    wandb.config.update(args)
    if args.train:
        train(args)
    else:
        eval(args)
    wandb.finish()
