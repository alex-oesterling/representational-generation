

import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import sys
import os
os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, getMPR, feature_extraction, make_result_path, group_estimation, compute_similarity, check_log_dir


def eval(args):
    print(args.mpr_group)
    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)
    profession_set = query_loader.dataset.profession_set

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder, train=args.train)  
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    
    ## Make embedding vectors
    print('extract embeddings from the reference distribution')
    refer_embedding = feature_extraction(vision_encoder, refer_loader, args, query=False)
    print('extract embeddings from the query distribution')
    query_embedding, profession_labels = feature_extraction(vision_encoder, query_loader, args, query=True)

    ## Estimate group labels
    refer_g_embedding = group_estimation(refer_embedding, args.vision_encoder, args.mpr_group, args.mpr_onehot)
    query_g_embedding = group_estimation(query_embedding, args.vision_encoder, args.mpr_group, args.mpr_onehot)
    print('Complete estimating group labels')
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
    
    MPR_dic = []
    norMPR_dic =[]
    score_dic = []
    idx_dic = []

    if args.bootstrapping and args.n_resampling == 1:
        raise ValueError('the number of resampling should be larger 1 for bootstrapping')
    
    if args.pool_size != 1.0:
        total = query_embedding.shape[0]
        if args.pool_size < 1:
            n_samples = int(query_embedding.shape[0]*args.pool_size)
        else:
            n_samples = int(args.pool_size)

        # idx = np.random.choice(total, n_samples, replace=False)
        idx = np.arange(n_samples)
        print('Pool size is reduced to ', args.pool_size)

        query_embedding = query_embedding[idx]
        query_g_embedding = query_g_embedding[idx]
        profession_labels = profession_labels[idx]

    s = compute_similarity(query_embedding, profession_labels, profession_set, vision_encoder, args.vision_encoder)        
    if not args.bootstrapping:
        args.n_resampling = 1
    for j in range(args.n_resampling):
        if args.bootstrapping:
            n_samples = query_embedding.shape[0] // 2
            resampling_idx = np.random.choice(n_samples, n_samples, replace=True)
            query_embedding_split = query_embedding[resampling_idx]
            query_g_embedding_split = query_g_embedding[resampling_idx]
            profession_labels_split = profession_labels[resampling_idx]
            s_split = s[resampling_idx]
        else:
            query_embedding_split = query_embedding
            query_g_embedding_split = query_g_embedding
            profession_labels_split = profession_labels
            s_split = s

        # compute CLIP similarity


        # compute MPR

        if args.retrieve:
            retrieved_idx, _, MPR, score = _retriever.retrieve(query_g_embedding_split, refer_g_embedding, k=args.k, s=s)
            # score = np.sum(s[profession_idx][idx.astype(dtype=bool)])
            idx_dic.append(idx[retrieved_idx])
            # norMPR_dic.append(noretrieve_MPR_list)
            print(f'Profession: {args.target_profession}, MPR: {noretrieve_MPR_list[-1]}, score: {score_list[-1]}')
            
        else:
            score = np.sum(s_split)
            MPR, c = getMPR(query_g_embedding_split, curation_set=refer_g_embedding, modelname=args.functionclass, groups=args.mpr_group)
            print(f'Profession: {args.target_profession}, final MPR: {MPR}, score: {score}')
        MPR_dic.append(MPR)
        score_dic.append(score)

    log_path =  make_result_path(args)
    filename = args.time + '_' + wandb.run.id
    results = {}
    results['MPR'] = MPR_dic
    results['score'] = score_dic
    results['optimal_c'] = c
    if args.retrieve: 
        results['idx'] = idx_dic
    elif args.bootstrapping:
        results['idx'] = idx[resampling_idx]
    else:
        results['idx'] = idx

    if args.bootstrapping:
        mean = np.mean(MPR_dic)
        std = np.std(MPR_dic)
        print(f'Mean MPR: {mean}, std: {std}')
        
    with open(os.path.join(log_path,filename+'.pkl'), 'wb') as f:
        pickle.dump(results, f)
    # with open(os.path.join(log_path,filename+'_MPR.pkl'), 'wb') as f:
    #     pickle.dump(MPR_dic, f)
    # with open(os.path.join(log_path,filename+'_score.pkl'), 'wb') as f:
    #     pickle.dump(score_dic, f)
    # with open(os.path.join(log_path,filename+'_idx.pkl'), 'wb') as f:
    #     pickle.dump(idx_dic, f)

    ## Compute FID

    ## IS


def train(args):
    # train_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)

    dm = networks.ModelFactory.get_model(modelname=args.target_model, train=args.train)

    _trainer = trainer.TrainerFactory.get_trainer(trainername=args.trainer, model=dm, args=args)
    _trainer.train()

    model =_trainer.model
    # save model
    save_dir = f'trained_models/{args.trainer}'
    check_log_dir(save_dir)

    groupname = [_g[0] for _g in args.trainer_group]
    groupname = "".join(groupname)
    filename = f'{args.date}_{groupname}'
    torch.save(model, os.path.join(save_dir, f'{filename}.pt'))
    
    # Get the required model

    # Train the model

    wandb.finish()

if __name__ == '__main__':

    print(" ".join(sys.argv))

    # check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Print out additional information when using CUDA
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)

    now = datetime.datetime.now()
    # Format as 'ddmmyyHMS'
    formatted_time = now.strftime('%H%M')
    if args.date == 'default':
        args.date = now.strftime('%m%d%y')
    args.time = formatted_time

    run = wandb.init(
            project='mpr_generative',
            entity='sangwonjung-Harvard University',
            name=args.date+'_'+formatted_time,
            settings=wandb.Settings(start_method="fork")
    )
    print('wandb mode : ',run.settings.mode)
    
    wandb.config.update(args)

    if args.train:
        train(args)
    else:
        eval(args)

    wandb.finish()
