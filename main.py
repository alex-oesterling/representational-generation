

import torch
import numpy as np
import datetime
from argument import get_args
import pickle
import wandb

import os
os.environ["WANDB_MODE"]="offline"

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, getMPR, feature_extraction, make_result_path, group_estimation, compute_similarity, check_log_dir


def eval(args):

    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)
    profession_set = query_loader.dataset.profession_set

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
    vision_encoder = vision_encoder.cuda() if torch.cuda.is_available() else vision_encoder
    
    ## Make embedding vectors
    print('extract embeddings from the reference distribution')
    refer_embedding = feature_extraction(vision_encoder, refer_loader, args, query=False)
    print('extract embeddings from the query distribution')
    query_embedding, profession_labels = feature_extraction(vision_encoder, query_loader, args, query=True)
    print(query_embedding.shape, profession_labels.shape)

    ## Estimate group labels
    refer_g_embedding = group_estimation(refer_embedding, args.vision_encoder, args.group)
    query_g_embedding = group_estimation(query_embedding, args.vision_encoder, args.group)
    print('Complete estimating group labels')
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
    
    if args.pool_size != 1.0:
        total = query_embedding.shape[0]
        if args.pool_size < 1:
            n_samples = int(query_embedding.shape[0]*args.pool_size)
        else:
            n_samples = int(args.pool_size)
        idx = np.random.choice(total, n_samples, replace=False)
        query_embedding = query_embedding[idx]
        query_g_embedding = query_g_embedding[idx]
        profession_labels = profession_labels[idx]
        print('Pool size is reduced to ', args.pool_size)

    # compute CLIP similarity
    s = compute_similarity(query_embedding, profession_labels, profession_set, vision_encoder, args.vision_encoder)        
        
    # compute MPR
    MPR_dic = {}
    norMPR_dic = {}
    score_dic = {}
    idx_dic = {}
    for i, profession in enumerate(profession_set):
        if profession not in ['firefighter', 'CEO']:
           continue

        profession_idx = profession_labels == i
        if args.retrieve:
            idx, MPR_list, noretrieve_MPR_list, score_list = _retriever.retrieve(query_g_embedding[profession_idx], refer_g_embedding, k=args.k, s=s[profession_idx])
            # score = np.sum(s[profession_idx][idx.astype(dtype=bool)])
            idx_dic[profession] = idx
            MPR_dic[profession] = MPR_list
            score_dic[profession] = score_list
            norMPR_dic[profession] = noretrieve_MPR_list
            print(f'Profession: {profession}, MPR: {noretrieve_MPR_list[-1]}, score: {score_list[-1]}')
            
        else:
            idx = None                
            score = np.sum(s[profession_idx])
            MPR, c = getMPR(query_g_embedding[profession_idx], curation_set=refer_g_embedding, indices=idx, modelname=args.functionclass)
            MPR_dic[profession] = MPR
            score_dic[profession] = score
            print(f'Profession: {profession}, final MPR: {MPR}, score: {score}')

    log_path =  make_result_path(args)
    filename = args.time + '_' + wandb.run.id
    with open(os.path.join(log_path,filename+'_MPR.pkl'), 'wb') as f:
        pickle.dump(MPR_dic, f)
    with open(os.path.join(log_path,filename+'_score.pkl'), 'wb') as f:
        pickle.dump(score_dic, f)
    
    if args.retrieve:
        with open(os.path.join(log_path,filename+'_norMPR.pkl'), 'wb') as f:
            pickle.dump(norMPR_dic, f)
        with open(os.path.join(log_path,filename+'_idx.pkl'), 'wb') as f:
            pickle.dump(idx_dic, f)

    ## Compute FID

    ## IS


def train(args):
    # train_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)

    dm = networks.ModelFactory.get_model(modelname=args.target_model)

    _trainer = trainer.TrainerFactory.get_trainer(trainername=args.trainer, model=dm, args=args)
    _trainer.train()

    model =_trainer.model
    # save model
    save_dir = f'trained_models/{args.trainer}'
    check_log_dir(save_dir)

    groupname = [_g[0] for _g in args.group]
    groupname = "".join(groupname)
    filename = f'{args.date}_{groupname}'
    torch.save(model, os.path.join(save_dir, f'{filename}.pt'))

    # if args.save_model:
    #     args_, save_dir, log_name = check_dirs(args_)
    # else:
    #     save_dir = None
    #     log_name = None

    # 

    # Get the required model

    # Train the model

    # wandb.finish()

if __name__ == '__main__':

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
