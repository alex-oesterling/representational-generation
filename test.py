

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
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
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
    
    base_filename = f'group_labels/{args.target_model}_{args.target_profession}_{args.functionclass}'
    base_filename += '_onehot' if args.mpr_onehot else ''
    # with open(f'group_labels/{args.target_profession}_refer_group_labels.pkl', 'wb') as f:
        # pickle.dump(refer_g_embedding,f)
    # with open(f'group_labels/{args.target_profession}_refer_group_labels_true.pkl', 'wb') as f:
        # pickle.dump(refer_loader.dataset.labels.numpy(),f)
    with open(base_filename+'_group_labels.pkl', 'wb') as f:
        pickle.dump(query_g_embedding,f)
    s = compute_similarity(query_embedding, profession_labels, profession_set, vision_encoder, args.vision_encoder)        
    with open(base_filename+'_scores.pkl', 'wb') as f:
        pickle.dump(s,f)
    
    if args.retrieve: 
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
    
    MPR_dic = {}
    norMPR_dic = {}
    score_dic = {}
    idx_dic = {}

    # n_compute_mpr = args.n_compute_mpr if args.pool_size != 21000.0 or args.pool_size!=1.0  else 1
    n_compute_mpr = 1
    for _ in range(n_compute_mpr):
        if args.pool_size != 1.0:
            total = query_embedding.shape[0]
            if args.pool_size < 1:
                n_samples = int(query_embedding.shape[0]*args.pool_size)
            else:
                n_samples = int(args.pool_size)

            idx = np.random.choice(total, n_samples, replace=False)
            # print(idx[:20])
            print('Pool size is reduced to ', args.pool_size)

            query_embedding_split = query_embedding[idx]
            query_g_embedding_split = query_g_embedding[idx]
            profession_labels_split = profession_labels[idx]

        else:
            query_embedding_split = query_embedding
            query_g_embedding_split = query_g_embedding
            profession_labels_split = profession_labels

    #     # compute CLIP similarity
    #     s = compute_similarity(query_embedding_split, profession_labels_split, profession_set, vision_encoder, args.vision_encoder)        
        
    # compute MPR
        for i, profession in enumerate(profession_set):
            if profession not in ['firefighter', 'CEO']:
                continue
            if profession not in MPR_dic.keys():
                MPR_dic[profession] = []
                norMPR_dic[profession] = []
                score_dic[profession] = []
                idx_dic[profession] = []

            profession_idx = profession_labels_split == i
            if args.retrieve:
                idx, MPR_list, noretrieve_MPR_list, score_list = _retriever.retrieve(query_g_embedding_split[profession_idx], refer_g_embedding, k=args.k, s=s[profession_idx])
                # score = np.sum(s[profession_idx][idx.astype(dtype=bool)])
                idx_dic[profession].append(idx)
                MPR_dic[profession].append(MPR_list)
                score_dic[profession].append(score_list)
                norMPR_dic[profession].append(noretrieve_MPR_list)
                print(f'Profession: {profession}, MPR: {noretrieve_MPR_list[-1]}, score: {score_list[-1]}')
                
            else:
                idx = None                
                score = np.sum(s[profession_idx])
                MPR, reg = getMPR(query_g_embedding_split[profession_idx], curation_set=refer_g_embedding, indices=idx, modelname=args.functionclass)
                MPR_dic[profession].append(MPR)
                score_dic[profession].append(score)
                print(f'Profession: {profession}, final MPR: {MPR}, score: {score}')
                with open(base_filename+'_weight.pkl', 'wb') as f:
                    pickle.dump(reg,f)


    # log_path =  make_result_path(args)
    # filename = args.time + '_' + wandb.run.id
    # with open(os.path.join(log_path,filename+'_MPR.pkl'), 'wb') as f:
    #     pickle.dump(MPR_dic, f)
    # with open(os.path.join(log_path,filename+'_score.pkl'), 'wb') as f:
    #     pickle.dump(score_dic, f)
    
    # if args.retrieve:
    #     with open(os.path.join(log_path,filename+'_norMPR.pkl'), 'wb') as f:
    #         pickle.dump(norMPR_dic, f)
    #     with open(os.path.join(log_path,filename+'_idx.pkl'), 'wb') as f:
    #         pickle.dump(idx_dic, f)

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

    # run = wandb.init(
    #         project='mpr_generative',
    #         entity='sangwonjung-Harvard University',
    #         name=args.date+'_'+formatted_time,
    #         settings=wandb.Settings(start_method="fork")
    # )
    # print('wandb mode : ',run.settings.mode)
    
    # wandb.config.update(args)

    if args.train:
        train(args)
    else:
        eval(args)

    # wandb.finish()
