

import torch
import numpy as np
from argument import get_args
import pickle
import wandb

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, getMPR, feature_extraction, make_result_path, group_estimation, compute_similarity

def eval(args):
    
    log_path =  make_result_path(args)

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
    refer_g_embedding = group_estimation(refer_embedding, args.vision_encoder)
    query_g_embedding = group_estimation(query_embedding, args.vision_encoder)
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever, args)
        s = compute_similarity(query_embedding, profession_labels, profession_set, vision_encoder, args.vision_encoder)
        
    MPR_dic = {}
    score_dic = {}
    idx_dic = {}
    for i, profession in enumerate(profession_set):
        if profession not in ['firefighter', 'CEO']:
           continue
        profession_idx = profession_labels == i
        if args.retrieve:
            idx, MPR_list, score_list = _retriever.retrieve(query_g_embedding[profession_idx], refer_g_embedding, k=args.k, s=s[profession_idx])
            # score = np.sum(s[profession_idx][idx.astype(dtype=bool)])
            idx_dic[profession] = idx
            MPR_dic[profession] = MPR_list
            score_dic[profession] = score_list
            print(f'Profession: {profession}, MPR: {MPR_list[-1]}, score: {score_list[-1]}')
            
        else:
            idx = None                
            score = np.sum(s[profession_idx])
            MPR, c = getMPR(query_g_embedding[profession_idx], curation_set=refer_g_embedding, indices=idx, modelname=args.functionclass)
            MPR_dic[profession] = MPR
            print(f'Profession: {profession}, final MPR: {MPR}')
    
    with open(log_path+'_MPR.pkl', 'wb') as f:
        pickle.dump(MPR_dic, f)

    if args.retrieve:
        with open(log_path+'_score.pkl', 'wb') as f:
            pickle.dump(score_dic, f)
        with open(log_path+'_idx.pkl', 'wb') as f:
            pickle.dump(idx_dic, f)

    ## Compute FID

    ## IS


def train(args):
    train_loader, val_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)



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

    wandb.init(
            project='mpr_generative',
            entity='fair_continual',
            name=args.date,
            settings=wandb.Settings(start_method="fork")
    )

    wandb.config.update(args)

    if args.train:
        train(args)
    else:
        eval(args)

    wandb.finish()
