

import torch
import numpy as np
from argument import get_args
import pickle

import data_handler
import networks
import trainer
import retriever
from utils import set_seed, getMPR, feature_extraction, make_result_path

def eval(args_):
    
    refer_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)
    query_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.query_dataset, args=args)

    ## Compute MPR
    vision_encoder = networks.ModelFactory.get_model(modelname=args.vision_encoder)  
    
    print('extract embeddings from the reference distribution')
    refer_embedding = feature_extraction(vision_encoder, refer_loader, args, query=False)
    print('extract embeddings from the query distribution')
    query_embedding, profession_labels = feature_extraction(vision_encoder, query_loader, args, query=True)
    
    if args.retrieve:
        _retriever = retriever.RetrieverFactory.get_retriever(args.retriever)

    MPR_dic = {}
    if args.target_profession == 'all':
        for i, profession in enumerate(query_loader.dataset.profession_set):
            prfession_idx = profession_labels == i
            if args.retrieve:
                idx = _retriever.retrieve(query_embedding[prfession_idx], refer_embedding, k=args.k)
            MPR, c = getMPR(query_embedding[prfession_idx], curation_set=refer_embedding, indices=idx)
            MPR_dic[profession] = MPR
            print(f'Profession: {profession}, MPR: {MPR}')  
    else:
        if args.retrieve:
            MPR,_ = _retriever.fit(query_embedding, refer_embedding, k=args.k)
        else:
            MPR, c = getMPR(query_embedding, curation_set=refer_embedding)
        print(f'Profession: {args.target_profession}, MPR: {MPR}')
        
    log_path =  make_result_path(args)   
    
    ## Compute FID

    ## IS

    with open(log_path+'.pkl', 'wb') as f:
        pickle.dump(MPR_dic, f)


def train(args_):
    train_loader, val_loader, test_loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.refer_dataset, args=args)

    # wandb.init(
    #         project='fair_continual',
    #         entity='fair_continual',
    #         name=args.date,
    #         settings=wandb.Settings(start_method="fork")
    # )


    # if args.save_model:
    #     args_, save_dir, log_name = check_dirs(args_)
    # else:
    #     save_dir = None
    #     log_name = None

    # wandb.config.update(args_)

    # Get the required model

    # Train the model

    # wandb.finish()


if __name__ == '__main__':

    args = get_args()    

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    set_seed(args.seed)


    if args.train:
        train(args)
    else:
        eval(args)
