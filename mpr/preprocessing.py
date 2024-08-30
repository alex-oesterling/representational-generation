
import torch
import numpy as np 
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from transformers import BlipProcessor
import random 
import itertools
import pickle
from tqdm import tqdm
import clip
import cv2
import torchvision 

class CLIPExtractor:
    def __init__(self, encoder, args):
        self.encoder = encoder
        self.args = args

    def extract(self, images, query=True):
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            outputs = self.encoder.encode_image(images)
        return outputs
    
def identity_embedding(args, encoder, dataloader, groups, query=True):
    dataset_name = args.refer_dataset if not query else args.query_dataset
    path = dataloader.dataset.dataset_path
    
    feature_dic = {}
    for ver in ['normal', 'face_detect']:
        # make feature vectors
        if ver == 'face_detect':
            dataloader.dataset.turn_on_detect()

        filename = f'{args.vision_encoder}_{ver}_feature.pkl'
        filepath = os.path.join(path,filename)
        print(filepath)

        save_flag = False
        features = []
        if os.path.exists(filepath):
            with open(os.path.join(path,filename), 'rb') as f:
                feature_dic[ver] =  pickle.load(f)
            print(f'embedding vectors of {dataset_name} are successfully loaded in {path}')
            continue
        else:
            save_flag = True

        encoder.eval()
        encoder = encoder.cuda() if torch.cuda.is_available() else encoder

        feature_extractor = None

        if args.vision_encoder == 'BLIP':
            feature_extractor = BlipExtractor(encoder, args)
        
        elif args.vision_encoder == 'CLIP':
            feature_extractor =  CLIPExtractor(encoder, args)
    
        for batch in tqdm(dataloader):
            image, label, idxs =  batch

            if torch.cuda.is_available():
                image = image.cuda()

            feature = feature_extractor.extract(image)
            features.append(feature.cpu())
        
        features = torch.cat(features).numpy()
        feature_dic[ver] = features

        if ver == 'face_detect':
            dataloader.dataset.turn_off_detect()
        
        if save_flag:
            with open(filepath, 'wb') as f:
                pickle.dump(features, f)

    # group estimation
    estimated_groups = []
    for group in groups:
        if group in ['gender', 'race', 'age', 'face', 'skintone', 'emotion']:
            feature = feature_dic['face_detect']
        elif group in ['background', 'house']:
            feature = feature_dic['normal']
        else:
            raise ValueError(f'group {group} is not supported')

        estimated_group = group_estimation(feature,group, args.vision_encoder, onehot=args.mpr_onehot, loader=dataloader)
        estimated_groups.append(estimated_group)
    estimated_groups = np.concatenate(estimated_groups, axis=1)
    print(estimated_groups.shape)
    return estimated_groups, feature_dic
            
def group_estimation(features, group='gender', vision_encoder_name='CLIP', onehot=False, loader=None):
    path = '/n/holyscratch01/calmon_lab/Lab/datasets/mpr_stuffs/'
    if group in ['gender', 'age','race']:
        with open(os.path.join(path,'clfs',f'fairface_{vision_encoder_name}_clf_{group}.pkl'), 'rb') as f:
            clf = pickle.load(f)
            estimated_group = clf.predict_proba(features)
            if onehot:
                # if estimated_group.shape[-1] == 1:
                    # estimated_group = estimated_group>0.5
                # else:
                one_hot_indices = np.argmax(estimated_group, axis=1)
                estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
    elif group == 'face':
        with open(os.path.join(path,'clfs',f'celeba_{vision_encoder_name}_clf_{group}.pkl'), 'rb') as f:                
            clf = pickle.load(f)
        estimated_group_list = []
        attrs = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Double_Chin', 'Eyeglasses',
            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Mustache', 'No_Beard', 'Sideburns', 'Smiling', 'Wearing_Hat']
        for attr in attrs:
            estimated_group = clf[attr].predict_proba(features)
            print(attr)
            if onehot:
                one_hot_indices = np.argmax(estimated_group, axis=1)
                estimated_group = np.eye(estimated_group.shape[1])[one_hot_indices]
                # print statistics
                print(f'{attr} 1/0: {np.sum(one_hot_indices)}/{len(one_hot_indices)-np.sum(one_hot_indices)}')
            estimated_group_list.append(estimated_group)
        estimated_group = np.concatenate(estimated_group_list, axis=1)

    # elif g == 'skintone':
    #     from skintone_esti import FanCrop
    #     face_detector = FanCrop()
    #     for image, label in loader:
    #         valid_idxs, images = face_detector.crop_images(image)

            
    estimated_group = estimated_group * 2 - 1
    
    # normalization
    # outputs = outputs / np.linalg.norm(outputs, axis=-1, keepdims=True)    
    # for i in range(outputs.shape[0]):
        # outputs[i, :] = outputs[i, :] / np.linalg.norm(outputs[i, :])

    return estimated_group


# old version
# def _blip_extraction(encoder, dataloader, args, query=True):
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
#     raw_text=["What word best describes the person's appearance?"]
#     text = processor(text=raw_text, return_tensors='pt')
#     input_ids=text['input_ids']
#     attention_mask=text['attention_mask']
    
#     raw_dummy_label = ['0']
#     dummy_label = processor(text=raw_dummy_label, return_tensors='pt')['input_ids']

#     outputs = []
#     professions = []
        
#     # professions = []
#     for batch in dataloader:
#         with torch.no_grad():
#             image = batch[0]
#             batch_size = image.shape[0]
            
#             _input_ids = input_ids.repeat(batch_size, 1)
#             _attention_mask = attention_mask.repeat(batch_size, 1)
#             _dummy_label = dummy_label.repeat(batch_size, 1)

#             if torch.cuda.is_available():
#                 image = image.cuda()
#                 _input_ids = _input_ids.cuda()
#                 _attention_mask = _attention_mask.cuda()
#                 _dummy_label = _dummy_label.cuda()
            
#             output = encoder.forward(pixel_values=image, input_ids=_input_ids, attention_mask=_attention_mask, labels=_dummy_label, return_dict=True)
#             outputs.append(output['image_embeds'])
            
#             if query:
#                 professions.append(batch[1])

#     outputs = torch.cat(outputs)
#     outputs = outputs.mean(axis=1)
#     return outputs if not query else (outputs, torch.cat(professions))

