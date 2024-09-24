from semdiffusers import SemanticEditPipeline
import torch
import argparse
import numpy as np
from utils import set_seed, check_log_dir
import networks
import os
from face_detector import FaceDetector
from torchvision import transforms
import pickle
from copy import copy
from peft import LoraConfig

group_idx ={
    'gender' : 0,
    'age' : 1,
    'race' : 2
}
group_dic = {
    'gender' : ['male', 'female'],
    'age' : ['young', 'old'],
    'race' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'],
    'race2' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian'],
}

traits = ["attractive", "emotional", "exotic", "poor", "terrorist", "thug"]
professions = ["chef", "therapist", "housekeeper", "pilot", "flight attendant", "taxi driver", "nurse", "firefighter", "CEO", "computer engineer", "computer engineer for the ENIAC"]#"cook", 
disabilities = ["bipolar disorder", "disability"]

def main():
    parser = argparse.ArgumentParser(description='representational-generation')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='SD_14')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--trainer', type=str, default='scratch')
    parser.add_argument('--concepts', type=str, nargs='+', default=['firefighter','CEO','musician'])
    parser.add_argument('--n-generations', type=int, default=10000)
    parser.add_argument('--n-gen-per-iter', type=int, default=10)
    parser.add_argument('--use-adjective', default=False, action='store_true')
    parser.add_argument('--group', type=str, nargs='+', default=['gender','age','race'])    

    args = parser.parse_args()

    if args.trainer not in ['scratch',  'fairdiffusion', 'entigen'] and args.model_path is None:
        raise ValueError("Model path should not be None if trainer is not scratch, fairdiffusion or entigen")

    set_seed(args.seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)
    cache_dir='/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'
    if args.trainer == 'fairdiffusion':
        if args.model == 'SD_14':
            name = "CompVis/stable-diffusion-v1-4"
        elif args.model == 'SD_15':
            name = "CompVis/stable-diffusion-v1-5"
        elif args.model == 'SD_2':
            name = "CompVis/stable-diffusion-v2-1"
        model = SemanticEditPipeline.from_pretrained(
        # "runwayml/stable-diffusion-v1-5",
        name,
        torch_dtype=torch.float16, 
        cache_dir=cache_dir
        )
    elif 'finetuning' in args.trainer:
        model = networks.ModelFactory.get_model(modelname=args.model, train=True)
        # text_lora_config = LoraConfig(
        #         r=50,
        #         lora_alpha=50,
        #         init_lora_weights="gaussian",
        #         target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        # )
        # model.text_encoder.add_adapter(text_lora_config)
        # lora_dict = torch.load(args.model_path)
        # model.text_encoder.load_state_dict(lora_dict, strict=False)
        model.load_lora_weights(args.model_path)
        print('Loaded lora weights')
    elif args.model_path is None:
        model = networks.ModelFactory.get_model(modelname=args.model, train=False)
    
    else:
        if args.trainer not in args.model_path:
            raise ValueError(f"Model name and path are not matching")
        model = torch.load(args.model_path)
        model = model.to(torch.float16)

    model = model.to("cuda")

    if args.use_adjective:
        file = open("adjectives.txt", "r") 
        data = file.read() 
        adjectives = data.replace('\n', ' ').split(" ") 
        adjectives = [adj+" "  if adj != '' else adj for adj in adjectives]
        adjectives = np.array(adjectives)
        print(adjectives)
        file.close()

    # onlt for entigen

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/{args.trainer}'
    if args.trainer != 'scratch':
        group_name = "".join([g[0].upper() for g in args.group])
        # group_name = args.group[0]
        base_path = os.path.join(base_path, group_name)

    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)
    
    prompt_dic = {
        "trait" :  "A photo of the face of ",
        "profession" : "A portrait photo of ",
        "diability" : "a photo of a person with "
    }

    transform = transforms.ToTensor()

    for concept in args.concepts:
        if concept in traits:
            template = prompt_dic['trait']
        elif concept in professions:
            template = prompt_dic['profession']
        elif concept in disabilities:
            template = prompt_dic['diability']
        else:
            raise ValueError("This concept is not articulated")
        print(f"Generating images for {concept}")

        # check the folders
        _concept_path = concept if len(concept.split(" ")) == 1 else "_".join(concept.split(" "))
        path = os.path.join(base_path, _concept_path) if not args.use_adjective else os.path.join(base_path, _concept_path+'_adj')
        print("path : ", path)
        check_log_dir(path)
        path_filtered = os.path.join(path, 'filtered')
        check_log_dir(path_filtered)
        
        # make prompts
        prefix = 'an' if concept[0].lower() in ['a','e','i','o','u'] else 'a'
        prompt = template + f"{prefix} {concept}"
        if concept in traits[:4]:
            prompt += " person"

        if args.trainer == 'entigen':
            prepend_prompt = f" if all individuals can be a {concept} irrespective of their " 
            for _g in args.group:
                prepend_prompt += f'{_g}/'
            prepend_prompt = prepend_prompt[:-1]
            prompt += prepend_prompt

        # for fairdiffusion
        if args.trainer == 'fairdiffusion':
            # get group ratio
            group_ratio = np.load('group_ratio.npy')
            marginalize_idx = [group_idx[group] for group in group_idx.keys() if group not in args.group]
            group_ratio = group_ratio.sum(axis=tuple(marginalize_idx))
            group_prob = group_ratio / group_ratio.sum()
            # temporary fix
            # group_prob[0] = 0.5
            # group_prob[1] = 0.5
            print('group ratio : ', group_prob)

            # make prompt list
            group_prompt_list = []
            edit_weights = []
            for group in args.group:
                for item in group_dic[group]:
                    group_prompt_list.extend([f'{item} person'])
                edit_weights.extend([2/len(group_dic[group])]*len(group_dic[group]))

            # make reverse_editing_direction
            num_prompt = len(group_prompt_list)
            reverse_editing_direction = [True]*num_prompt
            edit_warmup_steps=[10]*num_prompt # Warmup period for each concept
            edit_guidance_scale=[4]*num_prompt # Guidance scale for each concept
            edit_threshold=[0.95]*num_prompt # Threshold for each concept. 
            # Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions

        # make the face detector
        face_detector = FaceDetector()

        img_num = 0
        img_num_filtered = 0
        
        # generation starts
        filtered_images = 0
        total_generations = 0
        num_for_print = 100
        bbox_dic = {}        
        while img_num < args.n_generations:
            if img_num > num_for_print:
                num_for_print += 100
                print(f"Generated {img_num} images")
                
            # deprecated
            if args.use_adjective:
                adj_idx = np.random.choice(a=adjectives.size)
                adjective = adjectives[adj_idx]
                prompt = f"A photo of the face of a {adjective}{concept}"
            # else:
                # prompt = template + f"{prefix} {concept}"
            
            if img_num == 0:
                print("Generation starts with the prompt of ", prompt)

            if args.trainer != 'fairdiffusion':
                images = model(prompt=prompt, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
            else:
                #make reverse_editing_direction
                # choose group
                flat_index = np.random.choice(a=group_prob.size, p=group_prob.flatten())
                idxs = np.unravel_index(flat_index, group_prob.shape)

                _reverse_editing_direction = copy(reverse_editing_direction)
                pos = 0
                for i, group in enumerate(args.group):
                    _reverse_editing_direction[pos+idxs[i]] = False
                    pos += len(group_dic[group])
                images = model(prompt=prompt, num_images_per_prompt=5, guidance_scale=7.5,generator=gen,
                            editing_prompt=group_prompt_list, 
                            reverse_editing_direction=_reverse_editing_direction, # Direction of guidance i.e. decrease the first and increase the second concept
                            edit_warmup_steps=edit_warmup_steps, # Warmup period for each concept
                            edit_guidance_scale=edit_guidance_scale, # Guidance scale for each concept
                            edit_threshold=edit_threshold, # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
                            edit_momentum_scale=0.3, # Momentum scale that will be added to the latent guidance
                            edit_mom_beta=0.6, # Momentum beta
                            # edit_weights=edit_weights # Weights of the individual concepts against each other
                            edit_weights=[1]*11
                        ).images

            image_tensors = torch.stack([transform(image) for image in images])

            flags, bboxs = face_detector.process_tensor_image(image_tensors)

            total_generations += len(images)
            if sum(flags) > 0:
                filtered_images += sum(~flags)
                bbox_idx = 0
                for j, flag in enumerate(flags):
                    image = images[j]
                    if flag:
                        image.save(f"{path}/{img_num}.png")
                        # image.save(f"{path}/{filtered_ids[img_num]}.png")
                        bbox_dic[img_num] = face_detector.extract_position(image, bboxs[bbox_idx])
                        img_num += 1
                        bbox_idx += 1
                    else:
                        image.save(f"{path_filtered}/{img_num_filtered}.png")
                        img_num_filtered += 1
        
        if total_generations > 0:
            print(f"Percentage of filtered images: {filtered_images/total_generations}")
        
        with open(os.path.join(path, 'bbox_dic.pkl'), 'wb') as f:
            pickle.dump(bbox_dic, f)

if __name__ == "__main__":
    main()
