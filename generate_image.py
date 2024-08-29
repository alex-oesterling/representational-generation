from diffusers import DiffusionPipeline
import torch
import argparse
import numpy as np
from utils import set_seed, check_log_dir
import networks
import os
from face_detector import FaceDetector
from torchvision import transforms
import pickle

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

    args = parser.parse_args()

    group_idx ={
        'gender' : 0,
        'age' : 1,
        'race' : 2
    }
    group_dic = {
        'gender' : ['male', 'female'],
        'age' : ['young', 'old'],
        'race' : ['East Asian', 'Indian', 'Black', 'White', 'Middle Eastern', 'Latino_Hispanic', 'Southeast Asian']
    }

    args = parser.parse_args()

    set_seed(args.seed)

    gen = torch.Generator(device='cuda')
    gen.manual_seed(0)

    if args.model_path is None:
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

    base_path = f'/n/holyscratch01/calmon_lab/Lab/datasets/{args.trainer}'
    if args.trainer != 'scratch':
        group_name = args.model_path.split("_")[-1].split(".")[0]
        base_path = os.path.join(base_path, group_name)
    base_path = os.path.join(base_path, args.model)
    check_log_dir(base_path)
    
    traits = ["attractive", "emotional", "exotic", "poor", "terrorist", "thug"]
    professions = ["chef", "therapist", "housekeeper", "pilot", "flight attendant", "taxi driver", "nurse", "firefighter", "CEO"]#"cook", 
    prompt_dic = {
        "trait" :  "A photo of the face of ",
        "profession" : "A portrait photo of "
    }

    transform = transforms.ToTensor()

    for concept in args.concepts:
        if concept in traits:
            template = prompt_dic['trait']
        elif concept in professions:
            template = prompt_dic['profession']
        else:
            raise ValueError("This concept is not articulated")
        print(f"Generating images for {concept}")

        # check the folders
        _concept_path = concept if len(concept.split(" ")) == 1 else "_".join(concept.split(" "))
        path = os.path.join(base_path, _concept_path) if not args.use_adjective else os.path.join(base_path, _concept_path+'_adj')
        check_log_dir(path)
        path_filtered = os.path.join(path, 'filtered')
        check_log_dir(path_filtered)
        # make prompts
        prefix = 'a' if concept[0].lower() in ['a','e','i','o','u'] else 'an'
        template += f"{prefix} {concept}"
        if concept in traits[:4]:
            template += " person"

        # make the face detector
        face_detector = FaceDetector()

        # if os.path.exists(os.path.join(path, 'filtered_ids.txt')):
        #     with open(os.path.join(path, 'filtered_ids.txt'), 'r') as f:
        #         filtered_ids = f.readlines()
        #     filtered_ids = [int(idx) for idx in filtered_ids]
        #     print(filtered_ids)
        #     with open(os.path.join(path, 'bbox_dic.pkl'), 'rb') as f:
        #         bbox_dic = pickle.load(f)
        #     args.n_generations = len(filtered_ids)
        
        img_num = 0
        img_num_filtered = 0
        
        # generation starts
        filtered_images = 0
        total_generations = 0
        while img_num < args.n_generations:
            if img_num % 100 == 0:
                print(f"Generated {img_num} images")
                
            if args.use_adjective:
                adj_idx = np.random.choice(a=adjectives.size)
                adjective = adjectives[adj_idx]
                images = model(prompt=f"A photo of the face of a {adjective}{concept}", num_images_per_prompt=args.n_gen_per_iter, generator=gen).images
            else:
                template += f"{prefix} {concept}"
                images = model(prompt=template, num_images_per_prompt=args.n_gen_per_iter, generator=gen).images

            image_tensors = torch.stack([transform(image) for image in images])

            flags, bboxs = face_detector.process_tensor_image(image_tensors)
            bbox_dic = {}
            total_generations += len(images)
            if sum(flags) > 0:
                filtered_images += sum(~flags)
                bbox_idx = 0
                for j, flag in enumerate(flags):
                    if args.n_generations > img_num:
                        image = images[j]
                        if flag:
                            image.save(f"{path}/{img_num}.png")
                            # image.save(f"{path}/{filtered_ids[img_num]}.png")
                            img_num += 1
                            bbox_dic[img_num] = face_detector.extract_position(image, bboxs[bbox_idx])
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
