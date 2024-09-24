import torch
from PIL import Image
import json
import argparse
from face_detector import FaceDetector
from aesthetic_scoring import AestheticScorer
from mpr.preprocessing import group_estimation
import clip
import numpy as np
from collections import defaultdict
import json
import os
from tqdm import tqdm

class MSCOCODataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, preprocess, facedetector):
        self.data_dir = data_dir
        self.img_ids = self.data_dict()
        self.preprocess = preprocess
        self.facedetector = facedetector

    def data_dict(self):
        f = open(self.data_dir + 'annotations/captions_train2017.json')
        data = json.load(f)
        f.close()

        img_ids = []

        # img_id_set = {}
        for x in data['annotations']:
            img_id = x["image_id"]
            img_ids.append(img_id)
            # if img_id in img_id_set:
            #     img_id_set[img_id].append(x["caption"])
            # else:
            #     img_id_set[img_id] = [x["caption"]]
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = self.data_dir + "train2017/" + str(self.img_ids[idx]).zfill(12) + ".jpg"
        image = Image.open(img_path)
        is_face = self.facedetector.process_pil_image(image)
        # print(batch_counter, flush=True)
        # if is_face:
        #     img[batch_counter] = preprocess(image).to(args.device)
        #     ids[batch_counter] = img_ids[id_counter]
        #     batch_counter += 1
        # id_counter += 1
        return self.preprocess(image), self.img_ids[idx], is_face

def data_dict(args):
    f = open(args.data_dir + 'annotations/captions_train2017.json')
    data = json.load(f)
    f.close()


    img_id_set = {}
    for x in data['annotations']:

        img_id = x["image_id"]

        if img_id in img_id_set:
            img_id_set[img_id].append(x["caption"])
        else:
            img_id_set[img_id] = [x["caption"]]

    print(len(img_id_set))
    return img_id_set

def embed_images(clipmodel, facedetector, aestheticscorer, aestheticthreshold, args):
    preprocess = aestheticscorer.get_preprocess()

    output = defaultdict(dict)

    dataloader = torch.utils.data.DataLoader(MSCOCODataset(args.data_dir, preprocess, facedetector), batch_size=args.batch_size, shuffle=False)

    for image, ids, faces in tqdm(dataloader):
        image = image.to(args.device)
        faces = faces.to(args.device)
        ids = ids.to(args.device)
        # faces, _ = facedetector.process_tensor_image(image)
        scores = aestheticscorer(image).squeeze()
        # print(faces)
        if not torch.any(torch.logical_and((scores > aestheticthreshold),(faces))):
            continue
        score_indices = torch.where((scores > aestheticthreshold) & (faces))

        newbatch = image[score_indices]
        ids = ids[score_indices]

        embeds = clipmodel.encode_image(newbatch).cpu().numpy()

        gender_pred = group_estimation(embeds, 'gender')
        race_pred = group_estimation(embeds, 'race')
        age_pred = group_estimation(embeds, 'age')

        # gender_pred = np.round((gender_pred+1)/2)
        # race_pred = np.round((race_pred+1)/2)
        # age_pred = np.round((age_pred+1)/2)
        gender_pred = np.argmax((gender_pred+1)/2, axis=-1)
        race_pred = np.argmax((race_pred+1)/2, axis=-1)
        age_pred = np.argmax((age_pred+1)/2, axis=-1)

        # keylist = ["negative", "positive"]

        for i in range(len(score_indices)):
            if str(gender_pred[i]) not in output["gender"].keys():
                output["gender"][str(gender_pred[i])] = []
            if str(race_pred[i]) not in output["race"].keys():
                output["race"][str(race_pred[i])] = []
            if str(age_pred[i]) not in output["age"].keys():
                output["age"][str(age_pred[i])] = []
            output["gender"][str(gender_pred[i])].append(ids[i].item())
            output["race"][str(race_pred[i])].append(ids[i].item())
            output["age"][str(age_pred[i])].append(ids[i].item())
            # print(output)
            # exit()
    with open(os.path.join(args.out_dir, "filtered_mscoco.json"), "w") as outfile:
        json.dump(output, outfile)

    print(output)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default="./datasets/mscoco/")
    parser.add_argument('-out_dir', type=str, default="./datasets/mscoco/")
    parser.add_argument('-device', type=str, default='cuda', help="cpu, cuda, or mps")
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()

    clipmodel, _ = clip.load("ViT-B/32", device=args.device)
    facedetector = FaceDetector()
    aestheticscorer = AestheticScorer()
    aestheticscorer.to(args.device)
    
    with torch.no_grad():
        embed_images(clipmodel, facedetector, aestheticscorer, 1, args)

if __name__ == "__main__":
    main()