
import torch
import numpy as np 
import os
from tqdm import tqdm
from insightface.app import FaceAnalysis
import cv2
import torchvision 
import argparse
import data_handler
from torchvision import transforms
import dlib
import pickle

dlib.DLIB_USE_CUDA = True

class FaceDetector:
    def __init__(self):
        # self.app = FaceAnalysis(name='buffalo_l', 
                # providers=[('CUDAExecutionProvider', {'device_id':'0'})],
# )
        # self.app.prepare(ctx_id=0, det_size=(640, 640))

        # self.app = dlib.get_frontal_face_detector()
        self.app = dlib.cnn_face_detection_model_v1('datasets/stuffs/dlib_models/mmod_human_face_detector.dat')

    
    def process_tensor_image(self, images, fill_value=-1):
        faces = []
        face_indicators = []
        face_bboxs = []
        images_np = (images*255).cpu().detach().permute(0,2,3,1).numpy().astype(np.uint8)
        
        # images_np = images.permute(0,2,3,1).float().numpy().astype(np.uint8)

        for idx, image_np in enumerate(images_np):
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # faces_from_app = self.app.get(image_np[:,:,[2,1,0]])
            # gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            # faces_from_app = self.app(gray_image)
            faces_from_app = self.app(image_np, 1)
            num_faces = len(faces_from_app)
            if num_faces >= 1:
                # face_from_app = self._get_largest_face_app(faces_from_app, dim_max=image_np.shape[0], dim_min=0)
                # faces = self._crop_face(images[idx], bbox, target_size=[224,224], fill_value=fill_value)
                
                # face_landmarks = np.array(face_from_app["kps"])
                # aligned_faces = image_pipeline(images[idx], face_landmarks)
                
                face_indicators.append(True)
                face_bboxs.extend(faces_from_app)
                # facess.append(faces.unsqueeze(dim=0))
                # face_landmarks_app.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                # aligned_facess_app.append(aligned_faces.unsqueeze(dim=0))
            # elif num_faces >= 1:
            #     face_indicators.append(True)
            #     max_area = 0
            #     for face_bbox in faces_from_app:
            #         bbox = face_bbox.rect
            #         left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
            #         area = (right-left)*(bottom-top)
            #         if area > max_area:
            #             max_area = area
            #             max_bbox = bbox
            #     face_bboxs.append(max_bbox)
            else:
                face_indicators.append(False)
        
        print(f"The number of images filtered : {len(images_np)-sum(face_indicators)}")
        
        face_indicators = torch.tensor(face_indicators).to(device=images.device)
        # face_bboxs = torch.tensor(face_bboxs).to(device=images.device)
        # facess = torch.cat(facess, dim=0)
        # face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
        # aligned_facess_app = torch.cat(aligned_facess_app, dim=0)
                
        return face_indicators, face_bboxs
    
    def process_pil_image(self, image):
        image_np = np.array(image)

        faces_from_app = self.app(image_np, 1)
        num_faces = len(faces_from_app)
        return num_faces >= 1
    
    def extract_position(self, image, bbox):
        bbox = bbox.rect
        # left, top, width, height = bbox.left(), bbox.top(), bbox.width(), bbox.height()
        left, top, right, bottom = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        left = max(left, 0)
        top = max(top, 0)
        if type(image) == torch.Tensor:
            right = min(right, image.shape[-1])
            bottom = min(bottom, image.shape[-2])
        # if image is PIL Image
        else:
            right = min(right, image.size[-2])
            bottom = min(bottom, image.size[-1])

        return [left, top, right, bottom]
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='representational-generation')
        
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='general')
    parser.add_argument('--dataset-path', type=str, default=None, help='it is only used when query-dataset is general')

    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--vision-encoder', type=str, default='CLIP',choices = ['BLIP', 'CLIP', 'PATHS'])

    args = parser.parse_args()

    face_detector = FaceDetector()
    loader = data_handler.DataloaderFactory.get_dataloader(dataname=args.dataset, args=args)
    
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    loader.dataset.processor = None
    loader.dataset.transform = transform

    filtered_ids = []
    unfiltered_ids = []
    bbox_dic = {}
    for image, label, idxs in loader:
        flags, bboxs = face_detector.process_tensor_image(image)
        if args.filter_w_aesthetic:
            with torch.no_grad():
                image_features = model2.encode_image(image)

        filtered_ids.extend(idxs[~flags].tolist())
        unfiltered_ids.extend(idxs[flags].tolist())
        for idx, bbox in zip(idxs[flags], bboxs):
            bbox_dic[idx.item()] = face_detector.extract_position(image, bbox)

    # Save the filtered and unfiltered IDs to files
    with open(os.path.join(args.dataset_path,'filtered_ids.txt'), 'w') as f:
        for id in filtered_ids:
            f.write(f"{id}\n")

    with open(os.path.join(args.dataset_path,'unfiltered_ids.txt'), 'w') as f:
        for id in unfiltered_ids:
            f.write(f"{id}\n")

    with open(os.path.join(args.dataset_path,'bbox_dic.pkl'), 'wb') as f:
        pickle.dump(bbox_dic, f)
        # for id, bbox in bbox_dic.items():
            # f.write(f"{id}: {bbox}\n")
    
    # percentage of filtered images
    print(f"Percentage of filtered images: {len(filtered_ids)/(len(filtered_ids)+len(unfiltered_ids))}")
