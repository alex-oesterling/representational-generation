import torch
import numpy as np 
from sklearn.linear_model import LinearRegression
from transformers import BlipProcessor
import random 

def set_seed(seed): 
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def oracle_function(indices, dataset, curation_set=None, model=None):
    if model is None:
        model = LinearRegression()

    k = int(np.sum(indices))
    if curation_set is not None:
        m = curation_set.shape[0]
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        curation_indicator = np.concatenate((np.zeros(dataset.shape[0]), np.ones(curation_set.shape[0])))
        a_expanded = np.concatenate((indices, np.zeros(curation_set.shape[0])))
        m = curation_set.shape[0]
        alpha = (a_expanded/k - curation_indicator/m)
        reg = model.fit(expanded_dataset, alpha)
    else:
        m = dataset.shape[0]
        alpha = (indices/k - 1/m)
        reg = model.fit(dataset, alpha)
    return reg

def getMPR(indices, dataset, k, curation_set=None, model=None):
    reg = oracle_function(indices, dataset, curation_set=curation_set, model=model)
    if curation_set is not None:
        expanded_dataset = np.concatenate((dataset, curation_set), axis=0)
        m = curation_set.shape[0]
        c = reg.predict(expanded_dataset)
        c /= np.linalg.norm(c)
        c *= np.sqrt(c.shape[0]) ## sqrt(n+m) = 141
        mpr = np.abs(np.sum((indices/k)*c[:dataset.shape[0]]) - np.sum((1/m)*c[dataset.shape[0]:]))
    else:
        m = dataset.shape[0]
        c = reg.predict(dataset)
        c /= np.linalg.norm(c)
        c *= np.sqrt(c.shape[0]) ## sqrt(n) = 100
        mpr = np.abs(np.sum((indices/k)*c) - np.sum((1/m)*c))
    
    return mpr, c


def feature_extraction(encoder, dataloader, args):
    encoder.eval()
    encoder = encoder.cuda() if torch.cuda.is_available() else encoder

    if args.vision_encoder == 'BLIP':
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        raw_text=["What word best describes the person's appearance?"]
        text = processor(text=raw_text, return_tensors='pt')
        input_ids=text['input_ids']
        attention_mask=text['attention_mask']
        
        raw_dummy_label = ['0']
        dummy_label = processor(text=raw_dummy_label, return_tensors='pt')['input_ids']

        outputs = []
        # professions = []
        for batch in dataloader:
            with torch.no_grad():
                image = batch[0]
                batch_size = image.shape[0]
                
                _input_ids = input_ids.repeat(batch_size, 1)
                _attention_mask = attention_mask.repeat(batch_size, 1)
                _dummy_label = dummy_label.repeat(batch_size, 1)

                if torch.cuda.is_available():
                    image = image.cuda()
                    _input_ids = _input_ids.cuda()
                    _attention_mask = _attention_mask.cuda()
                    _dummy_label = _dummy_label.cuda()
                
                output = encoder.forward(pixel_values=image, input_ids=_input_ids, attention_mask=_attention_mask, labels=_dummy_label, return_dict=True)
                output = output['image_embeds'].mean(axis=1)
                outputs.append(output)
                
                # professions.append(profession)
        outputs = torch.cat(outputs)
        outputs = outputs.mean(axis=1)


    return outputs
