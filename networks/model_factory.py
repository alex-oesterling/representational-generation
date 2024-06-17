import torch


class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(modelname):
        if modelname == 'BLIP':
            from transformers import BlipForQuestionAnswering
            network = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        else:
            raise NotImplementedError

        return network
