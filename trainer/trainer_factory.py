from importlib import import_module

class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(trainer_name, args):
        module = import_module(f'trainer.{trainer_name}')
        return module.Trainer(args=args)

class GenericTrainer:
    '''
    Base class for retriever; to implement a new retriever, inherit from this.
    '''
    def __init__(self, args):
        self.args = args