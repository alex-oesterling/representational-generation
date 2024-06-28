from importlib import import_module

class RetrieverFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_retriever(retriever_name, args):
        module= import_module(f'retriever.{retriever_name}')
        return module.Retriever(args=args)


class GenericRetriever:
    '''
    Base class for retriever; to implement a new retriever, inherit from this.
    '''
    def __init__(self, args):
        self.args = args