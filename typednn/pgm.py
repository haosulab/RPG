# probablistic types 
# what we want to do:
#   - generate distribution with neural networks during the forward, sample or evaluate log prob for any data type  
#   - generative models: conditions on any data and generate any distributions that we can sample from; learning it with sampled $x$ and condition $y$
#   - density estimation: condition on any data, evaluate $log p(x|y)$

# maintain a probablisitc graphic model
from .basetypes import Type


class PGMType(Type):
    # probablistic distribution of the base_type 
    def __init__(self, *args) -> None:
        super().__init__()

        self.out = args[-1]
        self.arguments = args[:-1]


class PGM:
    def __init__(self) -> None:
        pass

    def sample(self):
        raise NotImplementedError
        
    def rsample(self):
        raise NotImplementedError

    def log_prob(self, element):
        raise NotImplementedError

    def condition(self, condition): # condition on something ..
        # lazy
        raise NotImplementedError

    def merge(self, other): # merge two pgm
        raise NotImplementedError