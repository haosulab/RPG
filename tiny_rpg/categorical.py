import torch
import torch.distributions as thdist
import torch.nn.functional as thfunc

class Categorical(thdist.Categorical):

    """ random variable is one-hot encoded """

    def log_prob(self, value):
        value = value.argmax(-1)
        return super().log_prob(value).unsqueeze(-1)
    
    def sample(self, sample_shape=torch.Size()):
        n_classes = self.logits.shape[-1]
        classes = super().sample(sample_shape)
        return thfunc.one_hot(classes, n_classes).float()
