import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self, embedding_dim=64, n_domains=3, lambda_=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(32, n_domains)
        )
    def forward(self, x):
        return self.net(self.grl(x))
