import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_adv
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_adv=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_adv = lambda_adv

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_adv)

class TDOAAdversaryHead(nn.Module):
    def __init__(self, in_features, hidden_features=128):
        super(TDOAAdversaryHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
