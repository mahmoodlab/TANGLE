import torch 
from torch import nn
import torch.nn.functional as F


class BatchedABMIL(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=256, dropout=False, n_classes=1, activation='softmax'):
        """
        Attention Network with Sigmoid Gating (3 fc layers). Supports batching 
        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(input_dim, hidden_dim),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, return_raw_attention=False):
        """
        Forward pass 
        x List[(torch.Tensor)]: List of [patches x d] w/ len(x) = bs
        """

        # gated attention 
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes

        if self.activation == 'softmax':
            activated_A = F.softmax(A, dim=1)
        elif self.activation == 'leaky_relu':  # enable "counting" 
            activated_A = F.leaky_relu(A)
        elif self.activation == 'relu':
            activated_A = F.relu(A)
        elif self.activation == 'sigmoid':  # enable "counting"
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError('Activation not implemented.')

        if return_raw_attention:
            return activated_A, A

        return activated_A
