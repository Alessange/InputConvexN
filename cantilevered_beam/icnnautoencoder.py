import torch
import torch.nn as nn
import torch.nn.functional as F
from autoencoder import Encoder, Decoder


class InputConvexDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(InputConvexDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.w1 = nn.Parameter(torch.randn(hidden_dim, latent_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))

        self.linear2 = nn.Linear(latent_dim, output_dim, bias=False)
        self.w2 = nn.Parameter(torch.randn(output_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        batch = x.size(0)
        x_prev = torch.zeros(batch, self.linear1.in_features, 
                             device=x.device, dtype=x.dtype)

        wx1_positive = F.softplus(self.w1)
        wx2_positive = F.softplus(self.w2)

        pre_act = self.linear1(x) + self.b1 + F.linear(x_prev, wx1_positive)
        x1 = F.elu(pre_act)
        
        out = self.linear2(x) + self.b2 + F.linear(x1, wx2_positive)

        return out

class InputConvexEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(InputConvexEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w1 = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.b1 = nn.Parameter(torch.randn(hidden_dim))

        self.linear2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.w2 = nn.Parameter(torch.randn(latent_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.randn(latent_dim))

    def forward(self, x):
        batch = x.size(0)
        x_prev = torch.zeros(batch, self.linear1.in_features,
                             device=x.device, dtype=x.dtype)

        wx1_positive = F.softplus(self.w1)
        wx2_positive = F.softplus(self.w2)
        
        pre_act = self.linear1(x) + self.b1 + F.linear(x_prev, wx1_positive)

        x1 = torch.exp(pre_act)
        
        out = self.linear2(x1) + self.b2 + F.linear(x1, wx2_positive)

        return out
    
class InputConvexAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(InputConvexAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = InputConvexDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon



if __name__ == "__main__":
    input_dim  = 100
    hidden_dim = 50
    latent_dim = 10

    model = InputConvexAutoencoder(input_dim, hidden_dim, latent_dim)

    x = torch.randn(32, input_dim)
    x_recon = model(x)

    print("Input shape:       ", x.shape)
    print("Reconstructed shape:", x_recon.shape)