import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x1 = F.elu(self.hidden1(x))
        x2 = F.elu(self.hidden2(x1))
        return self.out(x2)



class InputConvexNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_x1 = nn.Linear(1, 32, bias=True)
        self.hidden_z1 = nn.Linear(32, 32, bias=False)

        self.hidden_x2 = nn.Linear(1, 64, bias=True)
        self.hidden_z2 = nn.Linear(32, 64, bias=False)

        self.W_out = nn.Parameter(torch.randn(1, 64))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.shape[0]
        z0 = torch.zeros(batch_size, self.hidden_z1.in_features,
                         device=x.device, dtype=x.dtype)
        z1 = F.elu(self.hidden_z1(F.softplus(z0)) + self.hidden_x1(x))
        z2 = F.elu(self.hidden_z2(F.softplus(z1)) + self.hidden_x2(x))
        out = F.linear(z2, F.softplus(self.W_out)) + self.b_out
        return out


def train_model(model, x_train, y_train):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    model.eval()

if __name__ == "__main__":
    x_train = torch.linspace(-2, 2, 1000).unsqueeze(1)
    y_train = x_train**2

    model1 = InputConvexNN()
    model2 = MLP()
    train_model(model1, x_train, y_train)
    train_model(model2, x_train, y_train)

    x_test = torch.linspace(-5, 5, 2000).unsqueeze(1)
    y_test = x_test**2

    with torch.no_grad():
        y_pred1 = model1(x_test)
        y_pred2 = model2(x_test)


    plt.figure(figsize=(6,3))
    plt.plot(x_test.numpy(), y_test.numpy(),   'k--',         label='True $x^2$')
    plt.plot(x_test.numpy(), y_pred1.numpy(), linestyle='-',  color='C0', label='ICNN')
    plt.plot(x_test.numpy(), y_pred2.numpy(), linestyle='-.', color='C1', label='MLP')  
    plt.xlim(-5, 5)
    plt.ylim(0, 25)
    plt.legend()
    plt.title("Generalization")
    plt.tight_layout()
    plt.show()

