import torch
from torch.utils.data import TensorDataset, DataLoader
from autoencoder import Autoencoder



def train(model, loader, num_epochs=100, lr=0.001, device=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for epoch in range(num_epochs):
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return model


if __name__ == "__main__":

    all_disp = torch.load("cantilevered_beam/displacements.pt")                

    num_samples, n_vert, n_dim = all_disp.shape 
    X = all_disp.view(num_samples, -1).float()

    dataset = TensorDataset(X)                       
    loader  = DataLoader(dataset,batch_size=32,shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder(input_dim=X.shape[1], hidden_dim=500, latent_dim=5)
    trained_model = train(model, loader, num_epochs=100, lr=1e-3, device=device)
    torch.save(trained_model.state_dict(), "cantilevered_beam/autoencoder.pth")

