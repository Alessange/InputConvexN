import torch
from torch.utils.data import TensorDataset, DataLoader
from autoencoder import Autoencoder
from sklearn.decomposition import PCA
import pandas as pd

def train(model, loader, num_epochs=100, lr=0.001, device=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    loss_list = []

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
        loss_list.append(loss.item())

    return model, loss_list


if __name__ == "__main__":

    all_disp = torch.load("cantilevered_beam/displacements.pt")                
    T, N, dim = all_disp.shape
    num_samples, n_vert, n_dim = all_disp.shape 
    X = all_disp.view(num_samples, -1).float()


    
    latent_dim = 30
    hidden_dim = 100
    pca = PCA(n_components=hidden_dim)
    pca.fit(X)
    U_pca = pca.components_ 

    
    model = Autoencoder(input_dim=X.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
    dataset = TensorDataset(X)                       
    loader  = DataLoader(dataset,batch_size=64,shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.encoder.encoder[0].weight.copy_(torch.tensor(U_pca))
        model.encoder.encoder[0].bias.copy_(torch.zeros(hidden_dim))
        model.decoder.decoder[-1].weight.copy_(torch.tensor(U_pca).T)
        model.decoder.decoder[-1].bias.copy_(torch.zeros(X.shape[1]))
    
    trained_model, loss_pca_init = train(model, loader, num_epochs=2000, lr=1e-3, device=device)
    torch.save(trained_model.state_dict(), "cantilevered_beam/autoencoder_pca_init.pth")
    pd.DataFrame(loss_pca_init).to_csv("cantilevered_beam/loss_pca_init.csv", index=False)


    # trained_model, loss_init = train(model, loader, num_epochs=2000, lr=1e-3, device=device)
    # pd.DataFrame(loss_init).to_csv("cantilevered_beam/loss_init.csv", index=False)
    # torch.save(trained_model.state_dict(), "cantilevered_beam/autoencoder_init.pth")
    


