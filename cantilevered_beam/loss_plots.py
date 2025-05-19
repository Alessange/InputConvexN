import pandas as pd
import matplotlib.pyplot as plt


loss_init_ic = pd.read_csv("cantilevered_beam/loss_init_ic.csv").values
loss_init = pd.read_csv("cantilevered_beam/loss_init.csv").values
loss_pca_init = pd.read_csv("cantilevered_beam/loss_pca_init.csv").values
loss_pca_ic = pd.read_csv("cantilevered_beam/loss_pca_init_ic.csv").values


plt.figure(figsize=(10, 6))
plt.plot(loss_init, label="Loss - Init", linewidth=2)
plt.plot(loss_pca_init, label="Loss -Init PCA", linewidth=2)
plt.plot(loss_init_ic, label="Loss - IC Init", linewidth=2)
plt.plot(loss_pca_ic, label="Loss - IC PCA", linewidth=2)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("Training Loss Comparison (Log Scale)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
plt.savefig("cantilevered_beam/loss_comparison.jpg", dpi=300)
