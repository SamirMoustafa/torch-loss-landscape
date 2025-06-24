import torch
from torch import nn
from torch.optim import SGD

from torch_landscape import plot_loss_landscape_3d
from torch_landscape.utils import seed_everything

# XOR truth table
seed_everything(0)
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([0., 1., 1., 0.])  # XOR output

# Two-layer model for XOR
model = nn.Sequential(nn.Linear(2, 8),
                      nn.ReLU(),
                      nn.Linear(8, 1))
criterion = nn.BCEWithLogitsLoss()
optimizer = SGD(model.parameters(), lr=0.5)

# Full-batch training loop
for epoch in range(1000):
    optimizer.zero_grad()
    out = model(X).squeeze(1)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Loss evaluation function
@torch.no_grad()
def evaluate():
    out = model(X).squeeze(1)
    return criterion(out, y).item()

plot_loss_landscape_3d(optimal_parameters=model.parameters(),
                       model=model,
                       evaluate_function=evaluate,
                       file_directory="xor_two_layers_3D_plot")