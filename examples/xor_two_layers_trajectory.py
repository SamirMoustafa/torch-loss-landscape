import torch
from torch import nn
from torch.optim import SGD

from torch_landscape import plot_loss_landscape_trajectory
from torch_landscape.utils import clone_parameters, seed_everything

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

# Store parameter trajectory
parameters_with_loss = []

# Full-batch training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(X).squeeze(1)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    # Store a snapshot every 2 epochs
    if (epoch + 1) % 2 == 0:
        parameters_with_loss.append((clone_parameters(model.parameters()), loss.item()))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Loss evaluation function
@torch.no_grad()
def evaluate():
    out = model(X).squeeze(1)
    return criterion(out, y).item()

plot_loss_landscape_trajectory(optimal_parameters=model.parameters(),
                               parameters_snapshot_with_respective_loss=parameters_with_loss,
                               model=model,
                               evaluate_function=evaluate,
                               file_directory="xor_two_layers_trajectory_plot")