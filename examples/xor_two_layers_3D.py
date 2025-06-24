import torch
from torch import nn
from torch.optim import SGD

from torch_landscape.directions import RandomDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.utils import seed_everything
from torch_landscape.visualize import Plotly3dVisualization, VisualizationData, VisualizationOptions

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

# Loss landscape calculation and plot
params = list(model.parameters())
directions = RandomDirections(params).calculate_directions()
options = VisualizationOptions(num_points=100)
landscape_calculator = LinearLandscapeCalculator(params, directions, options=options)
landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate)

plotly_3d = Plotly3dVisualization(VisualizationOptions())
plotly_3d.plot(VisualizationData(landscape), 'xor_two_layers_3D', title="XOR Loss Landscape Surface")