import torch
from torch import nn
from torch.optim import SGD

from torch_landscape.directions import SvdDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.utils import clone_parameters, seed_everything
from torch_landscape.visualize import Plotly2dVisualization, VisualizationData
from torch_landscape.visualize_options import VisualizationOptions

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

# Calculate SVD directions from parameter trajectory
params = list(model.parameters())
directions = SvdDirections(params, parameters_with_loss).calculate_directions()


# Calculate trajectory in the SVD plane
options = VisualizationOptions(num_points=50, use_log_z=True)
trajectory = TrajectoryCalculator(params, directions).project_with_loss(parameters_with_loss)
trajectory.set_range_to_fit_trajectory(options)

# Loss surface calculation
@torch.no_grad()
def evaluate():
    out = model(X).squeeze(1)
    return criterion(out, y).item()

landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options)
landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate)


# Plot both loss landscape and trajectory
plotly_2d = Plotly2dVisualization(options)
plotly_2d.plot(VisualizationData(landscape, trajectory), 'xor_landscape_trajectory', title="XOR Loss Landscape and Trajectory")