from typing import Iterator, Callable, Union, List, Tuple

from torch import Tensor
from torch.nn import Parameter, Module

from torch_landscape.directions import SvdDirections, RandomDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.visualize import Plotly2dVisualization, VisualizationData, Plotly3dVisualization
from torch_landscape.visualize_options import VisualizationOptions


def plot_loss_landscape_trajectory(optimal_parameters: Iterator[Parameter],
                                   parameters_snapshot_with_respective_loss: Union[List[List[Tensor]], List[Tuple[List[Tensor], float]]],
                                   model: Module,
                                   evaluate_function: Callable[[], float],
                                   file_directory: str = "trajectory_plot"):
    # Calculate SVD directions from parameter trajectory
    optimal_parameters = [*optimal_parameters]
    directions = SvdDirections(optimal_parameters, parameters_snapshot_with_respective_loss).calculate_directions()
    # Calculate trajectory in the SVD plane
    options = VisualizationOptions(num_points=50)
    trajectory = TrajectoryCalculator(optimal_parameters, directions).project_with_loss(parameters_snapshot_with_respective_loss)
    trajectory.set_range_to_fit_trajectory(options)

    landscape_calculator = LinearLandscapeCalculator(model.parameters(), directions, options=options)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate_function)

    # Plot both loss landscape and trajectory
    plotly_2d = Plotly2dVisualization(options)
    plotly_2d.plot(VisualizationData(landscape, trajectory), file_directory)

def plot_loss_landscape_3d(optimal_parameters: Iterator[Parameter],
                           model: Module,
                           evaluate_function: Callable[[], float],
                           file_directory: str = "3D_plot"):
    # Loss landscape calculation and plot
    optimal_parameters = [*optimal_parameters]
    directions = RandomDirections(optimal_parameters).calculate_directions()
    options = VisualizationOptions(num_points=100)
    landscape_calculator = LinearLandscapeCalculator(optimal_parameters, directions, options=options)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate_function)

    plotly_3d = Plotly3dVisualization(VisualizationOptions())
    plotly_3d.plot(VisualizationData(landscape), file_directory)