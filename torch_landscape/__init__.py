from typing import Iterator, Callable, Union, List, Tuple, Optional

from torch import Tensor
from torch.nn import Parameter, Module

from torch_landscape.directions import RandomDirections, SvdDirections, PcaDirections
from torch_landscape.landscape_linear import LinearLandscapeCalculator
from torch_landscape.trajectory import TrajectoryCalculator
from torch_landscape.visualize import Plotly2dVisualization, VisualizationData, Plotly3dVisualization, DEFAULT_PLOTLY_TEMPLATE
from torch_landscape.visualize_options import VisualizationOptions


def plot_loss_landscape_trajectory(optimal_parameters: Iterator[Parameter],
                                   parameters_snapshot_with_respective_loss: Union[List[List[Tensor]], List[Tuple[List[Tensor], float]]],
                                   model: Module,
                                   evaluate_function: Callable[[], float],
                                   directions_cls: type = SvdDirections,
                                   min_x_value: int = -1,
                                   max_x_value: int = 1,
                                   min_y_value: int = -1,
                                   max_y_value: int = 1,
                                   num_points: int = 20,
                                   use_log_z: bool = False,
                                   show_title: bool = True,
                                   show_axes_labels: bool = True,
                                   n_jobs: int = 1,
                                   parallel_backend: str = "loky",
                                   mark_zero: bool = True,
                                   plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
                                   title: Optional[str] = None,
                                   file_directory: str = "trajectory_plot",
                                   file_extension: str = "html"):
    # Calculate SVD directions from parameter trajectory
    optimal_parameters = [*optimal_parameters]
    directions = directions_cls(optimal_parameters, parameters_snapshot_with_respective_loss).calculate_directions()
    # Calculate trajectory in the SVD plane
    options = VisualizationOptions(num_points=num_points,
                                   min_x_value=min_x_value,
                                   max_x_value=max_x_value,
                                   min_y_value=min_y_value,
                                   max_y_value=max_y_value,
                                   use_log_z=use_log_z,
                                   show_title=show_title,
                                   show_axes_labels=show_axes_labels)
    trajectory = TrajectoryCalculator(optimal_parameters, directions).project_with_loss(parameters_snapshot_with_respective_loss)
    trajectory.set_range_to_fit_trajectory(options)

    landscape_calculator = LinearLandscapeCalculator(optimized_parameters=optimal_parameters,
                                                     directions=directions,
                                                     min_x_value=min_x_value,
                                                     max_x_value=max_x_value,
                                                     min_y_value=min_y_value,
                                                     max_y_value=max_y_value,
                                                     options=options,
                                                     n_jobs=n_jobs,
                                                     parallel_backend=parallel_backend)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate_function)

    # Plot both loss landscape and trajectory
    plotly_2d = Plotly2dVisualization(options=options, plotly_template=plotly_template, mark_zero=mark_zero)
    plotly_2d.plot(VisualizationData(landscape, trajectory),
                   file_path=file_directory,
                   title=title,
                   file_extension=file_extension)

def plot_loss_landscape_3d(optimal_parameters: Iterator[Parameter],
                           model: Module,
                           evaluate_function: Callable[[], float],
                           apply_normalization: bool = True,
                           min_x_value: int = -1,
                           max_x_value: int = 1,
                           min_y_value: int = -1,
                           max_y_value: int = 1,
                           num_points: int = 50,
                           use_log_z: bool = False,
                           show_title: bool = True,
                           show_axes_labels: bool = True,
                           n_jobs: int = 1,
                           parallel_backend: str = "loky",
                           z_axis_lines: int = 50,
                           opacity: float = 0.8,
                           plotly_template: str = DEFAULT_PLOTLY_TEMPLATE,
                           show_color_scale: bool = False,
                           file_directory: str = "3D_plot",
                           title: Optional[str] = None,
                           file_extension: str = "html",
                           ):
    # Loss landscape calculation and plot
    optimal_parameters = [*optimal_parameters]
    directions = RandomDirections(optimal_parameters).calculate_directions(apply_normalization=apply_normalization)
    options = VisualizationOptions(num_points=num_points,
                                   min_x_value=min_x_value,
                                   max_x_value=max_x_value,
                                   min_y_value=min_y_value,
                                   max_y_value=max_y_value,
                                   use_log_z=use_log_z,
                                   show_title=show_title,
                                   show_axes_labels=show_axes_labels)
    landscape_calculator = LinearLandscapeCalculator(optimized_parameters=optimal_parameters,
                                                     directions=directions,
                                                     min_x_value=min_x_value,
                                                     max_x_value=max_x_value,
                                                     min_y_value=min_y_value,
                                                     max_y_value=max_y_value,
                                                     options=options,
                                                     n_jobs=n_jobs,
                                                     parallel_backend=parallel_backend)
    landscape = landscape_calculator.calculate_loss_surface_data_model(model, evaluate_function)

    plotly_3d = Plotly3dVisualization(options = VisualizationOptions(),
                                      z_axis_lines=z_axis_lines,
                                      opacity=opacity,
                                      plotly_template=plotly_template,
                                      show_color_scale=show_color_scale)
    plotly_3d.plot(VisualizationData(landscape),
                   file_path=file_directory,
                   title=title,
                   file_extension=file_extension
                   )