import itertools
from typing import Iterable, List, Tuple, Optional, Callable

from joblib import delayed
from torch import Tensor, no_grad
from torch.nn import Module
from torch.nn.utils import vector_to_parameters

from torch_landscape.landscape import LossSurfaceData, setup_surface_data_linear
from torch_landscape.parallel_progress_bar import ProgressParallel
from torch_landscape.subspace import Subspace
from torch_landscape.utils import clone_parameters, reset_params
from torch_landscape.visualize_options import VisualizationOptions


class SubspaceLandscapeCalculator:
    def __init__(
        self,
        subspace: Subspace,
        min_x_value: int = -1,
        max_x_value: int = 1,
        min_y_value: int = -1,
        max_y_value: int = 1,
        num_data_point: int = 50,
        options: Optional[VisualizationOptions] = None,
        n_jobs: int = 1,
        parallel_backend: str = "loky",
    ):
        if options is not None:
            min_x_value = options.min_x_value
            min_y_value = options.min_y_value
            max_x_value = options.max_x_value
            max_y_value = options.max_y_value
            num_data_point = options.num_points

        self._min_x_value = min_x_value
        self._max_x_value = max_x_value
        self._min_y_value = min_y_value
        self._max_y_value = max_y_value
        self._num_data_point = num_data_point
        self._subspace = subspace

        # set the number of jobs to use for parallelization
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend


    def calculate_loss_surface_data_model(
        self,
        model: Module,
        evaluation_function: Callable[[], float],
    ) -> LossSurfaceData:
        """
        Calculates the loss landscape for the specified model using the specified evaluation function.

        :param model: The torch model to use to calculate the loss landscape dictionary.
        :param evaluation_function: The evaluation function which takes one parameter (the model) and returns the loss.
        :return: The loss landscape in a dictionary.
        """
        surface_dict = self.calculate_loss_surface_data([*model.parameters()], lambda p: evaluation_function())
        return surface_dict

    def calculate_loss_surface_data(
            self,
            model_parameters: List[Tensor],
            evaluation_function: Callable[[Iterable[Tensor]], float],
    ) -> LossSurfaceData:
        """
        Calculates the loss surface for the specified model.
        :param model_parameters: the parameters of the model.
        :param evaluation_function: a function which should evaluate the loss of the model with the specified
        parameters, provided as first argument.
        :return: the loss surface data.
        """
        original_parameters = clone_parameters(model_parameters)
        surface_data = setup_surface_data_linear(
            self._min_x_value, self._max_x_value, self._num_data_point, self._min_y_value, self._max_y_value
        )

        def evaluate_row(y):
            with no_grad():
                row_results = []
                for x in surface_data.x_coordinates:
                    point = self._subspace.get_point(Tensor([[x, y]]).T)
                    vector_to_parameters(point, model_parameters)
                    row_results.append(evaluation_function(model_parameters))
                return row_results

        row_results = ProgressParallel(n_jobs=self.n_jobs, batch_size=1, backend=self.parallel_backend)(
            delayed(evaluate_row)(y)
            for y in surface_data.y_coordinates
        )

        for y_index, row_result in enumerate(row_results):
            for x_index, z in enumerate(row_result):
                surface_data.z_coordinates[y_index, x_index] = z

        # reset parameters of the model to the original parameters.
        reset_params(model_parameters, original_parameters)

        return surface_data

