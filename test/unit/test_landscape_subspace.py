import unittest

from torch import Tensor, hstack
from torch.nn import Module, Linear
from torch.testing import assert_close

from torch_landscape.landscape_subspace import SubspaceLandscapeCalculator
from torch_landscape.subspace import LinearSubspace


class SubspaceLandscapeCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.b1 = Tensor([[1., 0., 2.]]).T
        self.b1 /= self.b1.norm()
        self.b2 = Tensor([[4., 5., -2.]]).T
        self.b2 /= self.b2.norm()
        self.basis = hstack([self.b1, self.b2])
        self.zero_point = Tensor([[2., 1., 3.]]).T
        self.space = LinearSubspace(self.basis, self.zero_point)

    def test_calculate(self):
        calculator = SubspaceLandscapeCalculator(
            subspace=self.space,
            num_data_point=12,
        )

        def func(x, y, z):
            return (x - 2) ** 3 + (y - 1) ** 3

        def dummy_function(params):
            x, y, z = params
            return func(x, y, z)

        model_params = [Tensor([4.]), Tensor([9.]), Tensor([1.])]
        data = calculator.calculate_loss_surface_data(model_params, dummy_function)

        for x_index, x in enumerate(data.x_coordinates.tolist()):
            for y_index, y in enumerate(data.y_coordinates.tolist()):
                point = self.space.get_point(Tensor([[x, y]]).T)
                expected = func(point[0], point[1], point[2]).item()
                assert_close(expected, data.z_coordinates[y_index, x_index])



    def test_calculate_torch_model(self):
        calculator = SubspaceLandscapeCalculator(
            subspace=self.space,
            num_data_point=12,
        )

        class TestModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.linear = Linear(3, 1, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = TestModel()
        test_data = Tensor([[1., 2., 3.]])
        data = calculator.calculate_loss_surface_data_model(model, lambda: model(test_data))

        for x_index, x in enumerate(data.x_coordinates.tolist()):
            for y_index, y in enumerate(data.y_coordinates.tolist()):
                point = self.space.get_point(Tensor([[x, y]]).T)
                expected = (test_data @ point.reshape(1, 3).T).item()
                assert_close(expected, data.z_coordinates[y_index, x_index])



if __name__ == '__main__':
    unittest.main()
