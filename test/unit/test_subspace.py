import unittest

from torch import Tensor, hstack
from torch.testing import assert_close
from torch_landscape.subspace import LinearSubspace


class SubspaceTest(unittest.TestCase):
    def setUp(self):
        self.b1 = Tensor([[1., 0., 2.]]).T
        self.b1 /= self.b1.norm()
        self.b2 = Tensor([[4., 5., -2.]]).T
        self.b2 /= self.b2.norm()
        self.basis = hstack([self.b1, self.b2])
        self.zero_point = Tensor([[1., 2., 3.]]).T
        self.space = LinearSubspace(self.basis, self.zero_point)

    """
    Tests a projection to a linear subspace when the solution is unique.
    """
    def test_project_non_orthogonal_when_solution_unique(self):
        test_vector = self.zero_point + 2*self.b1 - 3*self.b2
        actual_coordinates = self.space.project(test_vector)
        expected_coordinates = Tensor([[2, -3]]).T
        assert_close(expected_coordinates, actual_coordinates)

    def test_project_orthogonal_when_solution_unique(self):
        test_vector = self.zero_point + 2*self.b1 - 3*self.b2
        assert self.b1.T @ self.b2 == 0.0
        self.space.is_orthogonal = True
        actual_coordinates = self.space.project(test_vector)
        expected_coordinates = Tensor([[2, -3]]).T
        assert_close(expected_coordinates, actual_coordinates)

    def test_project_non_orthogonal_with_get_point(self):
        test_vector = self.zero_point + 2*self.b1 - 3*self.b2
        actual_coordinates = self.space.project(test_vector)
        reconstructed = self.space.get_point(actual_coordinates)
        assert_close(test_vector, reconstructed)

    """
    Tests a projection to the linear subspace when the solution is not unique.
    """
    def test_project_non_orthogonal_when_solution_not_unique(self):
        test_vector = self.zero_point + 2 * self.b1 - 3 * self.b2
        test_vector += Tensor([[0.001, 0.002, 0.005]]).T
        actual_coordinates = self.space.project(test_vector)
        expected_coordinates = Tensor([[2, -3]]).T
        assert_close(expected_coordinates, actual_coordinates, rtol=0.01, atol=0.05)

    """
    Tests projection method when multiple vectors should be projected.
    """
    def test_project_non_orthogonal_when_multiple_vectors(self):
        test_vector = hstack([self.zero_point + self.b1 + self.b2,
                              self.zero_point - self.b1 - self.b2])
        actual_coordinates = self.space.project(test_vector)
        expected_coordinates = Tensor([[1., 1.], [-1., -1.]]).T
        assert_close(expected_coordinates, actual_coordinates)

    def test_project_orthogonal_when_multiple_vectors(self):
        self.space.is_orthogonal = True
        test_vector = hstack([self.zero_point + self.b1 + self.b2,
                              self.zero_point - self.b1 - self.b2])
        actual_coordinates = self.space.project(test_vector)
        expected_coordinates = Tensor([[1., 1.], [-1., -1.]]).T
        assert_close(expected_coordinates, actual_coordinates)

    """
    Tests get_point function when multiple coordinate vectors are provided.
    """
    def test_get_point_when_multiple_coordinate_vectors(self):
        points = hstack([self.zero_point + self.b1 + self.b2,
                              self.zero_point - self.b1 - self.b2])
        coordinates = hstack([Tensor([[1., 1.]]).T, Tensor([[-1., -1.]]).T])
        reconstructed = self.space.get_point(coordinates)
        assert_close(points, reconstructed)


if __name__ == '__main__':
    unittest.main()
