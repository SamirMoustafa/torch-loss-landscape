from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import torch.nn.functional
from torch import Tensor
from torch.linalg import lstsq


class Subspace(ABC):

    @abstractmethod
    def project(self, vector: Tensor) -> Tensor:
        pass

    @abstractmethod
    def get_point(self, coordinate_vector: Tensor) -> Tensor:
        pass


class LinearSubspace(Subspace):
    def __init__(self, basis: Tensor, zero_point: Optional[Tensor], is_orthogonal: bool = False) -> None:
        """
        Describes a linear subspace with basis vectors.
        :param basis: A matrix where each column represents a basis vector.
        :param zero_point: The zero point of the subspace.
        :param is_orthogonal: A boolean indicating if the subspace's basis is orthogonal.
        """
        super().__init__()
        self.basis = basis
        if zero_point is not None:
            assert basis.shape[0] == zero_point.shape[0]
        self.zero_point = zero_point
        self.is_orthogonal = is_orthogonal

    def project(self, vector: Tensor) -> Tensor:
        """
        Project one or multiple column vectors to the subspace. If multiple vectors
        should be projected, each column of the matrix should represent a vector.
        :param vector: A column vector or multiple columns vectors in a matrix to project.
        :return: The coordinates of the projected vectors. Each column is one vector.
        """
        shifted_vector = vector
        if self.zero_point is not None:
            shifted_vector = vector - self.zero_point
        if self.is_orthogonal:
            projected_vector = self.basis.T @ shifted_vector
        else:
            projected_vector = lstsq(self.basis, shifted_vector).solution
        return projected_vector

    def get_point(self, coordinate_vector: Tensor) -> Tensor:
        """
        Gets the point in the subspace by the specified coordinates.
        :param coordinate_vector: A column vector with the coordinates or multiple coordinate
            columns vectors in a matrix.
        :return: The points in the subspace by the specified coordinates.
        """
        result = self.basis @ coordinate_vector
        if self.zero_point is not None:
            result += self.zero_point
        return result


class NonlinearSubspace(Subspace):
    def __init__(self, subspaces: List[Subspace], activation: Callable = torch.nn.functional.relu) -> None:
        self.subspaces = subspaces
        self.activation = activation

    def project(self, vector: Tensor) -> Tensor:
        # project and apply activation function but for the last layer.
        for subspace in self.subspaces[:-1]:
            vector = subspace.project(vector)
            vector = self.activation(vector)
        # only project on last layer, do not apply activation.
        vector = self.subspaces[-1].project(vector)
        return vector

    def get_point(self, coordinate_vector: Tensor) -> Tensor:
        point = coordinate_vector

        for subspace in self.subspaces[::-1][:-1]:
            point = subspace.get_point(point)
            point = self.activation(point)

        point = self.subspaces[0].get_point(point)
        return point
