import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Sequence


class _Function:
    """
    Base class for vector-valued functions f: R^n -> R or R^n -> R^m with optional Jacobian computation.

    Attributes:
        ndim (int): x dimensionality.
        mdim (int): y dimensionality.
        requires_grad (bool): If True, compute and store the Jacobian on each call.
        dtype (np.dtype): Data type used for internal computations.
        _item (Optional[np.ndarray]): Last computed function values.
        _grad (Optional[np.ndarray]): Last computed Jacobian matrix.
        _hess (Optional[np.ndarray]): Last computed Hessian matrix.
    """
    def __init__(
            self,
            ndim: int,
            mdim: int = 1,
            requires_grad: bool = True,
            requires_hess: bool = True,
            dtype: np.dtype = np.float64
    ) -> None:
        """
        Initialise the function

        :param requires_grad: Whether to compute the Jacobian on call
        :param requires_hess: Whether to compute the Hessian on call
        :param dtype: Data type in computation
        """

        assert requires_grad or (not requires_hess), \
            'Cannot compute Hessian without Jacobian computation!'

        self._item: Optional[np.ndarray] = None
        self._grad: Optional[np.ndarray] = None
        self._hess: Optional[np.ndarray] = None

        # If requires additional computation
        self.requires_grad: bool = requires_grad
        self.requires_hess: bool = requires_hess

        # Digits nums type
        self.dtype: np.dtype = dtype

        # Input and output dimensions
        self.ndim = ndim
        self.mdim = mdim

    def _compute_item(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Compute function values for each point in x.

        Must be overridden in subclasses.

        :param x: Input array of shape (batch_size, ndim).
        :return: Array of function values of shape (batch_size, mdim) or (batch_size,).
        """
        raise NotImplementedError("_compute_item must be implemented in subclasses")

    def _compute_grad(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Compute the Jacobian matrices for each point in x.

        Must be overridden in subclasses if requires_grad=True.

        :param x: Input array of shape (batch_size, ndim).
        :return: Array of Jacobians of shape (batch_size, ndim) or (batch_size, ndim, mdim).
        """
        raise NotImplementedError("_compute_grad must be implemented in subclasses")

    def _compute_hess(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        """
        Compute the Hessian matrices for each point in x.

        Must be overridden in subclasses if second-order information is needed.

        :param x: Input array of shape (batch_size, ndim).
        :return: Array of Hessians of shape
                 - (batch_size, ndim, ndim) for scalar-valued functions, or
                 - (batch_size, ndim, ndim, mdim) for vector-valued functions.
        """
        raise NotImplementedError("_compute_hessian must be implemented in subclasses")

    def __call__(self, x: Union[Sequence[float], np.ndarray]) -> '_Function':
        """
        Method computes the function value at a given point x (and optionally the Jacobian)

        :param x: A sequence or NumPy array of length 2
        :return: self
        """

        # Sequence to array
        x = np.asarray(x, dtype=self.dtype)

        # Check correctness
        assert x.shape[-1] == self.ndim, f'Input x must be a array of length {self.ndim}, got shape {x.shape}'

        # Extract coordinates
        x = x.reshape(-1, self.ndim)

        # Compute values
        self._item = self._compute_item(x).squeeze()

        # Compute Jacobian Matrix
        if self.requires_grad:
            self._grad = self._compute_grad(x).squeeze()

            # Compute Hessian
            if self.requires_hess:
                self._hess = self._compute_hess(x).squeeze()

        return self

    @property
    def item(self) -> Union[float, np.ndarray]:
        """
        Return the last computed value of the function
        """
        assert self._item is not None, f'Item not calculated yet!'

        return self._item

    @property
    def grad(self) -> Union[float, np.ndarray]:
        """
        Returns last computed Jacobian matrix(or derivative) of the function
        """
        assert self.requires_grad, f"Gradient wasn't requested!"
        assert self._grad is not None, f'Gradient not calculated yet!'

        return self._grad

    @property
    def hess(self) -> Union[float, np.ndarray]:
        """
        Returns last computed Jacobian matrix(or derivative) of the function
        """
        assert self.requires_hess, f"Hessian wasn't requested!"
        assert self._hess is not None, f'Hessian not calculated yet!'

        return self._hess

    def __repr__(self) -> str:
        """
        Returns representation of the function
        """
        class_name = self.__class__.__name__
        item_str = (
            f"Item:\n{self._item}\n" if self._item is not None
            else "Item: not computed\n"
        )
        grad_str = (
            f"Grad:\n{self._grad}" if self.requires_grad and self._grad is not None
            else "Grad: not computed or not requested"
        )

        return (f"{class_name}(requires_grad={self.requires_grad}, dtype={self.dtype}):\n"
                f"  - {item_str}\n  - {grad_str}")


class Rosenbrock(_Function):
    """
    Rosenbrock function implementation with computing
    function value, Jacobian and Hessian matrices.
    """

    def _compute_item(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]
        item = 100 * (x1 - x0 ** 2) ** 2 + 5 * (1 - x0) ** 2

        return item

    def _compute_grad(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]

        df_dx0 = 10 * (x0 - 1) - 400 * x0 * (x1 - x0 ** 2)
        df_dx1 = 200 * (x1 - x0 ** 2)

        grad = np.stack([df_dx0, df_dx1], axis=1)

        return grad

    def _compute_hess(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]

        hess00 = 1200 * x0 ** 2 - 400 * x1 + 10
        hess11 = 200 * np.ones_like(x0)
        hess01 = -400 * x0

        row0 = np.stack([hess00, hess01], axis=1)
        row1 = np.stack([hess01, hess11], axis=1)

        hess = np.stack([row0, row1], axis=1)

        return hess


class Himmelblau(_Function):
    """
    Himmelblau function implementation with computing
    function value, Jacobian and Hessian matrices.
    """

    def _compute_item(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]
        item = (x0 ** 2 + x1 - 11) ** 2 + (x0 + x1 ** 2 - 7) ** 2

        return item

    def _compute_grad(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]

        df_dx0 = 4 * x0 * (x0 ** 2 + x1 - 11) + 2 * (x0 + x1 ** 2 - 7)
        df_dx1 = 2 * (x0 ** 2 + x1 - 11) + 4 * x1 * (x0 + x1 ** 2 - 7)

        grad = np.stack([df_dx0, df_dx1], axis=1)

        return grad

    def _compute_hess(self, x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        x0, x1 = x[:, 0], x[:, 1]

        hess00 = 12 * x0 ** 2 + 4 * x1 - 42
        hess11 = 4 * x0 + 12 * x1 ** 2 - 26
        hess01 = 4 * x0 + 4 * x1

        row0 = np.stack([hess00, hess01], axis=1)
        row1 = np.stack([hess01, hess11], axis=1)

        hess = np.stack([row0, row1], axis=1)

        return hess
