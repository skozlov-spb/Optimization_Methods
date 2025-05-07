import numpy as np

from tqdm.auto import tqdm

from functools import wraps

from typing import Union, Sequence, Dict, Callable
from packages.functions import _Function


def _counter(func):
    """
    Decorator to count calls to a method (shared across instances)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        return func(*args, **kwargs)

    wrapper.count = 0

    return wrapper


class _Optimizer:
    """
    Base class for iterative optimization methods minimizing a scalar function

    Attributes:
        _f (_Function): Target function (must output scalar)
        _x (np.ndarray): Current estimate of the minimizer
        _grad (np.ndarray): Gradient at current x
        _alpha (np.ndarray): Constant step size
        _lambda (np.ndarray): Compression ratio
        _eps (np.ndarray): Armijo parameter
        _grad (np.ndarray): Gradient at current x
        _max_iter (int): Maximum number of iterations
        _delta (float): Gradient-norm threshold for convergence
        _logging_steps (int): Number of steps between logs
        history (Dict[str, list]): Logs of 'points', 'alphas', and 'grad_norm'
    """
    def __init__(
            self,
            function: _Function,
            x_start: Union[Sequence[float], np.ndarray],
            backtrack_alpha: float = 1,
            backtrack_lambda: float = 0.5,
            backtrack_eps: float = 1e-4,
            delta: float = 1e-6,
            max_iter: int = 10 ** 6,
            logging_steps: int = 0
    ) -> None:
        """
        Initialize the optimizer.

        :param function: Callable object with .grad property returning gradient
        :param x_start: Starting point for optimization (shape: [ndim])
        :param backtrack_alpha: Initial backtracking step size
        :param backtrack_lambda: Initial backtracking compression ratio
        :param backtrack_eps: Initial backtracking Armijo parameter
        :param delta: Convergence threshold on gradient norm
        :param max_iter: Maximum allowed iterations
        :param logging_steps: If >0, update progress bar every logging_steps iterations
        """

        assert function.mdim == 1, "Output dimension must be scalar!"

        self._f: _Function = function
        self._x: np.ndarray = x_start.copy()
        self._grad: np.ndarray = self._f(self._x).grad

        # Backtracking init
        self._alpha: float = backtrack_alpha
        self._lambda: float = backtrack_lambda
        self._eps: float = backtrack_eps

        # Stopping Criterion
        self._max_iter: int = max_iter
        self._delta: float = delta
        self._logging_steps: int = logging_steps

        # Logging List
        self.history: Dict[str, list] = {
            'points': [x_start.copy()],
            'alphas': [],
            'grad_norm': []
        }

    def _backtracking(self) -> float:
        """
        Line-search with backtracking
        Must be implemented in subclasses

        :return: alpha
        """

        raise NotImplementedError("_choose_alpha must be implemented in subclasses")

    @_counter
    def _step(self) -> None:
        """
        Perform a single update: choose step size and update x and grad
        Must be implemented in subclasses
        """

        raise NotImplementedError("_step must be implemented in subclasses")

    def argmin(self) -> np.ndarray:
        """
        Run optimization until convergence or max_iter reached.

        :return: Estimated minimizer (ndarray)
        """
        grad_norm = np.linalg.norm(self._grad)

        progress_bar = tqdm(desc=f"‖grad‖ = {grad_norm:.2e} | alpha = None") if self._logging_steps > 0 else None

        while grad_norm > self._delta and self._step.count < self._max_iter:  # Stopping criterion
            self._step()

            # Save grad and point
            self.history['points'].append(self._x.copy())
            self.history['grad_norm'].append(grad_norm)

            # Calculate current gradient norm
            grad_norm = np.linalg.norm(self._grad)

            if self._logging_steps > 0:  # Update progress bar
                if self._step.count % self._logging_steps == 0:
                    progress_bar.set_description_str(f"‖grad‖ = {self.history['grad_norm'][-1]:.2e} | "
                                                     f"alpha = {self.history['alphas'][-1]:.2e}")

                progress_bar.update(1)

        # Reset steps counts
        self.__class__._step.count = 0

        if self._logging_steps > 0:
            progress_bar.close()

        return self._x

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NelderMead(_Optimizer):
    """
    Nelder–Mead (downhill simplex) method for unconstrained optimization without derivatives.

    Builds and deforms a simplex in n-dimensional space by repeatedly performing
    reflection, expansion, contraction, and shrink steps to approach a local minimum.

    History:
        'points': list of simplex best vertex after each iteration
    """
    def __init__(
            self,
            function: _Function,
            x_start: Union[Sequence[float], np.ndarray],
            alpha: float = 1.0,
            gamma: float = 2.0,
            rho: float = 0.5,
            sigma: float = 0.5,
            delta: float = 1e-6,
            max_iter: int = 10 ** 6,
            logging_steps: int = 0
    ) -> None:
        """
        Initialize the Nelder–Mead optimizer.

        :param function: Target _Function to minimize (must output scalar)
        :param x_start: Initial point, array-like of shape (ndim)
        :param alpha: Reflection coefficient (default 1.0)
        :param gamma: Expansion coefficient (default 2.0)
        :param rho: Contraction coefficient (default 0.5)
        :param sigma: Shrink (reduction) coefficient (default 0.5)
        :param delta: Convergence tolerance on simplex spread (default 1e-6)
        :param max_iter: Maximum number of iterations (default 1e6)
        :param logging_steps: If >0, update a tqdm progress bar every this many steps
        """

        super().__init__(
            function,
            x_start,
            delta=delta,
            max_iter=max_iter,
            logging_steps=logging_steps
        )

        # Turn off gradient and hessian calculating
        self._f.requires_grad = False
        self._f.requires_hess = False

        # Method parameters
        self._alpha: float = alpha
        self._gamma: float = gamma
        self._rho: float = rho
        self._sigma: float = sigma

        # Logging List
        self.history: Dict[str, list] = {
            'points': [x_start.copy()]
        }

        # Init simplex
        self._simplex = [self._x.copy()]

        for i in range(self._f.ndim):
            val = self._x.copy()
            val[i] += 0.05 * (val[i] if val[i] != 0 else 1.0)

            self._simplex.append(val)

        self._simplex = np.array(self._simplex)
        self._f_simplex = self._f(self._simplex).item

    @_counter
    def _step(self) -> None:
        """
        Perform one Nelder–Mead update on the current simplex:

        1. Sort vertices by function value.
        2. Compute centroid of all but the worst point.
        3. Reflect worst point across centroid.
        4. If reflection improves beyond best, try expansion.
        5. Else if reflection is intermediate, accept it.
        6. Else if reflection is better than worst, perform external contraction.
        7. Else perform internal contraction or shrink entire simplex.
        8. Update current estimate to the best vertex.
        """

        sorted_ids = np.argsort(self._f_simplex)
        self._simplex = self._simplex[sorted_ids]
        self._f_simplex = self._f_simplex[sorted_ids]

        x_min, x_max = self._simplex[[0, -1]]
        x_centre = np.mean(self._simplex[:-1], axis=0)

        f_min, f_second, f_max = self._f_simplex[[0, -2, -1]]

        x_r = (1 + self._alpha) * x_centre - self._alpha * x_max
        f_r = self._f(x_r).item

        if f_r < f_min:  # Расширение
            x_e = (1 - self._gamma) * x_centre + self._gamma * x_r
            f_e = self._f(x_e).item

            self._simplex[-1], self._f_simplex[-1] = (x_e, f_e) if f_e < f_r else (x_r, f_r)

        elif f_min <= f_r < f_second:  # Отражение
            self._simplex[-1], self._f_simplex[-1] = x_r, f_r

        elif f_second <= f_r < f_max:
            # Внешнее сжатие
            x_c = (1 - self._rho) * x_centre + self._rho * x_r
            f_c = self._f(x_c).item

            self._simplex[-1], self._f_simplex[-1] = x_c, f_c

        else:  # Сжатие
            x_c = (1 - self._rho) * x_centre + self._rho * x_max
            f_c = self._f(x_c).item

            if f_c < self._f_simplex[-1]:
                self._simplex[-1], self._f_simplex[-1] = x_c, f_c
            else:
                for i in range(1, self._f.ndim + 1):
                    self._simplex[i] = (1 - self._sigma) * x_min + self._sigma * self._simplex[i]
                    self._f_simplex[i] = self._f(self._simplex[i]).item

        # Update point
        self._x = self._simplex[min(range(self._f_simplex.shape[0]), key=lambda idx: self._f_simplex[idx])]

    def argmin(self) -> np.ndarray:
        """
        Run the Nelder–Mead algorithm until convergence or max_iter reached.

        :return: Estimated minimizer as ndarray of shape (ndim)
        """

        item_norm = np.linalg.norm(self._f(self._x).item)
        progress_bar = tqdm(desc=f"‖value‖ = {item_norm:.2e}") if self._logging_steps > 0 else None

        # Stopping criterion
        while self._f_simplex.max() - self._f_simplex.min() > self._delta and self._step.count < self._max_iter:
            self._step()

            # Save point
            self.history['points'].append(self._x.copy())

            # Calculate current item norm
            item_norm = np.linalg.norm(self._f(self._x).item)

            if self._logging_steps > 0:  # Update progress bar
                if self._step.count % self._logging_steps == 0:
                    progress_bar.set_description_str(f"‖value‖ = {item_norm:.2e}")

                progress_bar.update(1)

        # Reset steps counts
        self.__class__._step.count = 0

        if self._logging_steps > 0:
            progress_bar.close()

        return self._x


class GradientDescent(_Optimizer):
    """
    Classic gradient descent with optional exact line search
    """

    def __init__(
            self,
            function: _Function,
            x_start: Union[Sequence[float], np.ndarray],
            backtrack_alpha: float = 1,
            backtrack_lambda: float = 0.5,
            backtrack_eps: float = 1e-4,
            delta: float = 1e-6,
            max_iter: int = 10 ** 6,
            logging_steps: int = 0
    ) -> None:

        # Turn on gradient calculating
        function.requires_grad = True
        function.requires_hess = False

        super().__init__(
            function,
            x_start,
            backtrack_alpha,
            backtrack_lambda,
            backtrack_eps,
            delta,
            max_iter,
            logging_steps
        )

    def _backtracking(self) -> float:
        """
        Line-search with backtracking
        """
        alpha = self._alpha
        curr_item = self._f.item

        squared_grad_norm = np.linalg.norm(self._grad) ** 2
        x = self._x - alpha * self._grad

        while self._f(x).item > curr_item - alpha * self._eps * squared_grad_norm:
            alpha *= self._lambda
            x = self._x - alpha * self._grad

        return alpha

    @_counter
    def _step(self) -> None:
        """
        Perform one gradient descent update with optimal alpha in [0,1]
        """

        # Find optimal step size using Backtracking
        alpha = self._backtracking()

        # Update point and gradient
        self._x -= alpha * self._grad
        self._grad = self._f(self._x).grad

        # Save alpha
        self.history['alphas'].append(alpha)


class ConjugateGradient(_Optimizer):
    """
    Fletcher-Reeves nonlinear conjugate gradient method

    Attributes:
        like a base class _Optimizer attributes
        +
        _dir (np.ndarray): Current direction gradient
    """

    def __init__(
            self,
            function: _Function,
            x_start: Union[Sequence[float], np.ndarray],
            backtrack_alpha: float = 1,
            backtrack_lambda: float = 0.5,
            backtrack_eps: float = 1e-4,
            delta: float = 1e-6,
            max_iter: int = 10 ** 6,
            logging_steps: int = 0
    ) -> None:

        # Turn on gradient calculating
        function.requires_grad = True
        function.requires_hess = False

        super().__init__(
            function,
            x_start,
            backtrack_alpha,
            backtrack_lambda,
            backtrack_eps,
            delta,
            max_iter,
            logging_steps
        )

        # Keep direction vector, combines current gradient and last chosen direction
        self._dir: np.ndarray = -self._f(self._x).grad.copy()

    def _backtracking(self) -> float:
        """
        Line-search with backtracking
        """
        alpha = self._alpha
        curr_item = self._f.item

        squared_grad_norm = self._grad.dot(self._dir)
        x = self._x + alpha * self._dir

        while self._f(x).item > curr_item - alpha * self._eps * squared_grad_norm:
            alpha *= self._lambda
            x = self._x + alpha * self._dir

        return alpha

    @_counter
    def _step(self) -> None:
        """
        One iteration of the Fletcher–Reeves conjugate gradient method
        """

        # Find optimal step size using Backtracking
        alpha = self._backtracking()

        # Update point and direction
        new_x = self._x + alpha * self._dir
        new_grad = self._f(new_x).grad

        # Fletcher-Rives method
        if self._step.count % self._f.ndim == 0:
            self._dir = -new_grad.copy()
        else:
            beta = (np.linalg.norm(new_grad) /
                    np.linalg.norm(self._grad)) ** 2

            self._dir = -new_grad + beta * self._dir

        self._grad = new_grad
        self._x = new_x

        # Save alpha
        self.history['alphas'].append(alpha)


class NewtonMethod(_Optimizer):
    """
    Newton's method with exact line search (when Hessian may be indefinite).

    Attributes:
        like a base class _Optimizer attributes
        +
        _inv_hess (np.ndarray): Current inverse Hessian matrix
    """
    def __init__(
            self,
            function: _Function,
            x_start: Union[Sequence[float], np.ndarray],
            backtrack_alpha: float = 1,
            backtrack_lambda: float = 0.5,
            backtrack_eps: float = 1e-4,
            delta: float = 1e-6,
            max_iter: int = 10 ** 6,
            logging_steps: int = 0
    ) -> None:

        # Turn on gradient and hessian calculating
        function.requires_grad = True
        function.requires_hess = True

        super().__init__(
            function,
            x_start,
            backtrack_alpha,
            backtrack_lambda,
            backtrack_eps,
            delta,
            max_iter,
            logging_steps
        )

        # Keep direction vector, combines current jacobian and current hessian
        self._dir: np.ndarray = -np.linalg.solve(self._f(self._x).hess, self._grad)

    def _backtracking(self) -> float:
        """
        Line-search with backtracking
        """
        alpha = self._alpha
        curr_item = self._f.item

        squared_grad_norm = self._grad.dot(self._dir)
        x = self._x + alpha * self._dir

        while self._f(x).item > curr_item - alpha * self._eps * squared_grad_norm:
            alpha *= self._lambda
            x = self._x + alpha * self._dir

        return alpha

    @_counter
    def _step(self) -> None:
        """
        One iteration of Newton's method with line search.
        """

        # Find optimal step size using Backtracking
        alpha = self._backtracking()

        # Update point and gradient
        self._x += alpha * self._dir
        self._grad = self._f(self._x).grad
        self._dir = -np.linalg.solve(self._f.hess, self._grad)

        # Save alpha
        self.history['alphas'].append(alpha)
