import time

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Union, Sequence, Tuple, List, Dict, Type

from packages.functions import _Function
from packages.optim import _Optimizer


def plot_surface(
    fig: plt.Figure,
    ax: plt.Axes,
    function: _Function,
    xlim: Optional[Sequence[float]],
    ylim: Optional[Sequence[float]],
    num_minima: int = 1,
    resolution: int = 200,
    title: str = "Function Surface",
) -> None:
    """
    Visualizes function surface

    :param fig: Figure
    :param ax: Axes
    :param function: Target function
    :param xlim: x borders
    :param ylim: y borders
    :param num_minima: number of minima
    :param resolution: Points amount to be displayed
    :param title: Graphic's title
    """
    # Compute Points Net
    x = np.linspace(*xlim, resolution)
    y = np.linspace(*ylim, resolution)
    xx, yy = np.meshgrid(x, y)

    net = np.stack([xx, yy], axis=-1).reshape(-1, 2)

    # Calculate net points function values
    req_grad = function.requires_grad
    function.requires_grad = False

    zz = np.log1p(function(net).item)

    function.requires_grad = req_grad

    # Plot surface
    ids = np.argpartition(zz, kth=num_minima)[:num_minima]

    contour = ax.contourf(xx, yy, zz.reshape(resolution, resolution), levels=50, cmap='coolwarm')
    ax.scatter(net[ids, 0], net[ids, 1], color='red', edgecolor='black', marker='X', s=100, label='Optimal', zorder=11)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(contour, ax=ax)


def _plot_trajectory(
    ax: plt.Axes,
    trajectory: Sequence[float],
    label: str = "Trajectory",
) -> None:
    """
    Visualizes function surface and gradient descent trajectory

    :param ax: Matplotlib Axes to plot on (must be provided if plot is True).
    :param history: Optimizer argument trajectory
    :param label: -> Label for the trajectory line in the plot legend
    """
    # Plot trajectory
    points = np.array(trajectory)
    xs, ys = points[:, 0], points[:, 1]

    ax.plot(xs, ys, marker='o', color='red', markersize=4, linewidth=2, label=label)
    ax.scatter(xs[0], ys[0], color='white', label='Start', edgecolor='black', zorder=10)
    ax.scatter(xs[-1], ys[-1], color='black', label='End', zorder=10)
    ax.legend()


def pipeline(
        function: _Function,
        optim_method: Type[_Optimizer],
        x_start: Union[Sequence[float], np.ndarray],
        plot: bool = False,
        ax: Optional[plt.Axes] = None,
        **kwargs
) -> None:
    """
    Pipeline of finding function minimum

    :param function: Target function to minimize
    :param optim_method: Optimization method class (must accept function and starting point)
    :param x_start: Initial point for optimization
    :param plot: If True, plot the optimization trajectory
    :param ax: Matplotlib Axes to plot on (must be provided if plot is True)
    """

    assert plot or (ax is None), "Axes must be provided if plot is True!"

    start = time.time()

    optimizer = optim_method(function, x_start, **kwargs)
    x_optim = optimizer.argmin()

    end = time.time()

    print('\nРезультаты:')
    print(f'\tЭкстремум: {np.round(x_optim, 2)}'
          f'\n\tЗначение функции: {function(x_optim).item:2e}'
          f'\n\tЗатраченное время {end - start:.2f}s\n\n')

    if plot:  # Plot trajectory
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        _plot_trajectory(
            ax=ax,
            trajectory=optimizer.history['points'],
            label=f"{optimizer.__class__.__name__} Trajectory"
        )
