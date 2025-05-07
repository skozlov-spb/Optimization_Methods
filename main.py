import matplotlib.pyplot as plt

from packages.helpers import plot_surface, pipeline
from packages.functions import Rosenbrock, Himmelblau
from packages import config
from packages.optim import GradientDescent, ConjugateGradient, NewtonMethod, NelderMead


def main():
    # Задаем функции
    rb = Rosenbrock(ndim=2, requires_hess=False)
    hb = Himmelblau(ndim=2, requires_hess=False)

    # Выполняем методы
    named_methods = {
        'Метод деформируемого многогранника': (
            NelderMead,
            config.NELDERMEAD_RB_PARAMS,
            config.NELDERMEAD_HB_PARAMS
        ),
        'Метод Градиентного Спуска': (
            GradientDescent,
            config.GRADIENT_DESCENT_RB_PARAMS,
            config.GRADIENT_DESCENT_HB_PARAMS
        ),
        'Метод сопряженных направлений': (
            ConjugateGradient,
            config.CONJUGATE_GRADIENT_RB_PARAMS,
            config.CONJUGATE_GRADIENT_HB_PARAMS
        ),
        'Метод Ньютона': (
            NewtonMethod,
            config.NEWTON_METHOD_RB_PARAMS,
            config.NEWTON_METHOD_HB_PARAMS
        ),
    }

    for name, (method, rb_params, hb_params) in named_methods.items():
        print(f"{75 * '='}\n{name}\n{75 * '='}\n")

        # Разделим на два графика
        fig, (ax_rb, ax_hb) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(name)

        print('Функция Розенброка')
        # Отображаем поверхность
        plot_surface(
            fig=fig,
            ax=ax_rb,
            function=rb,
            xlim=(-2, 2),
            ylim=(-1, 3),
            num_minima=1,
            title=f"{rb.__class__.__name__} Trajectories"
        )

        # Прогон метода и отображение траектории
        pipeline(
            function=rb,
            optim_method=method,
            x_start=[-1.54, -0.32],
            plot=True,
            ax=ax_rb,
            **rb_params
        )

        print('Функция Химмельблау')
        plot_surface(
            fig=fig,
            ax=ax_hb,
            function=hb,
            xlim=(-6, 6),
            ylim=(-6, 6),
            num_minima=4,
            title=f"{hb.__class__.__name__} Trajectories"
        )

        pipeline(
            function=hb,
            optim_method=method,
            x_start=[2.23, -4.57],
            plot=True,
            ax=ax_hb,
            **hb_params
        )

        # Отобразим траектории
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
