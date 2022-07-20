# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm


class PersistablePlot:
    def _init_plot(self):
        if not plt.fignum_exists(self._fig_num):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            self._fig = fig
            self._hilbert_ax = axes[0]
            self._vineyard_ax = axes[1]
            self._fig_num = fig.number

    def __init__(self, hilbert_call_on_click):
        self._fig_num = -1
        self._fig = None
        self._hilbert_ax = None
        self._vineyard_ax = None
        self._hilbert_current_points = None
        self._hilbert_call_on_click = hilbert_call_on_click

        # prominence vineyard
        self._artists = []
        self._labels = []

        self._init_plot()

        def hilbert_on_click(event):
            ax = self._hilbert_ax
            if event.inaxes != ax:
                return
            if event.button == 1:
                vineyard_parameters = self._hilbert_call_on_click(
                    [event.xdata, event.ydata]
                )
                points = np.array(list(vineyard_parameters.values()))
                if self._hilbert_current_points is not None:
                    self._hilbert_current_points.remove()
                self._hilbert_current_points = ax.scatter(points[:, 0], points[:, 1], c="blue")
                ax.figure.canvas.draw_idle()
                ax.figure.canvas.flush_events()

        self._hilbert_ax.figure.canvas.mpl_connect(
            "button_press_event", hilbert_on_click
        )

    def plot_hilbert_function(self, xs, ys, max_dim, dimensions, colormap="binary"):
        ax = self._hilbert_ax
        cmap = cm.get_cmap(colormap)
        im = ax.imshow(
            dimensions[::-1],
            cmap=cmap,
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
        )
        # ntics = 10
        # bounds = list(range(0, max_dim, max_dim // ntics))
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="max")
        # ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax)
        im.set_clim(0, max_dim)
        ax.set_xlabel("distance scale")
        ax.set_ylabel("density threshold")
        ax.set_title("Hilbert function")
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()

    def add_artist_labels(self, artist, label):
        ax = self._vineyard_ax

        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self._artists += [artist]
        self._labels += [label]

        def hover(event):
            if event.inaxes != ax:
                return
            info = ""
            for aa, artist in enumerate(self._artists):
                cont, _ = artist.contains(event)
                if not cont:
                    continue
                lbl = self._labels[aa]
                info += str(lbl) + ";    "

            ax.format_coord = lambda x, y: info

        ax.figure.canvas.mpl_connect("motion_notify_event", hover)

    def plot_prominence_vineyard(
        self,
        vineyard,
        interpolate=True,
        areas=True,
        points=False,
        log_prominence=True,
        colormap="viridis",
    ):
        ax = self._vineyard_ax
        times = vineyard._parameter_indices
        vines = vineyard._vineyard_to_vines()
        num_vines = min(len(vines), vineyard._firstn)
        cmap = cm.get_cmap(colormap)
        colors = list(cmap(np.linspace(0, 1, num_vines)[::-1]))
        last = colors[-1]
        colors.extend([last for _ in range(num_vines - vineyard._firstn)])
        if areas:
            for i in range(len(vines) - 1):
                artist = ax.fill_between(
                    times, vines[i][1], vines[i + 1][1], color=colors[i]
                )
                self.add_artist_labels(artist, "gap " + str(i + 1))
            ax.fill_between(
                times, vines[len(vines) - 1][1], 0, color=colors[len(vines) - 1]
            )
            self.add_artist_labels(artist, "gap " + str(i + 1))
        for i, tv in enumerate(vines):
            times, vine = tv
            for vine_part, time_part in vineyard._vine_parts(vine):
                if interpolate:
                    artist = ax.plot(time_part, vine_part, c="black")
                if points:
                    artist = ax.plot(time_part, vine_part, "o", c="black")
                vineyard._values.extend(vine_part)
        ymax = max(vineyard._values)
        for t in times:
            artist = ax.vlines(x=t, ymin=0, ymax=ymax, color="black", alpha=0.1)
            self.add_artist_labels(
                artist,
                "parameter: (({:.3e},{:.3e}),({:.3e},{:.3e}))".format(
                    vineyard._parameters[t][0][0],
                    vineyard._parameters[t][0][1],
                    vineyard._parameters[t][1][0],
                    vineyard._parameters[t][1][1],
                ),
            )
        ax.set_xticks([])
        ax.set_xlabel("parameter")
        if log_prominence:
            ax.set_ylabel("log-prominence")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("prominence")
        ax.set_ylim(
            [np.quantile(np.array(vineyard._values), 0.05), max(vineyard._values)]
        )
        ax.set_title("Prominence vineyard")
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()


# class ProminenceVineyardHoverManager:
#    def __init__(self, ax):
#        assert isinstance(ax, mpl.axes.Axes)
#
#        # def onclick(event):
#        #    if event.button == 1:
#        #         ax.scatter(event.xdata),event.ydata)
#        # ax.figure.canvas.mpl_connect('button_press_event',onclick)
#        # plt.show()
#        # plt.draw()
#
#        self.ax = ax
#
