# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


class PersistablePlot:
    def _init_plot(self):
        if not plt.fignum_exists(self._fig_num):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            plt.subplots_adjust(bottom=0.2)
            self._fig = fig
            self._hilbert_ax = axes[0]
            self._vineyard_ax = axes[1]
            self._fig_num = fig.number

    def __init__(
        self, hilbert_call_on_click, vineyard_call_on_click, compute_prominence_vineyard
    ):
        self._fig_num = -1
        self._fig = None
        self._hilbert_ax = None
        self._vineyard_ax = None
        self._hilbert_current_points_plotted_on = None
        self._hilbert_current_lines_plotted_on = []
        self._hilbert_current_polygon_plotted_on = None
        self._vineyard_current_points_plotted_on = None
        self._vineyard_values = []
        self._hilbert_call_on_click = hilbert_call_on_click
        self._vineyard_call_on_click = vineyard_call_on_click
        self._compute_prominence_vineyard = compute_prominence_vineyard

        # prominence vineyard
        self._gaps = []
        self._gap_numbers = []
        self._lines = []
        self._line_index = []

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
                if self._hilbert_current_points_plotted_on is not None:
                    self._hilbert_current_points_plotted_on.remove()
                if len(self._hilbert_current_lines_plotted_on) > 0:
                    for x in self._hilbert_current_lines_plotted_on:
                        x.pop(0).remove()
                    self._hilbert_current_lines_plotted_on = []
                if self._hilbert_current_polygon_plotted_on is not None:
                    self._hilbert_current_polygon_plotted_on.remove()
                    self._hilbert_current_polygon_plotted_on = None
                self._hilbert_current_points_plotted_on = ax.scatter(
                    points[:, 0], points[:, 1], c="blue", s=10
                )
                if len(points) >= 2:
                    self._hilbert_current_lines_plotted_on.append(
                        ax.plot(
                            [points[0, 0], points[1, 0]],
                            [points[0, 1], points[1, 1]],
                            c="blue",
                            linewidth=1,
                        )
                    )
                if len(points) >= 4:
                    self._hilbert_current_lines_plotted_on.append(
                        ax.plot(
                            [points[2, 0], points[3, 0]],
                            [points[2, 1], points[3, 1]],
                            c="blue",
                            linewidth=1,
                        )
                    )
                    polygon = Polygon(
                        [points[0], points[1], points[3], points[2]],
                        True,
                        color="red",
                        alpha=0.1,
                    )
                    ax.add_patch(polygon)
                    self._hilbert_current_polygon_plotted_on = polygon
                if len(points) >= 4:
                    info = "Prominence vineyard with ({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}) to ({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}) selected.".format(
                        points[0, 0],
                        points[0, 1],
                        points[1, 0],
                        points[1, 1],
                        points[2, 0],
                        points[2, 1],
                        points[3, 0],
                        points[3, 1],
                    )
                    ax.format_coord = lambda x, y: info

                ax.figure.canvas.draw_idle()
                ax.figure.canvas.flush_events()

        self._hilbert_ax.figure.canvas.mpl_connect(
            "button_press_event", hilbert_on_click
        )

        def vineyard_on_click(event):
            ax = self._vineyard_ax
            if event.inaxes != ax:
                return

            if event.button == 1:
                # info = ""

                # gaps
                gap = None
                aas = []
                for aa, artist in enumerate(self._gaps):
                    cont, _ = artist.contains(event)
                    if not cont:
                        continue
                    aas.append(aa)
                if len(aas) > 0:
                    # aa = aas[-1]
                    gap = aas[-1]
                    # lbl = self._gap_numbers[aa]
                    # info += "gap: " + str(lbl) + ";    "

                # lines
                line_index = None
                aas = []
                for aa, artist in enumerate(self._lines):
                    cont, _ = artist.contains(event)
                    if not cont:
                        continue
                    aas.append(aa)
                if len(aas) > 0:
                    # aa = aas[-1]
                    line_index = aas[-1]
                    # lbl = self._line_index[aa]
                    # info += "line: " + str(lbl) + ";    "

                if gap is not None and line_index is not None:
                    parameters, n_clusters = self._vineyard_call_on_click(
                        gap + 1, line_index
                    )
                    if self._vineyard_current_points_plotted_on is not None:
                        self._vineyard_current_points_plotted_on.remove()
                    self._vineyard_current_points_plotted_on = ax.scatter(
                        [event.xdata], [event.ydata], c="blue", s=40
                    )

                    info = "Parameter ({:.2f}, {:.2f}) -> ({:.2f}, {:.2f}), with n_clusters = {:d} selected.".format(
                        parameters[0][0],
                        parameters[0][1],
                        parameters[1][0],
                        parameters[1][1],
                        n_clusters,
                    )
                    ax.format_coord = lambda x, y: info

                    ax.figure.canvas.draw_idle()
                    ax.figure.canvas.flush_events()

        self._vineyard_ax.figure.canvas.mpl_connect(
            "button_press_event", vineyard_on_click
        )

        def plot_prominence_vineyard_button(event):
            vineyard = self._compute_prominence_vineyard()
            self.plot_prominence_vineyard(vineyard)

        self._ax_button = plt.axes([0.05, 0.05, 0.15, 0.075])
        self._button_compute_and_plot = Button(self._ax_button, "Compute vineyard")
        self._button_compute_and_plot.on_clicked(plot_prominence_vineyard_button)

    def plot_hilbert_function(self, xs, ys, max_dim, dimensions, colormap="binary"):
        ax = self._hilbert_ax
        cmap = cm.get_cmap(colormap)
        im = ax.imshow(
            dimensions[::-1],
            cmap=cmap,
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
        )
        im.set_clim(0, max_dim)
        ax.set_xlabel("distance scale")
        ax.set_ylabel("density threshold")
        ax.set_title("component counting function")
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()

    def _add_gap(self, artist, number):

        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self._gaps += [artist]
        self._gap_numbers += [number]

    def _add_line_index(self, artist, number):

        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self._lines += [artist]
        self._line_index += [number]

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

        # TODO: abstract this
        ax.clear()
        self._gaps = []
        self._gap_numbers = []
        self._lines = []
        self._line_index = []

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
                self._add_gap(artist, i + 1)
            ax.fill_between(
                times, vines[len(vines) - 1][1], 0, color=colors[len(vines) - 1]
            )
            self._add_gap(artist, i + 1)
        for i, tv in enumerate(vines):
            times, vine = tv
            for vine_part, time_part in vineyard._vine_parts(vine):
                if interpolate:
                    artist = ax.plot(time_part, vine_part, c="black")
                if points:
                    artist = ax.plot(time_part, vine_part, "o", c="black")
                self._vineyard_values.extend(vine_part)
        ymax = max(self._vineyard_values)
        for t in times:
            artist = ax.vlines(x=t, ymin=0, ymax=ymax, color="black", alpha=0.1)
            self._add_line_index(artist, t)
        ax.set_xticks([])
        ax.set_xlabel("parameter")
        if log_prominence:
            ax.set_ylabel("log-prominence")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("prominence")
        values = np.array(self._vineyard_values)
        ax.set_ylim([np.quantile(values[values > 0], 0.05), max(values)])
        ax.set_title("Prominence vineyard")
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()
