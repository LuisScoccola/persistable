# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from matplotlib.patches import Polygon


class PersistableInteractive:
    def _init_plot(self):
        if not plt.fignum_exists(self._fig_num):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            plt.subplots_adjust(bottom=0.2)
            self._fig = fig
            self._hilbert_ax = axes[0]
            self._vineyard_ax = axes[1]
            self._fig_num = fig.number

    def __init__(
        self,
        persistable
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
        self._persistable = persistable

        ## to be passed/computed later:
        # user-selected bounds for the prominence vineyard
        self._vineyard_parameter_bounds = {}
        # user-selected start and end for a line
        self._line_parameters = None
        # user-selected number of clusters
        self._n_clusters = None
        # the computed prominence vineyard
        self._vineyard = None

        # prominence vineyard
        self._gaps = []
        self._gap_numbers = []
        self._lines = []
        self._line_index = []

        # initialize the plots
        self._init_plot()


        self._hilbert_ax.figure.canvas.mpl_connect(
            "button_press_event", self._hilbert_on_parameter_selection
        )

        self._vineyard_ax.figure.canvas.mpl_connect(
            "button_press_event", self._vineyard_on_parameter_selection
        )

        self._ax_button_clear = plt.axes([0.10, 0.0, 0.15, 0.075])
        self._button_clear_and_plot = Button(self._ax_button_clear, "clear parameters")
        self._button_clear_and_plot.on_clicked(self._hilbert_on_clear_parameter)

        self._ax_button_compute_vineyard = plt.axes([0.30, 0.0, 0.15, 0.075])
        self._button_compute_and_plot = Button(
            self._ax_button_compute_vineyard, "compute vineyard"
        )
        self._button_compute_and_plot.on_clicked(self._plot_prominence_vineyard_button)


    def _update_line_parameters(self, gap, line_index):
        self._line_parameters = self._vineyard._parameters[line_index]
        self._n_clusters = gap
        return self._line_parameters, gap

    def _clear_vineyard_parameter_bounds(self):
        self._vineyard_parameter_bounds = {}
        return self._vineyard_parameter_bounds

    def _update_vineyard_parameter_bounds(self, point):
        if "start1" not in self._vineyard_parameter_bounds:
            self._vineyard_parameter_bounds["start1"] = point
        elif "end1" not in self._vineyard_parameter_bounds:
            st1 = self._vineyard_parameter_bounds["start1"]
            if point[0] < st1[0] or point[1] > st1[1]:
                return self._vineyard_parameter_bounds
            self._vineyard_parameter_bounds["end1"] = point
        elif "start2" not in self._vineyard_parameter_bounds:
            self._vineyard_parameter_bounds["start2"] = point
        elif "end2" not in self._vineyard_parameter_bounds:
            st2 = self._vineyard_parameter_bounds["start2"]
            if point[0] < st2[0] or point[1] > st2[1]:
                return self._vineyard_parameter_bounds
            self._vineyard_parameter_bounds["end2"] = point
        else:
            self._vineyard_parameter_bounds = {}
            self._update_vineyard_parameter_bounds(point)
        return self._vineyard_parameter_bounds


    def parameter_selection(
        self,
        max_dim=20,
        max_k=None,
        bounds_s=None,
        granularity=50,
        n_jobs=4,
    ):
        ss, ks, max_dim, hf = self._persistable.hilbert_function(
            max_dim=max_dim,
            max_k=max_k,
            bounds_s=bounds_s,
            granularity=granularity,
            n_jobs=n_jobs,
        )
        self.plot_hilbert_function(ss, ks, max_dim, hf)

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


    def cluster(self):
        if self._line_parameters is None:
            raise Exception("No parameters for the line were given!")
        else:
            start, end = self._line_parameters
            n_clusters = self._n_clusters
        return self._persistable.cluster(n_clusters, start, end)


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

        ax.set_title("prominence vineyard")

        # TODO: warn that vineyard is empty
        if num_vines == 0:
            return

        cmap = cm.get_cmap(colormap)
        colors = list(cmap(np.linspace(0, 1, num_vines)[::-1]))
        last = colors[-1]
        colors.extend([last for _ in range(num_vines - vineyard._firstn)])
        if areas:
            for i in range(len(vines) - 1):
                artist = ax.fill_between(
                    times, vines[i][1], vines[i + 1][1], color=colors[i]
                )
                self._add_gap_prominence_vineyard(artist, i + 1)
            artist = ax.fill_between(
                times, vines[len(vines) - 1][1], 0, color=colors[len(vines) - 1]
            )
            self._add_gap_prominence_vineyard(artist, len(vines))
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
            self._add_line_prominence_vineyard(artist, t)
        ax.set_xticks([])
        ax.set_xlabel("parameter")
        if log_prominence:
            ax.set_ylabel("log-prominence")
            ax.set_yscale("log")
        else:
            ax.set_ylabel("prominence")
        values = np.array(self._vineyard_values)

        ax.set_ylim([np.quantile(values[values > 0], 0.05), max(values)])
        ax.figure.canvas.draw_idle()
        ax.figure.canvas.flush_events()

    def _plot_prominence_vineyard_button(self, event):
        if len(self._vineyard_parameter_bounds.values()) < 4:
            raise Exception("No parameters chosen!")
        start1 = self._vineyard_parameter_bounds["start1"]
        end1 = self._vineyard_parameter_bounds["end1"]
        start2 = self._vineyard_parameter_bounds["start2"]
        end2 = self._vineyard_parameter_bounds["end2"]

        self._vineyard = self._persistable.compute_prominence_vineyard([start1,end1], [start2,end2])
        self.plot_prominence_vineyard(self._vineyard)

    def _vineyard_on_parameter_selection(self, event):
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
                parameters, n_clusters = self._update_line_parameters(
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

    def _clear_hilbert_parameters(self):
        if self._hilbert_current_points_plotted_on is not None:
            self._hilbert_current_points_plotted_on.remove()
            self._hilbert_current_points_plotted_on = None
        if len(self._hilbert_current_lines_plotted_on) > 0:
            for x in self._hilbert_current_lines_plotted_on:
                x.pop(0).remove()
            self._hilbert_current_lines_plotted_on = []
        if self._hilbert_current_polygon_plotted_on is not None:
            self._hilbert_current_polygon_plotted_on.remove()
            self._hilbert_current_polygon_plotted_on = None

    def _draw_on_hilbert(self, vineyard_parameters):
        ax = self._hilbert_ax
        points = np.array(list(vineyard_parameters.values()))

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

    def _hilbert_on_parameter_selection(self, event):
        ax = self._hilbert_ax
        if event.inaxes != ax:
            return
        if event.button == 1:
            vineyard_parameters = self._update_vineyard_parameter_bounds(
                [event.xdata, event.ydata]
            )
            self._clear_hilbert_parameters()
            self._draw_on_hilbert(vineyard_parameters)

    def _hilbert_on_clear_parameter(self, event):
        _ = self._clear_vineyard_parameter_bounds()
        self._clear_hilbert_parameters()

    def _add_gap_prominence_vineyard(self, artist, number):

        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self._gaps += [artist]
        self._gap_numbers += [number]

    def _add_line_prominence_vineyard(self, artist, number):

        if isinstance(artist, list):
            assert len(artist) == 1
            artist = artist[0]

        self._lines += [artist]
        self._line_index += [number]

