# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Button
from matplotlib.patches import Polygon

import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
import pandas as pd


class PersistableInteractive:
    # def _init_plot(self):
    #    if not plt.fignum_exists(self._fig_num):
    #        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #        plt.subplots_adjust(bottom=0.2)
    #        self._fig = fig
    #        self._hilbert_ax = axes[0]
    #        self._vineyard_ax = axes[1]
    #        self._fig_num = fig.number

    def __init__(self, persistable, jupyter=False, debug=False):
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

        ## initialize the plots

        default_max_k = self._persistable._maxk
        default_k_step = default_max_k / 20
        default_min_s = self._persistable._connection_radius / 5
        default_max_s = self._persistable._connection_radius * 2
        default_s_step = (default_max_s - default_min_s) / 20
        default_log_granularity = 6
        default_num_jobs = 4
        default_max_dim = 15

        def blank_figure():
            fig = go.Figure(go.Scatter(x=[], y=[]))
            fig.update_layout(template=None)
            fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
            return fig

        if jupyter == True:
            self._app = JupyterDash(__name__)
        else:
            self._app = dash.Dash()
        self._app.layout = html.Div(
            children=[
                dcc.Store(id="stored-ccf"),
                html.H1("title"),
                html.P("prose"),
                html.Div(
                    className="grid",
                    children=[
                        html.Div(
                            children=[
                                html.H2("Component Counting Function"),
                                html.Div(
                                    className="parameters",
                                    children=[
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name", children="max k"
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="input-max-k",
                                                    type="number",
                                                    value=default_max_k,
                                                    min=0,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name", children="min s"
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="input-min-s",
                                                    type="number",
                                                    value=default_min_s,
                                                    min=0,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name", children="max s"
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="input-max-s",
                                                    type="number",
                                                    value=default_max_s,
                                                    min=0,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="number of cores",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="input-num-jobs",
                                                    type="number",
                                                    value=default_num_jobs,
                                                    min=1,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="granularity",
                                                ),
                                                dcc.Slider(
                                                    1,
                                                    9,
                                                    step=None,
                                                    marks={
                                                        i: str(2**i)
                                                        for i in range(1, 10)
                                                    },
                                                    value=default_log_granularity,
                                                    id="input-log-granularity",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                ),
                                                html.Button(
                                                    "compute CCF",
                                                    id="compute-ccf-button",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="max connected components",
                                                ),
                                                dcc.Input(
                                                    id="input-max-components",
                                                    type="number",
                                                    value=default_max_dim,
                                                    min=1,
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H2("Prominence Vineyard"),
                                html.Div(
                                    className="parameters",
                                ),
                            ]
                        ),
                        dcc.Graph(id="hilbert-plot", figure=blank_figure()),
                        html.Div(
                            className="fake-plot",
                        ),
                    ],
                ),
            ],
        )
        self._app.callback(
            dash.Output("stored-ccf", "data"),
            [
                dash.Input("compute-ccf-button", "n_clicks"),
                dash.State("input-max-k", "value"),
                dash.State("input-min-s", "value"),
                dash.State("input-max-s", "value"),
                dash.State("input-log-granularity", "value"),
                dash.State("input-num-jobs", "value"),
            ],
            True,
        )(self.compute_ccf)

        self._app.callback(
            dash.Output("hilbert-plot", "figure"),
            [
                dash.Input("stored-ccf", "data"),
                dash.Input("input-max-components", "value"),
            ],
            True,
        )(self.draw_ccf)

        # self._app.callback( dash.Output("my-output", "children"),
        #                    [dash.Input("print-button", "n_clicks") ], True)(self.test_print)
        if jupyter == True:
            self._app.run_server(mode="inline")
        else:
            self._app.run_server(debug=debug)

    def compute_ccf(
        self,
        n_clicks,
        max_k,
        min_s,
        max_s,
        log_granularity,
        num_jobs,
    ):
        max_k = float(max_k)
        min_s = float(min_s)
        max_s = float(max_s)
        granularity = 2**log_granularity
        num_jobs = int(num_jobs)
        ss, ks, hf = self._persistable.compute_hilbert_function(
            max_k,
            min_s,
            max_s,
            granularity,
            n_jobs=num_jobs,
        )
        return pd.DataFrame(hf, index=ks[:-1], columns=ss[:-1]).to_json(
            date_format="iso", orient="split"
        )

    def draw_ccf(self, ccf, max_components):
        max_components = 0 if max_components is None else int(max_components)
        ccf = pd.read_json(ccf, orient="split")

        def df_to_plotly(df):
            return {
                "z": df.values.tolist(),
                "x": df.columns.tolist(),
                "y": df.index.tolist(),
            }

        fig = go.Figure(
            layout=go.Layout(
                # height=500,
                # width=650,
                xaxis_title="distance scale",
                yaxis_title="density threshold",
            ),
        )
        fig.add_trace(go.Heatmap(df_to_plotly(ccf), zmin=0, zmax=max_components))
        fig.update_traces(colorscale="greys")
        fig.update_layout(coloraxis_showscale=False)
        return fig

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

        self._vineyard = self._persistable.compute_prominence_vineyard(
            [start1, end1], [start2, end2]
        )
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
