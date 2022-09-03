# Authors: Luis Scoccola
# License: 3-clause BSD

import sys
from re import A
import numpy as np
import warnings
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.widgets import Button
#from matplotlib.patches import Polygon

import plotly.graph_objects as go
import plotly
from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
from dash.long_callback import DiskcacheLongCallbackManager

import pandas as pd
import json
import diskcache

PERSISTABLE_STDERR = "./persistable-stderr"
PERSISTABLE_DASH_CACHE = "./persistable-dash-cache"
#WARNINGS_GLOBAL = "test default"

def empty_figure():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    return fig


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

        ### to be passed/computed later:
        ## user-selected bounds for the prominence vineyard
        #self._vineyard_parameter_bounds = {}
        ## user-selected start and end for a line
        #self._line_parameters = None
        ## user-selected number of clusters
        #self._n_clusters = None
        ## the computed prominence vineyard
        #self._vineyard = None

        ## prominence vineyard
        #self._gaps = []
        #self._gap_numbers = []
        #self._lines = []
        #self._line_index = []


        ## initialize the plots

        default_min_k = 0
        default_max_k = self._persistable._maxk
        default_k_step = default_max_k / 100
        default_min_s = self._persistable._connection_radius / 5
        default_max_s = self._persistable._connection_radius * 2
        default_s_step = (default_max_s - default_min_s) / 100
        default_log_granularity = 6
        default_num_jobs = 4
        default_max_dim = 15
        min_granularity = 4
        max_granularity = 8
        defr = 6
        default_x_start_first_line = (default_min_s + default_max_s) * (1 / defr)
        default_y_start_first_line = (default_min_k + default_max_k) * (1 / 2)
        default_x_end_first_line = (default_max_s + default_min_s) * (1 / 2)
        default_y_end_first_line = (default_min_k + default_max_k) * (1 / defr)
        default_x_start_second_line = (default_min_s + default_max_s) * (1 / 2)
        default_y_start_second_line = (default_min_k + default_max_k) * (
            (defr - 1) / defr
        )
        default_x_end_second_line = (default_max_s + default_min_s) * (
            (defr - 1) / defr
        )
        default_y_end_second_line = (default_min_k + default_max_k) * (1 / 2)

        # set temporary files
        cache = diskcache.Cache(PERSISTABLE_DASH_CACHE)
        long_callback_manager = DiskcacheLongCallbackManager(cache)
        with open(PERSISTABLE_STDERR, "w") as file:
                file.close()



        if jupyter == True:
            self._app = JupyterDash(
                __name__, long_callback_manager=long_callback_manager
            )
        else:
            self._app = dash.Dash(__name__, long_callback_manager=long_callback_manager)
        self._app.layout = html.Div(
            children=[
                # contains the component counting function as a pandas dataframe
                dcc.Store(id="stored-ccf"),
                # contains the basic component counting function plot as a plotly figure
                dcc.Store(id="stored-ccf-drawing"),
                dcc.Interval(
                    id="warnings-polling-interval",
                    interval=(1/2)*1000,
                    n_intervals=0
                ),
                ### contains the current selected slice endpoint as a list
                ##dcc.Store(id="selected-line-endpoint", data=json.dumps([])),
                html.H1("Interactive parameter selection for Persistable"),
                html.Details(
                    [
                        html.Summary("Quick help"),
                        dcc.Markdown('''
                        - When setting a field, press enter to TODO
                        - Check the log, below, for warnings.
                        - The app takes a second or so to update the graphical interface after an interaction.
                        - Computing the component counting function and prominence vineyard can take a while, depending on the size and dimensionality of the dataset as well as other factors.
                        - TODO
                        ''')
                    ]
                ),
                # html.Pre(id="test", style= {'border': 'thin lightgrey solid', 'overflowX': 'scroll' }),
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
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="density threshold min/max",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="min-density-threshold",
                                                    type="number",
                                                    value=default_min_k,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_k_step,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="max-density-threshold",
                                                    type="number",
                                                    value=default_max_k,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_k_step,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="distance scale min/max",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="min-dist-scale",
                                                    type="number",
                                                    value=default_min_s,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_s_step,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="max-dist-scale",
                                                    type="number",
                                                    value=default_max_s,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_s_step,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
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
                                                    step=1,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="granularity",
                                                ),
                                                dcc.Slider(
                                                    min_granularity,
                                                    max_granularity,
                                                    step=None,
                                                    marks={
                                                        i: str(2**i)
                                                        for i in range(
                                                            1, max_granularity + 1
                                                        )
                                                    },
                                                    value=default_log_granularity,
                                                    id="input-log-granularity",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                ),
                                                html.Button(
                                                    "(re)compute component counting function",
                                                    id="compute-ccf-button",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
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
                                                    step=1,
                                                    debounce=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="lines selection",
                                                ),
                                                dcc.RadioItems(
                                                    ["on", "off"],
                                                    "off",
                                                    id="display-lines-selection",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            id="endpoint-selection-div",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="endpoint selection",
                                                ),
                                                dcc.RadioItems(
                                                    [
                                                        "1st line start",
                                                        "1st line end",
                                                        "2nd line start",
                                                        "2nd line end",
                                                    ],
                                                    "1st line start",
                                                    id="endpoint-selection",
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
                                    children=[
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="first line start",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="x-start-first-line",
                                                    type="number",
                                                    value=default_x_start_first_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="y-start-first-line",
                                                    type="number",
                                                    value=default_y_start_first_line,
                                                    min=0,
                                                    step=default_k_step,
                                                    debounce=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="first line end",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="x-end-first-line",
                                                    type="number",
                                                    value=default_x_end_first_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="y-end-first-line",
                                                    type="number",
                                                    value=default_y_end_first_line,
                                                    min=0,
                                                    step=default_k_step,
                                                    debounce=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="second line start ",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="x-start-second-line",
                                                    type="number",
                                                    value=default_x_start_second_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="y-start-second-line",
                                                    type="number",
                                                    value=default_y_start_second_line,
                                                    min=0,
                                                    step=default_k_step,
                                                    debounce=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="second line end",
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="x-end-second-line",
                                                    type="number",
                                                    value=default_x_end_second_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className="value",
                                                    id="y-end-second-line",
                                                    type="number",
                                                    value=default_y_end_second_line,
                                                    min=0,
                                                    step=default_k_step,
                                                    debounce=True,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                ),
                                                html.Button(
                                                    "compute prominence vineyard",
                                                    id="compute-pv-button",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="prominence scale",
                                                ),
                                                dcc.RadioItems(
                                                    ["linear", "logarithmic"],
                                                    "logarithmic",
                                                    id="input-prom-vin-scale",
                                                    className="value",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id="cff-plot",
                            config={
                                "responsive":True,
                                "displayModeBar": False,
                                "modeBarButtonsToRemove": [
                                    "toImage",
                                    "pan",
                                    "zoomIn",
                                    "zoomOut",
                                    "resetScale",
                                    "lasso",
                                    "select",
                                ],
                                # "modeBarButtonsToAdd": [
                                #    "drawline",
                                #    "eraseshape",
                                # ],
                                "displaylogo": False,
                            },
                        ),
                        html.Div(
                            className="fake-plot",
                        ),
                    ],
                ),
                html.Details(
                    [
                        html.Summary("Warnings"),
                        html.Pre(
                            id="log",
                            style={
                                "border": "thin lightgrey solid",
                                "overflowX": "scroll",
                            },
                        ),
                    ],
                    open=True,
                ),
            ],
        )

        self._app.callback(
            [
                #dash.Output("log", "children"),
                dash.Output("x-start-first-line", "value"),
                dash.Output("y-start-first-line", "value"),
                dash.Output("x-end-first-line", "value"),
                dash.Output("y-end-first-line", "value"),
                dash.Output("x-start-second-line", "value"),
                dash.Output("y-start-second-line", "value"),
                dash.Output("x-end-second-line", "value"),
                dash.Output("y-end-second-line", "value"),
            ],
            # dash.Output("selected-line-endpoint", "data"),
            [
                dash.Input("cff-plot", "clickData"),
                dash.State("display-lines-selection", "value"),
                dash.State("endpoint-selection", "value"),
                dash.State("x-start-first-line", "value"),
                dash.State("y-start-first-line", "value"),
                dash.State("x-end-first-line", "value"),
                dash.State("y-end-first-line", "value"),
                dash.State("x-start-second-line", "value"),
                dash.State("y-start-second-line", "value"),
                dash.State("x-end-second-line", "value"),
                dash.State("y-end-second-line", "value"),
            ],
            True,
        )(self.on_ccf_click)

        self._app.callback(
            dash.Output("endpoint-selection-div", "hidden"),
            dash.Input("display-lines-selection", "value")
        )(
            lambda val: False if val=="on" else True
        )

        self._app.long_callback(
            dash.Output("stored-ccf", "data"),
            [
                dash.Input("compute-ccf-button", "n_clicks"),
                dash.State("min-density-threshold", "value"),
                dash.State("max-density-threshold", "value"),
                dash.State("min-dist-scale", "value"),
                dash.State("max-dist-scale", "value"),
                dash.State("input-log-granularity", "value"),
                dash.State("input-num-jobs", "value"),
            ],
            True,
            running=[(dash.Output("compute-ccf-button", "disabled"), True, False)],
        )(self.compute_ccf)

        self._app.callback(
            dash.Output("stored-ccf-drawing", "data"),
            # dash.Output("cff-plot", "figure"),
            [
                dash.Input("stored-ccf", "data"),
                dash.Input("input-max-components", "value"),
            ],
            False,
        )(self.draw_ccf)

        self._app.callback(
            dash.Output("cff-plot", "figure"),
            # dash.Output("selected-line-endpoint", "data"),
            [
                dash.State("stored-ccf", "data"),
                dash.Input("stored-ccf-drawing", "data"),
                dash.Input("min-dist-scale", "value"),
                dash.Input("max-dist-scale", "value"),
                dash.Input("min-density-threshold", "value"),
                dash.Input("max-density-threshold", "value"),
                dash.Input("display-lines-selection", "value"),
                dash.Input("x-start-first-line", "value"),
                dash.Input("y-start-first-line", "value"),
                dash.Input("x-end-first-line", "value"),
                dash.Input("y-end-first-line", "value"),
                dash.Input("x-start-second-line", "value"),
                dash.Input("y-start-second-line", "value"),
                dash.Input("x-end-second-line", "value"),
                dash.Input("y-end-second-line", "value"),
                dash.Input("endpoint-selection", "value"),
            ],
            False,
        )(self.draw_ccf_enclosing_box)

        self._app.callback(
            dash.Output("log", "children"),
            dash.Input("warnings-polling-interval", "n_intervals"),
            False,
        )(self.print_log)

        if jupyter:
            self._app.run_server(mode="inline")
        else:
            self._app.run_server(debug=debug)

    def print_log(
        self,
        n_intervals,
    ):
        with open(PERSISTABLE_STDERR, "r") as file:
            out = file.read()
        return out

    def compute_ccf(
        self,
        n_clicks,
        min_k,
        max_k,
        min_s,
        max_s,
        log_granularity,
        num_jobs,
    ):
        #open(PERSISTABLE_STDERR, 'w').close()
        min_k = float(min_k)
        max_k = float(max_k)
        min_s = float(min_s)
        max_s = float(max_s)
        granularity = 2**log_granularity
        num_jobs = int(num_jobs)

        out = ""
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            # warnings.simplefilter("always")
            warnings.warn("test warning")
            ss, ks, hf = self._persistable.compute_hilbert_function(
                min_k,
                max_k,
                min_s,
                max_s,
                granularity,
                n_jobs=num_jobs,
            )
            for a in w:
                out += warnings.formatwarning(a.message, a.category, a.filename, a.lineno)
            print("the contents of out are:")
            print(out)
            with open(PERSISTABLE_STDERR, "w") as file:
                file.write(out)


        return pd.DataFrame(hf, index=ks[:-1], columns=ss[:-1]).to_json(
            date_format="iso", orient="split"
        )

    def on_ccf_click(
        self,
        click_data,
        display_lines_selection,
        endpoint,
        x_start_first_line,
        y_start_first_line,
        x_end_first_line,
        y_end_first_line,
        x_start_second_line,
        y_start_second_line,
        x_end_second_line,
        y_end_second_line,
    ):
        if display_lines_selection=="on":
            new_x, new_y = click_data["points"][0]["x"], click_data["points"][0]["y"]
            if endpoint == "1st line start":
                x_start_first_line = new_x
                y_start_first_line = new_y
            elif endpoint == "1st line end":
                x_end_first_line = new_x
                y_end_first_line = new_y
            elif endpoint == "2nd line start":
                x_start_second_line = new_x
                y_start_second_line = new_y
            elif endpoint == "2nd line end":
                x_end_second_line = new_x
                y_end_second_line = new_y
        return (
            #json.dumps(click_data),
            x_start_first_line,
            y_start_first_line,
            x_end_first_line,
            y_end_first_line,
            x_start_second_line,
            y_start_second_line,
            x_end_second_line,
            y_end_second_line,
        )

    def draw_ccf_enclosing_box(
        self,
        ccf,
        ccf_drawing,
        min_dist_scale,
        max_dist_scale,
        min_density_threshold,
        max_density_threshold,
        display_line_selection,
        x_start_first_line,
        y_start_first_line,
        x_end_first_line,
        y_end_first_line,
        x_start_second_line,
        y_start_second_line,
        x_end_second_line,
        y_end_second_line,
        endpoint,
    ):
        if ccf is None:
            return empty_figure()

        fig = plotly.io.from_json(ccf_drawing)

        ccf = pd.read_json(ccf, orient="split")

        def generate_red_box(xs, ys, text):
            return go.Scatter(
                x=xs,
                y=ys,
                fillcolor="rgba(255, 0, 0, 0.1)",
                fill="toself",
                mode="none",
                text=text,
                name="",
            )

        # draw left side of new enclosing box
        fig.add_trace(
            generate_red_box(
                [min(ccf.columns), min_dist_scale, min_dist_scale, min(ccf.columns)],
                [min(ccf.index), min(ccf.index), max(ccf.index), max(ccf.index)],
                "Left side of new enclosing box",
            )
        )

        # draw right side of new enclosing box
        fig.add_trace(
            generate_red_box(
                [max_dist_scale, max(ccf.columns), max(ccf.columns), max_dist_scale],
                [min(ccf.index), min(ccf.index), max(ccf.index), max(ccf.index)],
                text="Right side of new enclosing box",
            )
        )

        # draw top side of new enclosing box
        fig.add_trace(
            generate_red_box(
                [min_dist_scale, max_dist_scale, max_dist_scale, min_dist_scale],
                [
                    max_density_threshold,
                    max_density_threshold,
                    max(ccf.index),
                    max(ccf.index),
                ],
                text="Top side of new enclosing box",
            )
        )

        # draw bottom side of new enclosing box
        fig.add_trace(
            generate_red_box(
                [min_dist_scale, max_dist_scale, max_dist_scale, min_dist_scale],
                [
                    min_density_threshold,
                    min_density_threshold,
                    min(ccf.index),
                    min(ccf.index),
                ],
                text="Bottom side of new enclosing box",
            )
        )

        if display_line_selection == "on":

            # draw polygon
            fig.add_trace(go.Scatter(
                x=[x_start_first_line, x_end_first_line, x_end_second_line, x_start_second_line ],
                y=[y_start_first_line, y_end_first_line, y_end_second_line, y_start_second_line ],
                fillcolor="rgba(0, 0, 255, 0.05)",
                fill="toself",
                mode="none",
                hoverinfo="skip",
                #text=text,
                name="",
            ))

            def generate_blue_line(xs, ys, text, different_marker=None):
                if different_marker == None:
                    marker_styles = ["circle", "circle"]
                elif different_marker == 0:
                    marker_styles = ["circle-open", "circle"]
                elif different_marker == 1:
                    marker_styles = ["circle", "circle-open"]
                return go.Scatter(
                    x=xs,
                    y=ys,
                    name=text + " line",
                    text=[text + " line start", text + " line end"],
                    marker=dict(size=20, color="blue"),
                    # hoverinfo="name+text",
                    marker_symbol=marker_styles,
                    hoverinfo="skip",
                    showlegend=False,
                    mode="markers+lines+text",
                    textposition="top center",
                )

            if endpoint == "1st line start":
                first_line_endpoints = 0
                second_line_endpoints = None
            elif endpoint == "1st line end":
                first_line_endpoints = 1
                second_line_endpoints = None
            elif endpoint == "2nd line start":
                first_line_endpoints = None
                second_line_endpoints = 0
            elif endpoint == "2nd line end":
                first_line_endpoints = None
                second_line_endpoints = 1

            fig.add_trace(
                generate_blue_line(
                    [x_start_first_line, x_end_first_line],
                    [y_start_first_line, y_end_first_line],
                    "1st",
                    first_line_endpoints,
                )
            )
            fig.add_trace(
                generate_blue_line(
                    [x_start_second_line, x_end_second_line],
                    [y_start_second_line, y_end_second_line],
                    "2nd",
                    second_line_endpoints,
                )
            )

        return fig
        ### also return an empty list represeting the fact that no endpoint is now selected in the ccf plot
        ##return fig, json.dumps([])

    def draw_ccf(
        self,
        ccf,
        max_components,
    ):

        if ccf is None:
            return empty_figure()

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
                xaxis={"fixedrange": True},
                yaxis={"fixedrange": True},
            ),
        )
        max_components = 0 if max_components is None else int(max_components)
        fig.add_trace(
            go.Heatmap(
                df_to_plotly(ccf),
                hovertemplate="<b># comp.: %{z:d}</b><br>x: %{x:.3e} <br>y: %{y:.3e} ",
                zmin=0,
                zmax=max_components,
                showscale=False,
                name="",
            )
        )
        fig.update_traces(colorscale="greys")

        fig.update_layout(showlegend=False)
        fig.update_layout(autosize=True)
        fig.update_xaxes(range=[min(ccf.columns), max(ccf.columns)])
        fig.update_yaxes(range=[min(ccf.index), max(ccf.index)])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig.update_layout(clickmode="event+select")
        # fig.update_layout(
        #    dragmode="drawline",
        #    newshape=dict(opacity=0.3, line=dict(color="darkblue", width=5)),
        # )
        # print(type(plotly.io.to_json(fig)))
        return plotly.io.to_json(fig)

    # def cluster(self):
    #    if self._line_parameters is None:
    #        raise Exception("No parameters for the line were given!")
    #    else:
    #        start, end = self._line_parameters
    #        n_clusters = self._n_clusters
    #    return self._persistable.cluster(n_clusters, start, end)

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
