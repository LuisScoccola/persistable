# Authors: Luis Scoccola
# License: 3-clause BSD

import warnings
import plotly.graph_objects as go
import plotly
import pandas as pd
import json
import diskcache
import dash
from dash import dcc, html, DiskcacheManager, ctx
from ._prominence_vineyard import ProminenceVineyard

from plotly.express.colors import sample_colorscale

import numpy as np

###
from dash.long_callback.managers import BaseLongCallbackManager

import uuid


def monkeypatched_hash_function(fn):
    return uuid.uuid4()


BaseLongCallbackManager.hash_function = monkeypatched_hash_function
###


PERSISTABLE_DASH_CACHE = "./persistable-dash-cache"

X_START_FIRST_LINE = "x-start-first-line-"
Y_START_FIRST_LINE = "y-start-first-line-"
X_END_FIRST_LINE = "x-end-first-line-"
Y_END_FIRST_LINE = "y-end-first-line-"
X_START_SECOND_LINE = "x-start-second-line-"
Y_START_SECOND_LINE = "y-start-second-line-"
X_END_SECOND_LINE = "x-end-second-line-"
Y_END_SECOND_LINE = "y-end-second-line-"
CFF_PLOT = "cff-plot-"
DISPLAY_LINES_SELECTION = "display-lines-selection-"
ENDPOINT_SELECTION = "endpoint-selection-"
STORED_CCF = "stored-ccf-"
STORED_CCF_DRAWING = "stored-ccf-drawing-"
MIN_DIST_SCALE = "min-dist-scale-"
MAX_DIST_SCALE = "max-dist-scale-"
MIN_DENSITY_THRESHOLD = "min-density-threshold-"
MAX_DENSITY_THRESHOLD = "max-density-threshold-"
ENDPOINT_SELECTION_DIV = "endpoint-selection-div-"
PARAMETER_SELECTION_DIV = "parameter-selection-div-"
DISPLAY_PARAMETER_SELECTION = "display-parameter-selection-"
COMPUTE_CCF_BUTTON = "compute-ccf-button-"
INPUT_LOG_GRANULARITY_CCF = "input-log-granularity-ccf-"
INPUT_NUM_JOBS_CCF = "input-num-jobs-ccf-"
INPUT_MAX_COMPONENTS = "input-max-components-"
LOG = "log-"
STORED_PV = "stored-pv-"
INPUT_MAX_VINES = "input-max-vines-"
INPUT_PROM_VIN_SCALE = "input-prom-vin-scale-"
COMPUTE_PV_BUTTON = "compute-pv-button-"
PV_PLOT = "pv-plot-"
STORED_PV_DRAWING = "stored-pv-drawing-"
INPUT_LOG_GRANULARITY_PV = "input-log-granularity-pv-"
INPUT_NUM_JOBS_PV = "input-num-jobs-pv-"
INPUT_LINE = "input-line-"
INPUT_GAP = "gap-"
EXPORT_PARAMETERS_BUTTON = "export-parameters-"
DUMMY_OUTPUT = "dummy-output-"

STORED_CCF_COMPUTATION_WARNINGS = "stored-ccf-computation-warnings-"
STORED_PV_COMPUTATION_WARNINGS = "stored-pv-computation-warnings-"

VALUE = "value"
CLICKDATA = "clickData"
HIDDEN = "hidden"
DATA = "data"
N_CLICKS = "n_clicks"
DISABLED = "disabled"
FIGURE = "figure"
CHILDREN = "children"
N_INTERVALS = "n_intervals"

IN = "input"
ST = "state"


def empty_figure():
    fig = go.Figure(
        layout=go.Layout(
            xaxis={"fixedrange": True},
            yaxis={"fixedrange": True},
        )
    )
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_layout(autosize=True)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_layout(clickmode="none")
    return fig


class PersistableInteractive:
    def __init__(self, persistable, jupyter=False, inline=False, debug=False):
        self._parameters = None
        self._persistable = persistable

        default_min_k = 0
        default_max_k = self._persistable._maxk
        default_k_step = default_max_k / 100
        default_min_s = self._persistable._connection_radius / 5
        default_max_s = self._persistable._connection_radius * 2
        default_s_step = (default_max_s - default_min_s) / 100
        default_log_granularity = 6
        default_num_jobs = 4
        default_max_dim = 15
        default_max_vines = 15
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
        background_callback_manager = DiskcacheManager(cache)

        if jupyter == True:
            from jupyter_dash import JupyterDash

            self._app = JupyterDash(
                __name__, background_callback_manager=background_callback_manager
            )
        else:
            self._app = dash.Dash(
                __name__, background_callback_manager=background_callback_manager
            )
        self._app.layout = html.Div(
            children=[
                # contains the component counting function as a pandas dataframe
                dcc.Store(id=STORED_CCF),
                # contains the basic component counting function plot as a plotly figure
                dcc.Store(id=STORED_CCF_DRAWING),
                # contains the vineyard as a vineyard object
                dcc.Store(id=STORED_PV),
                # contains the basic prominence vineyard plot as a plotly figure
                dcc.Store(id=STORED_PV_DRAWING),
                #
                dcc.Store(id=STORED_CCF_COMPUTATION_WARNINGS),
                dcc.Store(id=STORED_PV_COMPUTATION_WARNINGS),
                #
                html.Div(id=DUMMY_OUTPUT),
                # dcc.Interval(
                #    id=WARNINGS_POLLING_INTERVAL,
                #    interval=(1 / 2) * 1000,
                #    n_intervals=0,
                # ),
                html.Details(
                    [
                        html.Summary("Quick help"),
                        dcc.Markdown(
                            """
                        ### Interactive parameter selection for Persistable
                        - When setting a field, press enter to TODO
                        - Check the log, below, for warnings.
                        - The app can take a second or so to update the graphical interface after an interaction.
                        - Computing the component counting function and prominence vineyard can take a while, depending on the size and dimensionality of the dataset as well as other factors.
                        - Make sure to leave your pointer still when clicking on the component counting function plot, otherwise your interaction may not be registered.
                        - TODO
                        """
                        ),
                    ]
                ),
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
                                                    children="distance scale min/max",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=MIN_DIST_SCALE,
                                                    type="number",
                                                    value=default_min_s,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_s_step,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=MAX_DIST_SCALE,
                                                    type="number",
                                                    value=default_max_s,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_s_step,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="density threshold min/max",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=MIN_DENSITY_THRESHOLD,
                                                    type="number",
                                                    value=default_min_k,
                                                    min=0,
                                                    debounce=True,
                                                    step=default_k_step,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=MAX_DENSITY_THRESHOLD,
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
                                                    children="granularity/#cores computation",
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
                                                    id=INPUT_LOG_GRANULARITY_CCF,
                                                    className=VALUE,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_NUM_JOBS_CCF,
                                                    type="number",
                                                    value=default_num_jobs,
                                                    min=1,
                                                    step=1,
                                                ),
                                            ],
                                        ),
                                        # html.Div(
                                        #    className="parameter-single",
                                        #    children=[
                                        #        html.Span(
                                        #            className="name",
                                        #            children="granularity comp count func",
                                        #        ),
                                        #        dcc.Slider(
                                        #            min_granularity,
                                        #            max_granularity,
                                        #            step=None,
                                        #            marks={
                                        #                i: str(2**i)
                                        #                for i in range(
                                        #                    1, max_granularity + 1
                                        #                )
                                        #            },
                                        #            value=default_log_granularity,
                                        #            id=INPUT_LOG_GRANULARITY_CCF,
                                        #            className=VALUE,
                                        #        ),
                                        #    ],
                                        # ),
                                        html.Div(
                                            className="large-button",
                                            children=[
                                                html.Button(
                                                    "(re)compute component counting function",
                                                    id=COMPUTE_CCF_BUTTON,
                                                    className=VALUE,
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
                                                    id=INPUT_MAX_COMPONENTS,
                                                    type="number",
                                                    value=default_max_dim,
                                                    min=1,
                                                    className=VALUE,
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
                                                    id=DISPLAY_LINES_SELECTION,
                                                    className=VALUE,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-single",
                                            id=ENDPOINT_SELECTION_DIV,
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
                                                    id=ENDPOINT_SELECTION,
                                                    className=VALUE,
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
                                                    children="first line start x/y",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=X_START_FIRST_LINE,
                                                    type="number",
                                                    value=default_x_start_first_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=Y_START_FIRST_LINE,
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
                                                    children="first line end x/y",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=X_END_FIRST_LINE,
                                                    type="number",
                                                    value=default_x_end_first_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=Y_END_FIRST_LINE,
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
                                                    children="second line start x/y",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=X_START_SECOND_LINE,
                                                    type="number",
                                                    value=default_x_start_second_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=Y_START_SECOND_LINE,
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
                                                    children="second line end x/y",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=X_END_SECOND_LINE,
                                                    type="number",
                                                    value=default_x_end_second_line,
                                                    min=0,
                                                    step=default_s_step,
                                                    debounce=True,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=Y_END_SECOND_LINE,
                                                    type="number",
                                                    value=default_y_end_second_line,
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
                                                    children="#lines vineyard/#cores computation",
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
                                                    value=default_log_granularity - 1,
                                                    id=INPUT_LOG_GRANULARITY_PV,
                                                    className=VALUE,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_NUM_JOBS_PV,
                                                    type="number",
                                                    value=default_num_jobs,
                                                    min=1,
                                                    step=1,
                                                ),
                                            ],
                                        ),
                                        # html.Div(
                                        #    className="parameter-single",
                                        #    children=[
                                        #        html.Span(
                                        #            className="name",
                                        #            children="#lines vineyard",
                                        #        ),
                                        #        dcc.Slider(
                                        #            min_granularity,
                                        #            max_granularity,
                                        #            step=None,
                                        #            marks={
                                        #                i: str(2**i)
                                        #                for i in range(
                                        #                    1, max_granularity + 1
                                        #                )
                                        #            },
                                        #            value=default_log_granularity - 1,
                                        #            id=INPUT_LOG_GRANULARITY_PV,
                                        #            className=VALUE,
                                        #        ),
                                        #    ],
                                        # ),
                                        html.Div(
                                            className="large-button",
                                            children=[
                                                html.Button(
                                                    "(re)compute prominence vineyard",
                                                    id=COMPUTE_PV_BUTTON,
                                                    className=VALUE,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="max #vines/prominence scale",
                                                ),
                                                dcc.Input(
                                                    id=INPUT_MAX_VINES,
                                                    type="number",
                                                    value=default_max_vines,
                                                    min=1,
                                                    className=VALUE,
                                                    step=1,
                                                ),
                                                dcc.RadioItems(
                                                    ["linear", "logarithmic"],
                                                    "logarithmic",
                                                    id=INPUT_PROM_VIN_SCALE,
                                                    className=VALUE,
                                                ),

                                            ],
                                        ),
                                        #html.Div(
                                        #    className="parameter-single",
                                        #    children=[
                                        #        html.Span(
                                        #            className="name",
                                        #            children="prominence scale",
                                        #        ),
                                        #        dcc.RadioItems(
                                        #            ["linear", "logarithmic"],
                                        #            "logarithmic",
                                        #            id=INPUT_PROM_VIN_SCALE,
                                        #            className=VALUE,
                                        #        ),
                                        #    ],
                                        #),
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="parameter selection",
                                                ),
                                                dcc.RadioItems(
                                                    ["on", "off"],
                                                    "off",
                                                    id=DISPLAY_PARAMETER_SELECTION,
                                                    className=VALUE,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="parameter-double-button",
                                            id=PARAMETER_SELECTION_DIV,
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="line number/gap number",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_LINE,
                                                    type="number",
                                                    value=1,
                                                    min=1,
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_GAP,
                                                    type="number",
                                                    value=1,
                                                    min=1,
                                                ),
                                                html.Button(
                                                    "export parameters",
                                                    id=EXPORT_PARAMETERS_BUTTON,
                                                    className="button",
                                                    disabled=True
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id=CFF_PLOT,
                            figure=empty_figure(),
                            config={
                                "responsive": True,
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
                                "displaylogo": False,
                            },
                        ),
                        dcc.Graph(
                            id=PV_PLOT,
                            figure=empty_figure(),
                            config={
                                "responsive": True,
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
                                "displaylogo": False,
                            },
                        ),
                    ],
                ),
                html.Details(
                    [
                        html.Summary("Warnings"),
                        html.Pre(
                            id=LOG,
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

        def dash_callback(
            inputs, outputs, prevent_initial_call=False, background=False, running=None
        ):
            def cs(l):
                return l[0] + l[1]

            def out_function(function):
                dash_inputs = [
                    dash.Input(i, v) if t == IN else dash.State(i, v)
                    for i, v, t in inputs
                ]
                dash_outputs = (
                    [dash.Output(i, v) for i, v in outputs]
                    if len(outputs) > 1
                    else dash.Output(outputs[0][0], outputs[0][1])
                )

                if running is None:
                    dash_running_outputs = None
                else:
                    dash_running_outputs = [
                        (dash.Output(i, v), value_start, value_end)
                        for i, v, value_start, value_end in running
                    ]

                def callback_function(*argv):
                    d = {}
                    for n, arg in enumerate(argv):
                        d[cs(inputs[n])] = arg
                    d = function(d)
                    return (
                        tuple(d[cs(o)] for o in outputs)
                        if len(outputs) > 1
                        else d[cs(outputs[0])]
                    )

                if background:
                    self._app.long_callback(
                        dash_outputs,
                        dash_inputs,
                        prevent_initial_call,
                        running=dash_running_outputs,
                    )(callback_function)
                else:
                    self._app.callback(
                        dash_outputs,
                        dash_inputs,
                        prevent_initial_call,
                    )(callback_function)

                return function

            return out_function

        @dash_callback(
            [[DISPLAY_PARAMETER_SELECTION, VALUE, IN]],
            [[PARAMETER_SELECTION_DIV, HIDDEN]],
        )
        def toggle_parameter_selection_pv(d):
            if d[DISPLAY_PARAMETER_SELECTION + VALUE] == "on":
                d[PARAMETER_SELECTION_DIV + HIDDEN] = False
            else:
                d[PARAMETER_SELECTION_DIV + HIDDEN] = True
            return d

        @dash_callback(
            [
                [STORED_CCF_COMPUTATION_WARNINGS, DATA, IN],
                [STORED_PV_COMPUTATION_WARNINGS, DATA, IN],
            ],
            [[LOG, CHILDREN]],
            True,
        )
        def print_log(d):
            if ctx.triggered_id == STORED_CCF_COMPUTATION_WARNINGS:
                d[LOG + CHILDREN] = json.loads(
                    d[STORED_CCF_COMPUTATION_WARNINGS + DATA]
                )
            elif ctx.triggered_id == STORED_PV_COMPUTATION_WARNINGS:
                d[LOG + CHILDREN] = json.loads(d[STORED_PV_COMPUTATION_WARNINGS + DATA])
            else:
                raise Exception(
                    "print_log was triggered by unknown id: " + str(ctx.triggered_id)
                )
            return d

        @dash_callback(
            [
                [CFF_PLOT, CLICKDATA, IN],
                [DISPLAY_LINES_SELECTION, VALUE, ST],
                [ENDPOINT_SELECTION, VALUE, ST],
                [X_START_FIRST_LINE, VALUE, ST],
                [Y_START_FIRST_LINE, VALUE, ST],
                [X_END_FIRST_LINE, VALUE, ST],
                [Y_END_FIRST_LINE, VALUE, ST],
                [X_START_SECOND_LINE, VALUE, ST],
                [Y_START_SECOND_LINE, VALUE, ST],
                [X_END_SECOND_LINE, VALUE, ST],
                [Y_END_SECOND_LINE, VALUE, ST],
            ],
            [
                [X_START_FIRST_LINE, VALUE],
                [Y_START_FIRST_LINE, VALUE],
                [X_END_FIRST_LINE, VALUE],
                [Y_END_FIRST_LINE, VALUE],
                [X_START_SECOND_LINE, VALUE],
                [Y_START_SECOND_LINE, VALUE],
                [X_END_SECOND_LINE, VALUE],
                [Y_END_SECOND_LINE, VALUE],
            ],
            True,
        )
        def on_ccf_click(d):
            if d[DISPLAY_LINES_SELECTION + VALUE] == "on":
                new_x, new_y = (
                    d[CFF_PLOT + CLICKDATA]["points"][0]["x"],
                    d[CFF_PLOT + CLICKDATA]["points"][0]["y"],
                )
                if d[ENDPOINT_SELECTION + VALUE] == "1st line start":
                    d[X_START_FIRST_LINE + VALUE] = new_x
                    d[Y_START_FIRST_LINE + VALUE] = new_y
                elif d[ENDPOINT_SELECTION + VALUE] == "1st line end":
                    d[X_END_FIRST_LINE + VALUE] = new_x
                    d[Y_END_FIRST_LINE + VALUE] = new_y
                elif d[ENDPOINT_SELECTION + VALUE] == "2nd line start":
                    d[X_START_SECOND_LINE + VALUE] = new_x
                    d[Y_START_SECOND_LINE + VALUE] = new_y
                elif d[ENDPOINT_SELECTION + VALUE] == "2nd line end":
                    d[X_END_SECOND_LINE + VALUE] = new_x
                    d[Y_END_SECOND_LINE + VALUE] = new_y
            return d

        @dash_callback(
            [[DISPLAY_LINES_SELECTION, VALUE, IN]], [[ENDPOINT_SELECTION_DIV, HIDDEN]]
        )
        def toggle_parameter_selection_ccf(d):
            if d[DISPLAY_LINES_SELECTION + VALUE] == "on":
                d[ENDPOINT_SELECTION_DIV + HIDDEN] = False
            else:
                d[ENDPOINT_SELECTION_DIV + HIDDEN] = True
            return d

        @dash_callback(
            [[STORED_CCF, DATA, IN], [INPUT_MAX_COMPONENTS, VALUE, IN]],
            [[STORED_CCF_DRAWING, DATA]],
            False,
        )
        def draw_ccf(d):
            ccf = d[STORED_CCF + DATA]

            if ccf is None:
                d[STORED_CCF_DRAWING + DATA] = empty_figure()
                return d

            ccf = pd.read_json(ccf, orient="split")

            def df_to_plotly(df):
                return {
                    "z": df.values.tolist(),
                    "x": df.columns.tolist(),
                    "y": df.index.tolist(),
                }

            fig = go.Figure(
                layout=go.Layout(
                    xaxis_title="distance scale",
                    yaxis_title="density threshold",
                    xaxis={"fixedrange": True},
                    yaxis={"fixedrange": True},
                ),
            )
            max_components = d[INPUT_MAX_COMPONENTS + VALUE]
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

            d[STORED_CCF_DRAWING + DATA] = plotly.io.to_json(fig)
            return d

        @dash_callback(
            [
                [STORED_CCF, DATA, ST],
                [STORED_CCF_DRAWING, DATA, IN],
                [MIN_DIST_SCALE, VALUE, IN],
                [MAX_DIST_SCALE, VALUE, IN],
                [MIN_DENSITY_THRESHOLD, VALUE, IN],
                [MAX_DENSITY_THRESHOLD, VALUE, IN],
                [DISPLAY_LINES_SELECTION, VALUE, IN],
                [X_START_FIRST_LINE, VALUE, IN],
                [Y_START_FIRST_LINE, VALUE, IN],
                [X_END_FIRST_LINE, VALUE, IN],
                [Y_END_FIRST_LINE, VALUE, IN],
                [X_START_SECOND_LINE, VALUE, IN],
                [Y_START_SECOND_LINE, VALUE, IN],
                [X_END_SECOND_LINE, VALUE, IN],
                [Y_END_SECOND_LINE, VALUE, IN],
                [ENDPOINT_SELECTION, VALUE, IN],
            ],
            [[CFF_PLOT, FIGURE]],
            False,
        )
        def draw_ccf_enclosing_box(d):
            ccf = d[STORED_CCF + DATA]

            if ccf is None:
                d[CFF_PLOT + FIGURE] = empty_figure()
                return d

            fig = plotly.io.from_json(d[STORED_CCF_DRAWING + DATA])

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
                    [
                        min(ccf.columns),
                        d[MIN_DIST_SCALE + VALUE],
                        d[MIN_DIST_SCALE + VALUE],
                        min(ccf.columns),
                    ],
                    [min(ccf.index), min(ccf.index), max(ccf.index), max(ccf.index)],
                    "Left side of new enclosing box",
                )
            )

            # draw right side of new enclosing box
            fig.add_trace(
                generate_red_box(
                    [
                        d[MAX_DIST_SCALE + VALUE],
                        max(ccf.columns),
                        max(ccf.columns),
                        d[MAX_DIST_SCALE + VALUE],
                    ],
                    [min(ccf.index), min(ccf.index), max(ccf.index), max(ccf.index)],
                    text="Right side of new enclosing box",
                )
            )

            # draw top side of new enclosing box
            fig.add_trace(
                generate_red_box(
                    [
                        d[MIN_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MIN_DIST_SCALE + VALUE],
                    ],
                    [
                        d[MAX_DENSITY_THRESHOLD + VALUE],
                        d[MAX_DENSITY_THRESHOLD + VALUE],
                        max(ccf.index),
                        max(ccf.index),
                    ],
                    text="Top side of new enclosing box",
                )
            )

            # draw bottom side of new enclosing box
            fig.add_trace(
                generate_red_box(
                    [
                        d[MIN_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MIN_DIST_SCALE + VALUE],
                    ],
                    [
                        d[MIN_DENSITY_THRESHOLD + VALUE],
                        d[MIN_DENSITY_THRESHOLD + VALUE],
                        min(ccf.index),
                        min(ccf.index),
                    ],
                    text="Bottom side of new enclosing box",
                )
            )

            if d[DISPLAY_LINES_SELECTION + VALUE] == "on":

                # draw polygon
                fig.add_trace(
                    go.Scatter(
                        x=[
                            d[X_START_FIRST_LINE + VALUE],
                            d[X_END_FIRST_LINE + VALUE],
                            d[X_END_SECOND_LINE + VALUE],
                            d[X_START_SECOND_LINE + VALUE],
                        ],
                        y=[
                            d[Y_START_FIRST_LINE + VALUE],
                            d[Y_END_FIRST_LINE + VALUE],
                            d[Y_END_SECOND_LINE + VALUE],
                            d[Y_START_SECOND_LINE + VALUE],
                        ],
                        fillcolor="rgba(0, 0, 255, 0.05)",
                        fill="toself",
                        mode="none",
                        hoverinfo="skip",
                        # text=text,
                        name="",
                    )
                )

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
                        marker=dict(size=15, color="blue"),
                        # hoverinfo="name+text",
                        marker_symbol=marker_styles,
                        hoverinfo="skip",
                        showlegend=False,
                        mode="markers+lines+text",
                        textposition="top center",
                    )

                if d[ENDPOINT_SELECTION + VALUE] == "1st line start":
                    first_line_endpoints = 0
                    second_line_endpoints = None
                elif d[ENDPOINT_SELECTION + VALUE] == "1st line end":
                    first_line_endpoints = 1
                    second_line_endpoints = None
                elif d[ENDPOINT_SELECTION + VALUE] == "2nd line start":
                    first_line_endpoints = None
                    second_line_endpoints = 0
                elif d[ENDPOINT_SELECTION + VALUE] == "2nd line end":
                    first_line_endpoints = None
                    second_line_endpoints = 1

                fig.add_trace(
                    generate_blue_line(
                        [d[X_START_FIRST_LINE + VALUE], d[X_END_FIRST_LINE + VALUE]],
                        [d[Y_START_FIRST_LINE + VALUE], d[Y_END_FIRST_LINE + VALUE]],
                        "1st",
                        first_line_endpoints,
                    )
                )
                fig.add_trace(
                    generate_blue_line(
                        [d[X_START_SECOND_LINE + VALUE], d[X_END_SECOND_LINE + VALUE]],
                        [d[Y_START_SECOND_LINE + VALUE], d[Y_END_SECOND_LINE + VALUE]],
                        "2nd",
                        second_line_endpoints,
                    )
                )

            d[CFF_PLOT + FIGURE] = fig
            return d

        @dash_callback(
            [[STORED_PV, DATA, ST], [STORED_PV_DRAWING, DATA, IN]], [[PV_PLOT, FIGURE]]
        )
        def draw_pv_post(d):
            # pv,
            # pv_drawing,

            if d[STORED_PV + DATA] is None:
                d[PV_PLOT + FIGURE] = empty_figure()
                return d

            d[PV_PLOT + FIGURE] = plotly.io.from_json(d[STORED_PV_DRAWING + DATA])

            return d

        @dash_callback(
            [
                [COMPUTE_CCF_BUTTON, N_CLICKS, IN],
                [MIN_DENSITY_THRESHOLD, VALUE, ST],
                [MAX_DENSITY_THRESHOLD, VALUE, ST],
                [MIN_DIST_SCALE, VALUE, ST],
                [MAX_DIST_SCALE, VALUE, ST],
                [INPUT_LOG_GRANULARITY_CCF, VALUE, ST],
                [INPUT_NUM_JOBS_CCF, VALUE, ST],
            ],
            [
                [STORED_CCF, DATA],
                [STORED_CCF_COMPUTATION_WARNINGS, DATA],
            ],
            prevent_initial_call=True,
            background=True,
            running=[[COMPUTE_CCF_BUTTON, DISABLED, True, False]],
        )
        def compute_ccf(d):
            granularity = 2 ** d[INPUT_LOG_GRANULARITY_CCF + VALUE]
            num_jobs = int(d[INPUT_NUM_JOBS_CCF + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ss, ks, hf = self._persistable.compute_hilbert_function(
                    d[MIN_DENSITY_THRESHOLD + VALUE],
                    d[MAX_DENSITY_THRESHOLD + VALUE],
                    d[MIN_DIST_SCALE + VALUE],
                    d[MAX_DIST_SCALE + VALUE],
                    granularity,
                    n_jobs=num_jobs,
                )
                for a in w:
                    out += warnings.formatwarning(
                        a.message, a.category, a.filename, a.lineno
                    )

            d[STORED_CCF + DATA] = pd.DataFrame(
                hf, index=ks[:-1], columns=ss[:-1]
            ).to_json(date_format="iso", orient="split")

            d[STORED_CCF_COMPUTATION_WARNINGS + DATA] = json.dumps(out)

            return d

        @dash_callback(
            [
                [COMPUTE_PV_BUTTON, N_CLICKS, IN],
                [X_START_FIRST_LINE, VALUE, ST],
                [Y_START_FIRST_LINE, VALUE, ST],
                [X_END_FIRST_LINE, VALUE, ST],
                [Y_END_FIRST_LINE, VALUE, ST],
                [X_START_SECOND_LINE, VALUE, ST],
                [Y_START_SECOND_LINE, VALUE, ST],
                [X_END_SECOND_LINE, VALUE, ST],
                [Y_END_SECOND_LINE, VALUE, ST],
                [INPUT_LOG_GRANULARITY_PV, VALUE, ST],
                [INPUT_NUM_JOBS_PV, VALUE, ST],
            ],
            [
                [STORED_PV, DATA],
                [STORED_PV_COMPUTATION_WARNINGS, DATA],
                [INPUT_LINE, "max"],
                [INPUT_LINE, VALUE],
                [EXPORT_PARAMETERS_BUTTON, DISABLED]
            ],
            prevent_initial_call=True,
            background=True,
            running=[[COMPUTE_PV_BUTTON, DISABLED, True, False]],
        )
        def compute_pv(d):

            granularity = 2 ** d[INPUT_LOG_GRANULARITY_PV + VALUE]
            num_jobs = int(d[INPUT_NUM_JOBS_PV + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warnings.warn("need to check that prominence vineyard succedeed enable the button correctly!")
                pv = self._persistable.compute_prominence_vineyard(
                    [
                        [d[X_START_FIRST_LINE + VALUE], d[Y_START_FIRST_LINE + VALUE]],
                        [d[X_END_FIRST_LINE + VALUE], d[Y_END_FIRST_LINE + VALUE]],
                    ],
                    [
                        [
                            d[X_START_SECOND_LINE + VALUE],
                            d[Y_START_SECOND_LINE + VALUE],
                        ],
                        [d[X_END_SECOND_LINE + VALUE], d[Y_END_SECOND_LINE + VALUE]],
                    ],
                    n_parameters=granularity,
                    n_jobs=num_jobs,
                )
                for a in w:
                    out += warnings.formatwarning(
                        a.message, a.category, a.filename, a.lineno
                    )

            d[STORED_PV + DATA] = json.dumps(pv.__dict__)
            d[STORED_PV_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[INPUT_LINE + "max"] = granularity
            d[INPUT_LINE + VALUE] = granularity//2
            if True:
                d[EXPORT_PARAMETERS_BUTTON+DISABLED] = False
            else:
                d[EXPORT_PARAMETERS_BUTTON+DISABLED] = True

            return d

        @dash_callback(
            [
                [STORED_PV, DATA, IN],
                [INPUT_MAX_VINES, VALUE, IN],
                [INPUT_PROM_VIN_SCALE, VALUE, IN],
                [DISPLAY_PARAMETER_SELECTION, VALUE, IN],
                [INPUT_LINE, VALUE, IN],
                [INPUT_GAP, VALUE, IN],
            ],
            [[STORED_PV_DRAWING, DATA]],
            False,
        )
        def draw_pv(d):
            firstn = d[INPUT_MAX_VINES + VALUE]

            if d[STORED_PV + DATA] is None:
                d[STORED_PV_DRAWING + DATA] = empty_figure()
                return d

            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = ProminenceVineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_prominence_diagrams"],
            )

            _gaps = []
            _gap_numbers = []
            _lines = []
            _line_index = []
            _vineyard_values = []

            times = np.array(vineyard.parameter_indices()) + 1
            vines = vineyard._vineyard_to_vines()
            num_vines = min(len(vines), firstn)

            fig = go.Figure(
                layout=go.Layout(
                    xaxis_title="parameter line",
                    yaxis_title="prominence",
                    xaxis={
                        "fixedrange": True,
                        "showgrid": False,
                    },
                    yaxis={
                        "fixedrange": True,
                        "showgrid": False,
                    },
                    plot_bgcolor="rgba(0, 0, 0, 0.1)",
                ),
            )

            colors = sample_colorscale(
                "viridis", list(np.linspace(0, 1, num_vines))[::-1]
            )
            if num_vines > 0:
                for i in range(num_vines - 1, -1, -1):
                    till = "tozeroy" if i == num_vines - 1 else "tonexty"
                    color = (
                        "red"
                        if (
                            d[DISPLAY_PARAMETER_SELECTION + VALUE] == "on"
                            and i + 1 == d[INPUT_GAP + VALUE]
                        )
                        else colors[i]
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=vines[i][1],
                            fill=till,
                            # hoveron="fills",
                            text="vine " + str(i + 1),
                            hoverinfo="text",
                            line_color=color,
                        )
                    )
            for i, tv in enumerate(vines):
                times, vine = tv
                for vine_part, time_part in vineyard._vine_parts(vine):
                    #        #if interpolate:
                    #        #    artist = ax.plot(time_part, vine_part, c="black")
                    #        #if points:
                    #        #    artist = ax.plot(time_part, vine_part, "o", c="black")
                    _vineyard_values.extend(vine_part)
            # ymax = max(self._vineyard_values)
            # for t in times:
            #    artist = ax.vlines(x=t, ymin=0, ymax=ymax, color="black", alpha=0.1)
            #    self._add_line_prominence_vineyard(artist, t)
            # ax.set_xticks([])
            # ax.set_xlabel("parameter")
            # if log_prominence:
            #    ax.set_ylabel("log-prominence")
            #    ax.set_yscale(LOG)
            # else:
            #    ax.set_ylabel("prominence")
            values = np.array(_vineyard_values)

            if d[DISPLAY_PARAMETER_SELECTION + VALUE] == "on":
                fig.add_vline(
                    x=d[INPUT_LINE + VALUE], line_dash="dash", line_color="red"
                )  # line_width=3, , line_color="green")

            if d[INPUT_PROM_VIN_SCALE + VALUE] == "logarithmic":
                fig.update_layout(yaxis_type="log")
                fig.update_layout(
                    yaxis_range=[
                        np.log10(np.quantile(values[values > 0], 0.05)),
                        np.log10(max(values)),
                    ]
                )
            else:
                fig.update_layout(yaxis_range=[0, max(values)])

            fig.update_layout(showlegend=False)
            fig.update_layout(autosize=True)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

            d[STORED_PV_DRAWING + DATA] = plotly.io.to_json(fig)

            return d

        @dash_callback(
            [
                [EXPORT_PARAMETERS_BUTTON, N_CLICKS, IN],
                [INPUT_GAP, VALUE, ST],
                [INPUT_LINE, VALUE, ST],
                [STORED_PV, DATA, ST]
            ],
            [[DUMMY_OUTPUT, CHILDREN]],
            True
        )
        def export_parameters(d):
            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = ProminenceVineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_prominence_diagrams"],
            )
            line = vineyard._parameters[d[INPUT_LINE+VALUE]]
            self._parameters = {"n_clusters": d[INPUT_GAP+VALUE], "start": line[0], "end": line[1]}
            d[DUMMY_OUTPUT + CHILDREN] = None
            return d

        if jupyter:
            if inline:
                self._app.run_server(mode="inline")
            else:
                self._app.run_server()
        else:
            self._app.run_server(debug=debug)
