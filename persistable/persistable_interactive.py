# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import warnings
import traceback
import plotly.graph_objects as go
import plotly
from plotly.colors import sample_colorscale
import json
#import diskcache
import dash
from dash import dcc, html, DiskcacheManager, ctx
from dash.exceptions import PreventUpdate
from jupyter_dash import JupyterDash
import click
import socket
from ._vineyard import Vineyard

import threading


# monkeypatch the hashing function of dash, so that
# we can use a decorator to register callbacks
from dash.long_callback.managers import BaseLongCallbackManager

import uuid


def monkeypatched_hash_function(fn):
    return uuid.uuid4()


BaseLongCallbackManager.hash_function = monkeypatched_hash_function
###


X_START_FIRST_LINE = "x-start-first-line-"
Y_START_FIRST_LINE = "y-start-first-line-"
X_END_FIRST_LINE = "x-end-first-line-"
Y_END_FIRST_LINE = "y-end-first-line-"
X_START_SECOND_LINE = "x-start-second-line-"
Y_START_SECOND_LINE = "y-start-second-line-"
X_END_SECOND_LINE = "x-end-second-line-"
Y_END_SECOND_LINE = "y-end-second-line-"
CCF_PLOT = "ccf-plot-"
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
STOP_COMPUTE_CCF_BUTTON = "stop-compute-ccf-button-"
INPUT_GRANULARITY_CCF = "input-granularity-ccf-"
INPUT_NUM_JOBS_CCF = "input-num-jobs-ccf-"
INPUT_MAX_COMPONENTS = "input-max-components-"
CCF_PLOT_CONTROLS_DIV = "ccf-plot-controls-div-"
CCF_DETAILS = "ccf-details-"
PV_DETAILS = "pv-details-"
PV_PLOT_CONTROLS_DIV = "pv-plot-controls-div-"
LOG = "log-"
LOG_DIV = "log-div-"
STORED_PV = "stored-pv-"
INPUT_MAX_VINES = "input-max-vines-"
INPUT_PROM_VIN_SCALE = "input-prom-vin-scale-"
COMPUTE_PV_BUTTON = "compute-pv-button-"
STOP_COMPUTE_PV_BUTTON = "stop-compute-pv-button-"
PV_PLOT = "pv-plot-"
STORED_PV_DRAWING = "stored-pv-drawing-"
INPUT_GRANULARITY_PV = "input-granularity-pv-"
INPUT_NUM_JOBS_PV = "input-num-jobs-pv-"
INPUT_LINE = "input-line-"
INPUT_GAP = "gap-"
EXPORT_PARAMETERS_BUTTON = "export-parameters-"
FIXED_PARAMETERS = "fixed-parameters-"
STORED_PD = "stored-pd-"

STORED_CCF_COMPUTATION_WARNINGS = "stored-ccf-computation-warnings-"
STORED_PV_COMPUTATION_WARNINGS = "stored-pv-computation-warnings-"
# STORED_PD_COMPUTATION_WARNINGS = "stored-pd-computation-warnings-"

EXPORTED_PARAMETER = "exported-parameter-"

SHUTDOWN_BUTTON = "shutdown-button-"

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
    """A graphical user interface for doing parameter selection for Persistable.

    persistable: Persistable
        Persistable instance with which to interact with the user interface.

    """

    def __init__(self, persistable):
        self._persistable = persistable
        self._app = None
        self._debug = False
        self._parameters_sem = threading.Semaphore()
        self._parameters = None


    def start_UI(self, port=8050, debug=False, inline=False):
        """Serves the GUI with a given persistable instance.

        port: int, optional, default is 8050
            Integer representing which port of localhost to try use to run the GUI.
            If port is not available, we look for one that is available, starting
            from the given one.

        debug: bool, optional, default is False
            Whether to run Dash in debug mode.

        inline: bool, optional, default is False
            Boolean representing whether or not to display the GUI inside
            the Jupyter notebook. Only works if using Persistable in a Jupyter notebook.

        return: int
            If not run in inline mode, returns the port of localhost used to serve the UI.

        """
        if debug:
            self._debug = debug
        max_port = 65535
        for possible_port in range(port, max_port+1):
            # check if port is in use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("localhost", int(possible_port))) == 0
            if in_use:
                continue
            else:
                port = possible_port
                break
        if possible_port == max_port:
            raise Exception(
                "All ports are already in use. Cannot start the GUI."
            )

        background_callback_manager = DiskcacheManager()

        def suppress_warnings(app):
            import logging
            app.logger.setLevel(logging.WARNING)
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            def secho(text, file=None, nl=None, err=None, color=None, **styles):
                pass
            def echo(text, file=None, nl=None, err=None, color=None, **styles):
                pass
            click.echo = echo
            click.secho = secho

        if inline == True:

            self._app = JupyterDash(
                __name__,
                background_callback_manager=background_callback_manager,
                update_title = "Persistable is computing..."
            )
            self._layout_gui()
            self._register_callbacks(self._persistable, self._debug)

            if not debug:
                suppress_warnings(self._app)

            self._app.run_server(port=port, mode="inline", debug=debug)
        else:
            self._app = dash.Dash(
                __name__,
                background_callback_manager=background_callback_manager,
                update_title = "Persistable is computing..."
            )
            self._layout_gui()
            self._register_callbacks(self._persistable, self._debug)

            if not debug:
                suppress_warnings(self._app)

            def run():
                self._app.run_server(port=port, debug=debug, use_reloader=False)

            self._thread = threading.Thread(target=run)
            self._thread.daemon = True
            self._thread.start()

            return port


    def cluster(self, **kwargs):
        """Clusters the dataset with which the Persistable instance that was
        passed through ``run_with`` was initialized.

        ``**kwargs``:
            Passed to ``Persistable.cluster``.

        returns:
            A numpy array of length the number of points in the dataset containing
            integers from -1 to the number of clusters minus 1, representing the
            labels of the final clustering. The label -1 represents noise points,
            i.e., points deemed not to belong to any cluster by the algorithm.

        """
        params = self._chosen_parameters()
        if params == None:
            raise ValueError(
                "No parameters where chosen. Please use the graphical user interface to choose parameters."
            )
        else:
            return self._persistable.cluster(**params, **kwargs)


    def _chosen_parameters(self):
        self._parameters_sem.acquire()
        params = self._parameters
        self._parameters_sem.release()
        return params


    def _layout_gui(self):
        default_min_k = 0
        end = self._persistable._find_end()
        default_max_k = end[1]
        #default_k_step = default_max_k / 100
        default_min_s = 0
        default_max_s = end[0]
        #default_s_step = (default_max_s - default_min_s) / 100
        default_granularity_ccf = 100
        default_granularity_pv = 40
        default_num_jobs = 4
        default_max_dim = 15
        default_max_vines = 15
        min_granularity = 4
        max_granularity = 512
        min_granularity_vineyard = 1
        max_granularity_vineyard = max_granularity
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

        self._app.title = "Persistable"
        self._app.layout = html.Div(
            className="root",
            children=[
                # contains the component counting function as a dictionary of lists
                dcc.Store(id=STORED_CCF),
                # contains the basic component counting function plot as a plotly figure
                dcc.Store(id=STORED_CCF_DRAWING),
                # contains the vineyard as a vineyard object
                dcc.Store(id=STORED_PV),
                # contains the basic prominence vineyard plot as a plotly figure
                dcc.Store(id=STORED_PV_DRAWING),
                #
                dcc.Store(id=STORED_PD, data=json.dumps([])),
                #
                dcc.Store(id=STORED_CCF_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                dcc.Store(id=STORED_PV_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                # dcc.Store(id=STORED_PD_COMPUTATION_WARNINGS),
                #
                dcc.Store(id=FIXED_PARAMETERS, data=json.dumps([])),
                #
                html.Div(id=EXPORTED_PARAMETER, hidden=True),
                #html.Div(id=EXPORTED_PARAMETER_2, hidden=True),
                html.Div(
                    className="grid",
                    children=[
                        html.Div(
                            children=[
                                html.H2("Component Counting Function"),
                                html.Div(
                                    className="parameters",
                                    children=[
                                        html.Details(
                                            id = CCF_DETAILS,
                                            children = [
                                                html.Summary("Inputs"),
                                                html.Div(
                                                    className="parameters",
                                                    children=[
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="Distance scale min/max",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=MIN_DIST_SCALE,
                                                                    type="number",
                                                                    value=default_min_s,
                                                                    min=0,
                                                                    #step=default_s_step,
                                                                    debounce=True,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=MAX_DIST_SCALE,
                                                                    type="number",
                                                                    value=default_max_s,
                                                                    min=0,
                                                                    #step=default_s_step,
                                                                    debounce=True,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="Density threshold min/max",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=MIN_DENSITY_THRESHOLD,
                                                                    type="number",
                                                                    value=default_min_k,
                                                                    min=0,
                                                                    #step=default_k_step,
                                                                    debounce=True,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=MAX_DENSITY_THRESHOLD,
                                                                    type="number",
                                                                    value=default_max_k,
                                                                    min=0,
                                                                    #step=default_k_step,
                                                                    debounce=True,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-single",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="Granularity",
                                                                ),
                                                                dcc.Input(
                                                                    id=INPUT_GRANULARITY_CCF,
                                                                    className="small-value",
                                                                    type="number",
                                                                    value=default_granularity_ccf,
                                                                    min=min_granularity,
                                                                    max=max_granularity,
                                                                    debounce=True,
                                                                ),
                                                                html.Span(
                                                                    className="name",
                                                                    children="Max connected components",
                                                                ),
                                                                dcc.Input(
                                                                    id=INPUT_MAX_COMPONENTS,
                                                                    type="number",
                                                                    value=default_max_dim,
                                                                    min=1,
                                                                    className="small-value",
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
                                                                    children="Number of cores computation",
                                                                ),
                                                                dcc.Input(
                                                                    className="small-value",
                                                                    id=INPUT_NUM_JOBS_CCF,
                                                                    type="number",
                                                                    value=default_num_jobs,
                                                                    min=1,
                                                                    step=1,
                                                                    max=16,
                                                                    debounce=True,
                                                                ),
                                                                html.Div(
                                                                    className="space"
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="large-buttons",
                                            children=[
                                                html.Button(
                                                    "Compute",
                                                    id=COMPUTE_CCF_BUTTON,
                                                    className="button1",
                                                ),
                                                html.Button(
                                                    "Stop computation",
                                                    id=STOP_COMPUTE_CCF_BUTTON,
                                                    className="button2",
                                                    disabled=True,
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
                                        html.Details(
                                            id = PV_DETAILS,
                                            children = [
                                                html.Summary("Inputs"),
                                                html.Div(
                                                    className="parameters",
                                                    children=[
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="1st line start x/y",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=X_START_FIRST_LINE,
                                                                    type="number",
                                                                    value=default_x_start_first_line,
                                                                    min=0,
                                                                    # step=default_s_step,
                                                                    debounce=True,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=Y_START_FIRST_LINE,
                                                                    type="number",
                                                                    value=default_y_start_first_line,
                                                                    min=0,
                                                                    # step=default_k_step,
                                                                    debounce=True,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="1st line end x/y",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=X_END_FIRST_LINE,
                                                                    type="number",
                                                                    value=default_x_end_first_line,
                                                                    min=0,
                                                                    # step=default_s_step,
                                                                    debounce=True,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=Y_END_FIRST_LINE,
                                                                    type="number",
                                                                    value=default_y_end_first_line,
                                                                    min=0,
                                                                    # step=default_k_step,
                                                                    debounce=True,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="2nd line start x/y",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=X_START_SECOND_LINE,
                                                                    type="number",
                                                                    value=default_x_start_second_line,
                                                                    min=0,
                                                                    # step=default_s_step,
                                                                    debounce=True,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=Y_START_SECOND_LINE,
                                                                    type="number",
                                                                    value=default_y_start_second_line,
                                                                    min=0,
                                                                    # step=default_k_step,
                                                                    debounce=True,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-double",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="2nd line end x/y",
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=X_END_SECOND_LINE,
                                                                    type="number",
                                                                    value=default_x_end_second_line,
                                                                    min=0,
                                                                    # step=default_s_step,
                                                                ),
                                                                dcc.Input(
                                                                    className=VALUE,
                                                                    id=Y_END_SECOND_LINE,
                                                                    type="number",
                                                                    value=default_y_end_second_line,
                                                                    min=0,
                                                                    # step=default_k_step,
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            className="parameter-single",
                                                            children=[
                                                                html.Span(
                                                                    className="name",
                                                                    children="Number of lines vineyard",
                                                                ),
                                                                dcc.Input(
                                                                    id=INPUT_GRANULARITY_PV,
                                                                    className="small-value",
                                                                    type="number",
                                                                    value=default_granularity_pv,
                                                                    min=min_granularity_vineyard,
                                                                    max=max_granularity_vineyard,
                                                                    debounce=True,
                                                                ),
                                                                html.Span(
                                                                    className="name",
                                                                    children="Max number vines to display",
                                                                ),
                                                                dcc.Input(
                                                                    id=INPUT_MAX_VINES,
                                                                    type="number",
                                                                    value=default_max_vines,
                                                                    min=1,
                                                                    className="small-value",
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
                                                                    children="Number of cores computation",
                                                                ),
                                                                dcc.Input(
                                                                    className="small-value",
                                                                    id=INPUT_NUM_JOBS_PV,
                                                                    type="number",
                                                                    value=default_num_jobs,
                                                                    min=1,
                                                                    step=1,
                                                                    max=16,
                                                                    debounce=True,
                                                                ),
                                                                html.Span(
                                                                    className="name",
                                                                    children="Prominence scale",
                                                                ),
                                                                dcc.RadioItems(
                                                                    ["Lin", "Log"],
                                                                    "Lin",
                                                                    id=INPUT_PROM_VIN_SCALE,
                                                                    className="small-value",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            className="parameters",
                                            children=[
                                                html.Div(
                                                    className="large-buttons",
                                                    children=[
                                                        html.Button(
                                                            "Compute",
                                                            id=COMPUTE_PV_BUTTON,
                                                            className="button1",
                                                        ),
                                                        html.Button(
                                                            "Stop computation",
                                                            id=STOP_COMPUTE_PV_BUTTON,
                                                            className="button2",
                                                            disabled=True,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        dcc.Graph(
                            id=CCF_PLOT,
                            className="graph",
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
                            className="graph",
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
                        html.Div(
                            className="plot-tools",
                            children=[
                                html.Div(
                                    id=CCF_PLOT_CONTROLS_DIV,
                                    className="parameters",
                                    hidden=True,
                                    children=[
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="Vineyard inputs selection",
                                                ),
                                                dcc.RadioItems(
                                                    ["On", "Off"],
                                                    "Off",
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
                                                    children="Endpoint",
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
                            ],
                        ),
                        html.Div(
                            className="plot-tools",
                            children=[
                                html.Div(
                                    id=PV_PLOT_CONTROLS_DIV,
                                    className="parameters",
                                    hidden=True,
                                    children=[
                                        html.Div(
                                            className="parameter-single",
                                            children=[
                                                html.Span(
                                                    className="name",
                                                    children="Parameter selection",
                                                ),
                                                dcc.RadioItems(
                                                    ["On", "Off"],
                                                    "Off",
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
                                                    children="Line number",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_LINE,
                                                    type="number",
                                                    value=1,
                                                    min=1,
                                                    debounce=True,
                                                ),
                                                html.Span(
                                                    className="name",
                                                    children="Gap number",
                                                ),
                                                dcc.Input(
                                                    className=VALUE,
                                                    id=INPUT_GAP,
                                                    type="number",
                                                    value=1,
                                                    min=1,
                                                    debounce=True,
                                                ),
                                                html.Button(
                                                    "Choose parameter",
                                                    id=EXPORT_PARAMETERS_BUTTON,
                                                    className="button",
                                                    disabled=True,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Details(
                    id=LOG_DIV,
                    className="warnings",
                    children=[
                        html.Summary("Warnings"),
                        html.Pre(
                            id=LOG,
                            children=[" "],
                        ),
                    ],
                    open=False,
                ),
                html.Details(
                    className="help",
                    children=[
                        html.Summary("Tips"),
                        dcc.Markdown(
                            """
            - Check the documentation at [persistable.readthedocs.io](https://persistable.readthedocs.io/).
            - Check the log, above, for warnings.
            - Press enter after you modify numerical fields.
            - Computing the component counting function and prominence vineyard can take a while, depending on the size and dimensionality of the dataset as well as other factors.
            - Make sure to leave your pointer still when clicking on the component counting function plot, otherwise your interaction may not be registered.
            """
                        ),
                    ],
                ),
            ],
        )


    # when using long callbacks, dash will pickle all the objects that are used
    # in the function that is being registered as a long callback. In particular,
    # using self._persistable will also pickle self, which in this case contains
    # also a dash app, which cannot be pickled. We get around this by passing
    # self._persistable to the _register_callbacks function.
    def _register_callbacks(self, persistable, debug):
        def dash_callback(
            inputs,
            outputs,
            prevent_initial_call=False,
            background=False,
            running=None,
            cancel=None,
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

                if cancel is None:
                    dash_cancel = None
                else:
                    dash_cancel = [(dash.Input(i, v)) for i, v in cancel]

                def callback_function(*argv):
                    d = {}
                    for n, arg in enumerate(argv):
                        if arg is None:
                            raise PreventUpdate
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
                        cancel=dash_cancel,
                    )(callback_function)
                    # TODO: figure out why the following causes problems with joblib Parallel
                    # Note that callback with background=True is the recommended way of running
                    # background jobs in dash now.
                    #self._app.callback(
                    #    dash_outputs,
                    #    dash_inputs,
                    #    prevent_initial_call,
                    #    running=dash_running_outputs,
                    #    cancel=dash_cancel,
                    #    background=True,
                    #)(callback_function)
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
            if d[DISPLAY_PARAMETER_SELECTION + VALUE] == "On":
                d[PARAMETER_SELECTION_DIV + HIDDEN] = False
            else:
                d[PARAMETER_SELECTION_DIV + HIDDEN] = True
            return d

        @dash_callback(
            [
                [STORED_CCF_COMPUTATION_WARNINGS, DATA, IN],
                [STORED_PV_COMPUTATION_WARNINGS, DATA, IN],
            ],
            [[LOG, CHILDREN], [LOG_DIV, "open"]],
            True,
        )
        def print_log(d):
            if ctx.triggered_id == STORED_CCF_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_CCF_COMPUTATION_WARNINGS + DATA])
            elif ctx.triggered_id == STORED_PV_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_PV_COMPUTATION_WARNINGS + DATA])
            else:
                raise Exception(
                    "print_log was triggered by unknown id: " + str(ctx.triggered_id)
                )
            if len(message) > 0:
                d[LOG_DIV + "open"] = True
                d[LOG + CHILDREN] = message
            else:
                d[LOG_DIV + "open"] = False
                d[LOG + CHILDREN] = " "

            return d

        @dash_callback(
            [
                [CCF_PLOT, CLICKDATA, IN],
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
            if d[DISPLAY_LINES_SELECTION + VALUE] == "On":
                new_x, new_y = (
                    d[CCF_PLOT + CLICKDATA]["points"][0]["x"],
                    d[CCF_PLOT + CLICKDATA]["points"][0]["y"],
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
            if d[DISPLAY_LINES_SELECTION + VALUE] == "On":
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

            ccf = json.loads(ccf)

            max_components = d[INPUT_MAX_COMPONENTS + VALUE]

            fig = go.Figure(
                layout=go.Layout(
                    xaxis_title="Distance scale",
                    yaxis_title="Density threshold",
                    xaxis={"fixedrange": True},
                    yaxis={"fixedrange": True},
                ),
            )

            fig.add_trace(
                go.Heatmap(
                    **ccf,
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
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            fig.update_layout(clickmode="event+select")

            d[STORED_CCF_DRAWING + DATA] = plotly.io.to_json(fig)
            return d

        @dash_callback(
            [
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
                [DISPLAY_PARAMETER_SELECTION, VALUE, IN],
                [FIXED_PARAMETERS, DATA, IN],
                [STORED_PD, DATA, ST],
                [INPUT_GAP, VALUE, ST],
            ],
            [[CCF_PLOT, FIGURE]],
            False,
        )
        def draw_ccf_extras(d):
            fig = plotly.io.from_json(d[STORED_CCF_DRAWING + DATA])

            def generate_line(
                xs, ys, text, color="mediumslateblue", different_marker=None
            ):
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
                    marker=dict(size=15, color=color),
                    marker_symbol=marker_styles,
                    hoverinfo="skip",
                    showlegend=False,
                    mode="markers+lines+text",
                    textposition=["top center", "bottom center"],
                )

            if d[DISPLAY_LINES_SELECTION + VALUE] == "On":

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
                        showlegend=False,
                        # text=text,
                        name="",
                    )
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
                    generate_line(
                        [d[X_START_FIRST_LINE + VALUE], d[X_END_FIRST_LINE + VALUE]],
                        [d[Y_START_FIRST_LINE + VALUE], d[Y_END_FIRST_LINE + VALUE]],
                        "1st",
                        different_marker=first_line_endpoints,
                        color="mediumslateblue",
                    )
                )
                fig.add_trace(
                    generate_line(
                        [d[X_START_SECOND_LINE + VALUE], d[X_END_SECOND_LINE + VALUE]],
                        [d[Y_START_SECOND_LINE + VALUE], d[Y_END_SECOND_LINE + VALUE]],
                        "2nd",
                        different_marker=second_line_endpoints,
                        color="mediumslateblue",
                    )
                )

            params = json.loads(d[FIXED_PARAMETERS + DATA])
            if len(params) != 0 and d[DISPLAY_PARAMETER_SELECTION + VALUE] == "On":
                pd = json.loads(d[STORED_PD + DATA])
                if len(pd) != 0:

                    def generate_bar(xs, ys, color):
                        return go.Scatter(
                            x=xs,
                            y=ys,
                            marker=dict(color=color),
                            hoverinfo="skip",
                            showlegend=False,
                            mode="lines",
                            line=dict(width=6),
                        )

                    st_x = params["start"][0]
                    st_y = params["start"][1]
                    end_x = params["end"][0]
                    end_y = params["end"][1]
                    st = np.array([st_x, st_y])
                    end = np.array([end_x, end_y])
                    A = end - st

                    # ideally we would get the actual ratio of the rendered picture
                    # we are using an estimate given by the "usual" way in which persistable
                    # is rendered
                    ratio = np.array([3,2])
                    ratio = (ratio / np.linalg.norm(ratio)) * np.linalg.norm(np.array([1,1]))
                    alpha = [
                        (d[MAX_DIST_SCALE + VALUE] - d[MIN_DIST_SCALE + VALUE]),
                        ( d[MAX_DENSITY_THRESHOLD + VALUE] - d[MIN_DENSITY_THRESHOLD + VALUE]),
                    ]
                    alpha = np.array(alpha) / ratio

                    A = A/alpha
                    B = np.array([-A[1], A[0]])

                    shift = 50
                    B = B / np.linalg.norm(B)
                    tau = B * alpha/ shift

                    pd = np.array(pd)
                    lengths = pd[:, 1] - pd[:, 0]
                    pd = pd[np.argsort(lengths)[::-1]]
                    for i, point in enumerate(pd):
                        l = end_x - st_x
                        p_st = point[0]
                        p_end = point[1]
                        q_st_x = st_x + p_st
                        q_end_x = st_x + p_end
                        q_st_y = st_y - (st_y - end_y) * (p_st / l)
                        q_end_y = st_y - (st_y - end_y) * (p_end / l)
                        q_st = np.array([q_st_x, q_st_y])
                        q_end = np.array([q_end_x, q_end_y])
                        r_st = q_st + (i + 1) * tau
                        r_end = q_end + (i + 1) * tau
                        color = (
                            "rgba(34, 139, 34, 1)"
                            if i < d[INPUT_GAP + VALUE]
                            else "rgba(34, 139, 34, 0.3)"
                        )
                        fig.add_trace(
                            generate_bar(
                                [r_st[0], r_end[0]], [r_st[1], r_end[1]], color
                            )
                        )
                fig.add_trace(
                    generate_line(
                        [st_x, end_x],
                        [st_y, end_y],
                        "selected",
                        color="blue",
                    )
                )

            fig.update_layout(
                xaxis=dict(range=[d[MIN_DIST_SCALE + VALUE], d[MAX_DIST_SCALE + VALUE]]),
                yaxis=dict(range=[
                    d[MIN_DENSITY_THRESHOLD + VALUE],
                    d[MAX_DENSITY_THRESHOLD + VALUE],
                ])
            )

            d[CCF_PLOT + FIGURE] = fig
            return d

        @dash_callback(
            [[STORED_PV, DATA, ST], [STORED_PV_DRAWING, DATA, IN]], [[PV_PLOT, FIGURE]]
        )
        def draw_pv_post(d):

            d[PV_PLOT + FIGURE] = plotly.io.from_json(d[STORED_PV_DRAWING + DATA])

            return d

        @dash_callback(
            [
                [COMPUTE_CCF_BUTTON, N_CLICKS, IN],
                [MIN_DENSITY_THRESHOLD, VALUE, ST],
                [MAX_DENSITY_THRESHOLD, VALUE, ST],
                [MIN_DIST_SCALE, VALUE, ST],
                [MAX_DIST_SCALE, VALUE, ST],
                [INPUT_GRANULARITY_CCF, VALUE, ST],
                [INPUT_NUM_JOBS_CCF, VALUE, ST],
            ],
            [
                [STORED_CCF, DATA],
                [STORED_CCF_COMPUTATION_WARNINGS, DATA],
                [CCF_PLOT_CONTROLS_DIV, HIDDEN],
            ],
            prevent_initial_call=True,
            background=True,
            running=[
                [COMPUTE_CCF_BUTTON, DISABLED, True, False],
                [STOP_COMPUTE_CCF_BUTTON, DISABLED, False, True],
            ],
            cancel=[[STOP_COMPUTE_CCF_BUTTON, N_CLICKS]],
        )
        def compute_ccf(d):
            if debug:
                print("Compute ccf in background started.")

            granularity = d[INPUT_GRANULARITY_CCF + VALUE]
            num_jobs = int(d[INPUT_NUM_JOBS_CCF + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    ss, ks, hf = persistable._compute_hilbert_function(
                        d[MIN_DENSITY_THRESHOLD + VALUE],
                        d[MAX_DENSITY_THRESHOLD + VALUE],
                        d[MIN_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        granularity,
                        n_jobs=num_jobs,
                    )
                except ValueError:
                    out += traceback.format_exc()
                    d[STORED_CCF_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
                    d[STORED_CCF + DATA] = None
                    d[CCF_PLOT_CONTROLS_DIV + HIDDEN] = True
                    return d

            for a in w:
                out += warnings.formatwarning(
                    a.message, a.category, a.filename, a.lineno
                )

            d[STORED_CCF + DATA] = json.dumps(
                {"y": ks[:-1].tolist(), "x": ss[:-1].tolist(), "z" : hf.tolist()}
            )

            d[STORED_CCF_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[CCF_PLOT_CONTROLS_DIV + HIDDEN] = False

            if debug:
                print("Compute ccf in background finished.")

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
                [INPUT_GRANULARITY_PV, VALUE, ST],
                [INPUT_NUM_JOBS_PV, VALUE, ST],
            ],
            [
                [STORED_PV, DATA],
                [STORED_PV_COMPUTATION_WARNINGS, DATA],
                [INPUT_LINE, "max"],
                [INPUT_LINE, VALUE],
                [EXPORT_PARAMETERS_BUTTON, DISABLED],
                [PV_PLOT_CONTROLS_DIV, HIDDEN],
            ],
            prevent_initial_call=True,
            background=True,
            running=[
                [COMPUTE_PV_BUTTON, DISABLED, True, False],
                [STOP_COMPUTE_PV_BUTTON, DISABLED, False, True],
            ],
            cancel=[[STOP_COMPUTE_PV_BUTTON, N_CLICKS]],
        )
        def compute_pv(d):
            if debug:
                print("Compute pv in background started.")

            granularity = d[INPUT_GRANULARITY_PV + VALUE]
            num_jobs = int(d[INPUT_NUM_JOBS_PV + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    pv = persistable._compute_vineyard(
                        [
                            [
                                d[X_START_FIRST_LINE + VALUE],
                                d[Y_START_FIRST_LINE + VALUE],
                            ],
                            [d[X_END_FIRST_LINE + VALUE], d[Y_END_FIRST_LINE + VALUE]],
                        ],
                        [
                            [
                                d[X_START_SECOND_LINE + VALUE],
                                d[Y_START_SECOND_LINE + VALUE],
                            ],
                            [
                                d[X_END_SECOND_LINE + VALUE],
                                d[Y_END_SECOND_LINE + VALUE],
                            ],
                        ],
                        n_parameters=granularity,
                        n_jobs=num_jobs,
                    )
                except ValueError:
                    out += traceback.format_exc()
                    d[STORED_PV_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
                    d[STORED_PV + DATA] = None
                    d[INPUT_LINE + "max"] = granularity
                    d[INPUT_LINE + VALUE] = granularity // 2
                    d[EXPORT_PARAMETERS_BUTTON + DISABLED] = True
                    d[PV_PLOT_CONTROLS_DIV + HIDDEN] = True
                    return d

            for a in w:
                out += warnings.formatwarning(
                    a.message, a.category, a.filename, a.lineno
                )

            d[STORED_PV + DATA] = json.dumps(pv.__dict__)
            d[STORED_PV_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[INPUT_LINE + "max"] = granularity
            d[INPUT_LINE + VALUE] = granularity // 2
            d[EXPORT_PARAMETERS_BUTTON + DISABLED] = False

            d[PV_PLOT_CONTROLS_DIV + HIDDEN] = False

            if debug:
                print("Compute pv in background finished.")

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

            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = Vineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_persistence_diagrams"],
            )

            _vineyard_values = []

            times = np.array(vineyard.parameter_indices()) + 1
            vines = vineyard._vineyard_to_vines()
            num_vines = min(len(vines), firstn)

            fig = go.Figure(
                layout=go.Layout(
                    xaxis_title="Parameter line",
                    yaxis_title="Prominence",
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
                    color = colors[i]
                    if (
                        d[DISPLAY_PARAMETER_SELECTION + VALUE] == "On"
                        and i + 1 == d[INPUT_GAP + VALUE]
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=vines[i][1],
                                fill=till,
                                # hoveron="fills",
                                text="vine " + str(i + 1),
                                hoverinfo="text",
                                line_color=color,
                                fillpattern=go.scatter.Fillpattern(shape="x"),
                            )
                        )
                    else:
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
                    _vineyard_values.extend(vine_part)
            values = np.array(_vineyard_values)

            if d[DISPLAY_PARAMETER_SELECTION + VALUE] == "On":
                fig.add_vline(x=d[INPUT_LINE + VALUE], line_color="grey")

            if len(values) > 0:
                if d[INPUT_PROM_VIN_SCALE + VALUE] == "Log":
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
                [INPUT_GAP, VALUE, IN],
                [INPUT_LINE, VALUE, IN],
                [STORED_PV, DATA, IN],
            ],
            [
                [FIXED_PARAMETERS, DATA],
                [STORED_PD, DATA],
            ],
            True,
        )
        def fix_parameters(d):
            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = Vineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_persistence_diagrams"],
            )
            line = vineyard._parameters[d[INPUT_LINE + VALUE] - 1]
            params = {
                "n_clusters": d[INPUT_GAP + VALUE],
                "start": line[0],
                "end": line[1],
            }
            d[FIXED_PARAMETERS + DATA] = json.dumps(params)

            pd = vineyard._persistence_diagrams[d[INPUT_LINE + VALUE] - 1]

            d[STORED_PD + DATA] = json.dumps(pd)

            return d

        @dash_callback(
            [
                [EXPORT_PARAMETERS_BUTTON, N_CLICKS, IN],
                [FIXED_PARAMETERS, DATA, ST],
            ],
            [[EXPORTED_PARAMETER, CHILDREN]],
            True,
        )
        def export_parameters(d):
            self._parameters_sem.acquire()
            self._parameters = json.loads(d[FIXED_PARAMETERS + DATA])
            self._parameters_sem.release()
            d[EXPORTED_PARAMETER + CHILDREN] = json.dumps(self._parameters)
            return d
