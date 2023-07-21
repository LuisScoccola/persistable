# Authors: Luis Scoccola
# License: 3-clause BSD

import numpy as np
import warnings
import traceback
import plotly.graph_objects as go
import plotly
from plotly.colors import sample_colorscale
import json

# import diskcache
import dash
from dash import dcc, html, DiskcacheManager, ctx
from dash.exceptions import PreventUpdate
import click
import socket
from ._vineyard import Vineyard

import threading


X_POINT = "x-point-"
Y_POINT = "y-point-"
POINT_SELECTION_DIV = "point-selection-div-"
X_START_LINE = "x-start-line-"
Y_START_LINE = "y-start-line-"
X_END_LINE = "x-end-line-"
Y_END_LINE = "y-end-line-"
X_START_FIRST_LINE = "x-start-first-line-"
Y_START_FIRST_LINE = "y-start-first-line-"
X_END_FIRST_LINE = "x-end-first-line-"
Y_END_FIRST_LINE = "y-end-first-line-"
X_START_SECOND_LINE = "x-start-second-line-"
Y_START_SECOND_LINE = "y-start-second-line-"
X_END_SECOND_LINE = "x-end-second-line-"
Y_END_SECOND_LINE = "y-end-second-line-"
CCF_PLOT = "ccf-plot-"
INTERACTIVE_INPUTS_SELECTION = "interactive-inputs-selection-"
PV_ENDPOINT_SELECTION = "pv-endpoint-selection-"
PD_ENDPOINT_SELECTION = "pd-endpoint-selection-"
PV_DISPLAY_BARCODE = "pv-display-barcode-"
PD_DISPLAY_BARCODE = "pd-display-barcode-"
STORED_CCF = "stored-ccf-"
STORED_X_TICKS_CCF = "stored-x-ticks-ccf-"
STORED_Y_TICKS_CCF = "stored-y-ticks-ccf-"
STORED_X_TICKS_RI = "stored-x-ticks-ri-"
STORED_Y_TICKS_RI = "stored-y-ticks-ri-"
STORED_CCF_DRAWING = "stored-ccf-drawing-"
STORED_BETTI = "stored-betti-"
STORED_SIGNED_BARCODE_RECTANGLES = "stored-signed-barcode-rectangles-"
STORED_SIGNED_BARCODE_HOOKS = "stored-signed-barcode-hooks-"
MIN_DIST_SCALE = "min-dist-scale-"
MAX_DIST_SCALE = "max-dist-scale-"
MIN_DENSITY_THRESHOLD = "min-density-threshold-"
MAX_DENSITY_THRESHOLD = "max-density-threshold-"
PV_ENDPOINT_SELECTION_DIV = "pv-endpoint-selection-div-"
PD_ENDPOINT_SELECTION_DIV = "pd-endpoint-selection-div-"
PARAMETER_SELECTION_DIV_PV = "parameter-selection-div-pv-"
PARAMETER_SELECTION_DIV_PD = "parameter-selection-div-pd-"
DISPLAY_PARAMETER_SELECTION_PV = "display-parameter-selection-pv-"
DISPLAY_PARAMETER_SELECTION_PD = "display-parameter-selection-pd-"
DISPLAY_SELECTED_LINE = "display-selected-line-"
COMPUTE_CCF_BUTTON = "compute-ccf-button-"
STOP_COMPUTE_CCF_BUTTON = "stop-compute-ccf-button-"
COMPUTE_RI_BUTTON = "compute-ri-button-"
STOP_COMPUTE_RI_BUTTON = "stop-compute-ri-button-"
GRANULARITY = "granularity-"
MIN_GRANULARITY = "min-granularity-"
MAX_GRANULARITY = "max-granularity-"
MAX_GRANULARITY_RI = "max-granularity-ri-"
MIN_GRANULARITY_VINEYARD = "min-granularity-vineyard-"
MAX_GRANULARITY_VINEYARD = "max-granularity-vineyard-"
GRANULARITY_RI = "granularity-ri-"
NUM_JOBS_CCF = "num-jobs-ccf-"
NUM_JOBS_RI = "num-jobs-ri-"
MAX_COMPONENTS = "max-components-"
MAX_RI = "max-ri-"
MIN_LENGTH_RI = "min-length-bars-ri-"
SIGNED_BETTI_NUMBERS = "signed-betti-numbers-"
Y_COVARIANT = "y-covariant-"
DISPLAY_RI = "display-ri-"
DECOMPOSE_BY_RI = "decompose-by-ri-"
REDUCED_HOMOLOGY_RI = "reduced-homology-ri-"
CCF_PLOT_CONTROLS_DIV = "ccf-plot-controls-div-"
CCF_DETAILS = "ccf-details-"
CCF_EXTRAS = "ccf-extras-"
PV_PANEL = "pv-panel-"
PV_DETAILS = "pv-details-"
PD_DETAILS = "pd-details-"
PV_PLOT_CONTROLS_DIV = "pv-plot-controls-div-"
PD_PLOT_CONTROLS_DIV = "pd-plot-controls-div-"
PD_PANEL = "pd-panel-"
LOG = "log-"
LOG_DIV = "log-div-"
STORED_PV = "stored-pv-"
MAX_VINES = "max-vines-"
PROM_VIN_SCALE = "prom-vin-scale-"
COMPUTE_PV_BUTTON = "compute-pv-button-"
STOP_COMPUTE_PV_BUTTON = "stop-compute-pv-button-"
COMPUTE_PD_BUTTON = "compute-pd-button-"
STOP_COMPUTE_PD_BUTTON = "stop-compute-pd-button-"
PV_PLOT = "pv-plot-"
PD_PLOT = "pd-plot-"
STORED_PV_DRAWING = "stored-pv-drawing-"
GRANULARITY_PV = "granularity-pv-"
NUM_JOBS_PV = "num-jobs-pv-"
LINE = "line-"
PV_GAP = "pv-gap-"
PD_GAP = "pd-gap-"
EXPORT_PARAMETERS_BUTTON_PV = "export-parameters-button-pv-"
EXPORT_PARAMETERS_BUTTON_PD = "export-parameters-button-pd-"
EXPORT_PARAMETERS_BUTTON_DBSCAN = "export-parameters-button-dbscan-"
PV_FIXED_PARAMETERS = "fixed-parameters-"
STORED_PD_BY_PV = "stored-pd-by-pv-"
STORED_PARAMETERS_AND_PD_BY_PD = "stored-pd-by-pd-"

STORED_CCF_COMPUTATION_WARNINGS = "stored-ccf-computation-warnings-"
STORED_PV_COMPUTATION_WARNINGS = "stored-pv-computation-warnings-"
STORED_RI_COMPUTATION_WARNINGS = "stored-ri-computation-warnings-"
STORED_PD_COMPUTATION_WARNINGS = "stored-pd-computation-warnings-"

EXPORTED_PARAMETER = "exported-parameter-"
EXPORTED_STATE = "exported-state-"

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


def compute_defaults(end, default_granularity):
    bounds = {
        MIN_GRANULARITY: 2,
        MAX_GRANULARITY: 512,
        MAX_GRANULARITY_RI: 64,
        MIN_GRANULARITY_VINEYARD: 1,
        MAX_GRANULARITY_VINEYARD: 512,
    }
    d0 = {GRANULARITY: default_granularity}
    d1 = {
        MIN_DENSITY_THRESHOLD: 0,
        MAX_DENSITY_THRESHOLD: end[1],
        MIN_DIST_SCALE: 0,
        MAX_DIST_SCALE: end[0],
        GRANULARITY_RI: d0[GRANULARITY] // 5,
        GRANULARITY_PV: d0[GRANULARITY] // 2,
        NUM_JOBS_CCF: 1,
        NUM_JOBS_PV: 1,
        NUM_JOBS_RI: 1,
        MAX_COMPONENTS: 15,
        MAX_VINES: 15,
        Y_COVARIANT: "Cov",
        LINE: 1,
        PV_GAP: 1,
        PD_GAP: 1,
    }
    defr = 6
    d2 = {
        X_START_FIRST_LINE: (d1[MIN_DIST_SCALE] + d1[MAX_DIST_SCALE]) * (1 / defr),
        Y_START_FIRST_LINE: (d1[MIN_DENSITY_THRESHOLD] + d1[MAX_DENSITY_THRESHOLD])
        * (1 / 2),
        X_END_FIRST_LINE: (d1[MAX_DIST_SCALE] + d1[MIN_DIST_SCALE]) * (1 / 2),
        Y_END_FIRST_LINE: (d1[MIN_DENSITY_THRESHOLD] + d1[MAX_DENSITY_THRESHOLD])
        * (1 / defr),
        X_START_SECOND_LINE: (d1[MIN_DIST_SCALE] + d1[MAX_DIST_SCALE]) * (1 / 2),
        Y_START_SECOND_LINE: (d1[MIN_DENSITY_THRESHOLD] + d1[MAX_DENSITY_THRESHOLD])
        * ((defr - 1) / defr),
        X_END_SECOND_LINE: (d1[MAX_DIST_SCALE] + d1[MIN_DIST_SCALE])
        * ((defr - 1) / defr),
        Y_END_SECOND_LINE: (d1[MIN_DENSITY_THRESHOLD] + d1[MAX_DENSITY_THRESHOLD])
        * (1 / 2),
    }
    d3 = {
        X_START_LINE: (d2[X_START_FIRST_LINE] + d2[X_START_SECOND_LINE]) / 2,
        Y_START_LINE: (d2[Y_START_FIRST_LINE] + d2[Y_START_SECOND_LINE]) / 2,
        X_END_LINE: (d2[X_END_FIRST_LINE] + d2[X_END_SECOND_LINE]) / 2,
        Y_END_LINE: (d2[Y_END_FIRST_LINE] + d2[Y_END_SECOND_LINE]) / 2,
    }
    d4 = {
        X_POINT: (d3[X_START_LINE] + d3[X_END_LINE]) / 2,
        Y_POINT: (d3[Y_START_LINE] + d3[Y_END_LINE]) / 2,
    }

    return {**d0, **d1, **d2, **d3, **d4}, bounds


class PersistableInteractive:
    """Graphical user interface for doing parameter selection for ``Persistable``.

    persistable: Persistable
        Persistable instance with which to interact with the user interface.

    """

    def __init__(self, persistable):
        self._persistable = persistable
        self._app = None
        self._debug = False
        self._parameters_sem = threading.Semaphore()
        self._parameters = None
        self._ui_state = None

    def start_ui(self, ui_state=None, port=8050, debug=False, jupyter_mode="external"):
        """Serves the GUI with a given persistable instance.

        ui_state: dictionary, optional
            The state of a previous UI session, as a Python object, obtained
            by calling the method ``save_ui_state()``.

        port: int, optional, default is 8050
            Integer representing which port of localhost to try use to run the GUI.
            If port is not available, we look for one that is available, starting
            from the given one.

        debug: bool, optional, default is False
            Whether to run Dash in debug mode.

        jupyter_mode: string, optional, default is "external"
            How to display the application when running inside a jupyter notebook.
            Options are "external" to serve the app in a port returned by this function,
            "inline" to open the app inline in the jupyter notebook.
            "jupyterlab" to open the app in a separate tab in JupyterLab.

        return: int
            Returns the port of localhost used to serve the UI.

        """

        if debug:
            self._debug = debug
        max_port = 65535
        for possible_port in range(port, max_port + 1):
            # check if port is in use
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("localhost", int(possible_port))) == 0
            if in_use:
                continue
            else:
                port = possible_port
                break
        if possible_port == max_port:
            raise Exception("All ports are already in use. Cannot start the GUI.")

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

        self._app = dash.Dash(
            __name__,
            background_callback_manager=background_callback_manager,
            update_title="Persistable is computing...",
        )
        self._layout_gui(ui_state)
        self._register_callbacks(self._persistable, self._debug)

        if not debug:
            suppress_warnings(self._app)

        jupyter_height = 1000

        def run():
            self._app.run_server(
                port=port,
                debug=debug,
                use_reloader=False,
                jupyter_mode=jupyter_mode,
                jupyter_height=jupyter_height,
            )

        self._thread = threading.Thread(target=run)
        self._thread.daemon = True
        self._thread.start()

        return port

    def save_ui_state(self):
        """Save state of input fields in the UI as a Python object. The output
        can then be used as the optional input of the ``start_ui()`` method.

        returns: dictionary
        """
        self._parameters_sem.acquire()
        state = self._ui_state.copy()
        self._parameters_sem.release()
        return state

    def cluster(
        self, flattening_mode="conservative", keep_low_persistence_clusters=False
    ):
        """Clusters the dataset with which the Persistable instance that was initialized.

        flattening_mode: string, optional, default is "conservative"
            If "exhaustive", flatten the hierarchical clustering using the approach
            of 'Persistence-Based Clustering in Riemannian Manifolds' Chazal, Guibas,
            Oudot, Skraba.
            If "conservative", use the more stable approach of
            'Stable and consistent density-based clustering' Rolle, Scoccola.
            The conservative approach usually results in more unclustered points.

        keep_low_persistence_clusters: bool, optional, default is False
            Only has effect if ``flattening_mode`` is set to "exhaustive".
            Whether to keep clusters that are born below the persistence threshold
            associated to the selected n_clusters. If set to True, the number of clusters
            can be larger than the selected one.

        returns:
            A numpy array of length the number of points in the dataset containing
            integers from -1 to the number of clusters minus 1, representing the
            labels of the final clustering. The label -1 represents noise points,
            i.e., points deemed not to belong to any cluster by the algorithm.

        """
        params = self._chosen_parameters()
        if params == None:
            raise ValueError(
                "No parameters were chosen. Please use the graphical user interface to choose parameters."
            )
        else:
            if "point" in params:
                return self._persistable._dbscan_cluster(params["point"])
            else:
                return self._persistable.cluster(
                    **params,
                    flattening_mode=flattening_mode,
                    keep_low_persistence_clusters=keep_low_persistence_clusters
                )

    def _chosen_parameters(self):
        self._parameters_sem.acquire()
        if self._parameters is None:
            params = None
        else:
            params = self._parameters.copy()
        self._parameters_sem.release()
        return params

    def _layout_gui(self, ui_state):
        defaults, bounds = compute_defaults(
            self._persistable._find_end(), self._persistable._default_granularity()
        )
        if ui_state is None:
            ui_state = defaults

        self._app.title = "Persistable"

        ccf_inputs = (
            html.Details(
                id=CCF_DETAILS,
                children=[
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
                                        value=ui_state[MIN_DIST_SCALE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=MAX_DIST_SCALE,
                                        type="number",
                                        value=ui_state[MAX_DIST_SCALE],
                                        min=0,
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
                                        value=ui_state[MIN_DENSITY_THRESHOLD],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=MAX_DENSITY_THRESHOLD,
                                        type="number",
                                        value=ui_state[MAX_DENSITY_THRESHOLD],
                                        min=0,
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
                                        id=GRANULARITY,
                                        className="small-value",
                                        type="number",
                                        value=ui_state[GRANULARITY],
                                        min=bounds[MIN_GRANULARITY],
                                        max=bounds[MAX_GRANULARITY],
                                        debounce=True,
                                    ),
                                    html.Span(
                                        className="name",
                                        children="Max connected components",
                                    ),
                                    dcc.Input(
                                        id=MAX_COMPONENTS,
                                        type="number",
                                        value=ui_state[MAX_COMPONENTS],
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
                                        children="# cores computation",
                                    ),
                                    dcc.Input(
                                        className="small-value",
                                        id=NUM_JOBS_CCF,
                                        type="number",
                                        value=ui_state[NUM_JOBS_CCF],
                                        min=1,
                                        step=1,
                                        max=16,
                                        debounce=True,
                                    ),
                                    html.Span(
                                        className="name",
                                        children="Y axis",
                                    ),
                                    dcc.RadioItems(
                                        [
                                            "Cov",
                                            "Contr",
                                        ],
                                        value=ui_state[Y_COVARIANT],
                                        id=Y_COVARIANT,
                                        className="small-value",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )

        ccf_buttons = (
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
        )

        ccf_parameter_selection = (
            html.Div(
                className="plot-tools",
                children=[
                    html.Div(
                        id=CCF_PLOT_CONTROLS_DIV,
                        className="parameters",
                        hidden=True,
                        children=[
                            html.Div(
                                className="parameters",
                                children=[
                                    html.Div(
                                        className="parameter-single",
                                        children=[
                                            html.Span(
                                                className="name",
                                                children="Interactive inputs selection",
                                            ),
                                            dcc.RadioItems(
                                                [
                                                    "Off",
                                                    # "Single clustering",
                                                    "Line",
                                                    "Family of lines",
                                                ],
                                                "Off",
                                                id=INTERACTIVE_INPUTS_SELECTION,
                                                className=VALUE,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="parameters",
                                        id=PV_ENDPOINT_SELECTION_DIV,
                                        children=[
                                            html.Div(
                                                className="parameter-single",
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
                                                        id=PV_ENDPOINT_SELECTION,
                                                        className=VALUE,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="parameter-single",
                                                children=[
                                                    html.Span(
                                                        className="name",
                                                        children="Display barcode",
                                                    ),
                                                    dcc.RadioItems(
                                                        ["On", "Off"],
                                                        "Off",
                                                        id=PV_DISPLAY_BARCODE,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="parameters",
                                        id=PD_ENDPOINT_SELECTION_DIV,
                                        children=[
                                            html.Div(
                                                className="parameter-single",
                                                children=[
                                                    html.Span(
                                                        className="name",
                                                        children="Endpoint",
                                                    ),
                                                    dcc.RadioItems(
                                                        [
                                                            "Line start",
                                                            "Line end",
                                                        ],
                                                        "Line start",
                                                        id=PD_ENDPOINT_SELECTION,
                                                        className=VALUE,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="parameter-single",
                                                children=[
                                                    html.Span(
                                                        className="name",
                                                        children="Display barcode",
                                                    ),
                                                    dcc.RadioItems(
                                                        ["On", "Off"],
                                                        "Off",
                                                        id=PD_DISPLAY_BARCODE,
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="parameters",
                                        id=POINT_SELECTION_DIV,
                                        children=[
                                            html.Div(
                                                className="parameter-double",
                                                children=[
                                                    html.Span(
                                                        className="name",
                                                        children="x/y",
                                                    ),
                                                    dcc.Input(
                                                        className=VALUE,
                                                        id=X_POINT,
                                                        type="number",
                                                        value=ui_state[X_POINT],
                                                        min=0,
                                                        debounce=True,
                                                    ),
                                                    dcc.Input(
                                                        className=VALUE,
                                                        id=Y_POINT,
                                                        type="number",
                                                        value=ui_state[Y_POINT],
                                                        min=0,
                                                        debounce=True,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="parameter-single",
                                                children=[
                                                    html.Button(
                                                        "Choose parameter",
                                                        id=EXPORT_PARAMETERS_BUTTON_DBSCAN,
                                                        className="button",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                        ],
                    )
                ],
            ),
        )

        ccf_extras = (
            html.Div(
                hidden=True,
                children=[
                    html.Details(
                        id=CCF_EXTRAS,
                        children=[
                            html.Summary("Extras"),
                            html.H3("Rank decomposition"),
                            html.Div(
                                className="parameters",
                                children=[
                                    html.Div(
                                        className="parameter-single",
                                        children=[
                                            html.Span(
                                                className="name",
                                                children="Granularity",
                                            ),
                                            dcc.Input(
                                                id=GRANULARITY_RI,
                                                className="small-value",
                                                type="number",
                                                value=ui_state[GRANULARITY_RI],
                                                min=bounds[MIN_GRANULARITY],
                                                max=bounds[MAX_GRANULARITY_RI],
                                                debounce=True,
                                            ),
                                            html.Span(
                                                className="name",
                                                children="Max rank",
                                            ),
                                            dcc.Input(
                                                id=MAX_RI,
                                                type="number",
                                                value=ui_state[MAX_COMPONENTS],
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
                                                children="Min length bars",
                                            ),
                                            dcc.Input(
                                                id=MIN_LENGTH_RI,
                                                type="number",
                                                value=1,
                                                min=1,
                                                className="small-value",
                                                step=1,
                                                debounce=True,
                                            ),
                                            html.Span(
                                                className="name",
                                                children="# cores computation",
                                            ),
                                            dcc.Input(
                                                className="small-value",
                                                id=NUM_JOBS_RI,
                                                type="number",
                                                value=ui_state[NUM_JOBS_RI],
                                                min=1,
                                                step=1,
                                                max=16,
                                                debounce=True,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="parameter-single",
                                        children=[
                                            html.Span(
                                                className="name",
                                                children="Reduced homology",
                                            ),
                                            dcc.RadioItems(
                                                [
                                                    "Yes",
                                                    "No",
                                                ],
                                                "Yes",
                                                id=REDUCED_HOMOLOGY_RI,
                                                className="small-value",
                                            ),
                                            html.Span(
                                                className="name",
                                                children="Display",
                                            ),
                                            dcc.RadioItems(
                                                [
                                                    "Yes",
                                                    "No",
                                                ],
                                                "Yes",
                                                id=DISPLAY_RI,
                                                className="small-value",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="parameter-single",
                                        children=[
                                            html.Span(
                                                className="name",
                                                children="Decompose by",
                                            ),
                                            dcc.RadioItems(
                                                [
                                                    "Rect",
                                                    "Hook",
                                                ],
                                                "Rect",
                                                id=DECOMPOSE_BY_RI,
                                                className="value",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="large-buttons",
                                        children=[
                                            html.Button(
                                                "Compute",
                                                id=COMPUTE_RI_BUTTON,
                                                className="button1",
                                            ),
                                            html.Button(
                                                "Stop computation",
                                                id=STOP_COMPUTE_RI_BUTTON,
                                                className="button2",
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
        )

        pv_inputs = (
            html.Details(
                id=PV_DETAILS,
                children=[
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
                                        value=ui_state[X_START_FIRST_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_START_FIRST_LINE,
                                        type="number",
                                        value=ui_state[Y_START_FIRST_LINE],
                                        min=0,
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
                                        value=ui_state[X_END_FIRST_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_END_FIRST_LINE,
                                        type="number",
                                        value=ui_state[Y_END_FIRST_LINE],
                                        min=0,
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
                                        value=ui_state[X_START_SECOND_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_START_SECOND_LINE,
                                        type="number",
                                        value=ui_state[Y_START_SECOND_LINE],
                                        min=0,
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
                                        value=ui_state[X_END_SECOND_LINE],
                                        min=0,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_END_SECOND_LINE,
                                        type="number",
                                        value=ui_state[Y_END_SECOND_LINE],
                                        min=0,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="parameter-single",
                                children=[
                                    html.Span(
                                        className="name",
                                        children="# lines vineyard",
                                    ),
                                    dcc.Input(
                                        id=GRANULARITY_PV,
                                        className="small-value",
                                        type="number",
                                        value=ui_state[GRANULARITY_PV],
                                        min=bounds[MIN_GRANULARITY_VINEYARD],
                                        max=bounds[MAX_GRANULARITY_VINEYARD],
                                        debounce=True,
                                    ),
                                    html.Span(
                                        className="name",
                                        children="Max number vines to display",
                                    ),
                                    dcc.Input(
                                        id=MAX_VINES,
                                        type="number",
                                        value=ui_state[MAX_VINES],
                                        min=1,
                                        className="small-value",
                                        step=1,
                                        debounce=False,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="parameter-single",
                                children=[
                                    html.Span(
                                        className="name",
                                        children="# cores computation",
                                    ),
                                    dcc.Input(
                                        className="small-value",
                                        id=NUM_JOBS_PV,
                                        type="number",
                                        value=ui_state[NUM_JOBS_PV],
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
                                        id=PROM_VIN_SCALE,
                                        className="small-value",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )

        pv_buttons = (
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
        )

        pv_parameter_selection = (
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
                                id=DISPLAY_PARAMETER_SELECTION_PV,
                                className=VALUE,
                            ),
                        ],
                    ),
                    html.Div(
                        className="parameter-double-button",
                        id=PARAMETER_SELECTION_DIV_PV,
                        children=[
                            html.Span(
                                className="name",
                                children="Line number",
                            ),
                            dcc.Input(
                                className=VALUE,
                                id=LINE,
                                type="number",
                                value=ui_state[LINE],
                                min=1,
                                debounce=False,
                            ),
                            html.Span(
                                className="name",
                                children="Gap number",
                            ),
                            dcc.Input(
                                className=VALUE,
                                id=PV_GAP,
                                type="number",
                                value=ui_state[PV_GAP],
                                min=1,
                                debounce=False,
                            ),
                            html.Button(
                                "Choose parameter",
                                id=EXPORT_PARAMETERS_BUTTON_PV,
                                className="button",
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
        )

        pd_inputs = (
            html.Details(
                id=PD_DETAILS,
                children=[
                    html.Summary("Inputs"),
                    html.Div(
                        className="parameters",
                        children=[
                            html.Div(
                                className="parameter-double",
                                children=[
                                    html.Span(
                                        className="name",
                                        children="Line start x/y",
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=X_START_LINE,
                                        type="number",
                                        value=ui_state[X_START_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_START_LINE,
                                        type="number",
                                        value=ui_state[Y_START_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="parameter-double",
                                children=[
                                    html.Span(
                                        className="name",
                                        children="Line end x/y",
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=X_END_LINE,
                                        type="number",
                                        value=ui_state[X_END_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                    dcc.Input(
                                        className=VALUE,
                                        id=Y_END_LINE,
                                        type="number",
                                        value=ui_state[Y_END_LINE],
                                        min=0,
                                        debounce=True,
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        )

        pd_buttons = (
            html.Div(
                className="parameters",
                children=[
                    html.Div(
                        className="large-buttons",
                        children=[
                            html.Button(
                                "Compute",
                                id=COMPUTE_PD_BUTTON,
                                className="button1",
                            ),
                            html.Button(
                                "Stop computation",
                                id=STOP_COMPUTE_PD_BUTTON,
                                className="button2",
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
        )

        pd_parameter_selection = (
            html.Div(
                id=PD_PLOT_CONTROLS_DIV,
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
                                id=DISPLAY_PARAMETER_SELECTION_PD,
                                className=VALUE,
                            ),
                        ],
                    ),
                    html.Div(
                        className="parameter-double-button",
                        id=PARAMETER_SELECTION_DIV_PD,
                        children=[
                            html.Span(
                                className="name",
                                children="Gap number",
                            ),
                            dcc.Input(
                                className=VALUE,
                                id=PD_GAP,
                                type="number",
                                value=ui_state[PD_GAP],
                                min=1,
                                debounce=False,
                            ),
                            html.Button(
                                "Choose parameter",
                                id=EXPORT_PARAMETERS_BUTTON_PD,
                                className="button",
                                disabled=True,
                            ),
                        ],
                    ),
                ],
            ),
        )

        self._app.layout = html.Div(
            className="root",
            children=[
                # contains the component counting function as a dictionary of lists
                dcc.Store(id=STORED_CCF),
                dcc.Store(id=STORED_X_TICKS_CCF),
                dcc.Store(id=STORED_Y_TICKS_CCF),
                # contains the basic component counting function plot as a plotly figure
                dcc.Store(id=STORED_CCF_DRAWING),
                # contains the signed betti numbers as a list of lists
                dcc.Store(id=STORED_BETTI),
                # contains the signed barcode a list of lists of ...
                dcc.Store(id=STORED_SIGNED_BARCODE_RECTANGLES, data=json.dumps([])),
                dcc.Store(id=STORED_SIGNED_BARCODE_HOOKS, data=json.dumps([])),
                dcc.Store(id=STORED_X_TICKS_RI, data=json.dumps([])),
                dcc.Store(id=STORED_Y_TICKS_RI, data=json.dumps([])),
                # contains the vineyard as a vineyard object
                dcc.Store(id=STORED_PV),
                # contains the basic prominence vineyard plot as a plotly figure
                dcc.Store(id=STORED_PV_DRAWING),
                #
                dcc.Store(id=STORED_PD_BY_PV, data=json.dumps([])),
                dcc.Store(id=STORED_PARAMETERS_AND_PD_BY_PD, data=json.dumps([])),
                #
                dcc.Store(id=STORED_CCF_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                dcc.Store(id=STORED_RI_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                dcc.Store(id=STORED_PV_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                dcc.Store(id=STORED_PD_COMPUTATION_WARNINGS, data=json.dumps(" ")),
                #
                dcc.Store(id=PV_FIXED_PARAMETERS, data=json.dumps([])),
                dcc.Store(id=EXPORTED_PARAMETER, data=json.dumps([])),
                dcc.Store(id=EXPORTED_STATE, data=json.dumps([])),
                #
                #
                html.Div(
                    className="horizontal-grid",
                    children=[
                        html.Div(
                            className="vertical-grid",
                            children=[
                                html.Div(
                                    children=[
                                        html.H2("Component Counting Function"),
                                        html.Div(
                                            className="parameters",
                                            children=ccf_inputs + ccf_buttons,
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
                                html.Div(
                                    className="parameters",
                                    children=ccf_parameter_selection + ccf_extras,
                                ),
                            ],
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    hidden=True,
                                    id=PV_PANEL,
                                    className="vertical-grid",
                                    children=[
                                        html.Div(
                                            children=[
                                                html.H2("Prominence Vineyard"),
                                                html.Div(
                                                    className="parameters",
                                                    children=pv_inputs + pv_buttons,
                                                ),
                                            ]
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
                                            children=pv_parameter_selection,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    hidden=True,
                                    id=PD_PANEL,
                                    className="vertical-grid",
                                    children=[
                                        html.Div(
                                            children=[
                                                html.H2(
                                                    "Persistence Diagram of one-parameter slice"
                                                ),
                                                html.Div(
                                                    className="parameters",
                                                    children=pd_inputs + pd_buttons,
                                                ),
                                            ]
                                        ),
                                        dcc.Graph(
                                            id=PD_PLOT,
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
                                            children=pd_parameter_selection,
                                        ),
                                    ],
                                ),
                            ]
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
            prevent_update_with_none_input=True,
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
                        if prevent_update_with_none_input:
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
                    # self._app.callback(
                    #    dash_outputs,
                    #    dash_inputs,
                    #    prevent_initial_call,
                    #    running=dash_running_outputs,
                    #    cancel=dash_cancel,
                    #    background=True,
                    # )(callback_function)
                else:
                    self._app.callback(
                        dash_outputs,
                        dash_inputs,
                        prevent_initial_call,
                    )(callback_function)

                return function

            return out_function

        @dash_callback(
            [[DISPLAY_PARAMETER_SELECTION_PV, VALUE, IN]],
            [[PARAMETER_SELECTION_DIV_PV, HIDDEN]],
        )
        def toggle_parameter_selection_pv(d):
            if d[DISPLAY_PARAMETER_SELECTION_PV + VALUE] == "On":
                d[PARAMETER_SELECTION_DIV_PV + HIDDEN] = False
            else:
                d[PARAMETER_SELECTION_DIV_PV + HIDDEN] = True
            return d

        @dash_callback(
            [[DISPLAY_PARAMETER_SELECTION_PD, VALUE, IN]],
            [[PARAMETER_SELECTION_DIV_PD, HIDDEN]],
        )
        def toggle_parameter_selection_pd(d):
            if d[DISPLAY_PARAMETER_SELECTION_PD + VALUE] == "On":
                d[PARAMETER_SELECTION_DIV_PD + HIDDEN] = False
            else:
                d[PARAMETER_SELECTION_DIV_PD + HIDDEN] = True
            return d

        @dash_callback(
            [
                [STORED_CCF_COMPUTATION_WARNINGS, DATA, IN],
                [STORED_RI_COMPUTATION_WARNINGS, DATA, IN],
                [STORED_PV_COMPUTATION_WARNINGS, DATA, IN],
                [STORED_PD_COMPUTATION_WARNINGS, DATA, IN],
            ],
            [[LOG, CHILDREN], [LOG_DIV, "open"]],
            True,
        )
        def print_log(d):
            if ctx.triggered_id == STORED_CCF_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_CCF_COMPUTATION_WARNINGS + DATA])
            elif ctx.triggered_id == STORED_PV_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_PV_COMPUTATION_WARNINGS + DATA])
            elif ctx.triggered_id == STORED_RI_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_RI_COMPUTATION_WARNINGS + DATA])
            elif ctx.triggered_id == STORED_PD_COMPUTATION_WARNINGS:
                message = json.loads(d[STORED_PD_COMPUTATION_WARNINGS + DATA])
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
                [INTERACTIVE_INPUTS_SELECTION, VALUE, ST],
                [PV_ENDPOINT_SELECTION, VALUE, ST],
                [PD_ENDPOINT_SELECTION, VALUE, ST],
                [X_START_FIRST_LINE, VALUE, ST],
                [Y_START_FIRST_LINE, VALUE, ST],
                [X_END_FIRST_LINE, VALUE, ST],
                [Y_END_FIRST_LINE, VALUE, ST],
                [X_START_SECOND_LINE, VALUE, ST],
                [Y_START_SECOND_LINE, VALUE, ST],
                [X_END_SECOND_LINE, VALUE, ST],
                [Y_END_SECOND_LINE, VALUE, ST],
                [X_START_LINE, VALUE, ST],
                [Y_START_LINE, VALUE, ST],
                [X_END_LINE, VALUE, ST],
                [Y_END_LINE, VALUE, ST],
                [X_POINT, VALUE, ST],
                [Y_POINT, VALUE, ST],
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
                [X_START_LINE, VALUE],
                [Y_START_LINE, VALUE],
                [X_END_LINE, VALUE],
                [Y_END_LINE, VALUE],
                [X_POINT, VALUE],
                [Y_POINT, VALUE],
            ],
            True,
        )
        def on_ccf_click(d):
            new_x, new_y = (
                d[CCF_PLOT + CLICKDATA]["points"][0]["x"],
                d[CCF_PLOT + CLICKDATA]["points"][0]["y"],
            )
            if d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Family of lines":
                if d[PV_ENDPOINT_SELECTION + VALUE] == "1st line start":
                    d[X_START_FIRST_LINE + VALUE] = new_x
                    d[Y_START_FIRST_LINE + VALUE] = new_y
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "1st line end":
                    d[X_END_FIRST_LINE + VALUE] = new_x
                    d[Y_END_FIRST_LINE + VALUE] = new_y
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "2nd line start":
                    d[X_START_SECOND_LINE + VALUE] = new_x
                    d[Y_START_SECOND_LINE + VALUE] = new_y
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "2nd line end":
                    d[X_END_SECOND_LINE + VALUE] = new_x
                    d[Y_END_SECOND_LINE + VALUE] = new_y
            elif d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Line":
                if d[PD_ENDPOINT_SELECTION + VALUE] == "Line start":
                    d[X_START_LINE + VALUE] = new_x
                    d[Y_START_LINE + VALUE] = new_y
                elif d[PD_ENDPOINT_SELECTION + VALUE] == "Line end":
                    d[X_END_LINE + VALUE] = new_x
                    d[Y_END_LINE + VALUE] = new_y
            elif d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Single clustering":
                d[X_POINT + VALUE] = new_x
                d[Y_POINT + VALUE] = new_y
            return d

        @dash_callback(
            [[INTERACTIVE_INPUTS_SELECTION, VALUE, IN]],
            [
                [PV_ENDPOINT_SELECTION_DIV, HIDDEN],
                [PD_ENDPOINT_SELECTION_DIV, HIDDEN],
                [POINT_SELECTION_DIV, HIDDEN],
                [PD_PANEL, HIDDEN],
                [PV_PANEL, HIDDEN],
            ],
        )
        def toggle_parameter_selection_ccf(d):
            # TODO: abstract the following logic better
            if d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Family of lines":
                d[PV_ENDPOINT_SELECTION_DIV + HIDDEN] = False
                d[PD_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PV_PANEL + HIDDEN] = False
                d[PD_PANEL + HIDDEN] = True
                d[POINT_SELECTION_DIV + HIDDEN] = True
            elif d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Line":
                d[PV_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PD_ENDPOINT_SELECTION_DIV + HIDDEN] = False
                d[PV_PANEL + HIDDEN] = True
                d[PD_PANEL + HIDDEN] = False
                d[POINT_SELECTION_DIV + HIDDEN] = True
            elif d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Off":
                d[PV_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PD_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PV_PANEL + HIDDEN] = True
                d[PD_PANEL + HIDDEN] = True
                d[POINT_SELECTION_DIV + HIDDEN] = True
            elif d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Single clustering":
                d[PV_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PD_ENDPOINT_SELECTION_DIV + HIDDEN] = True
                d[PV_PANEL + HIDDEN] = True
                d[PD_PANEL + HIDDEN] = True
                d[POINT_SELECTION_DIV + HIDDEN] = False
            return d

        @dash_callback(
            [
                [STORED_CCF, DATA, IN],
                [STORED_X_TICKS_CCF, DATA, ST],
                [STORED_Y_TICKS_CCF, DATA, ST],
                [MAX_COMPONENTS, VALUE, IN],
            ],
            [[STORED_CCF_DRAWING, DATA]],
            False,
        )
        def draw_ccf(d):
            ccf = np.array(json.loads(d[STORED_CCF + DATA]))
            x_ticks = np.array(json.loads(d[STORED_X_TICKS_CCF + DATA]))
            y_ticks = np.array(json.loads(d[STORED_Y_TICKS_CCF + DATA]))
            delta_x_ccf = (x_ticks[1] - x_ticks[0]) / 2
            delta_y_ccf = (y_ticks[1] - y_ticks[0]) / 2

            def fn_x(x):
                return x + delta_x_ccf

            def fn_x_inverse(x):
                return x - delta_x_ccf

            def fn_y(y):
                return y + delta_y_ccf

            def fn_y_inverse(y):
                return y - delta_y_ccf

            max_components = d[MAX_COMPONENTS + VALUE]

            fig = go.Figure(
                layout=go.Layout(
                    xaxis_title="Distance scale",
                    yaxis_title="Density threshold",
                    xaxis={"fixedrange": True},
                    yaxis={"fixedrange": True},
                ),
            )

            # TODO: there is a small inconsistency (~ delta_x and delta_y) between what
            # is displayed on the heatmap and the numbers that one sees when hovering
            # we thus turn off the coordinates on hovering, below
            fig.add_trace(
                go.Heatmap(
                    transpose=True,
                    z=ccf,
                    x=fn_x(x_ticks),
                    y=fn_y(y_ticks),
                    # hovertemplate="<b># comp.: %{z:d}</b><br>x: %{x:.3e} <br>y: %{y:.3e} ",
                    hovertemplate="<b># comp.: %{z:d}</b> ",
                    zmin=0,
                    zmax=max_components,
                    showscale=False,
                    name="",
                )
            )

            fig.update_xaxes(tickson="boundaries")
            fig.update_yaxes(tickson="boundaries")

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
                [INTERACTIVE_INPUTS_SELECTION, VALUE, IN],
                [X_START_FIRST_LINE, VALUE, IN],
                [Y_START_FIRST_LINE, VALUE, IN],
                [X_END_FIRST_LINE, VALUE, IN],
                [Y_END_FIRST_LINE, VALUE, IN],
                [X_START_SECOND_LINE, VALUE, IN],
                [Y_START_SECOND_LINE, VALUE, IN],
                [X_END_SECOND_LINE, VALUE, IN],
                [Y_END_SECOND_LINE, VALUE, IN],
                [PV_ENDPOINT_SELECTION, VALUE, IN],
                [PD_ENDPOINT_SELECTION, VALUE, IN],
                [X_START_LINE, VALUE, IN],
                [Y_START_LINE, VALUE, IN],
                [X_END_LINE, VALUE, IN],
                [Y_END_LINE, VALUE, IN],
                [X_POINT, VALUE, IN],
                [Y_POINT, VALUE, IN],
                [DISPLAY_PARAMETER_SELECTION_PV, VALUE, IN],
                [DISPLAY_PARAMETER_SELECTION_PD, VALUE, IN],
                [PV_DISPLAY_BARCODE, VALUE, IN],
                [PD_DISPLAY_BARCODE, VALUE, IN],
                [INTERACTIVE_INPUTS_SELECTION, VALUE, IN],
                [PV_FIXED_PARAMETERS, DATA, IN],
                [STORED_PD_BY_PV, DATA, ST],
                [STORED_PARAMETERS_AND_PD_BY_PD, DATA, IN],
                [PV_GAP, VALUE, ST],
                [PD_GAP, VALUE, IN],
                [DISPLAY_RI, VALUE, IN],
                [Y_COVARIANT, VALUE, IN],
                [STORED_BETTI, DATA, ST],
                [STORED_X_TICKS_CCF, DATA, ST],
                [STORED_Y_TICKS_CCF, DATA, ST],
                [MAX_COMPONENTS, VALUE, IN],
                [MAX_RI, VALUE, IN],
                [STORED_SIGNED_BARCODE_RECTANGLES, DATA, IN],
                [STORED_SIGNED_BARCODE_HOOKS, DATA, IN],
                [STORED_X_TICKS_RI, DATA, ST],
                [STORED_Y_TICKS_RI, DATA, ST],
                [MIN_LENGTH_RI, VALUE, IN],
                [DECOMPOSE_BY_RI, VALUE, IN],
            ],
            [[CCF_PLOT, FIGURE]],
            False,
        )
        def draw_ccf_extras(d):
            fig = plotly.io.from_json(d[STORED_CCF_DRAWING + DATA])

            x_ticks_ccf = json.loads(d[STORED_X_TICKS_CCF + DATA])
            y_ticks_ccf = json.loads(d[STORED_Y_TICKS_CCF + DATA])
            delta_x_ccf = (x_ticks_ccf[1] - x_ticks_ccf[0]) / 2
            delta_y_ccf = (y_ticks_ccf[1] - y_ticks_ccf[0]) / 2
            x_ticks_ccf.append(np.array(x_ticks_ccf[-1]) + 2 * delta_x_ccf)
            y_ticks_ccf.append(np.array(y_ticks_ccf[-1]) + 2 * delta_y_ccf)

            max_components = d[MAX_COMPONENTS + VALUE]

            def _rgba(color, opacity):
                if color == "red":
                    red, green, blue = "255", "0", "0"
                if color == "green":
                    red, green, blue = "0", "255", "0"
                if color == "blue":
                    red, green, blue = "0", "0", "255"
                return (
                    "rgba(" + red + "," + green + "," + blue + "," + str(opacity) + ")"
                )

            def _draw_bar(xs, ys, color, width=6, endpoints=False, size=5):
                mode = "markers+lines" if endpoints else "lines"
                marker_styles = ["diamond", "diamond"]
                return go.Scatter(
                    x=xs,
                    y=ys,
                    marker=dict(color=color, size=size),
                    marker_symbol=marker_styles,
                    hoverinfo="skip",
                    showlegend=False,
                    mode=mode,
                    line=dict(width=width),
                )

            # draw signed barcode
            if d[DISPLAY_RI + VALUE] == "Yes":
                using_rectangles = (
                    True if d[DECOMPOSE_BY_RI + VALUE] == "Rect" else False
                )
                if using_rectangles:
                    sb = np.array(
                        json.loads(d[STORED_SIGNED_BARCODE_RECTANGLES + DATA])
                    )
                else:
                    sb = np.array(json.loads(d[STORED_SIGNED_BARCODE_HOOKS + DATA]))
                if len(sb) != 0:
                    max_components = d[MAX_RI + VALUE]
                    x_ticks = json.loads(d[STORED_X_TICKS_RI + DATA])
                    y_ticks = json.loads(d[STORED_Y_TICKS_RI + DATA])
                    delta_x = (x_ticks[1] - x_ticks[0]) / 2
                    delta_y = (y_ticks[1] - y_ticks[0]) / 2
                    x_ticks.append(np.array(x_ticks[-1]) + 2 * delta_x)
                    y_ticks.append(np.array(y_ticks[-1]) + 2 * delta_y)

                    lx = len(x_ticks)
                    ly = len(y_ticks)
                    traces = []
                    if using_rectangles:
                        total_width = min(lx, ly)
                    else:
                        total_width = max(lx, ly)
                    for i, j, i_, j_, mult in sb:
                        if mult != 0:
                            min_size = 3
                            if using_rectangles:
                                i_ += 1
                                j_ += 1
                                length = min((i_ - i), (j_ - j))
                                width = 10 * (length / total_width)
                                size = min_size + 5 * (length / total_width)
                            else:
                                length = max((i_ - i), (j_ - j))
                                width = 10 * (length / total_width)
                                size = min_size + 5 * (length / total_width)
                            min_opacity = 0.3
                            opacity = min_opacity + (
                                np.minimum(np.abs(mult), max_components)
                                / max_components
                            ) * (1 - min_opacity)
                            x_coords = np.array(
                                # [x_ticks[i] - delta_x, x_ticks[i_] - delta_x]
                                [x_ticks[i], x_ticks[i_]]
                            )
                            y_coords = np.array(
                                # [y_ticks[j] - delta_y, y_ticks[j_] - delta_y]
                                [y_ticks[j], y_ticks[j_]]
                            )
                            if length >= d[MIN_LENGTH_RI + VALUE]:
                                if mult < 0:
                                    color = _rgba("red", opacity)
                                    traces.append(
                                        _draw_bar(
                                            x_coords,
                                            y_coords,
                                            color,
                                            width,
                                            endpoints=True,
                                            size=size,
                                        )
                                    )
                                if mult > 0:
                                    color = _rgba("blue", opacity)
                                    traces.append(
                                        _draw_bar(
                                            x_coords,
                                            y_coords,
                                            color,
                                            width,
                                            endpoints=True,
                                            size=size,
                                        )
                                    )
                    fig.add_traces(traces)

            # draw Betti numbers
            if False:
                bn = np.array(json.loads(d[STORED_BETTI + DATA]))
                xs = x_ticks
                ys = y_ticks

                positive_bn = np.array(
                    [
                        # [xs[i] - delta_x, ys[j] - delta_y, bn[i, j]]
                        [xs[i], ys[j], bn[i, j]]
                        for i in range(len(xs))
                        for j in range(len(ys))
                        if bn[i, j] > 0
                    ]
                )
                negative_bn = np.array(
                    [
                        [xs[i], ys[j], -bn[i, j]]
                        for i in range(len(xs))
                        for j in range(len(ys))
                        if bn[i, j] < 0
                    ]
                )
                marker_size = 5
                min_opacity = 0.3
                positive_opacity = min_opacity + (
                    (np.minimum(positive_bn[:, 2], max_components)) / max_components
                ) * (1 - min_opacity)
                negative_opacity = min_opacity + (
                    (np.minimum(negative_bn[:, 2], max_components)) / max_components
                ) * (1 - min_opacity)

                # draw positive
                fig.add_trace(
                    go.Scatter(
                        x=positive_bn[:, 0],
                        y=positive_bn[:, 1],
                        name="",
                        marker=dict(
                            color="blue", size=marker_size, opacity=positive_opacity
                        ),
                        hoverinfo="skip",
                        mode="markers",
                    )
                )
                # draw negative
                fig.add_trace(
                    go.Scatter(
                        x=negative_bn[:, 0],
                        y=negative_bn[:, 1],
                        name="",
                        marker=dict(
                            color="red", size=marker_size, opacity=negative_opacity
                        ),
                        # color="green",
                        hoverinfo="skip",
                        mode="markers",
                    )
                )

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

            def generate_point(x, y, color="mediumslateblue"):
                return go.Scatter(
                    x=[x],
                    y=[y],
                    marker=dict(size=15, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                    mode="markers",
                )

            # draw single point
            if d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Single clustering":
                fig.add_trace(generate_point(d[X_POINT + VALUE], d[Y_POINT + VALUE]))

            # draw family of lines
            if d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Family of lines":
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

                if d[PV_ENDPOINT_SELECTION + VALUE] == "1st line start":
                    first_line_endpoints = 0
                    second_line_endpoints = None
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "1st line end":
                    first_line_endpoints = 1
                    second_line_endpoints = None
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "2nd line start":
                    first_line_endpoints = None
                    second_line_endpoints = 0
                elif d[PV_ENDPOINT_SELECTION + VALUE] == "2nd line end":
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

            # draw single line
            if d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Line":
                st_x = d[X_START_LINE + VALUE]
                st_y = d[Y_START_LINE + VALUE]
                end_x = d[X_END_LINE + VALUE]
                end_y = d[Y_END_LINE + VALUE]

                if d[PD_ENDPOINT_SELECTION + VALUE] == "Line start":
                    line_endpoints = 0
                elif d[PD_ENDPOINT_SELECTION + VALUE] == "Line end":
                    line_endpoints = 1

                fig.add_trace(
                    generate_line(
                        [st_x, end_x],
                        [st_y, end_y],
                        "selected",
                        color="blue",
                        different_marker=line_endpoints,
                    )
                )

                saved = json.loads(d[STORED_PARAMETERS_AND_PD_BY_PD + DATA])
                if len(saved) != 0:
                    saved_params, saved_pd = saved

                    saved_st_x, saved_st_y = saved_params[0]
                    saved_end_x, saved_end_y = saved_params[1]

                    if (
                        np.allclose(
                            np.array([st_x, st_y, end_x, end_y]),
                            np.array(
                                [saved_st_x, saved_st_y, saved_end_x, saved_end_y]
                            ),
                        )
                        and len(saved_pd) != 0
                        and d[PD_DISPLAY_BARCODE + VALUE] == "On"
                    ):
                        pd = np.array(saved_pd)
                        pd = pd - st_x
                        st = np.array([st_x, st_y])
                        end = np.array([end_x, end_y])
                        A = end - st

                        # ideally we would get the actual ratio of the rendered picture
                        # we are using an estimate given by the "usual" way in which
                        # persistable's GUI is rendered
                        ratio = np.array([3, 2])
                        ratio = (ratio / np.linalg.norm(ratio)) * np.linalg.norm(
                            np.array([1, 1])
                        )
                        alpha = [
                            (d[MAX_DIST_SCALE + VALUE] - d[MIN_DIST_SCALE + VALUE]),
                            (
                                d[MAX_DENSITY_THRESHOLD + VALUE]
                                - d[MIN_DENSITY_THRESHOLD + VALUE]
                            ),
                        ]
                        alpha = np.array(alpha) / ratio

                        A = A / alpha
                        B = np.array([-A[1], A[0]])

                        shift = 50
                        B = B / np.linalg.norm(B)
                        tau = B * alpha / shift

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
                                if i < d[PD_GAP + VALUE]
                                or d[DISPLAY_PARAMETER_SELECTION_PD + VALUE] == "Off"
                                else "rgba(34, 139, 34, 0.3)"
                            )
                            fig.add_trace(
                                _draw_bar(
                                    [r_st[0], r_end[0]], [r_st[1], r_end[1]], color
                                )
                            )

            # draw barcode in case of family of lines
            params = json.loads(d[PV_FIXED_PARAMETERS + DATA])
            if (
                len(params) != 0
                and d[INTERACTIVE_INPUTS_SELECTION + VALUE] == "Family of lines"
                and d[DISPLAY_PARAMETER_SELECTION_PV + VALUE] == "On"
            ):
                pd = json.loads(d[STORED_PD_BY_PV + DATA])

                st_x = params["start"][0]
                st_y = params["start"][1]
                end_x = params["end"][0]
                end_y = params["end"][1]

                fig.add_trace(
                    generate_line(
                        [st_x, end_x],
                        [st_y, end_y],
                        "selected",
                        color="blue",
                    )
                )

                if len(pd) != 0 and d[PV_DISPLAY_BARCODE + VALUE] == "On":
                    pd = np.array(pd)
                    pd = pd - st_x
                    st = np.array([st_x, st_y])
                    end = np.array([end_x, end_y])
                    A = end - st

                    # ideally we would get the actual ratio of the rendered picture
                    # we are using an estimate given by the "usual" way in which
                    # persistable's GUI is rendered
                    ratio = np.array([3, 2])
                    ratio = (ratio / np.linalg.norm(ratio)) * np.linalg.norm(
                        np.array([1, 1])
                    )
                    alpha = [
                        (d[MAX_DIST_SCALE + VALUE] - d[MIN_DIST_SCALE + VALUE]),
                        (
                            d[MAX_DENSITY_THRESHOLD + VALUE]
                            - d[MIN_DENSITY_THRESHOLD + VALUE]
                        ),
                    ]
                    alpha = np.array(alpha) / ratio

                    A = A / alpha
                    B = np.array([-A[1], A[0]])

                    shift = 50
                    B = B / np.linalg.norm(B)
                    tau = B * alpha / shift

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
                            if i < d[PV_GAP + VALUE]
                            else "rgba(34, 139, 34, 0.3)"
                        )
                        fig.add_trace(
                            _draw_bar([r_st[0], r_end[0]], [r_st[1], r_end[1]], color)
                        )

            if ctx.triggered_id in [
                MIN_DIST_SCALE,
                MAX_DIST_SCALE,
                MIN_DENSITY_THRESHOLD,
                MAX_DENSITY_THRESHOLD,
            ]:
                xbounds = [d[MIN_DIST_SCALE + VALUE], d[MAX_DIST_SCALE + VALUE]]
                ybounds = [
                    d[MIN_DENSITY_THRESHOLD + VALUE],
                    d[MAX_DENSITY_THRESHOLD + VALUE],
                ]
            else:
                xbounds = [x_ticks_ccf[0], x_ticks_ccf[-1]]
                ybounds = [y_ticks_ccf[-1], y_ticks_ccf[0]]

            if d[Y_COVARIANT + VALUE] == "Cov":
                ybounds = ybounds[::-1]
            fig.update_layout(
                xaxis=dict(range=xbounds),
                yaxis=dict(range=ybounds),
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
                [GRANULARITY, VALUE, ST],
                [NUM_JOBS_CCF, VALUE, ST],
            ],
            [
                [STORED_CCF, DATA],
                [STORED_X_TICKS_CCF, DATA],
                [STORED_Y_TICKS_CCF, DATA],
                [STORED_CCF_COMPUTATION_WARNINGS, DATA],
                [STORED_BETTI, DATA],
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
            granularity = d[GRANULARITY + VALUE]
            num_jobs = int(d[NUM_JOBS_CCF + VALUE])

            if debug:
                print(
                    "Compute ccf in background started with inputs ",
                    d[MIN_DIST_SCALE + VALUE],
                    d[MAX_DIST_SCALE + VALUE],
                    d[MAX_DENSITY_THRESHOLD + VALUE],
                    d[MIN_DENSITY_THRESHOLD + VALUE],
                    granularity,
                    num_jobs,
                )

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    ss, ks, hf, bn = persistable._hilbert_function(
                        d[MIN_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MAX_DENSITY_THRESHOLD + VALUE],
                        d[MIN_DENSITY_THRESHOLD + VALUE],
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

            d[STORED_CCF + DATA] = json.dumps(hf.tolist())
            d[STORED_X_TICKS_CCF + DATA] = json.dumps(ss.tolist())
            d[STORED_Y_TICKS_CCF + DATA] = json.dumps(ks.tolist())

            d[STORED_BETTI + DATA] = json.dumps(bn.tolist())

            d[STORED_CCF_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[CCF_PLOT_CONTROLS_DIV + HIDDEN] = False

            if debug:
                print("Compute ccf in background finished.")

            return d

        @dash_callback(
            [
                [COMPUTE_RI_BUTTON, N_CLICKS, IN],
                [MIN_DENSITY_THRESHOLD, VALUE, ST],
                [MAX_DENSITY_THRESHOLD, VALUE, ST],
                [MIN_DIST_SCALE, VALUE, ST],
                [MAX_DIST_SCALE, VALUE, ST],
                [GRANULARITY_RI, VALUE, ST],
                [NUM_JOBS_RI, VALUE, ST],
                [REDUCED_HOMOLOGY_RI, VALUE, ST],
            ],
            [
                [STORED_X_TICKS_RI, DATA],
                [STORED_Y_TICKS_RI, DATA],
                [STORED_RI_COMPUTATION_WARNINGS, DATA],
                [STORED_SIGNED_BARCODE_RECTANGLES, DATA],
                [STORED_SIGNED_BARCODE_HOOKS, DATA],
            ],
            prevent_initial_call=True,
            background=True,
            running=[
                [COMPUTE_RI_BUTTON, DISABLED, True, False],
                [STOP_COMPUTE_RI_BUTTON, DISABLED, False, True],
            ],
            cancel=[[STOP_COMPUTE_RI_BUTTON, N_CLICKS]],
        )
        def compute_rank_invariant(d):
            if debug:
                print("Compute rank invariant in background started.")

            granularity = d[GRANULARITY_RI + VALUE]
            num_jobs = int(d[NUM_JOBS_RI + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    reduced = True if d[REDUCED_HOMOLOGY_RI + VALUE] == "Yes" else False
                    ss, ks, ri, sbr, sbh = persistable._rank_invariant(
                        d[MIN_DIST_SCALE + VALUE],
                        d[MAX_DIST_SCALE + VALUE],
                        d[MAX_DENSITY_THRESHOLD + VALUE],
                        d[MIN_DENSITY_THRESHOLD + VALUE],
                        granularity,
                        reduced=reduced,
                        n_jobs=num_jobs,
                    )

                    lx = sbr.shape[0]
                    ly = sbr.shape[1]
                    sbr = [
                        [i, j, i_, j_, int(sbr[i, j, i_, j_])]
                        for i in range(lx)
                        for j in range(ly)
                        for i_ in range(i, lx)
                        for j_ in range(j, ly)
                        if sbr[i, j, i_, j_] != 0
                    ]
                    sbh = [
                        [i, j, i_, j_, int(sbh[i, j, i_, j_])]
                        for i in range(lx)
                        for j in range(ly)
                        for i_ in range(i, lx)
                        for j_ in range(j, ly)
                        if sbh[i, j, i_, j_] != 0
                    ]
                except ValueError:
                    out += traceback.format_exc()
                    d[STORED_RI_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
                    d[STORED_SIGNED_BARCODE_RECTANGLES + DATA] = None
                    d[STORED_SIGNED_BARCODE_HOOKS + DATA] = None
                    return d

            for a in w:
                out += warnings.formatwarning(
                    a.message, a.category, a.filename, a.lineno
                )

            d[STORED_SIGNED_BARCODE_RECTANGLES + DATA] = json.dumps(sbr)
            d[STORED_SIGNED_BARCODE_HOOKS + DATA] = json.dumps(sbh)

            d[STORED_X_TICKS_RI + DATA] = json.dumps(ss.tolist())
            d[STORED_Y_TICKS_RI + DATA] = json.dumps(ks.tolist())

            d[STORED_RI_COMPUTATION_WARNINGS + DATA] = json.dumps(out)

            if debug:
                print("Compute rank invariant in background finished.")

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
                [GRANULARITY_PV, VALUE, ST],
                [NUM_JOBS_PV, VALUE, ST],
            ],
            [
                [STORED_PV, DATA],
                [STORED_PV_COMPUTATION_WARNINGS, DATA],
                [LINE, "max"],
                [LINE, VALUE],
                [EXPORT_PARAMETERS_BUTTON_PV, DISABLED],
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

            granularity = d[GRANULARITY_PV + VALUE]
            num_jobs = int(d[NUM_JOBS_PV + VALUE])

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    pv = persistable._linear_vineyard(
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
                    d[LINE + "max"] = granularity
                    d[LINE + VALUE] = granularity // 2
                    d[EXPORT_PARAMETERS_BUTTON_PV + DISABLED] = True
                    d[PV_PLOT_CONTROLS_DIV + HIDDEN] = True
                    return d

            for a in w:
                out += warnings.formatwarning(
                    a.message, a.category, a.filename, a.lineno
                )

            d[STORED_PV + DATA] = json.dumps(pv.__dict__)
            d[STORED_PV_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[LINE + "max"] = granularity
            d[LINE + VALUE] = granularity // 2
            d[EXPORT_PARAMETERS_BUTTON_PV + DISABLED] = False

            d[PV_PLOT_CONTROLS_DIV + HIDDEN] = False

            if debug:
                print("Compute pv in background finished.")

            return d

        @dash_callback(
            [
                [COMPUTE_PD_BUTTON, N_CLICKS, IN],
                [X_START_LINE, VALUE, ST],
                [Y_START_LINE, VALUE, ST],
                [X_END_LINE, VALUE, ST],
                [Y_END_LINE, VALUE, ST],
            ],
            [
                [STORED_PARAMETERS_AND_PD_BY_PD, DATA],
                [STORED_PD_COMPUTATION_WARNINGS, DATA],
                [PD_PLOT_CONTROLS_DIV, HIDDEN],
                [EXPORT_PARAMETERS_BUTTON_PD, DISABLED],
            ],
            prevent_initial_call=True,
            background=True,
            running=[
                [COMPUTE_PD_BUTTON, DISABLED, True, False],
                [STOP_COMPUTE_PD_BUTTON, DISABLED, False, True],
            ],
            cancel=[[STOP_COMPUTE_PD_BUTTON, N_CLICKS]],
        )
        def compute_pd(d):
            if debug:
                print("Compute pd in background started.")

            out = ""
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    pv = persistable._linear_vineyard(
                        [
                            [
                                d[X_START_LINE + VALUE],
                                d[Y_START_LINE + VALUE],
                            ],
                            [d[X_END_LINE + VALUE], d[Y_END_LINE + VALUE]],
                        ],
                        [
                            [
                                d[X_START_LINE + VALUE],
                                d[Y_START_LINE + VALUE],
                            ],
                            [
                                d[X_END_LINE + VALUE],
                                d[Y_END_LINE + VALUE],
                            ],
                        ],
                        n_parameters=1,
                        n_jobs=1,
                    )
                except ValueError:
                    out += traceback.format_exc()
                    d[STORED_PARAMETERS_AND_PD_BY_PD + DATA] = json.dumps([])
                    d[STORED_PD_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
                    d[EXPORT_PARAMETERS_BUTTON_PD + DISABLED] = True
                    d[PD_PLOT_CONTROLS_DIV + HIDDEN] = True
                    return d

            for a in w:
                out += warnings.formatwarning(
                    a.message, a.category, a.filename, a.lineno
                )

            d[STORED_PARAMETERS_AND_PD_BY_PD + DATA] = json.dumps(
                (pv._parameters[0], pv._persistence_diagrams[0])
            )
            d[STORED_PD_COMPUTATION_WARNINGS + DATA] = json.dumps(out)
            d[EXPORT_PARAMETERS_BUTTON_PD + DISABLED] = False
            d[PD_PLOT_CONTROLS_DIV + HIDDEN] = False

            if debug:
                print("Compute pd in background finished.")

            return d

        @dash_callback(
            [
                [STORED_PARAMETERS_AND_PD_BY_PD, DATA, IN],
                [PD_PLOT, FIGURE, ST],
                [PD_GAP, VALUE, IN],
                [DISPLAY_PARAMETER_SELECTION_PD, VALUE, IN],
            ],
            [[PD_PLOT, FIGURE]],
            False,
        )
        def draw_pd(d):
            saved = json.loads(d[STORED_PARAMETERS_AND_PD_BY_PD + DATA])
            if len(saved) != 0:
                saved_params, saved_pd = saved

                if len(saved_pd) != 0:
                    saved_pd = np.array(saved_pd)

                    offset = min(saved_pd[:, 0])
                    start = 0
                    end = max(saved_pd[:, 1]) - offset

                    bit_more = 1 / 30
                    delta_x = (end - start) * bit_more
                    delta_y = (end - start) * bit_more * 3
                    x_range = [start - delta_x, end + delta_x]
                    y_range = [start - delta_y, end + delta_y]

                    fig = go.Figure(
                        layout=go.Layout(
                            xaxis=go.layout.XAxis(
                                title="Birth",
                                showticklabels=False,
                                fixedrange=True,
                                range=x_range,
                            ),
                            yaxis=go.layout.YAxis(
                                title="Death",
                                showticklabels=False,
                                fixedrange=True,
                                range=y_range,
                            ),
                        ),
                    )
                    # square background
                    fig.add_trace(
                        go.Scatter(
                            x=[
                                start - delta_x,
                                end + delta_x,
                                end + delta_x,
                                start - delta_x,
                            ],
                            y=[
                                start - delta_y,
                                start - delta_y,
                                end + delta_y,
                                end + delta_y,
                            ],
                            fill="toself",
                            fillcolor="white",
                            hoverinfo="skip",
                            mode="none",
                        )
                    )
                    # background below diagonal
                    fig.add_trace(
                        go.Scatter(
                            x=[start, end, end],
                            y=[start, start, end],
                            fill="toself",
                            fillcolor="grey",
                            hoverinfo="skip",
                            mode="none",
                        )
                    )
                    # background above diagonal
                    fig.add_trace(
                        go.Scatter(
                            x=[start, end, start],
                            y=[start, end, end],
                            fill="toself",
                            fillcolor="white",
                            hoverinfo="skip",
                            mode="none",
                        )
                    )
                    # enclosing box
                    fig.add_trace(
                        go.Scatter(
                            x=[start, end, end, start, start],
                            y=[start, start, end, end, start],
                            hoverinfo="skip",
                            fill="none",
                            mode="lines",
                            marker=dict(color="black"),
                            line=dict(width=1),
                        )
                    )

                    # draw gap
                    if d[DISPLAY_PARAMETER_SELECTION_PD + VALUE] == "On":
                        gap = d[PD_GAP + VALUE] - 1
                        prominences = saved_pd[:, 1] - saved_pd[:, 0]
                        prominences = np.sort(prominences)[::-1]
                        # only plot gap if gap makes sense
                        if gap < len(prominences):
                            gap_end = prominences[gap]
                            if gap + 1 >= len(prominences):
                                gap_start = 0
                            else:
                                gap_start = prominences[gap + 1]

                            fig.add_trace(
                                go.Scatter(
                                    x=[start, end - gap_start, end - gap_end, start],
                                    y=[gap_start, end, end, gap_end],
                                    fill="toself",
                                    fillcolor="rgba(255,0,0,0.2)",
                                    hoverinfo="skip",
                                    mode="none",
                                )
                            )

                    # draw points
                    prominences = saved_pd[:, 1] - saved_pd[:, 0]
                    saved_pd = saved_pd[np.argsort(prominences)[::-1]]
                    marker_size = 10
                    fig.add_trace(
                        go.Scatter(
                            x=saved_pd[:, 0] - offset,
                            y=saved_pd[:, 1] - offset,
                            mode="markers",
                            hoverinfo="text",
                            marker=dict(size=marker_size, color="green"),
                            text=["# " + str(i + 1) for i in range(len(saved_pd))],
                            showlegend=False,
                        )
                    )

                    fig.update_layout(showlegend=False)
                    fig.update_layout(autosize=True)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

                    d[PD_PLOT + FIGURE] = fig

            return d

        @dash_callback(
            [
                [STORED_PV, DATA, IN],
                [MAX_VINES, VALUE, IN],
                [PROM_VIN_SCALE, VALUE, IN],
                [DISPLAY_PARAMETER_SELECTION_PV, VALUE, IN],
                [LINE, VALUE, IN],
                [PV_GAP, VALUE, IN],
            ],
            [[STORED_PV_DRAWING, DATA]],
            False,
        )
        def draw_pv(d):
            firstn = d[MAX_VINES + VALUE]

            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = Vineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_persistence_diagrams"],
            )

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

            _vineyard_values = []
            colors = sample_colorscale(
                "viridis", list(np.linspace(0, 1, num_vines))[::-1]
            )
            if num_vines > 0:
                for i in range(num_vines - 1, -1, -1):
                    vine = vines[i][1]
                    for vine_part, _ in vineyard._vine_parts(vine):
                        vine_part_arr = np.array(vine_part)
                        vine_part_arr = vine_part_arr[vine_part_arr != 0]
                        _vineyard_values.extend(list(vine_part_arr))
                    till = "tozeroy" if i == num_vines - 1 else "tonexty"
                    color = colors[i]
                    if (
                        d[DISPLAY_PARAMETER_SELECTION_PV + VALUE] == "On"
                        and i + 1 == d[PV_GAP + VALUE]
                    ):
                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=vine,
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
                                y=vine,
                                fill=till,
                                # hoveron="fills",
                                text="vine " + str(i + 1),
                                hoverinfo="text",
                                line_color=color,
                            )
                        )
            values = np.array(_vineyard_values)

            if d[DISPLAY_PARAMETER_SELECTION_PV + VALUE] == "On":
                fig.add_vline(x=d[LINE + VALUE], line_color="grey")

            if len(values) > 0:
                if d[PROM_VIN_SCALE + VALUE] == "Log":
                    fig.update_layout(yaxis_type="log")
                    fig.update_layout(
                        yaxis_range=[
                            np.log10(np.quantile(values[values > 0], 0.05)),
                            np.log10(max(values)),
                        ]
                    )
                else:
                    fig.update_layout(yaxis_range=[min(values), max(values)])

            fig.update_layout(showlegend=False)
            fig.update_layout(autosize=True)
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

            d[STORED_PV_DRAWING + DATA] = plotly.io.to_json(fig)

            return d

        @dash_callback(
            [
                [PV_GAP, VALUE, IN],
                [LINE, VALUE, IN],
                [STORED_PV, DATA, IN],
            ],
            [
                [PV_FIXED_PARAMETERS, DATA],
                [STORED_PD_BY_PV, DATA],
            ],
            True,
        )
        def fix_parameters(d):
            vineyard_as_dict = json.loads(d[STORED_PV + DATA])
            vineyard = Vineyard(
                vineyard_as_dict["_parameters"],
                vineyard_as_dict["_persistence_diagrams"],
            )
            line = vineyard._parameters[d[LINE + VALUE] - 1]
            params = {
                "n_clusters": d[PV_GAP + VALUE],
                "start": line[0],
                "end": line[1],
            }
            d[PV_FIXED_PARAMETERS + DATA] = json.dumps(params)

            pd = vineyard._persistence_diagrams[d[LINE + VALUE] - 1]

            d[STORED_PD_BY_PV + DATA] = json.dumps(pd)

            return d

        @dash_callback(
            [
                [EXPORT_PARAMETERS_BUTTON_PV, N_CLICKS, IN],
                [EXPORT_PARAMETERS_BUTTON_PD, N_CLICKS, IN],
                [EXPORT_PARAMETERS_BUTTON_DBSCAN, N_CLICKS, IN],
                [PV_FIXED_PARAMETERS, DATA, ST],
                [PD_GAP, VALUE, ST],
                [X_START_LINE, VALUE, ST],
                [Y_START_LINE, VALUE, ST],
                [X_END_LINE, VALUE, ST],
                [Y_END_LINE, VALUE, ST],
                [X_POINT, VALUE, ST],
                [Y_POINT, VALUE, ST],
            ],
            [[EXPORTED_PARAMETER, DATA]],
            True,
            prevent_update_with_none_input=False,
        )
        def export_parameters(d):
            if ctx.triggered_id == EXPORT_PARAMETERS_BUTTON_PV:
                params = json.loads(d[PV_FIXED_PARAMETERS + DATA])
            elif ctx.triggered_id == EXPORT_PARAMETERS_BUTTON_PD:
                params = {
                    "n_clusters": d[PD_GAP + VALUE],
                    "start": [d[X_START_LINE + VALUE], d[Y_START_LINE + VALUE]],
                    "end": [d[X_END_LINE + VALUE], d[Y_END_LINE + VALUE]],
                }
            elif ctx.triggered_id == EXPORT_PARAMETERS_BUTTON_DBSCAN:
                params = {"point": (d[X_POINT + VALUE], d[Y_POINT + VALUE])}
            else:
                raise Exception(
                    "export_parameters was triggered by unknown id: "
                    + str(ctx.triggered_id)
                )
            self._parameters_sem.acquire()
            self._parameters = params
            self._parameters_sem.release()
            d[EXPORTED_PARAMETER + DATA] = json.dumps(self._parameters)
            return d

        @dash_callback(
            [
                [GRANULARITY, VALUE, IN],
                [GRANULARITY_RI, VALUE, IN],
                [GRANULARITY_PV, VALUE, IN],
                [MIN_DENSITY_THRESHOLD, VALUE, IN],
                [MAX_DENSITY_THRESHOLD, VALUE, IN],
                [MIN_DIST_SCALE, VALUE, IN],
                [MAX_DIST_SCALE, VALUE, IN],
                [NUM_JOBS_CCF, VALUE, IN],
                [NUM_JOBS_PV, VALUE, IN],
                [NUM_JOBS_RI, VALUE, IN],
                [Y_COVARIANT, VALUE, IN],
                [LINE, VALUE, IN],
                [PV_GAP, VALUE, IN],
                [PD_GAP, VALUE, IN],
                [MAX_COMPONENTS, VALUE, IN],
                [MAX_VINES, VALUE, IN],
                [X_START_FIRST_LINE, VALUE, IN],
                [Y_START_FIRST_LINE, VALUE, IN],
                [X_END_FIRST_LINE, VALUE, IN],
                [Y_END_FIRST_LINE, VALUE, IN],
                [X_START_SECOND_LINE, VALUE, IN],
                [Y_START_SECOND_LINE, VALUE, IN],
                [X_END_SECOND_LINE, VALUE, IN],
                [Y_END_SECOND_LINE, VALUE, IN],
                [X_START_LINE, VALUE, IN],
                [Y_START_LINE, VALUE, IN],
                [X_END_LINE, VALUE, IN],
                [Y_END_LINE, VALUE, IN],
                [X_POINT, VALUE, IN],
                [Y_POINT, VALUE, IN],
            ],
            [[EXPORTED_STATE, DATA]],
            prevent_initial_call=False,
            prevent_update_with_none_input=False,
        )
        def export_ui_state(d):
            def remove_trailing_value(word):
                return word[:-5]

            state = d.copy()
            self._parameters_sem.acquire()
            self._ui_state = {remove_trailing_value(w): v for w, v in state.items()}
            self._parameters_sem.release()
            d[EXPORTED_STATE + DATA] = json.dumps(state)
            return d
