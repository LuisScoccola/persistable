from distutils.log import debug
from playwright.sync_api import expect, Page
import re
import time
from sklearn.datasets import make_blobs
import persistable
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def setup():
    global pi
    global url
    global default_params

    X = make_blobs(100, centers=2, random_state=1)[0]
    # TODO: figure out why in github CI we can't use use the loky backend for joblib
    p = persistable.Persistable(X, debug=True, threading=True)
    pi = persistable.PersistableInteractive(p)
    port = pi.start_ui(debug=True)
    url = "http://localhost:" + str(port) + "/"

    default_params = {
        "n_clusters": 1,
        "start": [1.356663723496982, 0.250030517578125],
        "end": [4.069991170490947, 0.083343505859375],
    }

    yield


# buttons
def ccf_compute_button_locator(page):
    return page.locator("button#compute-ccf-button-")


def ccf_vineyard_input_selection_radio_on_locator(page):
    return page.locator(
        "div#ccf-plot-controls-div-.parameters div#interactive-inputs-selection- :text('Family of lines')"
    )


def ccf_details_locator(page):
    return page.locator("details#ccf-details-")


def pv_compute_button_locator(page):
    return page.locator("button#compute-pv-button-")


def pv_parameter_selection_radio_on_locator(page):
    return page.locator(
        "div#pv-plot-controls-div-.parameters div#display-parameter-selection-pv- :text('On')"
    )


def pv_export_parameters_button_locator(page):
    return page.locator("button#export-parameters-button-pv-")


def pv_details_locator(page):
    return page.locator("details#pv-details-")


# input fields
def ccf_granularity_input_locator(page):
    return page.locator("#granularity-")

def ccf_cores_input_locator(page):
    return page.locator("#num-jobs-ccf-")

def pv_granularity_input_locator(page):
    return page.locator("#granularity-pv-")

def pv_cores_input_locator(page):
    return page.locator("#num-jobs-pv-")


# other objects
def ccf_controls_div_locator(page):
    return page.locator("div#ccf-plot-controls-div-")


def ccf_density_threshold_label_locator(page):
    return page.locator("div#ccf-plot- text[data-unformatted='Density threshold']")


def ccf_1st_line_start_label_locator(page):
    return page.locator("div#ccf-plot- text[data-unformatted='1st line start']")


def pv_prominence_label_locator(page):
    return page.locator("div#pv-plot- text[data-unformatted='Prominence']")


def pv_parameter_selection_locator(page):
    return page.locator("div#pv-plot-controls-div- div#parameter-selection-div-pv-")


def test_app_title(page: Page):
    global pi
    global url

    page.goto(url)

    expect(page).to_have_title(re.compile("Persistable"))

def test_save_ui(page: Page):

    # TODO: set timeout globally
    timeout_milliseconds = 60000

    page.goto(url)

    # ccf
    ccf_details_locator(page).click()
    ccf_granularity_input_locator(page).fill("4")

    expect(ccf_density_threshold_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    expect(ccf_controls_div_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    ccf_compute_button_locator(page).click()
    expect(ccf_density_threshold_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)
    expect(ccf_controls_div_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    expect(ccf_1st_line_start_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    ccf_vineyard_input_selection_radio_on_locator(page).click(timeout=timeout_milliseconds)
    expect(ccf_1st_line_start_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    # pv
    pv_details_locator(page).click()
    pv_granularity_input_locator(page).fill("2")

    expect(pv_prominence_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    pv_compute_button_locator(page).click()
    expect(pv_prominence_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    expect(pv_parameter_selection_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    pv_parameter_selection_radio_on_locator(page).click()
    expect(pv_parameter_selection_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    pv_export_parameters_button_locator(page).click()

    # save and retreive ui state
    ui_state = pi.save_ui_state()

    X = make_blobs(100, centers=2, random_state=1)[0]
    p2 = persistable.Persistable(X, debug=True, threading=True)
    pi2 = persistable.PersistableInteractive(p2)
    port2  = pi2.start_ui(ui_state=ui_state, debug=True)
    url2 = "http://localhost:" + str(port2) + "/"

    time.sleep(2)

    page.goto(url2)

    ccf_details_locator(page).click()
    ccf_compute_button_locator(page).click()
    ccf_vineyard_input_selection_radio_on_locator(page).click(timeout=timeout_milliseconds)

    pv_details_locator(page).click()
    pv_compute_button_locator(page).click()
    pv_parameter_selection_radio_on_locator(page).click()
    pv_export_parameters_button_locator(page).click()

    time.sleep(2)

    params = pi2._chosen_parameters()

    assert params["n_clusters"] == default_params["n_clusters"]
    np.testing.assert_almost_equal(
        np.array(params["start"]), np.array(default_params["start"])
    )
    np.testing.assert_almost_equal(
        np.array(params["end"]), np.array(default_params["end"])
    )


def test_end_to_end(page: Page):

    # TODO: set timeout globally
    timeout_milliseconds = 60000

    page.goto(url)

    # ccf
    ccf_details_locator(page).click()
    ccf_granularity_input_locator(page).fill("4")

    expect(ccf_density_threshold_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    expect(ccf_controls_div_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    ccf_compute_button_locator(page).click()
    expect(ccf_density_threshold_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)
    expect(ccf_controls_div_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    expect(ccf_1st_line_start_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    ccf_vineyard_input_selection_radio_on_locator(page).click(timeout=timeout_milliseconds)
    expect(ccf_1st_line_start_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    # pv
    pv_details_locator(page).click()
    pv_granularity_input_locator(page).fill("2")

    expect(pv_prominence_label_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    pv_compute_button_locator(page).click()
    expect(pv_prominence_label_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    expect(pv_parameter_selection_locator(page)).not_to_be_visible(timeout=timeout_milliseconds)
    pv_parameter_selection_radio_on_locator(page).click()
    expect(pv_parameter_selection_locator(page)).to_be_visible(timeout=timeout_milliseconds)

    assert pi._chosen_parameters() is None
    pv_export_parameters_button_locator(page).click()
    time.sleep(1)
    params = pi._chosen_parameters()

    assert params["n_clusters"] == default_params["n_clusters"]
    np.testing.assert_almost_equal(
        np.array(params["start"]), np.array(default_params["start"])
    )
    np.testing.assert_almost_equal(
        np.array(params["end"]), np.array(default_params["end"])
    )
