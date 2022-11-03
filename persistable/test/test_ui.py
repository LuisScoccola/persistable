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
    p = persistable.Persistable(X)
    pi = persistable.PersistableInteractive(p)
    port = pi.start_UI()
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
        "div#ccf-plot-controls-div-.parameters div#display-lines-selection- :text('On')"
    )


def ccf_details_locator(page):
    return page.locator("details#ccf-details-")


def pv_compute_button_locator(page):
    return page.locator("button#compute-pv-button-")


def pv_parameter_selection_radio_on_locator(page):
    return page.locator(
        "div#pv-plot-controls-div-.parameters div#display-parameter-selection- :text('On')"
    )


def pv_export_parameters_button_locator(page):
    return page.locator("button#export-parameters-")


def pv_details_locator(page):
    return page.locator("details#pv-details-")


# input fields
def ccf_granularity_input_locator(page):
    return page.locator("#input-granularity-ccf-")

def ccf_cores_input_locator(page):
    return page.locator("#input-num-jobs-ccf-")

def pv_granularity_input_locator(page):
    return page.locator("#input-granularity-pv-")

def pv_cores_input_locator(page):
    return page.locator("#input-num-jobs-pv-")


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
    return page.locator("div#pv-plot-controls-div- div#parameter-selection-div-")


def test_app_title(page: Page):
    global pi
    global url

    page.goto(url)
    expect(page).to_have_title(re.compile("Persistable"))


def test_end_to_end(page: Page):

    page.goto(url)

    # ccf
    ccf_details_locator(page).click()
    ccf_granularity_input_locator(page).fill("4")
    # TODO: figure out why in github CI we can't use more than one core
    #ccf_cores_input_locator(page).fill("1")

    expect(ccf_density_threshold_label_locator(page)).not_to_be_visible()
    expect(ccf_controls_div_locator(page)).not_to_be_visible()
    ccf_compute_button_locator(page).click()
    expect(ccf_density_threshold_label_locator(page)).to_be_visible()
    expect(ccf_controls_div_locator(page)).to_be_visible()

    expect(ccf_1st_line_start_label_locator(page)).not_to_be_visible()
    ccf_vineyard_input_selection_radio_on_locator(page).click()
    expect(ccf_1st_line_start_label_locator(page)).to_be_visible()

    # pv
    pv_details_locator(page).click()
    pv_granularity_input_locator(page).fill("2")
    # TODO: figure out why in github CI we can't use more than one core
    #pv_cores_input_locator(page).fill("1")

    expect(pv_prominence_label_locator(page)).not_to_be_visible()
    pv_compute_button_locator(page).click()
    expect(pv_prominence_label_locator(page)).to_be_visible()

    expect(pv_parameter_selection_locator(page)).not_to_be_visible()
    pv_parameter_selection_radio_on_locator(page).click()
    expect(pv_parameter_selection_locator(page)).to_be_visible()

    assert pi._chosen_parameters() is None
    pv_export_parameters_button_locator(page).click()
    time.sleep(1)
    params = pi._chosen_parameters()

    print(params)
    assert params["n_clusters"] == default_params["n_clusters"]
    np.testing.assert_almost_equal(
        np.array(params["start"]), np.array(default_params["start"])
    )
    np.testing.assert_almost_equal(
        np.array(params["end"]), np.array(default_params["end"])
    )