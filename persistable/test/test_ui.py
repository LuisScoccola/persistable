from playwright.sync_api import expect, Page
import re
import time
from sklearn.datasets import make_blobs
import persistable
import pytest


@pytest.fixture(autouse=True)
def pipi():
    global pi
    global url

    X = make_blobs(2000, centers=4, random_state=1)[0]
    p = persistable.Persistable(X)
    pi = persistable.PersistableInteractive(p)
    port = pi.start_UI()
    url = "http://localhost:" + str(port) + "/"

    yield


def test_app_title(page : Page):
    global pi
    global url

    page.goto(url)
    expect(page).to_have_title(re.compile("Persistable"))


def ccf_density_threshold_label_locator(page):
    return page.locator("div#ccf-plot- text[data-unformatted='Density threshold']")

def ccf_compute_button_locator(page):
    return page.locator("button#compute-ccf-button-")

def pv_prominence_label_locator(page):
    return page.locator("div#pv-plot- text[data-unformatted='Prominence']")

def pv_compute_button_locator(page):
    return page.locator("button#compute-pv-button-")

def test_end_to_end(page : Page):

    page.goto(url)
    
    expect(ccf_density_threshold_label_locator(page)).not_to_be_visible()
    ccf_compute_button_locator(page).click()
    expect(ccf_density_threshold_label_locator(page)).to_be_visible()

    expect(pv_prominence_label_locator(page)).not_to_be_visible()
    pv_compute_button_locator(page).click()
    expect(pv_prominence_label_locator(page)).to_be_visible()

