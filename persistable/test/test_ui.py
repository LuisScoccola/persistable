from playwright.sync_api import expect
import re
import time
from sklearn.datasets import make_blobs
import persistable

def test_app_title(page):

    X = make_blobs(2000, centers=4, random_state=1)[0]
    p = persistable.Persistable(X)
    pi = persistable.PersistableInteractive(p)
    pi.start_UI(jupyter=False)

    time.sleep(1)
    page.goto("http://localhost:8050/")
    expect(page).to_have_title(re.compile("Persistable"))
