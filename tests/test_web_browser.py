"""Browser flows against a live, loopback-only instance and the saved model."""

import os
import re
import socket
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from zipfile import ZipFile

import pytest

playwright = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

from electrolyte_screening.web import ROOT, app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def local_url():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = monotonic() + 20
            while not server.started and thread.is_alive() and monotonic() < deadline:
                sleep(0.05)
            assert server.started, "The local test server did not start."
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as automation:
        browser = automation.chromium.launch(channel=os.environ.get("IONLAB_BROWSER", "msedge"), headless=True)
        try:
            yield browser
        finally:
            browser.close()


def screenshot(page, name):
    destination = os.environ.get("IONLAB_QA_SCREENSHOTS")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=path / name, full_page=True)


def test_desktop_upload_inspect_rank_export_and_benchmarks(local_url, browser):
    page = browser.new_page(viewport={"width": 1440, "height": 960}, accept_downloads=True)
    errors = []
    external_requests = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def local_only(route):
        if route.request.url.startswith(local_url + "/"):
            route.continue_()
        else:
            external_requests.append(route.request.url)
            route.abort()

    page.route("**/*", local_only)
    page.goto(local_url)
    playwright.expect(page.get_by_role("heading", name="Screen lithium solid electrolytes.")).to_be_visible()
    playwright.expect(page.locator("#snapshot-classifier")).to_have_text("Random forest · 300 trees")
    screenshot(page, "workspace.png")

    sample = page.get_by_role("button", name="Try a sample CIF")
    sample.focus()
    sample.press("Enter")
    playwright.expect(page.locator("#results-count")).to_have_text("1 scored · 0 rejected")
    playwright.expect(page.locator(".known-materials .reference-card")).to_have_count(3)
    playwright.expect(page.get_by_role("link", name="Source DOI for Li3ErI6")).to_have_attribute(
        "href", "https://doi.org/10.1021/acsaem.0c00147",
    )
    playwright.expect(page.locator(".reference-disclaimer")).to_contain_text("Similarity does not validate the prediction.")
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Choose files").click()
    chooser.value.set_files([
        str(FIXTURES / "partial_occupancy.cif"),
        str(FIXTURES / "invalid.cif"),
        str(FIXTURES / "lithium_free.cif"),
    ])
    playwright.expect(page.locator("#results-count")).to_have_text("2 scored · 2 rejected")
    playwright.expect(page.locator("#results-list")).to_contain_text("Could not parse")
    playwright.expect(page.locator("#results-list")).to_contain_text("must contain lithium")
    page.get_by_role("button", name="Inspect partial_occupancy.cif").click()
    playwright.expect(page.get_by_text("Partial occupancies are preserved in the model.", exact=False)).to_be_visible()
    canvas = page.locator("#material-detail canvas")
    canvas.scroll_into_view_if_needed()
    canvas.focus()
    before = canvas.evaluate("element => element.toDataURL()")
    canvas.press("ArrowLeft")
    assert canvas.evaluate("element => element.toDataURL()") != before
    canvas.press("Enter")
    playwright.expect(page.locator(".site-summary")).to_contain_text("Li 0.5")
    page.get_by_role("button", name="Unit cell", exact=True).click()
    playwright.expect(page.get_by_role("button", name="Unit cell", exact=True)).to_have_attribute("aria-pressed", "false")
    page.get_by_role("button", name="Reset view").click()
    page.get_by_text("Extracted model features (131)", exact=True).click()
    playwright.expect(page.get_by_text("Li atomic fraction", exact=True)).to_be_visible()
    screenshot(page, "inspection.png")

    page.get_by_role("link", name="Comparison").click()
    headings = page.locator("#comparison-content table").first.locator("thead th")
    playwright.expect(headings.nth(1)).to_contain_text("lithium-iodide.cif")
    playwright.expect(headings.nth(2)).to_contain_text("partial_occupancy.cif")
    page.get_by_text("All extracted features (131)", exact=True).click()
    playwright.expect(page.locator("#comparison-content details table tbody tr")).to_have_count(131)

    with page.expect_download() as event:
        page.get_by_role("button", name="Download report").click()
    download = event.value
    assert download.suggested_filename == "ionlab-screening-report.html"
    report = download.path().read_text(encoding="utf-8")
    for required in ("1e-4 S/cm", "RandomForestClassifier", "Training coverage", "partial_occupancy.cif",
                     "invalid.cif", "Limitations", "fast-class recall 0.50",
                     "Reported ionic conductivity", "https://doi.org/", "Similarity does not validate the prediction."):
        assert required in report

    page.get_by_role("link", name="Model results", exact=True).click()
    playwright.expect(page.get_by_role("heading", name="Official CSV benchmark · 121 test rows")).to_be_visible()
    playwright.expect(page.get_by_role("heading", name="Archived CIF evaluation · 67 test rows")).to_be_visible()
    playwright.expect(page.locator(".recall-callout strong")).to_have_text("0.50")
    for label in ("Logistic regression", "Random forest", "Histogram gradient boosting", "Majority baseline"):
        playwright.expect(page.get_by_role("rowheader", name=label, exact=True)).to_be_visible()
    playwright.expect(page.get_by_role("heading", name="Global feature importance")).to_be_visible()
    screenshot(page, "benchmarks.png")
    page.get_by_role("link", name="Screening", exact=True).click()
    page.get_by_role("button", name="Clear results").click()
    playwright.expect(page.locator("#screened-files")).to_be_hidden()
    playwright.expect(page.get_by_role("button", name="Download report")).to_be_disabled()
    assert not errors, errors
    assert not external_requests, external_requests
    page.close()


def test_mobile_layout_and_coverage_warning(local_url, browser):
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, accept_downloads=True)
    page.goto(local_url)
    screenshot(page, "mobile-workspace.png")
    stretched = (FIXTURES / "unseen_lii.cif").read_bytes().replace(b" 6.0", b" 60.0")
    transfer = page.evaluate_handle("""bytes => {
        const transfer = new DataTransfer();
        transfer.items.add(new File([new Uint8Array(bytes)], "large-cell.cif", {type: "chemical/x-cif"}));
        return transfer;
    }""", list(stretched))
    page.locator("#drop-zone").dispatch_event("drop", {"dataTransfer": transfer})
    playwright.expect(page.get_by_text("Training coverage warning", exact=True)).to_be_visible()
    playwright.expect(page.get_by_text("outside the training range", exact=False)).to_be_visible()
    page.get_by_role("button", name="View known neighbors").press("Enter")
    playwright.expect(page.get_by_role("heading", name="Known material neighbors")).to_be_focused()
    playwright.expect(page.locator(".known-materials .reference-card")).to_have_count(3)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    screenshot(page, "mobile-neighbors.png")
    page.get_by_role("link", name="Comparison").click()
    playwright.expect(page.get_by_text("Warning", exact=True)).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    with page.expect_download() as event:
        page.get_by_role("button", name="Download report").click()
    assert "outside the training range" in event.value.path().read_text(encoding="utf-8")
    page.close()


def test_cod_1545083_upload_prediction_and_mixed_site_viewer(local_url, browser):
    page = browser.new_page()
    page.goto(local_url)
    with page.expect_response("**/api/predict") as upload:
        page.locator("#file-input").set_input_files(FIXTURES / "1545083.cif")
    (result,) = upload.value.json()["results"]
    assert result["ok"], result
    playwright.expect(page.locator("#results-count")).to_have_text("1 scored · 0 rejected")
    playwright.expect(page.get_by_text(f"{result['fast_conductor_probability'] * 100:.1f}%", exact=True)).to_be_visible()

    mixed_index = next(i for i, site in enumerate(result["structure"]["sites"]) if set(site["species"]) == {"Zr", "Ta"})
    canvas = page.locator("#material-detail canvas")
    canvas.scroll_into_view_if_needed()
    canvas.focus()
    before = canvas.evaluate("element => element.toDataURL()")
    canvas.press("ArrowRight")
    assert canvas.evaluate("element => element.toDataURL()") != before
    canvas.evaluate("""(node, count) => {
        for (let i = 0; i < count; i++) node.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter"}));
    }""", mixed_index + 1)
    playwright.expect(page.locator(".site-summary")).to_contain_text("Zr 0.75824176")
    playwright.expect(page.locator(".site-summary")).to_contain_text("Ta 0.24175824")
    playwright.expect(page.locator(".known-materials").get_by_text("OBELiX flags a close-match CIF", exact=True)).to_have_count(2)
    playwright.expect(page.locator(".known-materials").get_by_role("link", name=re.compile("Source DOI"))).to_have_count(3)
    screenshot(page, "1545083.png")

    with ZipFile(ROOT / "data/raw/train_cifs.zip") as archive:
        known_cif = archive.read("train_randomized_cifs/jqc.cif")
    page.locator("#file-input").set_input_files({
        "name": "known-training.cif", "mimeType": "chemical/x-cif", "buffer": known_cif,
    })
    exact = page.locator(".exact-matches")
    playwright.expect(exact.get_by_role("heading", name="Matches an archived training structure")).to_be_visible()
    playwright.expect(exact.get_by_text("Identical archived training CIF")).to_be_visible()
    with page.expect_download() as event:
        page.get_by_role("button", name="Download report").click()
    assert "Identical archived training CIF" in event.value.path().read_text(encoding="utf-8")
    page.close()
