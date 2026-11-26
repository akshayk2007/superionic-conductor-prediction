import {crystalViewer, drawIllustration, elementColor} from "./crystal.js";
import {
  capitalize, classLabel, coverageMessages, exactMatchLabel, featureLabel, neighborReason,
  number, probability, proportion, reportedConductivity,
} from "./display.js";
import {downloadReport} from "./report.js";

const $ = id => document.getElementById(id);
const make = (tag, className = "", text = "") => {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
};

const state = {model: null, results: [], selected: null, busy: false, viewer: null};
const modelNames = {
  LogisticRegression: "Logistic regression",
  RandomForestClassifier: "Random forest",
  HistGradientBoostingClassifier: "Histogram gradient boosting",
  "Majority (most frequent)": "Majority baseline",
};

function list(items) {
  const container = make("ul");
  container.append(...items.map(item => make("li", "", item)));
  return container;
}

function definitions(rows, className) {
  const container = make("dl", className);
  for (const [label, value] of rows) {
    const pair = make("div");
    pair.append(make("dt", "", label), make("dd", "", value));
    container.append(pair);
  }
  return container;
}

function table(headers, rows, className = "", highlight = -1) {
  const wrapper = make("div", "table-scroll");
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", className === "comparison-table" ? "Scrollable material comparison" : "Scrollable model metrics");
  const grid = make("table", className);
  const head = grid.createTHead().insertRow();
  for (const header of headers) {
    const cell = make("th");
    cell.scope = "col";
    cell.append(header);
    head.append(cell);
  }
  const body = grid.createTBody();
  rows.forEach(([label, ...values], index) => {
    const row = body.insertRow();
    if (index === highlight) row.classList.add("selected-model");
    const name = make("th", "", label);
    name.scope = "row";
    row.append(name);
    for (const value of values) row.insertCell().append(value);
  });
  wrapper.append(grid);
  return wrapper;
}

function validResults() {
  return state.results.flatMap((result, index) => result.ok ? [{result, index}] : []);
}

function navigation() {
  const value = location.hash.slice(1);
  return ["screen", "compare", "model"].includes(value) ? value : "screen";
}

function showView() {
  const view = navigation();
  for (const panel of document.querySelectorAll("[data-panel]")) panel.hidden = panel.id !== view;
  for (const link of document.querySelectorAll("[data-view]")) {
    if (link.dataset.view === view) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (view === "compare") renderComparison();
  if (view === "model") renderModel();
}

function scorePill(result) {
  const label = classLabel(result);
  return make("span", "pill " + label.toLowerCase(), label);
}

function bar(value, className = "probability-bar") {
  const container = make("div", className);
  container.setAttribute("aria-hidden", "true");
  const fill = make("span");
  fill.style.width = (value * 100) + "%";
  container.append(fill);
  return container;
}

function openMaterial(index) {
  state.selected = index;
  location.hash = "screen";
  showView();
  renderResults();
  requestAnimationFrame(() => {
    const heading = $("material-title");
    heading.focus({preventScroll: true});
    heading.scrollIntoView({block: "start"});
  });
}

function renderResults() {
  const valid = validResults();
  const rejected = state.results.length - valid.length;
  $("screened-files").hidden = !state.results.length;
  $("download-report").disabled = !state.results.length || !state.model;
  $("clear-results").disabled = state.busy;
  $("compare-results").hidden = !valid.length;
  $("comparison-count").hidden = !valid.length;
  $("comparison-count").textContent = valid.length;
  $("results-count").textContent = valid.length + " scored · " + rejected + " rejected";
  $("results-list").replaceChildren(...state.results.map((result, index) => {
    const item = make("li", "result-row" + (!result.ok ? " rejected" : index === state.selected ? " selected" : ""));
    const meta = make("div");
    meta.append(make("span", "result-name", result.filename), make(
      "span", "result-meta",
      result.ok ? result.formula + " · " + capitalize(result.crystal_system) +
        " · " + probability(result.fast_conductor_probability) + " fast-class probability" +
        (result.training_coverage.warning ? " · Coverage warning" : "") : result.error,
    ));
    item.append(meta, result.ok ? scorePill(result) : make("span", "pill danger", "Rejected"));
    if (result.ok) {
      const inspect = make("button", "button button-outline", "Inspect");
      inspect.type = "button";
      inspect.setAttribute("aria-label", "Inspect " + result.filename);
      inspect.addEventListener("click", () => openMaterial(index));
      item.append(inspect);
    }
    return item;
  }));
  renderDetail(state.results[state.selected]);
  if (navigation() === "compare") renderComparison();
}

function structurePanel(result) {
  const section = make("div", "structure-panel");
  const frame = make("div", "structure-view");
  const toolbar = make("div", "structure-toolbar");
  const cell = make("button", "button button-outline", "Unit cell");
  cell.type = "button";
  cell.setAttribute("aria-pressed", "true");
  const reset = make("button", "button button-outline", "Reset view");
  reset.type = "button";
  toolbar.append(cell, reset);
  const canvas = make("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "Interactive crystal structure for " + result.formula);
  canvas.setAttribute("aria-describedby", "viewer-note");
  frame.append(toolbar, canvas);
  const legend = make("div", "structure-legend");
  for (const symbol of Object.keys(result.unit_cell_composition).sort()) {
    const key = make("span");
    const color = make("i");
    color.style.background = elementColor(symbol);
    key.append(color, symbol);
    legend.append(key);
  }
  const note = make("p", "viewer-note", "Drag or use arrow keys to rotate. Scroll or use +/− to zoom. Select a site or press Enter to inspect it.");
  note.id = "viewer-note";
  const occupancy = make("p", "viewer-note", result.has_partial_occupancies ?
    "Partial occupancies are preserved in the model. The viewer draws one marker per site, colored by its most occupied element." :
    "Markers show parsed sites within the unit cell.");
  const selection = make("p", "site-summary");
  selection.setAttribute("aria-live", "polite");
  section.append(frame, legend, note, occupancy, selection);
  if (result.structure.sites.length < result.structure.site_count) {
    section.append(make("p", "viewer-note", "Showing " + result.structure.sites.length + " of " +
      result.structure.site_count + " sites for display. Features use the full structure."));
  }
  return {section, attach() {
    const viewer = crystalViewer(canvas, result.structure, (site, index) => {
      selection.textContent = site ? "Marker " + (index + 1) + " occupancies · " +
        Object.entries(site.species).map(([symbol, value]) => symbol + " " + number(value, 8)).join(" · ") : "";
    });
    cell.addEventListener("click", () => {
      const visible = cell.getAttribute("aria-pressed") !== "true";
      cell.setAttribute("aria-pressed", visible);
      viewer.setCell(visible);
    });
    reset.addEventListener("click", () => {
      cell.setAttribute("aria-pressed", "true");
      viewer.setCell(true);
      viewer.reset();
    });
    return viewer;
  }};
}

function knownMaterials(result) {
  const section = make("section", "known-materials");
  const heading = make("h3", "", "Known material neighbors");
  heading.tabIndex = -1;
  section.append(heading, make("p", "muted",
    "Similar entries from the OBELiX training set, ranked by composition and crystal system."));
  const matches = result.known_materials;
  if (matches?.unavailable || !matches) {
    section.append(make("p", "callout neutral", matches?.unavailable || "Known materials are unavailable."));
    return {section, heading};
  }
  const renderCard = (material, exact) => {
    const card = make("li", "reference-card");
    const provenance = make("span", "pill " + (material.provenance === "close_match" ? "warn" : ""), material.provenance_label);
    card.append(make("h4", "", material.formula), provenance);
    card.append(make("span", "reference-measurement-label", "Reported ionic conductivity"),
      make("strong", "reference-conductivity", reportedConductivity(material)));
    if (material.doi_url) {
      const link = make("a", "reference-doi", "Source DOI");
      link.href = material.doi_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.setAttribute("aria-label", "Source DOI for " + material.formula + " (opens in a new tab)");
      card.append(link);
    } else {
      card.append(make("span", "muted", "Source DOI unavailable"));
    }
    card.append(make("p", "reference-reason", exact ? exactMatchLabel(material) : neighborReason(material)));
    return card;
  };
  if (matches.exact_matches.length) {
    const identity = make("div", "exact-matches");
    const cards = make("ul", "reference-grid");
    cards.append(...matches.exact_matches.map(material => renderCard(material, true)));
    identity.append(make("h4", "", "Matches an archived training structure"),
      make("p", "muted", "The upload matches a structure already represented in training. A close-match flag means the archived structure is a proxy for the measured specimen."),
      cards);
    if (matches.exact_match_count > matches.exact_matches.length) {
      identity.append(make("p", "muted", "Showing " + matches.exact_matches.length + " of " + matches.exact_match_count + " archive matches."));
    }
    section.append(identity);
  }
  if (matches.neighbors.length) {
    const cards = make("ul", "reference-grid");
    cards.append(...matches.neighbors.map(material => renderCard(material, false)));
    section.append(cards);
    if (new Set(matches.neighbors.map(material => material.formula)).size < matches.neighbors.length) {
      section.append(make("p", "muted reference-note", "Repeated formulas are separate OBELiX measurement records."));
    }
  } else {
    section.append(make("p", "callout neutral", "No additional training reference shared an element with this upload."));
  }
  if (!matches.archive_available) {
    section.append(make("p", "muted", "The local CIF archive is unavailable. Structure identity checks were skipped."));
  }
  section.append(make("p", "reference-disclaimer", state.model?.neighbor_disclaimer ||
    "These conductivity values were measured for the known materials, not for your uploaded material. Similarity does not validate the prediction."));
  const method = make("details", "reference-method");
  method.append(make("summary", "", "How neighbors are chosen"), make("p", "muted",
    (state.model?.neighbor_method || "Ranking compares fractional chemistry and crystal system.") +
    " A matching composition can describe more than one structure. Unflagged CIFs carry no close-match flag in OBELiX."));
  section.append(method);
  return {section, heading};
}

function renderDetail(result) {
  state.viewer?.destroy();
  state.viewer = null;
  const detail = $("material-detail");
  detail.replaceChildren();
  detail.hidden = !result?.ok;
  if (detail.hidden) return;
  const card = make("article", "card detail-card");
  const top = make("div", "detail-top");
  const title = make("h2", "", result.formula);
  title.id = "material-title";
  title.tabIndex = -1;
  const names = make("div");
  names.append(title, make("p", "muted", result.filename + " · " + capitalize(result.crystal_system) +
    " · Space group " + result.space_group_number));
  top.append(names, scorePill(result));
  const grid = make("div", "detail-grid");
  const structure = structurePanel(result);
  const known = knownMaterials(result);
  const score = make("div", "score-panel");
  const line = make("div", "score-line");
  line.append(make("span", "", "Estimated fast-class probability"), make("strong", "", probability(result.fast_conductor_probability)));
  score.append(line, bar(result.fast_conductor_probability),
    make("p", "muted", "Uncalibrated model estimate of the ≥ 1 × 10⁻⁴ S/cm class."),
    make("h3", "subheading", "Unit-cell composition"));
  const composition = Object.entries(result.unit_cell_composition).map(([el, amount]) => el + " " + number(amount, 8));
  score.append(make("p", "muted", composition.join(" · ")), make("h3", "subheading", "Lattice parameters"));
  score.append(definitions(Object.entries(result.lattice).map(([key, value]) => [
    ({alpha: "α", beta: "β", gamma: "γ"})[key] ?? key,
    number(value, 7) + (key.length === 1 ? " Å" : "°"),
  ]), "lattice"));
  const warnings = coverageMessages(result.training_coverage);
  const coverage = make("div", "callout" + (!warnings.length ? " neutral" : ""));
  coverage.append(make("strong", "", warnings.length ? "Training coverage warning" : "Within checked training ranges"));
  if (warnings.length) coverage.append(list(warnings));
  score.append(coverage);
  if (result.cif_parser_warnings.length) {
    const parser = make("details", "callout");
    parser.append(make("summary", "", "CIF parser warnings"), list(result.cif_parser_warnings));
    score.append(parser);
  }
  const jump = make("button", "text-button reference-jump", "View known neighbors");
  jump.type = "button";
  jump.addEventListener("click", () => {
    known.heading.focus({preventScroll: true});
    known.heading.scrollIntoView({block: "start"});
  });
  score.append(jump);
  const features = make("details", "feature-details");
  features.append(make("summary", "", "Extracted model features (" + Object.keys(result.features).length + ")"),
    make("p", "muted", "These values are the inputs to the classifier. Explaining an individual prediction would require a separate attribution analysis."),
    definitions(Object.entries(result.features).map(([key, value]) => [featureLabel(key), number(value, 8)]), "feature-grid"));
  grid.append(score, structure.section);
  card.append(top, grid, known.section, features);
  detail.append(card);
  state.viewer = structure.attach();
}

function renderComparison() {
  const target = $("comparison-content");
  const ranked = validResults().sort((a, b) => b.result.fast_conductor_probability - a.result.fast_conductor_probability);
  if (!ranked.length) {
    const empty = make("div", "card empty-state");
    const link = make("a", "button button-primary", "Upload CIFs");
    link.href = "#screen";
    empty.append(make("h2", "", "No materials to compare"),
      make("p", "muted", "Upload at least one valid crystal structure to view its rank and features."), link);
    target.replaceChildren(empty);
    return;
  }
  const headers = ["Feature", ...ranked.map(({result, index}, rank) => {
    const header = make("div");
    const inspect = make("button", "text-button", result.filename);
    inspect.type = "button";
    inspect.setAttribute("aria-label", "Inspect " + result.filename);
    inspect.addEventListener("click", () => openMaterial(index));
    header.append(make("strong", "", "#" + (rank + 1) + " · " + result.formula), document.createElement("br"), inspect);
    return header;
  })];
  const across = (label, format) => [label, ...ranked.map(({result}) => format(result))];
  const keyFeatures = [
    across("Fast-class probability", result => probability(result.fast_conductor_probability)),
    across("Predicted class", scorePill),
    across("Coverage", result => result.training_coverage.warning ? "Warning" : "Within checked ranges"),
    across("Crystal system", result => capitalize(result.crystal_system)),
    across("Lithium atomic fraction", result => number(result.features.fraction_Li, 7)),
    across("Cell volume per atom (Å³)", result => number(result.features.volume_per_atom_angstrom3, 7)),
    across("Element count", result => number(result.features.element_count)),
    across("Partial occupancies", result => result.has_partial_occupancies ? "Present" : "None detected"),
  ];
  const allFeatures = Object.keys(ranked[0].result.features);
  const expanded = make("details", "card comparison-details");
  expanded.append(make("summary", "", "All extracted features (" + allFeatures.length + ")"),
    table(headers.map(header => header instanceof Node ? header.querySelector("strong").textContent : header),
      allFeatures.map(feature => across(featureLabel(feature), result => number(result.features[feature], 8))),
      "comparison-table"));
  target.replaceChildren(table(headers, keyFeatures, "comparison-table"), expanded);
}

function metricsRow(label, metrics) {
  return [label, ...["accuracy", "precision", "recall", "f1"].map(key => metrics[key].toFixed(3)),
    JSON.stringify(metrics.confusion_matrix)];
}

function renderModel() {
  const target = $("model-content");
  const model = state.model;
  if (!model) {
    target.replaceChildren(make("p", "card muted", state.modelError || "Loading model results…"));
    return;
  }
  const csv = model.csv_benchmark;
  const cif = model.cif_evaluation;
  const metricHeaders = ["Model or input", "Accuracy", "Precision", "Recall", "F1", "Confusion matrix"];
  const original = make("section", "card benchmark-card");
  const entries = Object.entries(csv.test_scores);
  original.append(make("h2", "", "Official CSV benchmark · " + csv.test.rows + " test rows"),
    make("p", "muted", csv.test.fast + " fast · " + csv.test.slow + " slow · Random forest selected by mean F1 in five training folds."),
    table(metricHeaders, entries.map(([name, result]) => metricsRow(modelNames[name], result)), "",
      entries.findIndex(([name]) => name === csv.selection.selected)));
  const archived = make("section", "card benchmark-card");
  archived.append(make("h2", "", "Archived CIF evaluation · " + cif.counts.rows + " test rows"),
    make("p", "muted", "Saved random forest. Both rows below use the same " + cif.counts.rows + " entries."));
  const cifGrid = make("div", "cif-results");
  const recall = make("div", "recall-callout");
  recall.append(make("b", "", "CIF fast-class recall"), make("strong", "", proportion(cif.metrics.recall)),
    make("p", "", cif.metrics.confusion_matrix[1][1] + " of " + cif.counts.fast + " fast conductors detected"));
  cifGrid.append(table(metricHeaders, [
    metricsRow("Archived CIFs", cif.metrics),
    metricsRow("Paired CSV metadata", cif.paired_csv_metrics),
  ], "", 0), recall);
  archived.append(cifGrid, make("p", "muted", "Confusion matrices show true classes by row and predicted classes by column, both ordered slow, fast. Precision, recall, and F1 use the fast class."));

  const importance = make("section", "card");
  importance.append(make("h2", "", "Global feature importance"),
    make("p", "muted", "Mean decrease in impurity summarizes how the fitted forest used features across its trees. It can favor features with more split points. Bars are scaled to the largest displayed value. Individual predictions require a separate attribution analysis."));
  const strongest = model.global_feature_importance.slice(0, 8);
  const figure = make("ul", "importance-list");
  for (const {feature, importance: value} of strongest) {
    const line = make("li");
    line.append(make("span", "", featureLabel(feature)), bar(value / strongest[0].importance, "importance-bar"),
      make("span", "", probability(value)));
    figure.append(line);
  }
  importance.append(figure);
  const limitations = make("section", "card limitations");
  limitations.append(make("h2", "", "Model scope and limitations"), list(model.limitations));
  target.replaceChildren(original, archived, importance, limitations);
}

async function requestJson(url, options) {
  let response;
  try { response = await fetch(url, options); }
  catch { throw new Error("Could not reach IonLab. Check the connection or open the local demo."); }
  const body = await response.json().catch(() => null);
  if (!body) throw new Error("IonLab is unavailable at this address. On the demo laptop, open http://127.0.0.1:8000.");
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "The local API could not complete the request.");
  return body;
}

function uploadNotice(message = "") {
  $("upload-notice").textContent = message;
  $("upload-notice").hidden = !message;
}

async function submitFiles(files) {
  if (!files.length || state.busy) return;
  uploadNotice();
  const limit = state.model?.limits.files_per_batch ?? 20;
  if (files.length > limit) {
    uploadNotice("Choose at most " + limit + " CIF files per batch.");
    return;
  }
  const form = new FormData();
  for (const file of files) form.append("files", file);
  state.busy = true;
  $("drop-zone").setAttribute("aria-busy", "true");
  $("browse").disabled = $("try-example").disabled = $("clear-results").disabled = true;
  $("upload-status").textContent = "Screening " + files.length + (files.length === 1 ? " file…" : " files…");
  let completed = false;
  try {
    const {results} = await requestJson("/api/predict", {method: "POST", body: form});
    const firstValid = results.findIndex(result => result.ok);
    if (firstValid >= 0) state.selected = state.results.length + firstValid;
    state.results.push(...results);
    const ready = results.filter(result => result.ok).length;
    $("upload-status").textContent = ready + " scored · " + (results.length - ready) + " rejected";
    completed = true;
  } catch (error) {
    uploadNotice(error.message);
    $("upload-status").textContent = "";
  } finally {
    state.busy = false;
    $("drop-zone").setAttribute("aria-busy", "false");
    $("browse").disabled = $("try-example").disabled = false;
    renderResults();
    if (completed && navigation() === "screen") {
      $("screened-title").focus({preventScroll: true});
      $("screened-title").scrollIntoView({block: "start"});
    }
  }
}

$("browse").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", event => {
  const files = Array.from(event.target.files);
  event.target.value = "";
  submitFiles(files);
});
$("try-example").addEventListener("click", async () => {
  $("try-example").disabled = true;
  $("try-example").textContent = "Loading sample…";
  try {
    const response = await fetch("/static/lithium-iodide.cif");
    if (!response.ok) throw new Error("The sample CIF is unavailable.");
    await submitFiles([new File([await response.blob()], "lithium-iodide.cif")]);
  } catch {
    uploadNotice("The sample CIF could not be loaded. Check that the local IonLab server is running.");
  } finally {
    $("try-example").disabled = state.busy;
    $("try-example").textContent = "Try a sample CIF";
  }
});
for (const event of ["dragenter", "dragover", "dragleave", "drop"]) {
  $("drop-zone").addEventListener(event, drop => {
    drop.preventDefault();
    $("drop-zone").classList.toggle("dragging", event === "dragenter" || event === "dragover");
    if (event === "drop") submitFiles(Array.from(drop.dataTransfer.files));
  });
}
$("clear-results").addEventListener("click", () => {
  state.results = [];
  state.selected = null;
  $("upload-status").textContent = "";
  uploadNotice();
  renderResults();
});
$("download-report").addEventListener("click", () => downloadReport(state.model, state.results));
$("skip-navigation").addEventListener("click", event => {
  event.preventDefault();
  $("main").focus();
});
window.addEventListener("hashchange", showView);
drawIllustration($("hero-crystal"));
showView();
requestJson("/api/model").then(model => {
  state.model = model;
  $("snapshot-classifier").textContent = model.name + " · " + model.trees + " trees";
  $("snapshot-train").textContent = model.csv_benchmark.train.rows + " OBELiX rows";
  $("snapshot-recall").textContent = proportion(model.cif_evaluation.metrics.recall);
  $("upload-limits").textContent = "Up to " + model.limits.files_per_batch + " files per batch · " +
    model.limits.file_bytes / (1024 * 1024) + " MiB per file";
  $("download-report").disabled = !state.results.length;
  if (navigation() === "model") renderModel();
}).catch(() => {
  state.modelError = "Could not load model results. Check the local FastAPI server.";
  if (navigation() === "model") renderModel();
});
