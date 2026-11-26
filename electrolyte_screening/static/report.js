import {
  capitalize, classLabel, coverageMessages, exactMatchLabel, featureLabel, neighborReason,
  number, probability, reportedConductivity,
} from "./display.js";

function escape(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function rows(entries) {
  return entries.map(([label, value]) => "<tr><th scope='row'>" + escape(label) +
    "</th><td>" + escape(value) + "</td></tr>").join("");
}

const list = items => "<ul>" + items.map(item => "<li>" + escape(item) + "</li>").join("") + "</ul>";

function referenceTable(materials, reason) {
  return "<table><thead><tr><th scope='col'>Formula and provenance</th><th scope='col'>Reported ionic conductivity</th>" +
    "<th scope='col'>Why it is listed</th></tr></thead><tbody>" + materials.map(material => {
      const doi = material.doi_url?.startsWith("https://doi.org/") ?
        "<a href='" + escape(material.doi_url) + "' rel='noreferrer'>Source DOI</a>" : "Source DOI unavailable";
      return "<tr><th scope='row'>" + escape(material.formula) + "<br><small>" + escape(material.provenance_label) +
        "</small></th><td>" + escape(reportedConductivity(material)) + "<br>" + doi +
        "</td><td>" + escape(reason(material)) + "</td></tr>";
    }).join("") + "</tbody></table>";
}

function referenceSection(matches, model) {
  const title = "<h3>Known material neighbors</h3>";
  if (!matches || matches.unavailable) {
    return title + "<p>" + escape(matches?.unavailable || "Known materials are unavailable.") + "</p>";
  }
  const exact = matches.exact_matches.length ? "<h4>Matches to archived training structures</h4>" +
    "<p>Showing " + matches.exact_matches.length + " of " + matches.exact_match_count + " archive matches.</p>" +
    referenceTable(matches.exact_matches, exactMatchLabel) : "";
  const neighbors = matches.neighbors.length ? referenceTable(matches.neighbors, neighborReason) :
    "<p>No additional training reference shared an element with this upload.</p>";
  return title + exact + (exact ? "<h4>Other similar training materials</h4>" : "") + neighbors +
    (!matches.archive_available ? "<p>The local CIF archive was unavailable for identity checks.</p>" : "") +
    "<p><strong>" + escape(model.neighbor_disclaimer) + "</strong></p><p>" + escape(model.neighbor_method) +
    " A close-match flag marks a proxy structure associated with an experimental record. " +
    "A matching composition can describe more than one structure.</p>";
}

export function downloadReport(model, results) {
  const ranked = results.filter(result => result.ok).sort(
    (first, second) => second.fast_conductor_probability - first.fast_conductor_probability,
  );
  const rejected = results.filter(result => !result.ok);
  const summary = ranked.map((result, index) => [
    index + 1 + ". " + result.filename,
    result.formula + " · " + classLabel(result) + " · " + probability(result.fast_conductor_probability) +
      " · " + (result.training_coverage.warning ? "Coverage warning" : "Within checked ranges"),
  ]);
  const details = ranked.map(result => {
    const coverage = coverageMessages(result.training_coverage);
    const featureRows = Object.entries(result.features).map(([key, value]) => [featureLabel(key), value.toPrecision(10)]);
    const lattice = Object.entries(result.lattice).map(([key, value]) => [
      key, number(value, 8) + (key.length === 1 ? " Å" : "°"),
    ]);
    const chemistry = Object.entries(result.unit_cell_composition).map(([symbol, amount]) => (
      symbol + " " + number(amount, 8)
    )).join(", ");
    return "<section><h2>" + escape(result.filename) + "</h2><table>" + rows([
      ["Extracted composition", result.formula],
      ["Unit-cell element counts", chemistry],
      ["Crystal system", capitalize(result.crystal_system) + " (space group " + result.space_group_number + ")"],
      ["Predicted class", classLabel(result)],
      ["Estimated fast-class probability", probability(result.fast_conductor_probability)],
      ["Partial occupancies", result.has_partial_occupancies ? "Present and used in feature extraction" : "None detected"],
      ...lattice,
    ]) + "</table><h3>Training coverage</h3>" +
      (coverage.length ? list(coverage) : "<p>Within checked training ranges.</p>") +
      (result.cif_parser_warnings.length ? "<h3>CIF parser warnings</h3>" + list(result.cif_parser_warnings) : "") +
      referenceSection(result.known_materials, model) +
      "<details><summary>Extracted model features (" + featureRows.length + ")</summary><table>" +
      rows(featureRows) + "</table></details></section>";
  }).join("");

  const style = "body{font:16px/1.5 system-ui,sans-serif;color:#22233a;max-width:920px;margin:40px auto;padding:0 20px}" +
    "h1,h2,h3{line-height:1.15;color:#282047}section{margin-top:28px}table{border-collapse:collapse;width:100%}" +
    "th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #dedce9;overflow-wrap:anywhere}" +
    "th{width:39%;font-weight:650}details{margin-top:12px}summary{font-weight:650;cursor:pointer}" +
    "@media print{body{margin:0}h2,h3{break-after:avoid}tr{break-inside:avoid}}";
  const html = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>" +
    "<title>IonLab screening report</title><style>" + style + "</style><body><h1>IonLab screening report</h1>" +
    "<p>Predictions for " + ranked.length + " of " + results.length + " uploaded files.</p>" +
    "<section><h2>Model details</h2><table>" + rows([
      ["Classifier", model.classifier + " (" + model.trees + " trees)"],
      ["Dataset", model.dataset + " at revision " + model.dataset_revision],
      ["Artifact SHA-256", model.artifact_sha256],
      ["Fast-conductor threshold", "Ionic conductivity ≥ " + model.threshold_s_per_cm.toExponential() + " S/cm"],
      ["Training rows", model.csv_benchmark.train.rows],
      ["Official CSV benchmark", model.csv_benchmark.test.rows + " rows"],
      ["Archived CIF evaluation", model.cif_evaluation.counts.rows + " rows, fast-class recall " + model.cif_evaluation.metrics.recall.toFixed(2)],
    ]) + "</table></section>" +
    (summary.length ? "<section><h2>Ranked results</h2><table>" + rows(summary) + "</table></section>" : "") +
    (rejected.length ? "<section><h2>Rejected files</h2><table>" +
      rows(rejected.map(result => [result.filename, result.error])) + "</table></section>" : "") +
    details + "<section><h2>Limitations</h2>" + list([
      ...model.limitations,
      "The coverage check uses elemental fractions, cell volume per atom, and crystal system from the training distribution.",
      "The extracted feature values are model inputs. Global forest feature importance summarizes splits across the model. Individual predictions require a separate attribution analysis.",
    ]) + "</section></body></html>";
  const url = URL.createObjectURL(new Blob([html], {type: "text/html;charset=utf-8"}));
  const link = Object.assign(document.createElement("a"), {href: url, download: "ionlab-screening-report.html"});
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
