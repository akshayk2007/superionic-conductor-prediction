# Lithium solid-electrolyte screening with OBELiX

This Python pipeline screens lithium-containing crystal structures for experimental room-temperature ionic conductivity of at least **1 × 10⁻⁴ S/cm**. It reproduces a benchmark on the authors’ official split and saves one fitted pipeline for prediction from a CIF.

## Expo setup on Windows

**Prepare the demo laptop while online.** From the project root in Windows PowerShell with Python 3.12 (64-bit), run:

```powershell
if (!(Test-Path .\.venv\Scripts\python.exe)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m electrolyte_screening download
```

Keep the verified `artifacts/fast_conductor.joblib`, both benchmark reports, and the downloaded training CSV and CIF archive on the demo laptop. The download command verifies the pinned files. The app checks the saved model’s SHA-256 against the upload verification report.

**Start IonLab in PowerShell window 1.**

```powershell
.\.venv\Scripts\python.exe -m uvicorn electrolyte_screening.web:app --host 127.0.0.1 --port 8000
```

Wait for **Application startup complete**, then open [IonLab on this laptop](http://127.0.0.1:8000).

**Start a Cloudflare Quick Tunnel in PowerShell window 2.** Install `cloudflared` while online, then open a fresh PowerShell window if the command is still unavailable.

```powershell
winget install --id Cloudflare.cloudflared --exact
cloudflared --version
cloudflared tunnel --url http://127.0.0.1:8000
```

Share the temporary HTTPS `trycloudflare.com` URL printed by `cloudflared`. Anyone with that URL can reach the app. Quick Tunnels are temporary and the address can change when restarted. [Cloudflare Quick Tunnel instructions](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) · [Windows binary and MSI downloads](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)

**Stop or switch to offline.** Press **Ctrl+C** in window 2 to end the tunnel. Press **Ctrl+C** in window 1 when you also want to stop IonLab. If expo internet fails, leave window 1 running and open the local URL on the laptop:

```powershell
Start-Process 'http://127.0.0.1:8000/'
```

If the server stopped, rerun the window 1 command. The bundled sample, predictions, crystal viewer, training neighbors, comparisons, and HTML report work from the local files. Opening a DOI or the public tunnel link requires internet access.

### 60-second expo demo

| Time | What to show |
| --- | --- |
| 0–10 s | Open IonLab and explain that it screens a crystal structure from a CIF. Fast means ionic conductivity ≥ 1 × 10⁻⁴ S/cm. |
| 10–25 s | Click **Try a sample CIF**. Point out the predicted class, uncalibrated model estimate, and interactive viewer. |
| 25–40 s | Upload `tests/fixtures/1545083.cif` as your second material. Open **View known neighbors** and point out one source DOI and a close-match label. The reported measurements belong to the references. |
| 40–50 s | Open **Comparison** to show the ranking, then click **Download report**. |
| 50–60 s | Open **Model results** and point to the separate CIF evaluation. Recall was 0.50, which means the deployed model caught 10 of 20 fast conductors on those 67 CIFs. |

### App behavior

Upload one UTF-8, single-structure `.cif` or a batch of up to 20 files at 2 MiB per file. Each file gets its own result or rejection. The server also caps the total upload stream and bounds concurrent batches. If the app is busy, wait for a batch to finish and retry. The comparison ranks valid uploads by estimated fast-class probability. Download the standalone HTML report to keep the results after closing or reloading the page.

Known neighbors come only from the pinned OBELiX training CSV. Ranking uses 85% elemental-fraction overlap and 15% crystal-system agreement. Composition alone is used when reference symmetry is missing. The score uses bounded input features and omits cell volume because the audit found CSV and CIF cell-basis discrepancies. Conductivity, class labels, IDs, and DOI are excluded from ranking. Reported upper bounds stay upper bounds. Separate references can share a formula when the measurements come from different reports.

The training archive supports a separate identity check by byte equality or volume-preserving structural comparison on matching normalized compositions. The structural comparison uses 0.5% lattice-length tolerance, 0.5° angle tolerance, and a site tolerance of 0.01 times the mean free length per site. Exact and equivalent matches appear apart from the neighbor list. An OBELiX close-match flag indicates a proxy structure for the experimental record. An unflagged archive label communicates the status of that flag. Without the local training CSV, predictions still work and the app displays a neighbor-unavailable message. Without the CIF archive, the CSV can still provide neighbors and the app labels archive availability.

**These conductivity values were measured for the known materials, not for your uploaded material. Similarity does not validate the prediction.**

The browser viewer draws one marker per parsed site and colors mixed sites by their most occupied element. Select a marker to inspect the occupancy of each element. Large structures use a sampled display of at most 2,500 sites. The model features use the full parsed structure and its fractional occupancies. For small rounding excesses, pymatgen scales site totals above 1 and at most 1.002 proportionally to 1, preserving the ratio between elements. Larger totals produce a per-file error. The model-results page separates the 121-row CSV benchmark and the 67-row CIF evaluation, where fast-class recall was **0.50**. The global importance chart reports mean decrease in impurity across the fitted forest. An individual prediction would require separate attribution.

### Run the web tests

The browser tests use headless Microsoft Edge and start their own loopback server on an available port.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web-test.txt
.\.venv\Scripts\python.exe -m pytest -q tests/test_web_api.py tests/test_web_browser.py
```

To use installed Google Chrome, set `$env:IONLAB_BROWSER = 'chrome'` before running the tests.

The mixed-occupancy regression fixture is the original CIF for [Crystallography Open Database entry 1545083](https://www.crystallography.net/cod/1545083.html), from Hamao, Kataoka, Kijima, and Akimoto (2016).

## Reproduce the ML experiment

These commands rebuild the benchmark and fit a model. Use the web-app commands above to serve the verified artifact. From the project root in PowerShell with Python 3.12 (64-bit):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m electrolyte_screening download
.\.venv\Scripts\python.exe -m electrolyte_screening audit --report reports/data-audit.md
.\.venv\Scripts\python.exe -m electrolyte_screening train --report reports/benchmark.json
.\.venv\Scripts\python.exe -m pytest -q
```

The download verifies SHA-256 for the official train and test CSVs and both CIF archives. It stores them in `data/raw/`. Training saves the fitted featurizer, classifier, and training coverage profile together in `artifacts/fast_conductor.joblib`. The Python commands use `./.venv/bin/python` on POSIX systems. Use the same dependency versions when loading the artifact.

To try a CIF from the official training archive:

```powershell
Expand-Archive -LiteralPath data/raw/train_cifs.zip -DestinationPath data/examples -Force
.\.venv\Scripts\python.exe -m electrolyte_screening predict data/examples/train_randomized_cifs/jqc.cif
```

Prediction returns the class (`1` for fast and `0` for slow), an estimated fast-class probability, and any coverage or CIF parser warnings. Substitute the path to a UTF-8 CIF containing a single lithium-bearing crystal structure. The predictor derives composition from parsed atom sites and weights the atom count by site occupancy. Missing lithium and invalid structures produce input errors.

## Data and labels

The source datasets and archives are from NRC-Mila’s **OBELiX** at revision `4eaac889809dd489e3bae468c45d5c0e02b9147a` under CC BY 4.0. Cite Félix Therrien et al. (2025), *OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes*. [Repository](https://github.com/NRC-Mila/OBELiX) · [License](https://github.com/NRC-Mila/OBELiX/blob/4eaac889809dd489e3bae468c45d5c0e02b9147a/LICENSE.txt) · [Paper](https://arxiv.org/abs/2502.14234)

The label uses the primary `Ionic conductivity (S cm-1)` column. An exact value equal to or above the threshold is fast. A strict upper bound at or below the threshold is slow. An inclusive upper bound is slow only when strictly below the threshold. For example, `<1E-10` implies slow without assigning an exact conductivity. Bounds that could span both classes and blank measurements halt training so that every official row is accounted for. All 37 upper bounds in this revision imply slow.

| Split | Samples | Slow | Fast |
| --- | ---: | ---: | ---: |
| Train | 478 | 260 (54.4%) | 218 (45.6%) |
| Test | 121 | 68 (56.2%) | 53 (43.8%) |

The [data audit](reports/data-audit.md) records units, all 28 columns’ missing-value counts, censoring, repeated compositions, cross-split overlap, and the 254 train and 67 test CIFs matched to CSV rows.

## Features

The CSV allowlist is `True Composition`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, and `Space group #`. These eight fields are populated across both splits. CIF prediction obtains unit-cell composition and lattice parameters from the parsed structure and infers its space group with a symmetry tolerance of 0.01 Å and angle tolerance of 5°.

| Derived feature | Definition |
| --- | --- |
| Elemental fractions | 118 atomic fractions in atomic-number order, including Li. Absent elements have zero fraction. |
| Element count | Number of elements present. |
| Atomic-number summaries | Composition-weighted mean and population standard deviation. |
| Atomic-mass summaries | Composition-weighted mean and population standard deviation in amu. |
| Cell volume per atom | Volume from the cell lengths in Å and angles in degrees, divided by the unit-cell atom count in `True Composition` or the CIF. Units are Å³ per atom. |
| Crystal system | Seven one-hot features derived from the international space group number. |

Conductivity fields, DOI, IDs, notes, CIF presence, and all other curation metadata are excluded by the allowlist. The benchmark derives descriptors from complete CSV metadata so every official row enters the comparison, including rows without a CIF.

## Split and model selection

The official CSVs retain their original order and membership. Five-fold stratified group cross-validation runs only on the training rows with seed 42. Groups use elemental fractions from `True Composition` rounded to eight decimal places, so equivalent compositions stay in the same validation fold. The featurizer is deterministic and LogisticRegression fits its scaler inside each fold. The tree models use the same unscaled features.

Selection maximizes mean positive-class F1. An exact tie favors LogisticRegression, then RandomForestClassifier, then HistGradientBoostingClassifier. After selection, each candidate and the training-majority baseline fit on all 478 training rows and are scored once on the same 121 official test rows. The reported results use the classifiers’ default decision rules.

## Measured results

The confusion matrices use true classes as rows and predicted classes as columns, both ordered slow, fast. Precision, recall, and F1 refer to the fast class. The majority baseline predicts slow for every row, with precision set to zero because it predicts no positives.

| Model | Train CV F1 | Test accuracy | Test precision | Test recall | Test F1 | Test confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LogisticRegression | 0.847 | 0.653 | 0.587 | 0.698 | 0.638 | `[[42, 26], [16, 37]]` |
| **RandomForestClassifier (selected)** | **0.880** | **0.793** | **0.780** | **0.736** | **0.757** | `[[57, 11], [14, 39]]` |
| HistGradientBoostingClassifier | 0.863 | 0.736 | 0.698 | 0.698 | 0.698 | `[[52, 16], [16, 37]]` |
| Majority baseline | Fixed rule | 0.562 | 0.000 | 0.000 | 0.000 | `[[68, 0], [53, 0]]` |

The [benchmark report](reports/benchmark.json) contains full-precision scores and per-fold F1.

## Upload verification

Reproduce the saved model’s CIF evaluation and CSV overlap sensitivity check:

```powershell
.\.venv\Scripts\python.exe -m electrolyte_screening verify-uploads --report reports/upload-verification.json
```

All 67 archived test CIFs were scored using the same byte-based prediction function as the CLI. Of these, 47 have slow labels and 20 have fast labels. The paired CSV comparison uses those exact 67 rows and the same fitted model.

| RandomForestClassifier input | Rows | Accuracy | Precision | Recall | F1 | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Archived CIFs** | **67** | **0.836** | **0.909** | **0.500** | **0.645** | `[[46, 1], [10, 10]]` |
| CSV metadata for the same 67 rows | 67 | 0.851 | 0.917 | 0.550 | 0.688 | `[[46, 1], [9, 11]]` |

Across these pairs, 34 compositions differed by more than 0.0001 in at least one atomic fraction or by element presence. Cell volume per atom differed by more than 1% for `4xq`, `24k`, `9o6`, and `duf`. The largest difference was `24k`, which reports 92 atoms in the CSV unit cell and has 46 occupancy-weighted atoms in the archived CIF at the same cell volume. For `um1`, the reported space group was 167 and the inferred space group was 148. Both resolve to the trigonal crystal system, and all 67 crystal systems matched. The atomic-number or mass summaries differed for 35 rows, the number of elements matched in every row, and 36 rows differed on at least one model feature under the recorded tolerances.

`duf` was the sole prediction change, from fast using the CSV to slow using the CIF. OBELiX flags the archived structure as a close match, and it had the largest atomic-fraction difference in the paired comparison. The parser retained partial occupancies in 54 files and reported fractional-coordinate rounding in four. The [verification report](reports/upload-verification.json) records per-entry features, mismatches, and predictions.

The following pairs have equal normalized atomic fractions. The two overlapping test rows are both fast, and the saved model classified both correctly from CSV metadata.

| Train ID and formula | Test ID and formula |
| --- | --- |
| `52b` · Li10.05Ge1.05P1.95S12 | `2yl` · Li3.35Ge0.35P0.65S4 |
| `8po` · Li10.2Ge1.2P1.8S12 | `mvq` · Li3.4Ge0.4P0.6S4 |

The sensitivity check removes those two test rows only from scoring. It uses the saved model and rebuilds the other three models with the same fixed settings on the full official train set. The 121-row scores reproduce the original benchmark exactly.

| Model | Official accuracy (121) | Accuracy excluding overlap (119) | Official F1 (121) | F1 excluding overlap (119) |
| --- | ---: | ---: | ---: | ---: |
| LogisticRegression | 0.653 | 0.664 | 0.638 | 0.649 |
| RandomForestClassifier | 0.793 | 0.790 | 0.757 | 0.747 |
| HistGradientBoostingClassifier | 0.736 | 0.739 | 0.698 | 0.699 |
| Majority baseline | 0.562 | 0.571 | 0.000 | 0.000 |

The upload fixtures include an idealized LiI cell absent from training (class 0, probability 0.347), a lithium-free structure (rejected), invalid CIF text (rejected), and a half-occupied lithium site (class 0, probability 0.203). Tests confirm that changing a block name or formula header leaves the structural features and prediction identical. Changing the actual occupancy changes the features.

## Training coverage warning

The profile flags elements or crystal systems absent from training, cell volume per atom outside the training range (8.62–45.37 Å³ per atom), or a large composition distance. Composition distance is half the sum of the absolute differences in elemental fractions to the nearest training composition. The cutoff is the training 95th percentile of nearest-neighbor distances among distinct compositions, with each composition excluded from its own comparison. The cutoff was 0.20, and 14 archived test CIFs exceeded it.

The warning is an applicability heuristic. A new composition can still be within the observed coverage, as with the LiI fixture. The fast-class probability is a model estimate and calibration remains unmeasured. Older artifacts report `coverage_profile_missing`. The current train command builds the profile.

## Limitations

- Two normalized compositions occur in both official splits. Fourteen training composition groups and one test group contain both labels. The official test score measures performance on this exact split, and cross-validation groups compositions only, so related chemistries can still appear across folds.
- The archived test CIFs cover 67 of the 121 test rows, with 20 fast conductors in the subset. Some archived structures are close matches for the samples with measured conductivity. CIF-based results use the corresponding CSV conductivity labels and measure performance on this subset.
- Experimental conductivity can vary with sample preparation, measurement protocol, and bulk versus total reporting. The descriptors omit local coordination, defects, and grain boundaries. Performance on prospective materials remains unmeasured.
