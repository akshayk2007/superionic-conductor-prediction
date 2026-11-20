const tabButtons = document.querySelectorAll('.tab-btn');
const panels = document.querySelectorAll('.panel');

function switchTab(targetId) {
  if (!document.getElementById(targetId)) return;
  tabButtons.forEach(button => {
    const active = button.dataset.target === targetId;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
  panels.forEach(panel => panel.classList.toggle('active', panel.id === targetId));
}

tabButtons.forEach(button => button.addEventListener('click', () => switchTab(button.dataset.target)));
document.getElementById('startScreening').addEventListener('click', () => switchTab('manual'));
document.querySelectorAll('[data-go]').forEach(button => button.addEventListener('click', () => switchTab(button.dataset.go)));
document.querySelectorAll('.back-manual').forEach(button => button.addEventListener('click', () => switchTab('manual')));
document.getElementById('brandHome').addEventListener('click', event => {event.preventDefault();switchTab('home');});

// --- Manual Form Handling ---
const manualForm = document.getElementById('manualForm');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('resultBox');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const formError = document.getElementById('formError');

manualForm.addEventListener('reset', () => {
  resultBox.hidden = true;
  resultPlaceholder.hidden = false;
  formError.hidden = true;
});

manualForm.addEventListener('submit', async event => {
  event.preventDefault();
  formError.hidden = true;
  const fields = ['LASD','SBI','AFC','LLSD','LLB'];
  const payload = Object.fromEntries(fields.map(id => [id, Number(document.getElementById(id).value)]));
  if (fields.some(id => document.getElementById(id).value.trim() === '' || !Number.isFinite(payload[id]))) {
    formError.textContent = 'Please enter a valid number for all five descriptors.';
    formError.hidden = false;
    return;
  }
  predictBtn.disabled = true;
  predictBtn.textContent = 'Running model…';
  try {
    const response = await fetch('/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    const raw = await response.text();
    let data;
    try { data = JSON.parse(raw); } catch { throw new Error('The server returned an invalid response.'); }
    if (!response.ok) {
      const detail = typeof data.detail === 'string' ? data.detail : 'Prediction failed. Please try again.';
      throw new Error(detail);
    }
    const probability = Number(data.probability);
    if (!Number.isFinite(probability) || probability < 0 || probability > 100 || typeof data.is_superionic !== 'boolean') {
      throw new Error('Unexpected prediction result.');
    }
    resultBox.className = `result ${data.is_superionic ? 'good' : 'poor'}`;
    document.getElementById('resultStatus').textContent = data.is_superionic ? 'PREDICTED SUPERIONIC' : 'PREDICTED NON-SUPERIONIC';
    document.getElementById('verdictText').textContent = data.verdict || (data.is_superionic ? 'Superionic conductor' : 'Poor conductor');
    document.getElementById('scoreText').textContent = `Superionic classification probability: ${probability.toFixed(2)}%`;
    document.getElementById('probabilityFill').style.width = `${probability}%`;
    document.getElementById('probabilityTrack').setAttribute('aria-valuenow', String(probability));
    resultPlaceholder.hidden = true;
    resultBox.hidden = false;
  } catch (error) {
    formError.textContent = error.message || 'Something went wrong. Check that the backend is running.';
    formError.hidden = false;
  } finally {
    predictBtn.disabled = false;
    predictBtn.innerHTML = 'Run prediction <span aria-hidden="true">→</span>';
  }
});

// --- CIF Form Handling ---
const cifForm = document.getElementById('cifForm');
const cifPredictBtn = document.getElementById('cifPredictBtn');
const cifResultBox = document.getElementById('cifResultBox');
const cifResultPlaceholder = document.getElementById('cifResultPlaceholder');
const cifError = document.getElementById('cifError');

cifForm.addEventListener('reset', () => {
  cifResultBox.hidden = true;
  cifResultPlaceholder.hidden = false;
  cifError.hidden = true;
});

cifForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    cifError.hidden = true;

    const fileInput = document.getElementById("cifFile");
    const file = fileInput.files[0];
    if (!file) {
        cifError.textContent = "Please select a CIF file first.";
        cifError.hidden = false;
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    cifPredictBtn.disabled = true;
    cifPredictBtn.textContent = "Extracting & running…";

    try {
        const response = await fetch("/predict-cif", {
            method: "POST",
            body: formData
        });

        const text = await response.text();
        let data;

        try {
            data = JSON.parse(text);
        } catch {
            throw new Error("Server returned an invalid response.");
        }

        if (!response.ok) {
            throw new Error(data.detail || "CIF analysis failed");
        }

        const probability = Number(data.prediction.probability);
        const isSuperionic = data.prediction.is_superionic;

        cifResultBox.className = `result ${isSuperionic ? 'good' : 'poor'}`;
        document.getElementById('cifResultStatus').textContent = isSuperionic ? 'PREDICTED SUPERIONIC' : 'PREDICTED NON-SUPERIONIC';
        document.getElementById('cifVerdictText').textContent = data.prediction.verdict;
        document.getElementById('cifScoreText').textContent = `Superionic classification probability: ${probability.toFixed(2)}%`;
        document.getElementById('cifProbabilityFill').style.width = `${probability}%`;
        document.getElementById('cifProbabilityTrack').setAttribute('aria-valuenow', String(probability));

        document.getElementById('cifMetaText').innerHTML = `
          <strong>Formula:</strong> ${data.formula} &nbsp;|&nbsp; <strong>Sites:</strong> ${data.number_of_sites}<br>
          <strong>Lattice:</strong> <em>a</em> = ${data.lattice_parameters.a} Å, <em>b</em> = ${data.lattice_parameters.b} Å, <em>c</em> = ${data.lattice_parameters.c} Å<br>
          <strong>Sendek Features:</strong><br>
          • LASD: ${data.descriptors.LASD.toFixed(3)} Å &nbsp;• SBI: ${data.descriptors.SBI.toFixed(3)} &nbsp;• AFC: ${data.descriptors.AFC.toFixed(3)}<br>
          • LLSD: ${data.descriptors.LLSD.toFixed(3)} Å &nbsp;• LLB: ${data.descriptors.LLB.toFixed(3)}
        `;

        cifResultPlaceholder.hidden = true;
        cifResultBox.hidden = false;

    } catch (error) {
        cifError.textContent = error.message || "Something went wrong during CIF screening.";
        cifError.hidden = false;
    } finally {
        cifPredictBtn.disabled = false;
        cifPredictBtn.innerHTML = 'Run prediction <span aria-hidden="true">→</span>';
    }
});

// --- CSV Batch Form Handling ---
const csvForm = document.getElementById('csvForm');
const csvPredictBtn = document.getElementById('csvPredictBtn');
const csvResultBox = document.getElementById('csvResultBox');
const csvResultPlaceholder = document.getElementById('csvResultPlaceholder');
const csvError = document.getElementById('csvError');

csvForm.addEventListener('reset', () => {
  csvResultBox.hidden = true;
  csvResultPlaceholder.hidden = false;
  csvError.hidden = true;
});

csvForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    csvError.hidden = true;

    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];
    if (!file) {
        csvError.textContent = "Please select a CSV file first.";
        csvError.hidden = false;
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    csvPredictBtn.disabled = true;
    csvPredictBtn.textContent = "Processing batch…";

    try {
        const response = await fetch("/predict-csv", {
            method: "POST",
            body: formData
        });

        const text = await response.text();
        let data;

        try {
            data = JSON.parse(text);
        } catch {
            throw new Error("Server returned an invalid response.");
        }

        if (!response.ok) {
            throw new Error(data.detail || "CSV batch prediction failed");
        }

        document.getElementById('csvSummaryText').innerHTML = `
          Total Rows: ${data.total_rows} &nbsp;|&nbsp; Successful: <span style="color: #15803d;">${data.successful_predictions}</span> &nbsp;|&nbsp; Failed: <span style="color: #b91c1c;">${data.failed_rows}</span>
        `;

        const tbody = document.getElementById('csvTableBody');
        tbody.innerHTML = '';
        data.results.forEach(item => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid #edf2f7';
            
            const badgeColor = item.is_superionic ? '#dcfce7; color: #15803d;' : (item.error ? '#fee2e2; color: #b91c1c;' : '#fef3c7; color: #92400e;');
            const probText = item.probability !== null ? `${item.probability}%` : 'N/A';
            
            tr.innerHTML = `
              <td style="padding: 8px 6px; font-weight: 700;">${item.material}</td>
              <td style="padding: 8px 6px;">${probText}</td>
              <td style="padding: 8px 6px;"><span style="padding: 3px 6px; border-radius: 4px; font-size: 10px; font-weight: 800; background: ${badgeColor}">${item.verdict}</span></td>
            `;
            tbody.appendChild(tr);
        });

        csvResultPlaceholder.hidden = true;
        csvResultBox.hidden = false;

    } catch (error) {
        csvError.textContent = error.message || "Something went wrong during CSV batch screening.";
        csvError.hidden = false;
    } finally {
        csvPredictBtn.disabled = false;
        csvPredictBtn.innerHTML = 'Run batch prediction <span aria-hidden="true">→</span>';
    }
});