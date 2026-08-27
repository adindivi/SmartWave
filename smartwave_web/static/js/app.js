/**
 * SmartWave Web Frontend - Production Hardened
 * All API calls, chart rendering, and UI state management.
 */
const API_BASE = '/api';

// UI Elements
const btnLoad = document.getElementById('btn-load');
const btnTrain = document.getElementById('btn-train');
const btnEval = document.getElementById('btn-eval');
const logConsole = document.getElementById('log-console');

// Hyperparameters
const gammaInput = document.getElementById('gamma');
const nuInput = document.getElementById('nu');
const gammaVal = document.getElementById('gamma-val');
const nuVal = document.getElementById('nu-val');

gammaInput.addEventListener('input', e => gammaVal.textContent = e.target.value);
nuInput.addEventListener('input', e => nuVal.textContent = e.target.value);

// Chart instances
let waveChart, fpChart, pcaChart;

// Global processing lock to prevent double-clicks
let isProcessing = false;

function logMessage(msg) {
    const time = new Date().toLocaleTimeString();
    logConsole.textContent += `\n[${time}] ${msg}`;
    logConsole.scrollTop = logConsole.scrollHeight;
}

function setAllButtonsDisabled(disabled) {
    btnLoad.disabled = disabled;
    btnTrain.disabled = disabled;
    btnEval.disabled = disabled;
}

// --- Loading Overlay Helpers ---
function showOverlay(title, text) {
    const overlay = document.getElementById('loading-overlay');
    const pBar = document.getElementById('progress-bar');
    const lTitle = document.getElementById('loading-title');
    const lText = document.getElementById('loading-text');

    lTitle.textContent = title;
    lText.textContent = text;
    pBar.style.transition = 'none';
    pBar.style.width = '0%';
    overlay.classList.remove('hidden');

    // Start a slow fill animation
    setTimeout(() => {
        pBar.style.transition = 'width 20s cubic-bezier(0.1, 0.7, 1.0, 0.1)';
        pBar.style.width = '95%';
    }, 50);
}

function hideOverlay() {
    const overlay = document.getElementById('loading-overlay');
    const pBar = document.getElementById('progress-bar');
    pBar.style.transition = 'width 0.3s ease';
    pBar.style.width = '100%';
    setTimeout(() => overlay.classList.add('hidden'), 400);
}

// 6-Second Labor Illusion for Training
async function showLaborIllusion(title, durationMs) {
    const overlay = document.getElementById('loading-overlay');
    const pBar = document.getElementById('progress-bar');
    const lTitle = document.getElementById('loading-title');
    const lText = document.getElementById('loading-text');

    lTitle.textContent = title;
    overlay.classList.remove('hidden');

    const steps = 60;
    const interval = durationMs / steps;

    for (let i = 0; i <= steps; i++) {
        pBar.style.transition = 'none';
        pBar.style.width = `${(i / steps) * 100}%`;
        if (i === 10) lText.textContent = "Step 1: Standardizing Data Matrix...";
        if (i === 30) lText.textContent = "Step 2: Optimizing SVM Hyperplanes...";
        if (i === 50) lText.textContent = "Step 3: Extracting Support Vectors...";
        if (i === 58) lText.textContent = "Finalizing boundary...";
        await new Promise(r => setTimeout(r, interval));
    }

    lText.textContent = "Complete!";
    await new Promise(r => setTimeout(r, 300));
    overlay.classList.add('hidden');
}

// --- Extract Features ---
async function doExtract(type) {
    if (isProcessing) return;
    isProcessing = true;
    setAllButtonsDisabled(true);

    try {
        let target_dir = document.getElementById('target-dir').value.trim();

        if (type === 'abnormal') {
            target_dir = target_dir.replace(/normal/g, 'abnormal');
            document.getElementById('target-dir').value = target_dir;
        }

        if (!target_dir) {
            logMessage("Error: Dataset path is empty.");
            return;
        }

        logMessage(`Extracting ${type} features from: ${target_dir}`);
        showOverlay("EXTRACTING FEATURES...", `Processing ${type} .wav files... (10~30 seconds)`);

        const formData = new FormData();
        formData.append("target_dir", target_dir);
        formData.append("data_type", type);

        const res = await fetch(`${API_BASE}/extract`, {
            method: 'POST',
            body: formData
        });

        const data = await res.json();

        if (res.ok) {
            logMessage(`Success! ${data.count} extracted, ${data.skipped} skipped.`);
            if (type === 'normal') {
                btnTrain.disabled = false;
            } else {
                updateCharts();
            }
        } else {
            logMessage(`Error: ${data.error}`);
        }
    } catch (e) {
        logMessage(`Network Error: ${e.message}`);
    } finally {
        hideOverlay();
        isProcessing = false;
        btnLoad.disabled = false;
        // Re-enable buttons that should be active
        if (btnTrain.dataset.wasEnabled === 'true') btnTrain.disabled = false;
        if (btnEval.dataset.wasEnabled === 'true') btnEval.disabled = false;
    }
}

btnLoad.addEventListener('click', () => doExtract('normal'));
btnEval.addEventListener('click', () => doExtract('abnormal'));

// --- Train OCSVM ---
btnTrain.addEventListener('click', async () => {
    if (isProcessing) return;
    isProcessing = true;
    setAllButtonsDisabled(true);

    try {
        logMessage("Initiating training sequence...");

        const trainPromise = fetch(`${API_BASE}/train`, {
            method: 'POST',
            body: new URLSearchParams({
                'gamma': gammaInput.value,
                'nu': nuInput.value
            })
        });

        await showLaborIllusion("🧠 TRAINING OCSVM MODEL", 6000);

        const res = await trainPromise;
        const data = await res.json();

        if (res.ok) {
            logMessage(`Training Complete! Support Vectors: ${data.support_vectors}`);
            btnEval.disabled = false;
            btnEval.dataset.wasEnabled = 'true';
            updateCharts();
        } else {
            logMessage(`Training Error: ${data.error}`);
        }
    } catch (e) {
        logMessage(`Network Error: ${e.message}`);
    } finally {
        isProcessing = false;
        btnLoad.disabled = false;
        btnTrain.disabled = false;
    }
});

// --- Update Charts ---
async function updateCharts() {
    try {
        logMessage("Loading chart data...");
        const res = await fetch(`${API_BASE}/charts`);
        const data = await res.json();
        if (!res.ok) {
            logMessage(`Chart Error: ${data.error}`);
            return;
        }

        drawWaveChart(data.wave_normal, data.wave_abnormal);
        drawFpChart(data.fp_mean, data.fp_std, data.fp_abnormal);
        drawPcaChart(data.pca_normal, data.pca_sv, data.pca_abnormal);
        logMessage("Charts rendered successfully.");
    } catch (e) {
        logMessage(`Chart Error: ${e.message}`);
    }
}

// --- Chart.js Implementations ---
function drawWaveChart(normal, abnormal) {
    const ctx = document.getElementById('chart-wave').getContext('2d');
    if (waveChart) waveChart.destroy();

    const maxLen = Math.max(normal?.length || 0, abnormal?.length || 0);
    if (maxLen === 0) return;

    const labels = Array.from({length: maxLen}, (_, i) => i);

    const datasets = [];
    if (normal && normal.length) datasets.push({
        label: 'Normal Wave',
        data: normal,
        borderColor: '#707070',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1
    });
    if (abnormal && abnormal.length) datasets.push({
        label: 'Abnormal Wave',
        data: abnormal,
        borderColor: '#F43F5E',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1
    });

    waveChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { position: 'top' } },
            scales: { x: { display: false }, y: { min: -1, max: 1 } }
        }
    });
}

function drawFpChart(mean, std, abnormal) {
    const ctx = document.getElementById('chart-fp').getContext('2d');
    if (fpChart) fpChart.destroy();

    if (!mean || mean.length === 0) return;

    const labels = Array.from({length: mean.length}, (_, i) => i);
    const datasets = [
        {
            label: 'Normal Baseline',
            data: mean,
            borderColor: '#007aff',
            borderWidth: 2,
            pointRadius: 0
        }
    ];

    if (abnormal && abnormal.length) {
        datasets.push({
            label: 'Anomaly Signature',
            data: abnormal,
            borderColor: '#F43F5E',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0
        });
    }

    fpChart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: { x: { display: true }, y: { display: true } }
        }
    });
}

function drawPcaChart(normal, sv, abnormal) {
    const ctx = document.getElementById('chart-pca').getContext('2d');
    if (pcaChart) pcaChart.destroy();

    if (!normal || normal.length === 0) return;

    const normalPoints = normal.map(p => ({x: p[0], y: p[1]}));
    const svPoints = (sv || []).map(p => ({x: p[0], y: p[1]}));
    const abPoints = (abnormal || []).map(p => ({x: p[0], y: p[1]}));

    const datasets = [
        {
            label: 'Normal Data',
            data: normalPoints,
            backgroundColor: '#10B981',
            pointRadius: 4
        },
        {
            label: 'Support Vectors',
            data: svPoints,
            backgroundColor: '#F59E0B',
            borderColor: '#ffffff',
            borderWidth: 1,
            pointRadius: 6
        }
    ];

    if (abPoints.length > 0) {
        datasets.push({
            label: 'Abnormal (Anomalies)',
            data: abPoints,
            backgroundColor: '#000000',
            pointStyle: 'crossRot',
            pointRadius: 6,
            borderWidth: 2
        });
    }

    pcaChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            scales: { x: { display: true }, y: { display: true } }
        }
    });
}
