const API_BASE = '/api';

// UI Elements
const btnLoad = document.getElementById('btn-load');
const btnTrain = document.getElementById('btn-train');
const btnEval = document.getElementById('btn-eval');
const logConsole = document.getElementById('log-console');
const pathDisplay = document.getElementById('load-path');

// Hyperparameters
const gammaInput = document.getElementById('gamma');
const nuInput = document.getElementById('nu');
const gammaVal = document.getElementById('gamma-val');
const nuVal = document.getElementById('nu-val');

gammaInput.addEventListener('input', e => gammaVal.textContent = e.target.value);
nuInput.addEventListener('input', e => nuVal.textContent = e.target.value);

// Chart instances
let waveChart, fpChart, pcaChart;

function logMessage(msg) {
    const time = new Date().toLocaleTimeString();
    logConsole.textContent += `\n[${time}] ${msg}`;
    logConsole.scrollTop = logConsole.scrollHeight;
}

// 6-Second Labor Illusion Simulator
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
        pBar.style.width = `${(i / steps) * 100}%`;
        if (i === 10) lText.textContent = "Standardizing Matrix...";
        if (i === 30) lText.textContent = "Optimizing SVM Hyperplanes...";
        if (i === 50) lText.textContent = "Extracting Support Vectors...";
        await new Promise(r => setTimeout(r, interval));
    }
    
    lText.textContent = "Complete!";
    await new Promise(r => setTimeout(r, 300));
    overlay.classList.add('hidden');
}

// Extract Features
async function doExtract(type) {
    try {
        let target_dir = document.getElementById('target-dir').value.trim();
        
        if (type === 'abnormal') {
            // Automatically swap 'normal' to 'abnormal' in the path for convenience
            target_dir = target_dir.replace('normal', 'abnormal');
            document.getElementById('target-dir').value = target_dir;
        }

        if (!target_dir) {
            logMessage("Error: Path is empty.");
            return;
        }
        
        logMessage(`Starting extraction for ${type} from ${target_dir}...`);
        
        // SHOW LOADING MODAL
        const overlay = document.getElementById('loading-overlay');
        const pBar = document.getElementById('progress-bar');
        const lTitle = document.getElementById('loading-title');
        const lText = document.getElementById('loading-text');
        
        lTitle.textContent = "EXTRACTING FEATURES...";
        lText.textContent = `Reading .wav files from ${type} folder... (This may take 10~20 seconds)`;
        
        // Fake progress bar that fills over 15 seconds
        pBar.style.transition = 'none';
        pBar.style.width = '0%';
        overlay.classList.remove('hidden');
        
        setTimeout(() => {
            pBar.style.transition = 'width 15s cubic-bezier(0.1, 0.7, 1.0, 0.1)';
            pBar.style.width = '95%';
        }, 50);
        
        const formData = new FormData();
        formData.append("target_dir", target_dir);
        formData.append("type", type);
        
        const res = await fetch(`${API_BASE}/extract`, {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        
        // HIDE LOADING MODAL
        pBar.style.transition = 'none';
        pBar.style.width = '100%';
        setTimeout(() => overlay.classList.add('hidden'), 300);
        if (res.ok) {
            logMessage(`Success! Extracted ${data.count} items.`);
            if (type === 'normal') {
                btnTrain.disabled = false;
            } else {
                updateCharts();
            }
        } else {
            logMessage(`Error: ${data.error}`);
        }
    } catch (e) {
        logMessage(`Error: ${e.message}`);
    }
}

btnLoad.addEventListener('click', () => doExtract('normal'));
btnEval.addEventListener('click', () => doExtract('abnormal'));

// Train OCSVM
btnTrain.addEventListener('click', async () => {
    try {
        btnTrain.disabled = true;
        logMessage("Initiating training sequence...");
        
        // Parallel: show labor illusion on frontend, run training on backend
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
            logMessage("OCSVM Training Complete!");
            btnTrain.disabled = false;
            btnEval.disabled = false;
            updateCharts();
        } else {
            logMessage(`Training Error: ${data.error}`);
            btnTrain.disabled = false;
        }
    } catch (e) {
        logMessage(`Error: ${e.message}`);
        btnTrain.disabled = false;
    }
});

// Update Charts
async function updateCharts() {
    try {
        const res = await fetch(`${API_BASE}/charts`);
        const data = await res.json();
        if (!res.ok) return;
        
        drawWaveChart(data.wave_normal, data.wave_abnormal);
        drawFpChart(data.fp_mean, data.fp_std, data.fp_abnormal);
        drawPcaChart(data.pca_normal, data.pca_sv, data.pca_abnormal, data.contour_x, data.contour_y, data.contour_z);
    } catch (e) {
        logMessage(`Chart Error: ${e.message}`);
    }
}

// Chart.js Implementations
function drawWaveChart(normal, abnormal) {
    const ctx = document.getElementById('chart-wave').getContext('2d');
    if (waveChart) waveChart.destroy();
    
    const labels = Array.from({length: Math.max(normal.length, abnormal.length)}, (_, i) => i);
    
    const datasets = [];
    if (normal.length) datasets.push({
        label: 'Normal Wave',
        data: normal,
        borderColor: '#dddddd',
        borderWidth: 1,
        pointRadius: 0,
        tension: 0.1
    });
    if (abnormal.length) datasets.push({
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
    
    const labels = Array.from({length: mean.length}, (_, i) => i);
    const datasets = [
        {
            label: 'Normal Baseline',
            data: mean,
            borderColor: '#007aff', // Link color used as secondary accent
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

function drawPcaChart(normal, sv, abnormal, cx, cy, cz) {
    const ctx = document.getElementById('chart-pca').getContext('2d');
    if (pcaChart) pcaChart.destroy();
    
    const normalPoints = normal.map(p => ({x: p[0], y: p[1]}));
    const svPoints = sv.map(p => ({x: p[0], y: p[1]}));
    const abPoints = abnormal.map(p => ({x: p[0], y: p[1]}));
    
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
        },
        {
            label: 'Abnormal (Anomalies)',
            data: abPoints,
            backgroundColor: '#000000',
            pointStyle: 'crossRot',
            pointRadius: 6,
            borderWidth: 2
        }
    ];
    
    // Note: Chart.js doesn't natively support contour plots.
    // For a real production app, we would use Plotly.js for the PCA contour.
    // Here we just plot the scatter points as a representation.

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
