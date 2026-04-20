// ===== Global State =====
let allData = [];
let currentModel = null;
let currentIteration = 0; // 0 is the latest run
let availableModels = [];
let iterationsByModel = {}; // { modelName: maxIterations }

let availableFinetunes = [];
let currentFinetune = null;

let charts = {};
let tableData = [];
let currentSort = { column: null, direction: 'asc' };

const API_BASE = '/api';

document.addEventListener('DOMContentLoaded', () => {
    loadData();
});

function switchTab(event, tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    } else {
        document.querySelector(`[onclick*="${tabId}"]`)?.classList.add('active');
    }

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });

    const targetId = tabId === 'dashboard' ? 'dashboard' : 'trainDashboard';
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
        targetEl.classList.add('active');
        targetEl.style.display = 'block';
    }

    document.getElementById('errorSection').style.display = 'none';

    if (tabId === 'train' && availableFinetunes.length === 0) {
        loadFinetuneList();
    }
}

async function loadData() {
    try {
        showLoading();
        const response = await fetch(`${API_BASE}/data`);

        if (!response.ok) {
            document.getElementById('dashboard').innerHTML = `
                <div style="text-align:center; padding: 4rem; margin-top: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
                    <h2 style="color: var(--text-primary)">No Evaluation Data Found</h2>
                    <p style="color: var(--text-muted)">Could not find valid CSV data. You can still check Training Logs.</p>
                </div>`;
            hideLoading();
            return;
        }

        const result = await response.json();
        allData = result.data || [];

        // Group data to determine iterations per model
        iterationsByModel = {};
        const models = [...new Set(allData.map(row => row.model))];

        models.forEach(model => {
            const modelRows = allData.filter(row => row.model === model);
            const groups = {};
            modelRows.forEach(row => {
                const key = `${row.task_key}|${row.quant_method}`;
                if (!groups[key]) groups[key] = 0;
                groups[key]++;
            });

            // Iterations for this model is the max number of times any (task, quant) was run
            const counts = Object.values(groups);
            iterationsByModel[model] = counts.length > 0 ? Math.max(...counts) : 0;
        });

        availableModels = models.sort();

        if (availableModels.length === 0) {
            document.getElementById('dashboard').innerHTML = `
                <div style="text-align:center; padding: 4rem; margin-top: 2rem;">
                    <h2 style="color: var(--text-primary)">No Models Extracted</h2>
                    <p style="color: var(--text-muted)">The CSV files were found but no models could be identified.</p>
                </div>`;
            hideLoading();
            return;
        }

        currentModel = availableModels[0];
        initializeDashboard();
        hideLoading();

    } catch (error) {
        console.error('Error loading data:', error);
        showError(`Network Error: Ensure Flask API is running (${error.message})`);
    }
}

async function loadFinetuneList() {
    try {
        const response = await fetch(`${API_BASE}/finetunes`);
        const result = await response.json();

        if (result.success && result.files && result.files.length > 0) {
            availableFinetunes = result.files;
            currentFinetune = availableFinetunes[0];
            createFinetuneToggle();
            loadTrainData();
        } else {
            document.getElementById('trainDashboardContent').style.display = 'none';
            document.getElementById('trainDashboard').innerHTML += `
                <div id="noFinetuneMsg" style="text-align:center; padding: 4rem; margin-top: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📂</div>
                    <h2 style="color: var(--text-primary)">No Training Logs Found</h2>
                    <p style="color: var(--text-muted)">Place your .out or .log files inside the <strong>finetunes</strong> folder to view them here.</p>
                </div>`;
        }
    } catch (error) {
        console.error("Failed to load finetune list", error);
    }
}

function createFinetuneToggle() {
    const container = document.getElementById('finetuneToggle');
    if (!container) return;
    container.innerHTML = '';

    availableFinetunes.forEach(file => {
        const button = document.createElement('button');
        button.className = 'model-toggle-btn';
        button.textContent = file;

        if (file === currentFinetune) button.classList.add('active');

        button.addEventListener('click', () => {
            currentFinetune = file;
            document.querySelectorAll('#finetuneToggle .model-toggle-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            loadTrainData();
        });

        container.appendChild(button);
    });
}

async function loadTrainData() {
    if (!currentFinetune) return;

    try {
        const response = await fetch(`${API_BASE}/train_logs?file=${encodeURIComponent(currentFinetune)}`);
        const result = await response.json();

        if (!result.success) {
            document.getElementById('logTerminal').innerHTML = `<span class="error">Error: ${result.message}</span>`;
            return;
        }

        document.getElementById('logTerminal').textContent = result.raw_text;
        document.getElementById('currentJobStatus').textContent = `Log: ${currentFinetune}`;

        const trainData = result.train_metrics.map(d => ({ x: d.epoch, y: d.loss }));
        const evalData = result.eval_metrics.map(d => ({ x: d.epoch, y: d.eval_loss }));

        if (evalData.length > 0) {
            let bestEval = evalData[0];
            for (let d of evalData) {
                if (d.y < bestEval.y) bestEval = d;
            }
            document.getElementById('bestEvalValue').innerHTML = `Epoch ${bestEval.x} <br><span style="font-size:1rem;color:var(--text-muted)">Loss: ${bestEval.y.toFixed(4)}</span>`;
        } else {
            document.getElementById('bestEvalValue').textContent = `No Eval Data`;
        }

        renderTrainChart(trainData, evalData);
        fetchCheckpoints();

    } catch (error) {
        console.error('Error loading training data:', error);
        document.getElementById('logTerminal').innerHTML = `<span class="error">Failed to fetch training logs: ${error.message}</span>`;
    }
}

async function fetchCheckpoints() {
    try {
        const res = await fetch(`${API_BASE}/checkpoints`);
        const data = await res.json();
        const select = document.getElementById('checkpointSelect');
        select.innerHTML = '';

        if (data.success && data.checkpoints.length > 0) {
            data.checkpoints.forEach(cp => {
                const option = document.createElement('option');
                option.value = cp;
                option.textContent = cp;
                // Auto-select the final adapter if it is in the list
                if (cp.endsWith('final_adapter')) option.selected = true;
                select.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "No Checkpoints Found";
            select.appendChild(option);
        }
    } catch (err) {
        console.error("Failed to load checkpoints", err);
    }
}

function downloadSelectedCheckpoint() {
    const folderPath = document.getElementById('checkpointSelect').value;
    if (!folderPath) {
        alert("No checkpoint selected to download.");
        return;
    }
    // FIX: Added 'download_model' to the path so it hits the correct Flask route
    window.location.href = `${API_BASE}/download_model?folder=${encodeURIComponent(folderPath)}`;
}
// ===== UI State Helpers =====
function showLoading() {
    document.getElementById('loadingSection').style.display = 'flex';
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('tabsContainer').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('tabsContainer').style.display = 'flex';
    document.getElementById('dashboard').style.display = 'block';
}

function showError(message) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('tabsContainer').style.display = 'flex';
    document.getElementById('errorSection').style.display = 'flex';
    document.getElementById('errorMessage').textContent = message;
}

// ===== Dashboard Initialization =====
function initializeDashboard() {
    createModelToggle();
    currentIteration = 0;
    updateDashboard();
    setupEventListeners();
}

function createModelToggle() {
    const container = document.getElementById('modelToggle');
    if (!container) return;
    container.innerHTML = '';

    availableModels.forEach(model => {
        const button = document.createElement('button');
        button.className = 'model-toggle-btn';
        button.textContent = model;
        button.dataset.model = model;

        if (model === currentModel) button.classList.add('active');

        button.addEventListener('click', () => {
            currentModel = model;
            currentIteration = 0;
            document.querySelectorAll('#modelToggle .model-toggle-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            updateDashboard();
        });

        container.appendChild(button);
    });
}

function createRunToggle() {
    const container = document.getElementById('runToggle');
    const section = document.getElementById('runSelectorSection');
    if (!container || !section) return;

    const iterations = iterationsByModel[currentModel] || 0;

    if (iterations <= 1) {
        section.style.display = 'none';
        return;
    }

    section.style.display = 'block';
    container.innerHTML = '';

    // Create buttons for each iteration
    for (let i = 0; i < iterations; i++) {
        const button = document.createElement('button');
        button.className = 'model-toggle-btn';
        button.textContent = `Evaluation Run ${i + 1}`;

        if (i === currentIteration) button.classList.add('active');

        button.addEventListener('click', () => {
            currentIteration = i;
            document.querySelectorAll('#runToggle .model-toggle-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            updateDashboard();
        });

        container.appendChild(button);
    }
}

function setupEventListeners() {
    const searchInput = document.getElementById('searchInput');
    const quantFilter = document.getElementById('quantFilter');

    if (searchInput) searchInput.addEventListener('input', () => updateTable());
    if (quantFilter) quantFilter.addEventListener('change', () => updateTable());

    document.querySelectorAll('.data-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sortTable(th.dataset.sort));
    });
}

function aggregateMMLU(data) {
    const aggregated = [];
    const mmluData = { base: [], int8: [], int4: [] };
    const nonMmluData = [];

    data.forEach(row => {
        const task = row.task_key;
        if (task.startsWith('mmlu_') && task !== 'mmlu') {
            const quant = row.quant_method;
            const value = parseFloat(row.primary_value);
            if (!isNaN(value) && mmluData[quant]) mmluData[quant].push(value);
        } else {
            nonMmluData.push(row);
        }
    });

    ['base', 'int8', 'int4'].forEach(quant => {
        if (mmluData[quant].length > 0) {
            const avgValue = mmluData[quant].reduce((a, b) => a + b, 0) / mmluData[quant].length;
            aggregated.push({
                model: data[0]?.model || '',
                quant_method: quant,
                task_key: 'MMLU',
                primary_value: avgValue,
                primary_metric: 'acc'
            });
        }
    });

    return [...nonMmluData, ...aggregated];
}

function updateDashboard() {
    if (!currentModel) return;

    createRunToggle();

    // Group items for this model by (task, quant) and sort by source_file (timestamp)
    const modelData = allData.filter(row => row.model === currentModel);
    const grouped = {};
    modelData.forEach(row => {
        const key = `${row.task_key}|${row.quant_method}`;
        if (!grouped[key]) grouped[key] = [];
        grouped[key].push(row);
    });

    // Sort each group and pick the one for currentIteration
    const filteredRows = [];
    Object.values(grouped).forEach(rows => {
        // Sort descending by filename timestamp (latest first)
        rows.sort((a, b) => (b.source_file || "").localeCompare(a.source_file || ""));

        // Pick the N-th iteration, or the last available one if N is out of bounds
        const item = rows[currentIteration] || rows[0];
        if (item) filteredRows.push(item);
    });

    const aggregatedData = aggregateMMLU(filteredRows);

    updateCharts(aggregatedData);
    updateTable(aggregatedData);
    populateFilters();
}

// ===== Charts =====
function updateCharts(modelData) {
    renderQuantizationImpactChart(modelData);
    renderSensitivityChart(modelData, 'int8', 'sensitivityInt8Chart', 'INT8');
    renderSensitivityChart(modelData, 'int4', 'sensitivityInt4Chart', 'INT4');
}

function renderTrainChart(trainData, evalData) {
    const ctx = document.getElementById('trainLossChart');
    if (!ctx) return;

    if (charts.trainLoss) charts.trainLoss.destroy();

    charts.trainLoss = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Training Loss',
                    data: trainData,
                    borderColor: 'rgba(79, 172, 254, 1)',
                    backgroundColor: 'rgba(79, 172, 254, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: 'Evaluation Loss',
                    data: evalData,
                    borderColor: 'rgba(240, 147, 251, 1)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 6,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#b8b8d1' } } },
            scales: {
                x: { type: 'linear', title: { display: true, text: 'Epoch', color: '#b8b8d1' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b8b8d1' } },
                y: { title: { display: true, text: 'Loss', color: '#b8b8d1' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b8b8d1' } }
            }
        }
    });
}

function renderQuantizationImpactChart(modelData) {
    const ctx = document.getElementById('quantizationImpactChart');
    if (!ctx) return;

    const taskData = {};
    modelData.forEach(row => {
        const task = row.task_key;
        const quant = row.quant_method;
        const value = parseFloat(row.primary_value);
        if (!taskData[task]) taskData[task] = {};
        taskData[task][quant] = value;
    });

    const tasks = Object.entries(taskData)
        .filter(([_, quants]) => quants.base)
        .sort((a, b) => (b[1].base || 0) - (a[1].base || 0))
        .slice(0, 10)
        .map(([task]) => task);

    const datasets = [
        { label: 'Base', data: tasks.map(task => taskData[task].base || 0), backgroundColor: 'rgba(67, 233, 123, 0.8)', borderColor: 'rgba(67, 233, 123, 1)', borderWidth: 2, borderRadius: 8 },
        { label: 'INT8', data: tasks.map(task => taskData[task].int8 || 0), backgroundColor: 'rgba(79, 172, 254, 0.8)', borderColor: 'rgba(79, 172, 254, 1)', borderWidth: 2, borderRadius: 8 },
        { label: 'INT4', data: tasks.map(task => taskData[task].int4 || 0), backgroundColor: 'rgba(240, 147, 251, 0.8)', borderColor: 'rgba(240, 147, 251, 1)', borderWidth: 2, borderRadius: 8 }
    ];

    if (charts.quantizationImpact) charts.quantizationImpact.destroy();

    charts.quantizationImpact = new Chart(ctx, {
        type: 'bar',
        data: { labels: tasks, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { labels: { color: '#b8b8d1', padding: 20 } },
                tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%` } }
            },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#b8b8d1', callback: (value) => (value * 100).toFixed(0) + '%' } },
                x: { grid: { display: false }, ticks: { color: '#b8b8d1', maxRotation: 45, minRotation: 45 } }
            }
        }
    });
}

function renderSensitivityChart(modelData, quantMethod, elementId, label) {
    const ctx = document.getElementById(elementId);
    if (!ctx) return;

    const taskData = {};
    modelData.forEach(row => {
        const task = row.task_key;
        const quant = row.quant_method;
        const value = parseFloat(row.primary_value);
        if (!taskData[task]) taskData[task] = {};
        taskData[task][quant] = value;
    });

    const sensitivities = [];
    Object.entries(taskData).forEach(([task, quants]) => {
        const base = quants.base || 0;
        const quantized = quants[quantMethod];
        if (base > 0 && quantized) {
            sensitivities.push({ task, drop: ((base - quantized) / base) * 100 });
        }
    });

    sensitivities.sort((a, b) => b.drop - a.drop);
    const top = sensitivities.slice(0, 8);

    if (charts[elementId]) charts[elementId].destroy();

    const colors = quantMethod === 'int8' ? {
        high: 'rgba(79, 172, 254, 0.8)', mid: 'rgba(79, 172, 254, 0.6)', low: 'rgba(79, 172, 254, 0.4)', border: 'rgba(79, 172, 254, 1)'
    } : {
        high: 'rgba(245, 101, 101, 0.8)', mid: 'rgba(251, 191, 36, 0.8)', low: 'rgba(79, 172, 254, 0.8)', borderHigh: 'rgba(245, 101, 101, 1)', borderMid: 'rgba(251, 191, 36, 1)', borderLow: 'rgba(79, 172, 254, 1)'
    };

    charts[elementId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top.map(s => s.task),
            datasets: [{
                label: `${label} Drop (%)`,
                data: top.map(s => s.drop),
                backgroundColor: top.map(s => quantMethod === 'int8' ? (s.drop > 5 ? colors.high : (s.drop > 2 ? colors.mid : colors.low)) : (s.drop > 15 ? colors.high : (s.drop > 10 ? colors.mid : colors.low))),
                borderColor: top.map(s => quantMethod === 'int8' ? colors.border : (s.drop > 15 ? colors.borderHigh : (s.drop > 10 ? colors.borderMid : colors.borderLow))),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            indexAxis: 'y', responsive: true, maintainAspectRatio: true,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => `Drop: ${context.parsed.x.toFixed(2)}%` } } },
            scales: { x: { ticks: { color: '#b8b8d1', callback: (value) => value + '%' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }, y: { ticks: { color: '#b8b8d1' }, grid: { display: false } } }
        }
    });
}

function updateTable(modelData) {
    if (!modelData) {
        modelData = allData.filter(row => row.model === currentModel);
        modelData = aggregateMMLU(modelData);
    }

    const taskData = {};
    modelData.forEach(row => {
        const task = row.task_key;
        const quant = row.quant_method;
        if (!taskData[task]) taskData[task] = { task_key: task };
        taskData[task][quant] = parseFloat(row.primary_value);
    });

    tableData = Object.values(taskData).map(row => {
        const base = row.base || 0;
        const int8 = row.int8 || 0;
        const int4 = row.int4 || 0;
        return {
            task_key: row.task_key, base: base, int8: int8, int8_drop: base > 0 ? ((base - int8) / base) * 100 : 0, int4: int4, int4_drop: base > 0 ? ((base - int4) / base) * 100 : 0
        };
    });

    const searchTerm = document.getElementById('searchInput')?.value.toLowerCase() || '';
    const quantFilter = document.getElementById('quantFilter')?.value || '';

    let filtered = tableData.filter(row => {
        return (!searchTerm || row.task_key.toLowerCase().includes(searchTerm)) && (!quantFilter || row[quantFilter] > 0);
    });

    renderTable(filtered);
}

function renderTable(data) {
    const tbody = document.getElementById('tableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    data.forEach(row => {
        const tr = document.createElement('tr');
        const int8Class = row.int8_drop > 10 ? 'drop-high' : row.int8_drop > 5 ? 'drop-medium' : 'drop-low';
        const int4Class = row.int4_drop > 10 ? 'drop-high' : row.int4_drop > 5 ? 'drop-medium' : 'drop-low';

        tr.innerHTML = `
            <td><strong>${row.task_key}</strong></td>
            <td>${formatPercentage(row.base)}</td>
            <td>${formatPercentage(row.int8)}</td>
            <td class="${int8Class}">${row.int8_drop.toFixed(2)}%</td>
            <td>${formatPercentage(row.int4)}</td>
            <td class="${int4Class}">${row.int4_drop.toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
    });

    document.getElementById('rowCount').textContent = `Showing ${data.length} tasks`;
}

function sortTable(column) {
    currentSort.direction = (currentSort.column === column && currentSort.direction === 'asc') ? 'desc' : 'asc';
    currentSort.column = column;

    tableData.sort((a, b) => {
        const aVal = a[column], bVal = b[column];
        if (typeof aVal === 'number' && typeof bVal === 'number') {
            return currentSort.direction === 'asc' ? aVal - bVal : bVal - aVal;
        }
        return currentSort.direction === 'asc' ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal));
    });
    updateTable();
}

function populateFilters() {
    const quantFilter = document.getElementById('quantFilter');
    if (!quantFilter || quantFilter.options.length > 1) return;

    ['base', 'int8', 'int4'].forEach(q => {
        const option = document.createElement('option');
        option.value = q;
        option.textContent = q.toUpperCase();
        quantFilter.appendChild(option);
    });
}

function formatPercentage(value) {
    return (value === null || value === undefined || isNaN(value)) ? '-' : (value * 100).toFixed(2) + '%';
}