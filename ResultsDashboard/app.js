// ===== Global State =====
let allData = [];
let currentModel = null;
let currentIteration = 0; // 0 is the latest run
let availableModels = [];
let iterationsByModel = {}; // { modelName: maxIterations }

let detailedResults = {};
let currentDetailedModel = null;
let currentTestType = null;

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

    const targetId = tabId === 'dashboard' ? 'dashboard' : tabId === 'detailed' ? 'detailedDashboard' : 'trainDashboard';
    const targetEl = document.getElementById(targetId);
    if (targetEl) {
        targetEl.classList.add('active');
        targetEl.style.display = 'block';
    }

    document.getElementById('errorSection').style.display = 'none';

    if (tabId === 'detailed') {
        initDetailedDashboard();
    }

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

        const detailedResponse = await fetch(`${API_BASE}/detailed-results`);
        if (detailedResponse.ok) {
            const detailedResult = await detailedResponse.json();
            if (detailedResult.success) {
                detailedResults = detailedResult.data;
            }
        }

        const models = [...new Set(allData.map(row => row.model))];

        models.forEach(model => {
            const modelRows = allData.filter(row => row.model === model);
            const groups = {};
            modelRows.forEach(row => {
                const key = `${row.task_key}|${row.quant_method}`;
                if (!groups[key]) groups[key] = 0;
                groups[key]++;
            });

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
        currentDetailedModel = availableModels[0];
        initializeDashboard();
        hideLoading();

    } catch (error) {
        console.error('Error loading data:', error);
        showError(`Network Error: Ensure Flask API is running (${error.message})`);
    }
}

// ===== Detailed Question Analysis =====

const TEST_TYPE_LABELS = {
    'accuracy_custom': 'Q&A Accuracy',
    'coherence_custom': 'Coherence',
    'tool_calling': 'Tool Calling',
    'ocr_custom': 'OCR',
    'gsm8k': 'GSM8K',
    'gsm8k_tr': 'GSM8K (Trace Reasoning)',
    'arc_challenge': 'ARC Challenge',
    'truthfulqa_mc2': 'TruthfulQA',
    'hendrycks_math': 'Hendrycks Math',
    'wikitext': 'WikiText'
};

const LM_EVAL_TASKS = new Set(['gsm8k', 'gsm8k_tr', 'arc_challenge', 'truthfulqa_mc2', 'hendrycks_math', 'wikitext']);

let detailedPage = 1;
let detailedTotalPages = 1;
const detailedPageSize = 50;

function initDetailedDashboard() {
    createDetailedModelToggle();
    updateDetailedView();
}

function createDetailedModelToggle() {
    const container = document.getElementById('detailedModelToggle');
    if (!container) return;
    container.innerHTML = '';

    availableModels.forEach(model => {
        const button = document.createElement('button');
        button.className = 'model-toggle-btn';
        button.textContent = model;

        if (model === currentDetailedModel) button.classList.add('active');

        button.addEventListener('click', () => {
            currentDetailedModel = model;
            currentTestType = null;
            document.querySelectorAll('#detailedModelToggle .model-toggle-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            updateDetailedView();
        });

        container.appendChild(button);
    });
}

function updateDetailedView() {
    if (!currentDetailedModel) return;

    createTestTypeToggle();
    renderDetailedTable();
}

function createTestTypeToggle() {
    const container = document.getElementById('testTypeToggle');
    if (!container) return;
    container.innerHTML = '';

    const modelTests = detailedResults[currentDetailedModel] || {};
    const testTypes = Object.keys(modelTests);

    if (testTypes.length === 0) {
        container.innerHTML = '<p class="detailed-no-data"><span class="icon">📭</span>No detailed results available for this model. Run custom tests to see per-question data.</p>';
        currentTestType = null;
        return;
    }

    if (currentTestType === null || !testTypes.includes(currentTestType)) {
        currentTestType = testTypes[0];
    }

    testTypes.forEach(testType => {
        const button = document.createElement('button');
        button.className = 'model-toggle-btn';
        button.textContent = TEST_TYPE_LABELS[testType] || testType;

        if (testType === currentTestType) button.classList.add('active');

        button.addEventListener('click', () => {
            currentTestType = testType;
            document.querySelectorAll('#testTypeToggle .model-toggle-btn').forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            renderDetailedTable();
        });

        container.appendChild(button);
    });
}

function renderDetailedTable(resetPage) {
    const thead = document.getElementById('detailedTableHead');
    const tbody = document.getElementById('detailedTableBody');
    const rowcount = document.getElementById('detailedRowCount');

    if (!thead || !tbody) return;

    if (resetPage === undefined || resetPage) {
        detailedPage = 1;
    }

    if (!currentDetailedModel || !currentTestType) {
        thead.innerHTML = '';
        tbody.innerHTML = '<tr><td class="detailed-no-data" colspan="10">Select a test type above</td></tr>';
        rowcount.textContent = 'Showing 0 rows';
        updatePaginationControls(1);
        return;
    }

    const modelTests = detailedResults[currentDetailedModel] || {};
    const items = modelTests[currentTestType] || {};
    const itemKeys = Object.keys(items);

    if (itemKeys.length === 0) {
        thead.innerHTML = '';
        tbody.innerHTML = '<tr><td class="detailed-no-data" colspan="10">No items found</td></tr>';
        rowcount.textContent = 'Showing 0 rows';
        updatePaginationControls(1);
        return;
    }

    const { columns, rows } = buildDetailedRows(itemKeys, items, currentTestType);

    thead.innerHTML = `<tr>${columns.map(c => `<th>${c.label}</th>`).join('')}</tr>`;

    const searchTerm = document.getElementById('detailedSearchInput')?.value.toLowerCase() || '';
    const diffOnly = document.getElementById('diffOnlyFilter')?.checked || false;

    let filtered = rows.filter(row => {
        const matchesSearch = !searchTerm || row._searchText.toLowerCase().includes(searchTerm);
        const matchesDiff = !diffOnly || row._hasDiff;
        return matchesSearch && matchesDiff;
    });

    const totalPages = Math.max(1, Math.ceil(filtered.length / detailedPageSize));
    if (detailedPage > totalPages) detailedPage = totalPages;

    const startIdx = (detailedPage - 1) * detailedPageSize;
    const endIdx = startIdx + detailedPageSize;
    const paged = filtered.slice(startIdx, endIdx);

    tbody.innerHTML = '';
    paged.forEach(row => {
        const tr = document.createElement('tr');
        if (row._hasDiff) tr.classList.add('row-has-diff');

        tr.innerHTML = columns.map(col => {
            if (col.type === 'answer') {
                return `<td class="answer-cell" title="${escapeHtml(row[col.key] || 'No result')}">${escapeHtml(truncate(row[col.key] || 'No result', 80))}</td>`;
            }
            if (col.type === 'badge') {
                return `<td>${row[col.key]}</td>`;
            }
            if (col.type === 'metric') {
                return `<td><span class="detailed-metric">${escapeHtml(String(row[col.key] || '-'))}</span></td>`;
            }
            return `<td>${escapeHtml(String(row[col.key] || '-'))}</td>`;
        }).join('');

        tbody.appendChild(tr);
    });

    rowcount.textContent = `Showing ${startIdx + 1}–${Math.min(endIdx, filtered.length)} of ${filtered.length} items`;
    updatePaginationControls(totalPages);
}

function updatePaginationControls(totalPages) {
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const info = document.getElementById('pageInfo');
    detailedTotalPages = totalPages;
    if (!prevBtn || !nextBtn || !info) return;
    prevBtn.disabled = detailedPage <= 1;
    nextBtn.disabled = detailedPage >= totalPages;
    info.textContent = `Page ${detailedPage} of ${totalPages}`;
}

function detailedPrevPage() {
    if (detailedPage > 1) {
        detailedPage--;
        renderDetailedTable(false);
    }
}

function detailedNextPage() {
    if (detailedPage < detailedTotalPages) {
        detailedPage++;
        renderDetailedTable(false);
    }
}

function buildDetailedRows(itemKeys, items, testType) {
    const quants = ['base', 'int8', 'int4'];

    if (testType === 'accuracy_custom') {
        const columns = [
            { label: 'Question', key: '_question', type: 'text' },
            { label: 'Expected', key: '_expected', type: 'text' },
            { label: 'Base Answer', key: 'base_answer', type: 'answer' },
            { label: 'INT8 Answer', key: 'int8_answer', type: 'answer' },
            { label: 'INT4 Answer', key: 'int4_answer', type: 'answer' },
        ];

        const rows = itemKeys.map(itemKey => {
            const itemData = items[itemKey];
            const row = { _question: itemKey, _expected: '', _searchText: itemKey, _hasDiff: false };

            let hasAnswers = {};
            quants.forEach(q => {
                const qd = itemData[q];
                if (qd) {
                    row[`${q}_answer`] = qd.model_answer || '';
                    row[`_expected`] = qd.expected_answer || '';
                    hasAnswers[q] = true;
                    row._searchText += ' ' + (qd.model_answer || '') + ' ' + (qd.expected_answer || '');
                } else {
                    row[`${q}_answer`] = '';
                }
            });

            const answers = quants.filter(q => hasAnswers[q]).map(q => row[`${q}_answer`]);
            row._hasDiff = answers.length >= 2 && !allEqual(answers);

            const badgeCells = {};
            quants.forEach(q => {
                const qd = itemData[q];
                if (qd) {
                    if (qd.exact_match === true) {
                        badgeCells[`${q}_badge`] = `<span class="answer-badge badge-correct">✓ Exact</span>`;
                    } else if (qd.contains_match === true) {
                        badgeCells[`${q}_badge`] = `<span class="answer-badge badge-partial">~ Contains</span>`;
                    } else {
                        badgeCells[`${q}_badge`] = `<span class="answer-badge badge-wrong">✗ Wrong</span>`;
                    }
                }
            });

            Object.assign(row, badgeCells);
            return row;
        });

        return { columns, rows };
    }

    if (testType === 'coherence_custom') {
        const columns = [
            { label: 'Prompt ID', key: '_prompt_id', type: 'text' },
            { label: 'Domain', key: '_domain', type: 'text' },
            { label: 'Base Score', key: 'base_sim', type: 'metric' },
            { label: 'INT8 Score', key: 'int8_sim', type: 'metric' },
            { label: 'INT4 Score', key: 'int4_sim', type: 'metric' },
            { label: 'Base Preview', key: 'base_preview', type: 'answer' },
            { label: 'INT8 Preview', key: 'int8_preview', type: 'answer' },
            { label: 'INT4 Preview', key: 'int4_preview', type: 'answer' },
        ];

        const rows = itemKeys.map(itemKey => {
            const itemData = items[itemKey];
            const row = { _prompt_id: itemKey, _domain: '', _searchText: itemKey, _hasDiff: false };

            let sims = [];
            quants.forEach(q => {
                const qd = itemData[q];
                if (qd) {
                    row[`${q}_sim`] = qd.cosine_similarity !== undefined ? qd.cosine_similarity.toFixed(4) : '-';
                    row[`${q}_preview`] = qd.generated_preview || '';
                    row[`_domain`] = qd.domain || '';
                    sims.push(qd.cosine_similarity);
                    row._searchText += ' ' + (qd.generated_preview || '');
                } else {
                    row[`${q}_sim`] = '-';
                    row[`${q}_preview`] = '';
                }
            });

            const validSims = sims.filter(s => s !== undefined && s !== null);
            row._hasDiff = validSims.length >= 2 && !allEqual(validSims.map(s => parseFloat(s.toFixed(3))));

            return row;
        });

        return { columns, rows };
    }

    if (testType === 'tool_calling') {
        const columns = [
            { label: 'Prompt', key: '_prompt', type: 'text' },
            { label: 'Expected Tool', key: '_expected', type: 'text' },
            { label: 'Base Response', key: 'base_resp', type: 'answer' },
            { label: 'INT8 Response', key: 'int8_resp', type: 'answer' },
            { label: 'INT4 Response', key: 'int4_resp', type: 'answer' },
        ];

        const rows = itemKeys.map(itemKey => {
            const itemData = items[itemKey];
            const row = { _prompt: itemKey, _expected: '', _searchText: itemKey, _hasDiff: false };

            let responses = {};
            quants.forEach(q => {
                const qd = itemData[q];
                if (qd) {
                    row[`${q}_resp`] = qd.generated || '';
                    row[`_expected`] = qd.expected || '';
                    responses[q] = qd.generated || '';
                    row._searchText += ' ' + (qd.generated || '') + ' ' + (qd.expected || '');
                } else {
                    row[`${q}_resp`] = '';
                }
            });

            const respValues = Object.values(responses);
            row._hasDiff = respValues.length >= 2 && !allEqual(respValues);

            return row;
        });

        return { columns, rows };
    }

    if (testType === 'ocr_custom') {
        const columns = [
            { label: 'Image', key: '_image', type: 'text' },
            { label: 'Category', key: '_category', type: 'text' },
            { label: 'Question', key: '_question', type: 'text' },
            { label: 'Ground Truth', key: '_gt', type: 'text' },
            { label: 'Base Response', key: 'base_resp', type: 'answer' },
            { label: 'INT8 Response', key: 'int8_resp', type: 'answer' },
            { label: 'INT4 Response', key: 'int4_resp', type: 'answer' },
            { label: 'Base Sim', key: 'base_sim', type: 'metric' },
            { label: 'INT8 Sim', key: 'int8_sim', type: 'metric' },
            { label: 'INT4 Sim', key: 'int4_sim', type: 'metric' },
        ];

        const rows = itemKeys.map(itemKey => {
            const itemData = items[itemKey];
            const row = {
                _image: itemKey,
                _category: '',
                _question: '',
                _gt: '',
                _searchText: itemKey,
                _hasDiff: false
            };

            let responses = {};
            quants.forEach(q => {
                const qd = itemData[q];
                if (qd) {
                    row[`${q}_resp`] = qd.model_response || '';
                    row[`${q}_sim`] = qd.similarity !== undefined ? qd.similarity.toFixed(4) : '-';
                    row[`_category`] = qd.category || '';
                    row[`_question`] = qd.question || '';
                    row[`_gt`] = qd.ground_truth || '';
                    responses[q] = qd.model_response || '';
                    row._searchText += ' ' + (qd.model_response || '') + ' ' + (qd.ground_truth || '') + ' ' + (qd.question || '');
                } else {
                    row[`${q}_resp`] = '';
                    row[`${q}_sim`] = '-';
                }
            });

            const respValues = Object.values(responses);
            row._hasDiff = respValues.length >= 2 && !allEqual(respValues);

            return row;
        });

        return { columns, rows };
    }

    return { columns: [], rows: [] };
}

// ===== Helpers =====

function allEqual(arr) {
    if (arr.length < 2) return true;
    const first = arr[0];
    return arr.every(v => v === first);
}

function truncate(str, len) {
    if (str.length <= len) return str;
    return str.substring(0, len) + '...';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ===== Finetune Functions =====

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
    const detailedSearchInput = document.getElementById('detailedSearchInput');
    const diffOnlyFilter = document.getElementById('diffOnlyFilter');

    if (searchInput) searchInput.addEventListener('input', () => updateTable());
    if (quantFilter) quantFilter.addEventListener('change', () => updateTable());
    if (detailedSearchInput) detailedSearchInput.addEventListener('input', () => renderDetailedTable());
    if (diffOnlyFilter) diffOnlyFilter.addEventListener('change', () => renderDetailedTable());

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