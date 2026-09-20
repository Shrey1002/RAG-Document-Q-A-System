const API_BASE = '';

function showTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    if (tabName === 'sources') refreshSources();
    if (tabName === 'stats') refreshStats();
}

document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => showTab(tab.dataset.tab));
});

function showLoading(el) {
    el.innerHTML = `
        <div class="loading-overlay">
            <div>
                <div class="loading"></div>
                <div class="loading-bar"><div class="loading-bar-fill"></div></div>
            </div>
            <span>Processing...</span>
        </div>
    `;
    el.classList.add('visible');
}

function showResult(el, html, isError = false) {
    el.innerHTML = html;
    el.classList.add('visible');
    if (isError) el.classList.add('error');
    else el.classList.remove('error');
}

function formatCitations(citations) {
    return citations.map(c => `
        <div class="citation">
            <span class="source">📄 ${c.source}</span>
            (Page ${c.page}, Chunk ${c.chunk_index})
            <span class="score">Score: ${c.relevance_score.toFixed(4)}</span>
        </div>
    `).join('');
}

// File upload
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
let selectedFiles = [];

uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

function handleFiles(files) {
    for (let f of files) {
        if (!selectedFiles.find(s => s.name === f.name)) {
            selectedFiles.push(f);
        }
    }
    updateFileList();
}

function updateFileList() {
    fileList.innerHTML = selectedFiles.map(f => `<span class="file-tag">${f.name}</span>`).join('');
}

// Ingest
document.getElementById('ingestBtn').addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        showResult(document.getElementById('ingestResult'), '<h3>⚠️ No files selected</h3>');
        return;
    }

    const btn = document.getElementById('ingestBtn');
    btn.disabled = true;
    const resultEl = document.getElementById('ingestResult');
    showLoading(resultEl);

    const formData = new FormData();
    for (let f of selectedFiles) {
        formData.append('files', f);
    }
    formData.append('chunking_strategy', document.getElementById('strategy').value);

    try {
        const res = await fetch(API_BASE + '/ingest-multiple', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        showResult(resultEl, `
            <h3>✅ Ingestion Complete</h3>
            <p><strong>Chunks created:</strong> ${data.chunks_created}</p>
            <p><strong>Sources indexed:</strong> ${data.sources_indexed}</p>
            <p><strong>Time:</strong> ${data.time_taken_ms}ms</p>
        `);
        selectedFiles = [];
        updateFileList();
    } catch (err) {
        showResult(resultEl, `<h3>❌ Error: ${err.message}</h3>`, true);
    }

    btn.disabled = false;
});

// Query
document.getElementById('queryBtn').addEventListener('click', async () => {
    const question = document.getElementById('question').value.trim();
    if (!question) {
        showResult(document.getElementById('queryResult'), '<h3>⚠️ Please enter a question</h3>');
        return;
    }

    const btn = document.getElementById('queryBtn');
    btn.disabled = true;
    const resultEl = document.getElementById('queryResult');
    showLoading(resultEl);

    try {
        const res = await fetch(API_BASE + '/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                top_k: parseInt(document.getElementById('topK').value),
                filter_source: document.getElementById('filterSource').value || null
            })
        });
        const data = await res.json();
        showResult(resultEl, `
            <h3>🤖 Answer</h3>
            <div class="answer-text">${data.answer}</div>
            <h3 style="margin-top:16px">📚 Citations (${data.citations.length})</h3>
            ${formatCitations(data.citations)}
            <p style="margin-top:12px; color:#94a3b8; font-size:0.8rem">
                ⏱️ Retrieval: ${data.retrieval_time_ms}ms | Generation: ${data.generation_time_ms}ms
            </p>
        `);
    } catch (err) {
        showResult(resultEl, `<h3>❌ Error: ${err.message}</h3>`, true);
    }

    btn.disabled = false;
});

// Sources
async function refreshSources() {
    try {
        const res = await fetch(API_BASE + '/sources');
        const data = await res.json();
        const list = document.getElementById('sourcesList');
        if (data.sources.length === 0) {
            list.innerHTML = '<li>No sources indexed yet</li>';
        } else {
            list.innerHTML = data.sources.map(s => `<li>${s}</li>`).join('');
        }
    } catch (err) {
        document.getElementById('sourcesList').innerHTML = `<li style="color:#f87171">Error: ${err.message}</li>`;
    }
}

document.getElementById('refreshSources').addEventListener('click', refreshSources);

// Stats
async function refreshStats() {
    try {
        const res = await fetch(API_BASE + '/stats');
        const data = await res.json();
        const grid = document.getElementById('statsGrid');
        grid.innerHTML = `
            <div class="stat-item"><div class="value">${data.total_chunks}</div><div class="label">Total Chunks</div></div>
            <div class="stat-item"><div class="value">${data.total_sources}</div><div class="label">Total Sources</div></div>
            <div class="stat-item"><div class="value">${data.embedding_model}</div><div class="label">Embedding Model</div></div>
            <div class="stat-item"><div class="value">${data.llm_model}</div><div class="label">LLM Model</div></div>
        `;
    } catch (err) {
        document.getElementById('statsGrid').innerHTML = `<div class="stat-item" style="grid-column:1/-1"><div class="value" style="color:#f87171">Error</div><div class="label">${err.message}</div></div>`;
    }
}

document.getElementById('refreshStats').addEventListener('click', refreshStats);