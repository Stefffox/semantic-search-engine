"""
gui.py
------
Interface graphique web (Flask) pour le moteur de recherche sémantique.

Lancement :
    python gui.py

Puis ouvrir http://localhost:5000 dans votre navigateur.

Fonctionnalités :
    - Recherche sémantique (embeddings) ou lexicale (TF-IDF)
    - Indexation de fichiers via upload
    - Comparaison côte à côte des deux moteurs
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify

# ---------------------------------------------------------------------------
# Application Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# ---------------------------------------------------------------------------
# État global (chargé au démarrage)
# ---------------------------------------------------------------------------

_state: dict = {
    "index": None,
    "provider": None,
    "sem_engine": None,
    "tfidf_index": None,
    "error": None,
}


def _load_engines() -> None:
    """Charge l'index et les moteurs si un index existe sur disque."""
    try:
        from src.embedding.ollama_provider import OllamaEmbeddingProvider
        from src.indexer import VectorIndex
        from src.lexical_search import TFIDFIndex
        from src.search import SemanticSearch

        provider = OllamaEmbeddingProvider()
        _state["provider"] = provider

        try:
            index = VectorIndex.load()
            _state["index"] = index
            _state["sem_engine"] = SemanticSearch(index, provider)
            _state["tfidf_index"] = TFIDFIndex().build(index.metadata)
            _state["error"] = None
        except FileNotFoundError:
            _state["error"] = "no_index"
    except ConnectionError:
        _state["error"] = "ollama_down"
    except Exception as e:
        _state["error"] = str(e)


# ---------------------------------------------------------------------------
# Template HTML unique (tout-en-un)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moteur de recherche sémantique</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ===== Reset & Base ===== */
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --primary: #6C63FF;
            --primary-dark: #5A52E0;
            --primary-light: #8B83FF;
            --accent: #00D2FF;
            --accent-green: #00E89D;
            --bg: #0F1117;
            --bg-card: #1A1D2E;
            --bg-input: #232738;
            --border: rgba(108, 99, 255, 0.2);
            --border-hover: rgba(108, 99, 255, 0.5);
            --text: #E8EAF0;
            --text-muted: #8892A0;
            --text-dim: #5A6070;
            --danger: #FF6B6B;
            --warning: #FFBE5C;
            --radius: 12px;
            --radius-sm: 8px;
            --shadow: 0 4px 24px rgba(0,0,0,0.3);
            --transition: 0.2s ease;
        }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }

        /* ===== Layout ===== */
        .app-layout {
            display: grid;
            grid-template-columns: 320px 1fr;
            min-height: 100vh;
        }

        /* ===== Sidebar ===== */
        .sidebar {
            background: #141625;
            border-right: 1px solid var(--border);
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            overflow-y: auto;
        }

        .sidebar-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .sidebar-section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem;
        }

        /* ===== Stats ===== */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }

        .stat-box {
            background: rgba(108, 99, 255, 0.08);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.8rem;
            text-align: center;
        }

        .stat-value {
            font-size: 1.6rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-light), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .stat-label {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .source-list {
            list-style: none;
            font-size: 0.85rem;
        }
        .source-list li {
            padding: 0.3rem 0;
            color: var(--text-muted);
        }
        .source-list li::before {
            content: '📄 ';
        }

        /* ===== File Upload ===== */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: var(--primary-light);
            background: rgba(108, 99, 255, 0.05);
        }
        .upload-zone .icon { font-size: 2rem; margin-bottom: 0.3rem; }
        .upload-zone p { font-size: 0.85rem; color: var(--text-muted); }
        .upload-zone input[type="file"] {
            position: absolute; inset: 0; opacity: 0; cursor: pointer;
        }

        .file-list {
            margin-top: 0.8rem;
            font-size: 0.82rem;
        }
        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.4rem 0.6rem;
            background: var(--bg-input);
            border-radius: var(--radius-sm);
            margin-bottom: 0.4rem;
        }
        .file-item .remove-btn {
            background: none; border: none; color: var(--danger);
            cursor: pointer; font-size: 1rem; padding: 0 0.3rem;
        }

        /* ===== Params ===== */
        .param-group { margin-bottom: 0.8rem; }
        .param-group label {
            display: block;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
        }
        .param-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .param-row input[type="range"] {
            flex: 1;
            accent-color: var(--primary);
        }
        .param-row .param-value {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--primary-light);
            min-width: 2.5rem;
            text-align: right;
        }

        /* ===== Buttons ===== */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            padding: 0.7rem 1.2rem;
            border: none;
            border-radius: var(--radius-sm);
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            width: 100%;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: #fff;
        }
        .btn-primary:hover {
            background: linear-gradient(135deg, var(--primary-light), var(--primary));
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(108, 99, 255, 0.3);
        }
        .btn-primary:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        /* ===== Main Content ===== */
        .main-content {
            padding: 2.5rem 3rem;
            max-width: 960px;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .main-subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* ===== Search Bar ===== */
        .search-bar {
            display: flex;
            gap: 0.6rem;
            margin-bottom: 1.2rem;
        }
        .search-input-wrap {
            flex: 1;
            position: relative;
        }
        .search-input-wrap .search-icon {
            position: absolute;
            left: 1rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.1rem;
            color: var(--text-dim);
        }
        .search-input {
            width: 100%;
            padding: 0.9rem 1rem 0.9rem 2.8rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text);
            font-family: inherit;
            font-size: 0.95rem;
            transition: var(--transition);
            outline: none;
        }
        .search-input:focus {
            border-color: var(--primary-light);
            box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.15);
        }
        .search-input::placeholder { color: var(--text-dim); }

        .search-btn {
            padding: 0.9rem 1.5rem;
            width: auto;
        }

        /* ===== Mode Tabs ===== */
        .mode-tabs {
            display: flex;
            gap: 0.4rem;
            margin-bottom: 1.5rem;
            background: var(--bg-card);
            padding: 0.3rem;
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }
        .mode-tab {
            flex: 1;
            padding: 0.6rem 1rem;
            background: none;
            border: none;
            border-radius: var(--radius-sm);
            color: var(--text-muted);
            font-family: inherit;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }
        .mode-tab:hover { color: var(--text); }
        .mode-tab.active {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: #fff;
            box-shadow: 0 2px 8px rgba(108, 99, 255, 0.3);
        }

        /* ===== Top-K selector ===== */
        .topk-selector {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        .topk-selector select {
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            color: var(--text);
            padding: 0.4rem 0.6rem;
            font-family: inherit;
            outline: none;
        }

        /* ===== Results ===== */
        .results-header {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        .result-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            margin-bottom: 0.8rem;
            transition: var(--transition);
            animation: fadeInUp 0.3s ease forwards;
            opacity: 0;
        }
        .result-card:hover {
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(108, 99, 255, 0.1);
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }
        .result-rank {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--primary-light);
        }
        .result-source {
            display: inline-block;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #fff;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 10px;
            border-radius: 20px;
            margin-left: 0.6rem;
        }
        .result-score {
            font-size: 0.82rem;
            color: var(--text-muted);
        }
        .result-score strong {
            color: var(--accent-green);
            font-weight: 700;
        }

        .score-bar-bg {
            height: 4px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            margin-bottom: 0.8rem;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 2px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.6s ease;
        }

        .result-text {
            font-size: 0.88rem;
            color: #C0C5D0;
            line-height: 1.65;
        }

        /* ===== Comparison Mode ===== */
        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }
        .comparison-col h3 {
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }

        /* ===== Empty / Loading ===== */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-dim);
        }
        .empty-state .icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
        .empty-state h3 { color: var(--text-muted); font-size: 1.1rem; margin-bottom: 0.4rem; }

        .loading-spinner {
            display: inline-block;
            width: 20px; height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--primary-light);
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ===== Alert ===== */
        .alert {
            padding: 1rem 1.2rem;
            border-radius: var(--radius-sm);
            font-size: 0.88rem;
            margin-bottom: 1rem;
        }
        .alert-info {
            background: rgba(0, 210, 255, 0.08);
            border: 1px solid rgba(0, 210, 255, 0.2);
            color: var(--accent);
        }
        .alert-success {
            background: rgba(0, 232, 157, 0.08);
            border: 1px solid rgba(0, 232, 157, 0.2);
            color: var(--accent-green);
        }
        .alert-warning {
            background: rgba(255, 190, 92, 0.08);
            border: 1px solid rgba(255, 190, 92, 0.2);
            color: var(--warning);
        }
        .alert-danger {
            background: rgba(255, 107, 107, 0.08);
            border: 1px solid rgba(255, 107, 107, 0.2);
            color: var(--danger);
        }

        /* ===== Progress ===== */
        .progress-bar-wrap {
            height: 6px;
            background: rgba(255,255,255,0.06);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.6rem;
        }
        .progress-bar-inner {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            transition: width 0.4s ease;
        }

        /* ===== Responsive ===== */
        @media (max-width: 900px) {
            .app-layout { grid-template-columns: 1fr; }
            .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
            .main-content { padding: 1.5rem; }
            .comparison-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="app-layout">

    <!-- ========== SIDEBAR ========== -->
    <aside class="sidebar">
        <div>
            <div style="font-size:1.3rem;font-weight:800;margin-bottom:0.2rem;">
                🔍 <span style="background:linear-gradient(135deg,#6C63FF,#00D2FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">SearchVec</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-dim);">Moteur de recherche sémantique</div>
        </div>

        <!-- Stats -->
        <div class="sidebar-section">
            <div class="sidebar-title">📊 Index actuel</div>
            <div id="index-stats">
                {% if index %}
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{{ index.num_chunks }}</div>
                        <div class="stat-label">Chunks</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ index.embedding_dim }}</div>
                        <div class="stat-label">Dimensions</div>
                    </div>
                </div>
                <ul class="source-list">
                    {% for s in sources %}
                    <li>{{ s }}</li>
                    {% endfor %}
                </ul>
                {% else %}
                <div class="alert alert-info">Aucun index. Indexez des fichiers ci-dessous.</div>
                {% endif %}
            </div>
        </div>

        <!-- Upload -->
        <div class="sidebar-section">
            <div class="sidebar-title">📂 Indexer des fichiers</div>
            <div class="upload-zone" id="upload-zone">
                <div class="icon">📁</div>
                <p>Glissez vos fichiers .txt ici<br>ou cliquez pour parcourir</p>
                <input type="file" id="file-input" multiple accept=".txt">
            </div>
            <div class="file-list" id="file-list"></div>

            <div class="param-group" style="margin-top:1rem;">
                <label>Taille des chunks (mots)</label>
                <div class="param-row">
                    <input type="range" id="chunk-size" min="50" max="1000" step="50" value="300">
                    <span class="param-value" id="chunk-size-val">300</span>
                </div>
            </div>
            <div class="param-group">
                <label>Chevauchement (mots)</label>
                <div class="param-row">
                    <input type="range" id="overlap" min="0" max="200" step="10" value="50">
                    <span class="param-value" id="overlap-val">50</span>
                </div>
            </div>

            <button class="btn btn-primary" id="index-btn" disabled>🚀 Indexer</button>
            <div id="index-status"></div>
        </div>
    </aside>

    <!-- ========== MAIN ========== -->
    <main class="main-content">
        <h1 class="main-title">Recherche sémantique</h1>
        <p class="main-subtitle">Recherchez dans vos documents par le sens, pas seulement par les mots.</p>

        <div class="search-bar">
            <div class="search-input-wrap">
                <span class="search-icon">🔎</span>
                <input type="text" class="search-input" id="search-input"
                       placeholder="Ex : Comment fonctionne la photosynthèse ?"
                       autocomplete="off">
            </div>
            <button class="btn btn-primary search-btn" id="search-btn">Rechercher</button>
        </div>

        <div class="mode-tabs">
            <button class="mode-tab active" data-mode="semantic">🧠 Sémantique</button>
            <button class="mode-tab" data-mode="lexical">📝 Lexical (TF-IDF)</button>
            <button class="mode-tab" data-mode="compare">⚖️ Comparaison</button>
        </div>

        <div class="topk-selector">
            <span>Nombre de résultats :</span>
            <select id="topk">
                <option value="3">3</option>
                <option value="5" selected>5</option>
                <option value="10">10</option>
                <option value="15">15</option>
                <option value="20">20</option>
            </select>
        </div>

        <div id="results-area">
            <div class="empty-state">
                <div class="icon">🔎</div>
                <h3>Tapez une requête pour commencer</h3>
                <p>Les résultats apparaîtront ici.</p>
            </div>
        </div>
    </main>
</div>

<script>
// ===== State =====
let selectedFiles = [];
let currentMode = 'semantic';

// ===== DOM refs =====
const fileInput     = document.getElementById('file-input');
const fileListEl    = document.getElementById('file-list');
const uploadZone    = document.getElementById('upload-zone');
const indexBtn      = document.getElementById('index-btn');
const indexStatus   = document.getElementById('index-status');
const chunkSizeIn   = document.getElementById('chunk-size');
const overlapIn     = document.getElementById('overlap');
const chunkSizeVal  = document.getElementById('chunk-size-val');
const overlapVal    = document.getElementById('overlap-val');
const searchInput   = document.getElementById('search-input');
const searchBtn     = document.getElementById('search-btn');
const resultsArea   = document.getElementById('results-area');
const topkSelect    = document.getElementById('topk');

// ===== Param sliders =====
chunkSizeIn.addEventListener('input', () => { chunkSizeVal.textContent = chunkSizeIn.value; });
overlapIn.addEventListener('input', () => { overlapVal.textContent = overlapIn.value; });

// ===== File handling =====
fileInput.addEventListener('change', (e) => {
    for (const f of e.target.files) {
        if (!selectedFiles.find(sf => sf.name === f.name)) selectedFiles.push(f);
    }
    renderFileList();
});

uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => { uploadZone.classList.remove('dragover'); });
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    for (const f of e.dataTransfer.files) {
        if (f.name.endsWith('.txt') && !selectedFiles.find(sf => sf.name === f.name))
            selectedFiles.push(f);
    }
    renderFileList();
});

function renderFileList() {
    indexBtn.disabled = selectedFiles.length === 0;
    if (selectedFiles.length === 0) { fileListEl.innerHTML = ''; return; }
    fileListEl.innerHTML = selectedFiles.map((f, i) => `
        <div class="file-item">
            <span>📄 ${f.name} <span style="color:var(--text-dim)">(${(f.size/1024).toFixed(1)} Ko)</span></span>
            <button class="remove-btn" onclick="removeFile(${i})">✕</button>
        </div>
    `).join('');
}

function removeFile(i) {
    selectedFiles.splice(i, 1);
    renderFileList();
}

// ===== Indexation =====
indexBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;
    indexBtn.disabled = true;
    indexStatus.innerHTML = `
        <div style="margin-top:0.8rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;font-size:0.85rem;color:var(--text-muted);">
                <span class="loading-spinner"></span> Indexation en cours…
            </div>
            <div class="progress-bar-wrap"><div class="progress-bar-inner" style="width:30%"></div></div>
        </div>`;

    const formData = new FormData();
    for (const f of selectedFiles) formData.append('files', f);
    formData.append('chunk_size', chunkSizeIn.value);
    formData.append('overlap', overlapIn.value);

    try {
        const resp = await fetch('/api/index', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.success) {
            indexStatus.innerHTML = `<div class="alert alert-success" style="margin-top:0.8rem;">
                ✅ Index créé : ${data.num_chunks} chunks, dim=${data.embedding_dim}</div>`;
            selectedFiles = [];
            renderFileList();
            setTimeout(() => location.reload(), 1500);
        } else {
            indexStatus.innerHTML = `<div class="alert alert-danger" style="margin-top:0.8rem;">❌ ${data.error}</div>`;
            indexBtn.disabled = false;
        }
    } catch (e) {
        indexStatus.innerHTML = `<div class="alert alert-danger" style="margin-top:0.8rem;">❌ Erreur réseau : ${e.message}</div>`;
        indexBtn.disabled = false;
    }
});

// ===== Mode tabs =====
document.querySelectorAll('.mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        currentMode = tab.dataset.mode;
    });
});

// ===== Search =====
searchBtn.addEventListener('click', doSearch);
searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    const topk = parseInt(topkSelect.value);
    searchBtn.disabled = true;
    resultsArea.innerHTML = `<div style="text-align:center;padding:2rem;color:var(--text-muted);">
        <span class="loading-spinner"></span><br><br>Recherche en cours…</div>`;

    try {
        if (currentMode === 'compare') {
            const [semResp, lexResp] = await Promise.all([
                fetch(`/api/search?query=${encodeURIComponent(query)}&mode=semantic&top_k=${topk}`),
                fetch(`/api/search?query=${encodeURIComponent(query)}&mode=lexical&top_k=${topk}`)
            ]);
            const semData = await semResp.json();
            const lexData = await lexResp.json();

            if (semData.error) { showError(semData.error); return; }

            resultsArea.innerHTML = `
                <div class="results-header">Comparaison pour : « ${escapeHtml(query)} »</div>
                <div class="comparison-grid">
                    <div class="comparison-col">
                        <h3>🧠 Sémantique</h3>
                        ${renderResults(semData.results)}
                    </div>
                    <div class="comparison-col">
                        <h3>📝 Lexical (TF-IDF)</h3>
                        ${renderResults(lexData.results)}
                    </div>
                </div>`;
        } else {
            const resp = await fetch(`/api/search?query=${encodeURIComponent(query)}&mode=${currentMode}&top_k=${topk}`);
            const data = await resp.json();

            if (data.error) { showError(data.error); return; }

            const modeLabel = currentMode === 'semantic' ? '🧠 Sémantique' : '📝 Lexical';
            resultsArea.innerHTML = `
                <div class="results-header">${data.results.length} résultat(s) ${modeLabel} pour : « ${escapeHtml(query)} »</div>
                ${renderResults(data.results)}`;
        }
    } catch(e) {
        showError('Erreur réseau : ' + e.message);
    } finally {
        searchBtn.disabled = false;
    }
}

function renderResults(results) {
    if (!results || results.length === 0)
        return '<div class="alert alert-info">Aucun résultat trouvé.</div>';

    return results.map((r, i) => {
        const score = r.score || 0;
        const barWidth = Math.max(5, Math.min(100, Math.round(score * 100)));
        const preview = (r.text || '').substring(0, 400) + ((r.text || '').length > 400 ? '…' : '');
        const delay = i * 0.06;

        return `<div class="result-card" style="animation-delay:${delay}s">
            <div class="result-header">
                <div>
                    <span class="result-rank">#${i+1}</span>
                    <span class="result-source">${escapeHtml(r.source || '?')}</span>
                </div>
                <span class="result-score">score : <strong>${score.toFixed(4)}</strong></span>
            </div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:${barWidth}%"></div></div>
            <div class="result-text">${escapeHtml(preview)}</div>
        </div>`;
    }).join('');
}

function showError(msg) {
    resultsArea.innerHTML = `<div class="alert alert-danger">${escapeHtml(msg)}</div>`;
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index_page():
    """Page principale."""
    index = _state.get("index")
    sources = []
    if index:
        sources = sorted({m.get("source", "?") for m in index.metadata})

    return render_template_string(
        HTML_TEMPLATE,
        index=index,
        sources=sources,
        error=_state.get("error"),
    )


@app.route("/api/search")
def api_search():
    """API de recherche (GET)."""
    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", "semantic")
    top_k = int(request.args.get("top_k", 5))

    if not query:
        return jsonify({"error": "Requête vide."}), 400

    if _state.get("error") == "no_index":
        return jsonify({"error": "Aucun index trouvé. Indexez des fichiers d'abord."}), 400

    if _state.get("error") == "ollama_down":
        return jsonify({"error": "Ollama est inaccessible. Vérifiez qu'il est lancé."}), 500

    try:
        if mode == "semantic":
            results = _state["sem_engine"].search(query, top_k=top_k)
        else:
            results = _state["tfidf_index"].search(query, top_k=top_k)

        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/index", methods=["POST"])
def api_index():
    """API d'indexation (POST multipart)."""
    from src.indexer import VectorIndex

    files = request.files.getlist("files")
    chunk_size = int(request.form.get("chunk_size", 300))
    overlap = int(request.form.get("overlap", 50))

    if not files:
        return jsonify({"success": False, "error": "Aucun fichier envoyé."}), 400

    documents: dict[str, str] = {}
    for f in files:
        content = f.read().decode("utf-8")
        documents[f.filename] = content

    try:
        provider = _state.get("provider")
        if provider is None:
            from src.embedding.ollama_provider import OllamaEmbeddingProvider
            provider = OllamaEmbeddingProvider()
            _state["provider"] = provider

        index = VectorIndex.from_documents(
            documents, provider, chunk_size=chunk_size, overlap=overlap
        )
        index.save()

        # Recharger les moteurs
        from src.lexical_search import TFIDFIndex
        from src.search import SemanticSearch

        _state["index"] = index
        _state["sem_engine"] = SemanticSearch(index, provider)
        _state["tfidf_index"] = TFIDFIndex().build(index.metadata)
        _state["error"] = None

        return jsonify({
            "success": True,
            "num_chunks": index.num_chunks,
            "embedding_dim": index.embedding_dim,
        })

    except ConnectionError:
        return jsonify({"success": False, "error": "Ollama inaccessible. Lancez-le d'abord."}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print(" 🔍 Moteur de recherche sémantique — Interface web")
    print("=" * 60)
    print("\nChargement de l'index et des moteurs...")

    _load_engines()

    if _state["error"] == "no_index":
        print("⚠️  Aucun index trouvé — vous pourrez indexer via l'interface.")
    elif _state["error"] == "ollama_down":
        print("⚠️  Ollama inaccessible — vérifiez qu'il est lancé.")
    elif _state["error"]:
        print(f"⚠️  Erreur : {_state['error']}")
    else:
        idx = _state["index"]
        print(f"✅ Index chargé : {idx.num_chunks} chunks, dim={idx.embedding_dim}")

    print("\n🌐 Ouverture dans le navigateur : http://localhost:5000")
    print("   (Ctrl+C pour arrêter)\n")

    import webbrowser
    webbrowser.open("http://localhost:5000")

    app.run(debug=False, port=5000)
