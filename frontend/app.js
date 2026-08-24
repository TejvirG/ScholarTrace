const question = document.querySelector('#question');
const ask = document.querySelector('#ask');
const results = document.querySelector('#results');
const loading = document.querySelector('#loading');
const answer = document.querySelector('#answer');
const citations = document.querySelector('#citations');
const confidence = document.querySelector('#confidence');
const count = document.querySelector('#count');
const telemetry = document.querySelector('#telemetry');
const documentInput = document.querySelector('#document');
const uploadStatus = document.querySelector('#upload-status');

function escapeHTML(value) {
  return String(value).replace(/[&<>'"]/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character]));
}

function renderCitation(item) {
  const safeURL = escapeHTML(item.source_url);
  return `<article class="citation"><div class="meta">${escapeHTML(item.title)} / ${escapeHTML(item.section)} / p.${item.page}</div><p>${escapeHTML(item.text)}</p><div class="source">Match ${(item.score * 100).toFixed(0)}% · <a href="${safeURL}" target="_blank" rel="noreferrer">source ↗</a></div></article>`;
}

async function runQuery() {
  const value = question.value.trim();
  if (!value) return;
  ask.disabled = true;
  results.classList.add('hidden');
  loading.classList.remove('hidden');
  try {
    const response = await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:value, top_k:3})});
    if (!response.ok) throw new Error('Query failed');
    const data = await response.json();
    answer.textContent = data.answer;
    confidence.textContent = data.abstained ? 'ABSTAINED' : `CONFIDENCE ${(data.confidence * 100).toFixed(0)}%`;
    count.textContent = `${data.citations.length} citations`;
    telemetry.textContent = `${data.retrieval_count} passages ranked / ${data.latency_ms} ms local retrieval`;
    citations.innerHTML = data.citations.length ? data.citations.map(renderCitation).join('') : '<p>No supporting passages found.</p>';
    results.classList.remove('hidden');
  } catch (error) {
    answer.textContent = 'The local service could not answer this request.';
    results.classList.remove('hidden');
  } finally {
    loading.classList.add('hidden');
    ask.disabled = false;
  }
}

ask.addEventListener('click', runQuery);
documentInput.addEventListener('change', async () => {
  const file = documentInput.files[0];
  if (!file) return;
  uploadStatus.textContent = `Indexing ${file.name}...`;
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/api/documents', {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Upload failed');
    uploadStatus.textContent = `${file.name} indexed: ${data.chunks_added} passage(s) added.`;
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    documentInput.value = '';
  }
});
question.addEventListener('keydown', event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) runQuery(); });
fetch('/api/evaluation').then(response => response.json()).then(data => { document.querySelector('#metrics').textContent = `Recall@3 ${(data.recall_at_k * 100).toFixed(0)}%`; });
