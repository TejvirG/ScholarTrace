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
let uploadedDocumentText = '';

function escapeHTML(value) {
  return String(value).replace(/[&<>'"]/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[character]));
}

function renderCitation(item) {
  const safeURL = escapeHTML(item.source_url);
  const title = escapeHTML(item.title);
  const section = escapeHTML(item.section);
  return `<article class="citation"><div class="meta"><a class="citation-title" href="${safeURL}" target="_blank" rel="noreferrer">${title}</a> / ${section} / p.${item.page}</div><p>${escapeHTML(item.text)}</p><div class="source">Match ${(item.score * 100).toFixed(0)}% · <a href="${safeURL}" target="_blank" rel="noreferrer">source ↗</a></div></article>`;
}

async function runQuery() {
  const value = question.value.trim();
  const searchText = value || uploadedDocumentText.trim();
  if (!searchText) {
    answer.textContent = 'Upload a document or type a question to search the evidence index.';
    results.classList.remove('hidden');
    citations.innerHTML = '';
    count.textContent = '0 citations';
    confidence.textContent = 'NO QUERY';
    telemetry.textContent = 'Waiting for input';
    return;
  }

  ask.disabled = true;
  results.classList.add('hidden');
  loading.classList.remove('hidden');
  try {
    const response = await fetch('/api/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question: value || '', top_k: 4})});
    if (!response.ok) throw new Error('Query failed');
    const data = await response.json();
    answer.textContent = data.answer || 'No answer available for this query.';
    confidence.textContent = data.abstained ? 'ABSTAINED' : `CONFIDENCE ${(data.confidence * 100).toFixed(0)}%`;
    count.textContent = `${data.citations.length} citations`;
    telemetry.textContent = `${data.latency_ms} ms local retrieval`;
    citations.innerHTML = data.citations.length ? data.citations.map(renderCitation).join('') : '<p>No supporting passages found.</p>';
    results.classList.remove('hidden');
  } catch (error) {
    answer.textContent = 'The local service could not answer this request.';
    confidence.textContent = 'ERROR';
    count.textContent = '0 citations';
    telemetry.textContent = 'Query failed';
    citations.innerHTML = '<p>The query could not be completed. Please try again.</p>';
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
  uploadedDocumentText = '';
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch('/api/documents', {method:'POST', body:form});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Upload failed');
    if (file.type.startsWith('text/') || file.name.toLowerCase().endsWith('.md') || file.name.toLowerCase().endsWith('.txt') || file.name.toLowerCase().endsWith('.json')) {
      uploadedDocumentText = await file.text();
    }
    uploadStatus.textContent = `${file.name} indexed: ${data.chunks_added} passage(s) added.`;
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    documentInput.value = '';
  }
});
question.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    runQuery();
  }
});
fetch('/api/evaluation').then(response => response.json()).then(data => {
  document.querySelector('#metrics').textContent = `Recall@5 ${(data.recall_at_5 * 100).toFixed(0)}% / Recall@10 ${(data.recall_at_10 * 100).toFixed(0)}%`;
}).catch(() => {
  document.querySelector('#metrics').textContent = 'Metrics unavailable';
});
