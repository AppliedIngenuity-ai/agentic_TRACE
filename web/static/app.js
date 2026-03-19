/**
 * Agent Web UI - Main Application
 */

const UI_VERSION = '2026-03-16';

// State
let config = null;
let sessionId = null;
let pollOffset = 0;
let pollInterval = null;
let currentInputRequest = null;
let lastEventTime = null;
let eventCount = 0;
let executionSteps = [];  // Track steps for summary
let lastLlmMessage = null;  // Track last LLM message for execution summary
let pendingTokens = null;  // Tokens from most recent LLM iteration, awaiting attachment to a step
let prevResultElapsed = 0;  // elapsed_ms of previous tool_result (for wall-clock step duration)
const SESSION_TIMEOUT_MS = 300000; // 5 minutes with no updates = timeout

// DOM Elements
const titleEl = document.getElementById('title');
const configSelect = document.getElementById('config-select');
const queryInput = document.getElementById('query-input');
const runBtn = document.getElementById('run-btn');
const eventsLog = document.getElementById('events-log');
const eventsSection = document.getElementById('events-section');
const eventCountEl = document.getElementById('event-count');
const summarySection = document.getElementById('summary-section');
const executionSummary = document.getElementById('execution-summary');
const resultsSection = document.getElementById('results-section');
const tableContainer = document.getElementById('results-table-container');
const inputModal = document.getElementById('input-modal');
const inputContext = document.getElementById('input-context');
const inputQuestion = document.getElementById('input-question');
const inputOptions = document.getElementById('input-options');
const inputCustom = document.getElementById('input-custom');
const inputSubmit = document.getElementById('input-submit');
const sessionIdEl = document.getElementById('session-id');
const chartsSection = document.getElementById('charts-section');
const chartsContainer = document.getElementById('charts-container');
const stopBtn = document.getElementById('stop-btn');

// Display UI version
{
    const versionEl = document.createElement('div');
    versionEl.style.cssText = 'position:fixed;bottom:4px;right:8px;font-size:11px;color:#888;pointer-events:none;';
    versionEl.textContent = `UI ${UI_VERSION}`;
    document.body.appendChild(versionEl);
}

// Toggle collapsible section
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.toggle('collapsed');
    }
}

// Initialize
async function init() {
    try {
        const response = await fetch('/api/config');
        config = await response.json();

        // Update title
        titleEl.textContent = config.title;
        document.title = config.title;

        // Populate config dropdown
        configSelect.innerHTML = '';
        config.configs.forEach(cfg => {
            const option = document.createElement('option');
            option.value = cfg.id;
            option.textContent = `${cfg.name} (${cfg.model})`;
            if (cfg.id === config.default_config) {
                option.selected = true;
            }
            configSelect.appendChild(option);
        });

        // Enable query input
        queryInput.disabled = false;
        runBtn.disabled = false;
    } catch (error) {
        console.error('Failed to load config:', error);
        addEvent('error', { message: 'Failed to load configuration' });
    }
}

// Run query
async function runQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    const selectedConfig = configSelect.value;

    // Clear previous results
    eventsLog.innerHTML = '';
    resultsSection.style.display = 'none';
    tableContainer.innerHTML = '';
    chartsSection.style.display = 'none';
    chartsContainer.innerHTML = '';
    summarySection.style.display = 'none';
    executionSummary.innerHTML = '';
    sessionIdEl.style.display = 'none';
    pollOffset = 0;
    eventCount = 0;
    executionSteps = [];
    lastLlmMessage = null;
    pendingTokens = null;
    prevResultElapsed = 0;
    updateEventCount();

    // Expand execution log when starting new query
    eventsSection.classList.remove('collapsed');

    // Disable inputs, show stop button
    queryInput.disabled = true;
    runBtn.disabled = true;
    runBtn.textContent = 'Running...';
    stopBtn.style.display = 'inline-block';

    try {
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, config: selectedConfig }),
        });

        const data = await response.json();

        if (data.error) {
            addEvent('error', { message: data.error });
            resetUI();
            return;
        }

        sessionId = data.session_id;
        lastEventTime = Date.now(); // Initialize timeout tracker

        // Show session ID
        sessionIdEl.textContent = `Session: ${sessionId}`;
        sessionIdEl.style.display = 'inline';

        // Start polling
        startPolling();
    } catch (error) {
        console.error('Failed to start session:', error);
        addEvent('error', { message: 'Failed to start session' });
        resetUI();
    }
}

// Start polling for events
function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    const pollMs = config?.poll_interval_ms || 5000;

    // Poll immediately, then at interval
    pollEvents();
    pollInterval = setInterval(pollEvents, pollMs);
}

// Stop polling
function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Poll for events
async function pollEvents() {
    if (!sessionId) return;

    // Check for session timeout (no events for 5 minutes)
    if (lastEventTime && (Date.now() - lastEventTime > SESSION_TIMEOUT_MS)) {
        addEvent('error', { message: 'Session timeout: no updates for 5 minutes' });
        addEvent('complete', { message: 'Session complete: timeout', status: 'timeout' });
        stopPolling();
        resetUI();
        return;
    }

    try {
        const response = await fetch(`/api/sessions/${sessionId}/stream?offset=${pollOffset}`);
        const data = await response.json();

        if (data.error) {
            console.error('Poll error:', data.error);
            return;
        }

        // Process new events and update last event time
        if (data.events.length > 0) {
            lastEventTime = Date.now();
            data.events.forEach(event => {
                processEvent(event);
            });
        }

        pollOffset = data.offset;
    } catch (error) {
        console.error('Poll failed:', error);
    }
}

// Process a single event
function processEvent(event) {
    switch (event.type) {
        case 'start':
            addEvent('start', { message: `Session started: ${event.query}` });
            break;

        case 'iteration':
            addEvent('iteration', { message: `Iteration ${event.n}` });
            break;

        case 'tool_call':
            addEvent('tool_call', {
                message: `${event.name}(${formatArgs(event.args)})`
            });
            // Track for summary — attach pending LLM tokens to first tool call of each iteration
            {
                const step = {
                    type: 'call',
                    tool: event.name,
                    args: event.args,
                    call_elapsed_ms: event.elapsed_ms || 0,
                };
                if (pendingTokens) {
                    step.tokens = pendingTokens;
                    pendingTokens = null;
                }
                executionSteps.push(step);
            }
            break;

        case 'tool_result': {
            // Compute wall-clock step duration (includes LLM thinking time before this tool call)
            const resultElapsed = event.elapsed_ms || 0;
            const wallDuration = resultElapsed - prevResultElapsed;
            prevResultElapsed = resultElapsed;
            const durationStr = wallDuration > 0 ? `${(wallDuration / 1000).toFixed(1)}s` : '';
            const elapsedStr = resultElapsed > 0 ? `${(resultElapsed / 1000).toFixed(1)}s` : '';
            const timingStr = durationStr && elapsedStr ? ` [${durationStr} / ${elapsedStr}]` : '';
            const resultMsg = event.success
                ? `${event.name}${timingStr}: ${event.summary}${event.rows ? ` (${event.rows} rows)` : ''}`
                : `${event.name}${timingStr} ERROR: ${event.error}`;
            addEvent('tool_result', {
                message: resultMsg,
                success: event.success,
            });
            // Update last step with result
            if (executionSteps.length > 0) {
                const lastStep = executionSteps[executionSteps.length - 1];
                lastStep.success = event.success;
                lastStep.summary = event.summary;
                lastStep.rows = event.rows;
                lastStep.error = event.error;
                lastStep.elapsed_ms = resultElapsed;
                lastStep.duration_ms = wallDuration;
            }
            break;
        }

        case 'needs_input':
            showInputModal(event);
            break;

        case 'input_received':
            hideInputModal();
            addEvent('input_received', { message: `User input: ${event.value}` });
            break;

        case 'llm_message':
            addEvent('llm_message', { message: event.content });
            lastLlmMessage = event.content;
            break;

        case 'result':
            showResult(event);
            break;

        case 'chart':
            showChart(event);
            break;

        case 'complete': {
            const totalStr = event.elapsed_ms ? ` (${(event.elapsed_ms / 1000).toFixed(1)}s)` : '';
            addEvent('complete', {
                message: `Session complete: ${event.status}${totalStr}`,
                status: event.status,
            });
            // Build and show execution summary
            showExecutionSummary(event.status, event.llm_message, event.elapsed_ms, {
                tokens_in: event.tokens_in,
                tokens_out: event.tokens_out,
                tokens_reasoning: event.tokens_reasoning,
                max_context_tokens: event.max_context_tokens,
                context_window: event.context_window,
            });
            // Collapse execution log on completion
            eventsSection.classList.add('collapsed');
            stopPolling();
            resetUI();
            break;
        }

        case 'warning':
            addEvent('warning', { message: event.message });
            break;

        case 'pre_process':
            if (event.error) {
                addEvent('warning', { message: `Pre-process error: ${event.error}` });
            } else if (event.context) {
                addEvent('pre_process', { message: `Context injected: ${event.context.substring(0, 200)}...` });
            } else {
                addEvent('pre_process', { message: 'Pre-process: no context injected' });
            }
            // Track for summary
            executionSteps.push({
                type: 'call',
                tool: 'pre_process',
                args: {},
                success: !event.error,
                summary: event.error ? `Error: ${event.error}` : (event.context ? 'Injected context' : 'No context injected'),
                error: event.error,
                duration_ms: event.duration_ms || 0,
                elapsed_ms: event.elapsed_ms || 0,
            });
            break;

        case 'post_finalize':
            addEvent(event.action === 'pass' ? 'tool_result' : 'warning', {
                message: `Post-validation: ${event.message || 'PASS'}`,
                success: event.action === 'pass',
            });
            // Track for summary — attach pending tokens from the LLM call that triggered validation
            {
                const step = {
                    type: 'call',
                    tool: 'post_finalize',
                    args: {},
                    success: event.action === 'pass',
                    summary: event.action === 'pass' ? 'PASS' : event.message,
                    error: event.action !== 'pass' ? event.message : undefined,
                    duration_ms: event.duration_ms || 0,
                    elapsed_ms: event.elapsed_ms || 0,
                };
                if (pendingTokens) {
                    step.tokens = pendingTokens;
                    pendingTokens = null;
                }
                executionSteps.push(step);
            }
            break;

        case 'tokens': {
            const prompt = event.tokens_in ? event.tokens_in.toLocaleString() : '0';
            const reasoning = event.tokens_reasoning ? event.tokens_reasoning.toLocaleString() : '0';
            const output = event.tokens_out ? (event.tokens_out - (event.tokens_reasoning || 0)).toLocaleString() : '0';
            const total = ((event.tokens_in || 0) + (event.tokens_out || 0)).toLocaleString();
            addEvent('tokens', {
                message: `Iter ${event.iteration} tokens — prompt: ${prompt}, reasoning: ${reasoning}, output: ${output}, total: ${total}`
            });
            // Store for attachment to the next step(s)
            pendingTokens = {
                tokens_in: event.tokens_in || 0,
                tokens_out: event.tokens_out || 0,
                tokens_reasoning: event.tokens_reasoning || 0,
            };
            break;
        }

        case 'error':
            addEvent('error', { message: event.message });
            executionSteps.push({
                type: 'call',
                tool: 'error',
                args: {},
                success: false,
                summary: event.message,
                error: event.message,
            });
            break;

        default:
            console.log('Unknown event:', event);
    }
}

// Update event count display
function updateEventCount() {
    if (eventCountEl) {
        eventCountEl.textContent = eventCount > 0 ? `${eventCount}` : '';
    }
}

// Add event to log
function addEvent(type, data) {
    eventCount++;
    updateEventCount();

    const eventEl = document.createElement('div');
    eventEl.className = `event ${type}`;

    if (data.success === false) {
        eventEl.className += ' error';
    } else if (data.status === 'error') {
        eventEl.className += ' error';
    }

    const typeLabel = document.createElement('span');
    typeLabel.className = 'event-type';
    typeLabel.textContent = `[${type}]`;

    const messageText = document.createTextNode(` ${data.message || ''}`);

    eventEl.appendChild(typeLabel);
    eventEl.appendChild(messageText);

    // Auto-scroll only if user is already at (or near) the bottom
    const atBottom = eventsLog.scrollHeight - eventsLog.scrollTop - eventsLog.clientHeight < 50;

    eventsLog.appendChild(eventEl);

    if (atBottom) {
        eventsLog.scrollTop = eventsLog.scrollHeight;
    }
}

// Format args for display (no truncation)
function formatArgs(args) {
    if (!args) return '';
    const parts = [];
    for (const [key, value] of Object.entries(args)) {
        let valStr = typeof value === 'string' ? `"${value}"` : JSON.stringify(value);
        parts.push(`${key}=${valStr}`);
    }
    return parts.join(', ');
}

// Format timing for summary steps: "1.2s / 4.5s" (wall-clock since prev step / elapsed from start)
function formatStepTiming(step) {
    const parts = [];
    if (step.wall_ms > 0) parts.push(`${(step.wall_ms / 1000).toFixed(1)}s`);
    if (step.elapsed_ms > 0) parts.push(`${(step.elapsed_ms / 1000).toFixed(1)}s`);
    return parts.length > 0 ? ` [${parts.join(' / ')}]` : '';
}

// Format token count with locale separators
function fmtTokens(n) {
    return (n || 0).toLocaleString();
}

// Show execution summary
function showExecutionSummary(status, llmMessage, totalElapsedMs, tokenStats) {
    const msg = llmMessage || lastLlmMessage;
    if (executionSteps.length === 0 && !msg) return;

    summarySection.style.display = 'block';
    executionSummary.innerHTML = '';

    // Show LLM message at the top if present
    if (msg) {
        const labelEl = document.createElement('div');
        labelEl.className = 'summary-llm-label';
        labelEl.textContent = 'LLM-Generated Summary';
        executionSummary.appendChild(labelEl);

        const msgEl = document.createElement('div');
        msgEl.className = 'summary-llm-message';
        msgEl.textContent = msg;
        executionSummary.appendChild(msgEl);
    }

    // Display names for internal step types
    const stepDisplayNames = {
        'pre_process': 'pre-process (hook)',
        'post_finalize': 'validation (hook)',
    };

    // Compute wall-clock duration per step (includes LLM time between steps)
    let prevElapsed = 0;
    executionSteps.forEach(step => {
        if (step.type !== 'call') return;
        step.wall_ms = (step.elapsed_ms || 0) - prevElapsed;
        prevElapsed = step.elapsed_ms || 0;
    });

    let stepNum = 1;
    executionSteps.forEach(step => {
        if (step.type !== 'call') return;

        const stepEl = document.createElement('div');
        stepEl.className = `summary-step ${step.success ? 'success' : (step.success === false ? 'error' : '')}`;

        const numEl = document.createElement('span');
        numEl.className = 'step-number';
        numEl.textContent = `Step ${stepNum}${formatStepTiming(step)}:`;

        const toolEl = document.createElement('span');
        toolEl.className = 'step-tool';
        toolEl.textContent = stepDisplayNames[step.tool] || step.tool;

        const arrowEl = document.createElement('span');
        arrowEl.className = 'step-arrow';
        arrowEl.textContent = ' → ';

        const resultEl = document.createElement('span');
        resultEl.className = 'step-result';
        if (step.success === false) {
            resultEl.textContent = `ERROR: ${step.error}`;
        } else {
            resultEl.textContent = step.summary || 'Success';
        }

        stepEl.appendChild(numEl);
        stepEl.appendChild(toolEl);
        stepEl.appendChild(arrowEl);
        stepEl.appendChild(resultEl);

        if (step.rows) {
            const rowsEl = document.createElement('span');
            rowsEl.className = 'step-rows';
            rowsEl.textContent = ` (${step.rows} rows)`;
            stepEl.appendChild(rowsEl);
        }

        // Show per-step token counts if available
        if (step.tokens) {
            const t = step.tokens;
            const stepTotal = t.tokens_in + t.tokens_out;
            const tokEl = document.createElement('span');
            tokEl.className = 'step-tokens';
            tokEl.textContent = ` [${fmtTokens(stepTotal)} tokens]`;
            tokEl.title = `Prompt: ${fmtTokens(t.tokens_in)}, Output: ${fmtTokens(t.tokens_out - (t.tokens_reasoning || 0))}${t.tokens_reasoning ? `, Reasoning: ${fmtTokens(t.tokens_reasoning)}` : ''}`;
            stepEl.appendChild(tokEl);
        }

        executionSummary.appendChild(stepEl);
        stepNum++;
    });

    // Status + total time line
    const totalStr = totalElapsedMs ? ` in ${(totalElapsedMs / 1000).toFixed(1)}s` : '';
    const statusEl = document.createElement('div');
    statusEl.className = `summary-step ${status === 'success' ? 'success' : 'error'}`;
    statusEl.style.marginTop = '12px';
    statusEl.style.fontWeight = '500';
    statusEl.innerHTML = `<span class="step-result">Status: ${status} — ${stepNum - 1} steps${totalStr}</span>`;
    executionSummary.appendChild(statusEl);

    // Token usage summary
    if (tokenStats && (tokenStats.tokens_in || tokenStats.tokens_out)) {
        const tokensEl = document.createElement('div');
        tokensEl.className = 'summary-tokens';

        const totalAll = (tokenStats.tokens_in || 0) + (tokenStats.tokens_out || 0);
        const outputRegular = (tokenStats.tokens_out || 0) - (tokenStats.tokens_reasoning || 0);

        let lines = [
            `Total tokens: ${fmtTokens(totalAll)}`,
            `  Prompt (input): ${fmtTokens(tokenStats.tokens_in)}`,
            `  Output (completion): ${fmtTokens(tokenStats.tokens_out)}`,
        ];
        if (tokenStats.tokens_reasoning) {
            lines.push(`    Reasoning: ${fmtTokens(tokenStats.tokens_reasoning)}`);
            lines.push(`    Regular: ${fmtTokens(outputRegular)}`);
        }
        if (tokenStats.max_context_tokens) {
            let contextLine = `  Max in-context: ${fmtTokens(tokenStats.max_context_tokens)}`;
            if (tokenStats.context_window) {
                const pct = ((tokenStats.max_context_tokens / tokenStats.context_window) * 100).toFixed(1);
                contextLine += ` / ${fmtTokens(tokenStats.context_window)} (${pct}%)`;
            }
            lines.push(contextLine);
        }

        tokensEl.textContent = lines.join('\n');
        executionSummary.appendChild(tokensEl);
    }
}

// Show input modal
function showInputModal(event) {
    currentInputRequest = event.id;

    inputContext.textContent = event.context || '';
    inputContext.style.display = event.context ? 'block' : 'none';
    inputQuestion.textContent = event.question;

    // Build options
    inputOptions.innerHTML = '';
    if (event.options && event.options.length > 0) {
        event.options.forEach((opt) => {
            const btn = document.createElement('button');
            btn.className = 'option-btn';
            btn.textContent = opt;
            btn.onclick = () => selectOption(btn, opt);
            inputOptions.appendChild(btn);
        });

        // Add "Something else" option
        const otherBtn = document.createElement('button');
        otherBtn.className = 'option-btn';
        otherBtn.textContent = 'Something else...';
        otherBtn.onclick = () => {
            document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
            inputCustom.focus();
        };
        inputOptions.appendChild(otherBtn);
    }

    inputCustom.value = '';
    inputModal.style.display = 'flex';
}

// Select an option
let selectedOption = null;
function selectOption(btn, value) {
    document.querySelectorAll('.option-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedOption = value;
    inputCustom.value = '';
}

// Hide input modal
function hideInputModal() {
    inputModal.style.display = 'none';
    currentInputRequest = null;
    selectedOption = null;
}

// Submit input
async function submitInput() {
    if (!currentInputRequest) return;

    const value = inputCustom.value.trim() || selectedOption || '';
    if (!value) {
        alert('Please select an option or enter a custom answer');
        return;
    }

    try {
        await fetch(`/api/sessions/${sessionId}/input`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                request_id: currentInputRequest,
                value: value,
            }),
        });

        // Modal will be hidden when input_received event is processed
    } catch (error) {
        console.error('Failed to submit input:', error);
        alert('Failed to submit input');
    }
}

// Show result
function showResult(event) {
    resultsSection.style.display = 'block';

    // Each result gets its own self-contained block (supports multiple finalized views)
    const block = document.createElement('div');
    block.className = 'result-block';

    // Header: info + download link
    const header = document.createElement('div');
    header.className = 'result-header';
    const info = document.createElement('span');
    info.textContent = `${event.file} — ${event.rows} rows, ${event.columns.length} columns`;
    const dl = document.createElement('a');
    dl.href = `/api/sessions/${sessionId}/results/${event.file}`;
    dl.download = event.file;
    dl.className = 'download-btn';
    dl.textContent = 'Download CSV';
    header.appendChild(info);
    header.appendChild(dl);
    block.appendChild(header);

    // Table
    if (event.preview && event.preview.length > 0) {
        const tableWrap = document.createElement('div');
        tableWrap.className = 'table-container';

        const table = document.createElement('table');
        table.className = 'data-table';

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        event.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        event.preview.forEach(row => {
            const tr = document.createElement('tr');
            event.columns.forEach(col => {
                const td = document.createElement('td');
                let val = row[col];
                if (typeof val === 'number') {
                    val = val.toLocaleString(undefined, { maximumFractionDigits: 4 });
                }
                td.textContent = val ?? '';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        tableWrap.appendChild(table);

        if (event.preview.length < event.rows) {
            const more = document.createElement('p');
            more.style.color = 'var(--text-muted)';
            more.style.marginTop = '12px';
            more.style.fontSize = '0.9rem';
            more.textContent = `Showing ${event.preview.length} of ${event.rows} rows. Download CSV for full data.`;
            tableWrap.appendChild(more);
        }

        block.appendChild(tableWrap);
    }

    tableContainer.appendChild(block);
}

// Show chart
function showChart(event) {
    chartsSection.style.display = 'block';

    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'chart-wrapper';

    const img = document.createElement('img');
    img.src = `/api/sessions/${sessionId}/artifacts/${event.file}`;
    img.alt = event.file;
    img.className = 'chart-image';

    const caption = document.createElement('p');
    caption.className = 'chart-caption';
    caption.textContent = event.file;

    chartWrapper.appendChild(img);
    chartWrapper.appendChild(caption);
    chartsContainer.appendChild(chartWrapper);

    addEvent('chart', { message: `Chart created: ${event.file}` });
}

// Reset UI
function resetUI() {
    queryInput.disabled = false;
    runBtn.disabled = false;
    runBtn.textContent = 'Run';
    stopBtn.style.display = 'none';
    stopBtn.disabled = false;
    stopBtn.textContent = 'Stop';
}

// Stop the running session
async function stopSession() {
    if (!sessionId) return;
    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping...';
    try {
        await fetch(`/api/sessions/${sessionId}/stop`, { method: 'POST' });
    } catch (e) {
        console.error('Failed to stop session:', e);
    }
}

// Event listeners
runBtn.addEventListener('click', runQuery);
stopBtn.addEventListener('click', stopSession);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') runQuery();
});
inputSubmit.addEventListener('click', submitInput);
inputCustom.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') submitInput();
});

// Initialize on load
init();
