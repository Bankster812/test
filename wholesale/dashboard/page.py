"""The control-center UI (single-page app, vanilla JS, no build step)."""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Keystone Property Partners — CEO Control Center</title>
<style>
  :root{
    --bg:#0a0e14; --bg2:#0d131c; --panel:#121a25; --panel2:#0f1620; --line:#1f2c3a;
    --txt:#e8eef5; --dim:#8298ad; --accent:#4cc9f0; --accent2:#7b8cff;
    --win:#3ddc84; --warn:#ffb454; --esc:#c77dff; --danger:#ff6b6b;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%}
  body{background:var(--bg);color:var(--txt);
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    display:grid;grid-template-columns:226px 1fr;grid-template-rows:58px 1fr;
    grid-template-areas:"side top" "side main";height:100vh;overflow:hidden}

  .side{grid-area:side;background:linear-gradient(180deg,#0d141d,#0a0e14);
    border-right:1px solid var(--line);padding:16px 12px;overflow:auto}
  .brand{display:flex;align-items:center;gap:9px;padding:4px 8px 16px;font-weight:700;font-size:15px}
  .brand .logo{width:26px;height:26px;border-radius:7px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;
    color:#06101a;font-weight:900}
  .nav a{display:flex;align-items:center;gap:10px;padding:9px 11px;border-radius:9px;
    color:var(--dim);text-decoration:none;cursor:pointer;margin-bottom:2px;font-weight:500}
  .nav a:hover{background:#121b26;color:var(--txt)}
  .nav a.active{background:linear-gradient(90deg,rgba(76,201,240,.16),transparent);
    color:#fff;box-shadow:inset 2px 0 0 var(--accent)}
  .nav .ico{width:18px;text-align:center}
  .nav .badge{margin-left:auto;background:var(--accent);color:#06101a;font-size:10px;
    font-weight:800;border-radius:999px;padding:1px 7px}
  .side .sechead{color:#54677a;font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
    padding:14px 10px 6px}

  .top{grid-area:top;border-bottom:1px solid var(--line);background:var(--bg2);
    display:flex;align-items:center;gap:14px;padding:0 20px}
  .top h1{font-size:16px;margin:0}
  .top .sub{color:var(--dim);font-size:12px}
  .grow{flex:1}
  .pill{font-size:11px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
  .pill.on{color:var(--win);border-color:#1d5b39} .pill.off{color:var(--warn);border-color:#5b421d}
  .pill.live{color:var(--win);border-color:#1d5b39}
  .pill.demo{color:var(--warn);border-color:#5b421d}
  .btn{font:inherit;border:1px solid var(--line);background:#16212e;color:var(--txt);
    padding:7px 13px;border-radius:9px;cursor:pointer;font-weight:600}
  .btn:hover{filter:brightness(1.2)}
  .btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#06101a;border:0}
  .btn.approve{background:#143a26;border-color:#1d5b39;color:var(--win)}
  .btn.reject{background:#3a1a1a;border-color:#5b2424;color:#ff8b8b}
  .btn.sm{padding:5px 10px;font-size:12px}

  .main{grid-area:main;overflow:auto;padding:20px}
  .view{display:none;animation:fade .2s ease} .view.active{display:block}
  @keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
  h2.title{font-size:18px;margin:0 0 2px} .desc{color:var(--dim);margin:0 0 16px;font-size:13px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
  .panel h3{margin:0 0 12px;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--dim)}
  .grid{display:grid;gap:14px}
  .kpis{grid-template-columns:repeat(6,1fr)}
  .cards3{grid-template-columns:repeat(3,1fr)}
  .cards2{grid-template-columns:1.3fr .9fr}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  .kpi .v{font-size:24px;font-weight:700} .kpi.win .v{color:var(--win)}
  .kpi .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:3px}

  /* Atlas panel */
  .atlas-panel{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
  .atlas-header{display:flex;align-items:center;gap:12px;margin-bottom:10px}
  .atlas-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
  .atlas-dot.green{background:var(--win);box-shadow:0 0 8px var(--win)}
  .atlas-dot.amber{background:var(--warn);box-shadow:0 0 8px var(--warn)}
  .atlas-dot.red{background:var(--danger);box-shadow:0 0 8px var(--danger)}
  .atlas-name{font-weight:700;font-size:13px}
  .atlas-teams{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
  .atlas-team{font-size:11px;color:var(--dim);background:#16212e;padding:4px 9px;border-radius:7px;border:1px solid var(--line)}
  .atlas-stats{display:flex;gap:16px;margin-top:10px;font-size:12px;color:var(--dim)}
  .atlas-stat span{color:var(--warn)}

  /* Agents two-pane */
  .agent-layout{display:grid;grid-template-columns:280px 1fr;gap:14px;height:calc(100vh - 160px)}
  .agent-list{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:auto}
  .agent-row{display:flex;align-items:center;gap:10px;padding:11px 14px;cursor:pointer;border-bottom:1px solid var(--line);border-left:4px solid transparent}
  .agent-row:hover{background:#16212e}
  .agent-row.sel{background:linear-gradient(90deg,rgba(76,201,240,.1),transparent);border-left-color:var(--accent)}
  .agent-row .ar-dot{width:8px;height:8px;border-radius:50%;background:#3a4a5a;flex-shrink:0}
  .ar-dot.thinking{background:var(--warn);box-shadow:0 0 6px var(--warn)}
  .ar-dot.acting{background:var(--win);box-shadow:0 0 6px var(--win)}
  .agent-row .ar-name{font-weight:600;font-size:13px}
  .agent-row .ar-meta{color:var(--dim);font-size:11px}
  .agent-row .ar-count{margin-left:auto;font-size:11px;color:var(--dim)}
  .agent-detail{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:auto;display:flex;flex-direction:column}
  .ad-header{padding:18px 18px 14px;border-bottom:1px solid var(--line)}
  .ad-name{font-size:18px;font-weight:700}
  .ad-role{color:var(--dim);font-size:13px;margin-top:2px}
  .ad-badges{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
  .ad-badge{font-size:11px;padding:3px 9px;border-radius:7px;background:#16212e;border:1px solid var(--line)}
  .ad-body{flex:1;overflow:auto;padding:16px 18px}
  .ad-section{margin-bottom:16px}
  .ad-section-title{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);margin-bottom:8px}
  .ad-stat-row{display:flex;gap:20px;margin-bottom:12px}
  .ad-stat{font-size:12px} .ad-stat b{color:var(--txt)} .ad-stat span{color:var(--dim)}
  .ad-activity{max-height:220px;overflow:auto}
  .ad-chat{border-top:1px solid var(--line);padding:14px 18px}
  .ad-chat-log{max-height:160px;overflow:auto;margin-bottom:10px;font-size:12.5px}
  .ad-chat-log .msg-ceo{color:var(--accent);padding:3px 0} .ad-chat-log .msg-agent{color:var(--txt);padding:3px 0 8px}
  .ad-chat-input{display:flex;gap:8px}
  .ad-chat-input input{flex:1}
  .ad-empty{padding:40px;text-align:center;color:#3a4a5a;font-style:italic}

  /* Lead import */
  .onboard{background:linear-gradient(135deg,#0d1e2e,#111825);border:1px solid var(--accent);
    border-radius:14px;padding:24px;margin-bottom:16px;text-align:center}
  .onboard h3{color:var(--accent);margin:0 0 8px}
  .onboard p{color:var(--dim);margin:0 0 16px;font-size:13px}
  .onboard .onboard-actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}

  .agent{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px;border-left:4px solid #456}
  .dot{width:9px;height:9px;border-radius:50%;background:#3a4a5a;display:inline-block}
  .dot.thinking{background:var(--warn);box-shadow:0 0 9px var(--warn);animation:p 1s infinite}
  .dot.acting{background:var(--win);box-shadow:0 0 9px var(--win);animation:p 1s infinite}
  @keyframes p{50%{opacity:.35}}

  .board{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:9px;overflow-x:auto}
  .lane{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:9px;min-height:90px}
  .lane h4{font-size:10px;text-transform:uppercase;color:var(--dim);margin:0 0 7px;display:flex;justify-content:space-between}
  .lane h4 b{color:var(--txt)}
  .chip{background:#16212e;border:1px solid var(--line);border-radius:8px;padding:6px 8px;margin-bottom:6px;font-size:11px}
  .chip .a{color:var(--dim);font-size:10px} .f{color:var(--win);font-weight:700}

  .card{background:var(--panel2);border:1px solid var(--line);border-radius:11px;padding:12px;margin-bottom:10px}
  .card.esc{border-color:#4a2d6b}
  .card .h{font-weight:650;margin-bottom:4px} .row{display:flex;gap:8px;margin-top:9px;flex-wrap:wrap}
  .muted{color:var(--dim)} .empty{color:#3a4a5a;font-style:italic;font-size:13px}

  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;
    letter-spacing:.5px;padding:8px 10px;border-bottom:1px solid var(--line)}
  td{padding:9px 10px;border-bottom:1px solid #16202b}
  tr.clk{cursor:pointer} tr.clk:hover td{background:#131d28}
  .stg{font-size:10px;padding:2px 8px;border-radius:6px;background:#16212e;border:1px solid var(--line);color:var(--dim)}

  .feed{max-height:420px;overflow:auto;font-size:12.5px}
  .ev{display:flex;gap:9px;padding:5px 0;border-bottom:1px solid #16202b}
  .ev .t{color:#46596c;min-width:56px} .ev .who{color:var(--dim);min-width:150px}
  .ev.win .msg{color:var(--win)} .ev.warn .msg{color:var(--warn)} .ev.escalate .msg{color:var(--esc)}

  input,select,textarea{font:inherit;background:#0d1620;border:1px solid var(--line);color:var(--txt);
    border-radius:9px;padding:9px 11px;width:100%}
  textarea{min-height:280px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;white-space:pre}
  label{display:block;font-size:11px;color:var(--dim);margin:0 0 5px;text-transform:uppercase;letter-spacing:.5px}
  .field{margin-bottom:12px}
  .tag{font-size:9px;padding:2px 6px;border-radius:5px;border:1px solid var(--line);margin-right:6px;color:var(--dim)}
  .o{padding:7px 0;border-bottom:1px solid #16202b;font-size:12.5px}
  .tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
  .tabs button{background:#16212e;border:1px solid var(--line);color:var(--dim);padding:6px 11px;border-radius:8px;cursor:pointer;font:inherit}
  .tabs button.active{background:var(--accent);color:#06101a;border:0;font-weight:700}
</style>
</head>
<body>
  <aside class="side">
    <div class="brand"><div class="logo">K</div><div id="brand">Keystone</div></div>
    <div class="sechead">Operate</div>
    <nav class="nav" id="nav">
      <a data-v="command" class="active"><span class="ico">⚡</span>Command Center<span class="badge" id="nav-actions" style="display:none">0</span></a>
      <a data-v="agents"><span class="ico">🤖</span>Agents</a>
      <a data-v="pipeline"><span class="ico">📊</span>Pipeline</a>
      <a data-v="deals"><span class="ico">🏠</span>Deals</a>
    </nav>
    <div class="sechead">Grow</div>
    <nav class="nav">
      <a data-v="outreach"><span class="ico">✉️</span>Outreach</a>
      <a data-v="dispo"><span class="ico">📣</span>Disposition</a>
    </nav>
    <div class="sechead">Close</div>
    <nav class="nav">
      <a data-v="legal"><span class="ico">⚖️</span>Legal</a>
      <a data-v="contracts"><span class="ico">📄</span>Contracts</a>
      <a data-v="import"><span class="ico">📥</span>Import Leads</a>
      <a data-v="settings"><span class="ico">⚙️</span>Settings</a>
    </nav>
  </aside>

  <header class="top">
    <div><h1 id="co">Keystone Property Partners</h1><div class="sub" id="sub"></div></div>
    <span class="grow"></span>
    <span class="pill" id="mode-badge">LIVE</span>
    <span class="pill" id="llm">LLM —</span>
    <span class="pill" id="armed">dry-run</span>
    <span class="pill" id="tick">tick 0</span>
    <button class="btn sm" id="pausebtn" onclick="toggleRun()">⏸ Pause</button>
    <button class="btn sm" onclick="api('/api/tick',{}).then(load)">⏭ Step</button>
  </header>

  <main class="main">
    <!-- COMMAND CENTER -->
    <section class="view active" id="v-command">
      <h2 class="title">Command Center</h2>
      <p class="desc">The agents run the company. This is your last mile — approve, send, sign.</p>

      <!-- Atlas org health -->
      <div class="atlas-panel" id="atlas-panel">
        <div class="atlas-header">
          <div class="atlas-dot" id="atlas-dot"></div>
          <div>
            <div class="atlas-name">ATLAS · Chief of Staff</div>
            <div class="muted" style="font-size:12px" id="atlas-note">Loading…</div>
          </div>
        </div>
        <div class="atlas-teams" id="atlas-teams"></div>
        <div class="atlas-stats" id="atlas-stats"></div>
      </div>

      <!-- Onboarding card (shown only in live mode with empty pipeline) -->
      <div id="onboard-card" style="display:none">
        <div class="onboard">
          <h3>No leads yet — let's get started</h3>
          <p>Import a CSV of properties, add a lead manually, or let Scout monitor county records.</p>
          <div class="onboard-actions">
            <button class="btn primary" onclick="nav('import')">📥 Import leads CSV</button>
            <button class="btn" onclick="nav('import');setTimeout(()=>$('#tab-manual').click(),200)">✏️ Add one manually</button>
          </div>
        </div>
      </div>

      <div class="grid kpis" id="kpis"></div>
      <div class="panel" style="margin-top:16px">
        <h3>⚡ Action Queue — needs you</h3><div id="actions"></div>
      </div>
      <div class="grid cards2">
        <div class="panel"><h3>Live activity</h3><div class="feed" id="feed"></div></div>
        <div class="panel"><h3>Integration outbox (dry-run)</h3><div id="outbox" class="feed"></div></div>
      </div>
    </section>

    <!-- AGENTS — two pane layout -->
    <section class="view" id="v-agents">
      <h2 class="title">Agents</h2>
      <p class="desc">Click an agent to see detail and chat with them directly.</p>
      <div class="agent-layout">
        <div class="agent-list" id="agent-list"></div>
        <div class="agent-detail" id="agent-detail">
          <div class="ad-empty">Select an agent from the list to see details and chat.</div>
        </div>
      </div>
    </section>

    <!-- PIPELINE -->
    <section class="view" id="v-pipeline">
      <h2 class="title">Deal pipeline</h2><p class="desc">Every deal in its current stage.</p>
      <div class="panel"><div class="board" id="board"></div></div>
    </section>

    <!-- DEALS -->
    <section class="view" id="v-deals">
      <h2 class="title">Deals</h2><p class="desc">Click a row for detail, underwriting, and the legal memo.</p>
      <div id="dealdetail"></div>
      <div class="panel"><table><thead><tr><th>ID</th><th>Property</th><th>Stage</th>
        <th>Contract</th><th>Fee</th><th>Buyer</th></tr></thead><tbody id="dealrows"></tbody></table></div>
    </section>

    <!-- OUTREACH -->
    <section class="view" id="v-outreach">
      <h2 class="title">Outreach — Riley</h2><p class="desc">B2B contacts (realtors, partners, cash buyers). Drafts are dry-run.</p>
      <div class="grid cards2">
        <div class="panel"><h3>Queue & drafts</h3><div id="contacts" class="feed" style="max-height:520px"></div></div>
        <div class="panel"><h3>Add a contact</h3>
          <div class="field"><label>Name / business</label><input id="c-name" placeholder="e.g. DFW Cash Buyers LLC"></div>
          <div class="field"><label>Type</label><select id="c-kind">
            <option value="cashbuyer">Cash buyer</option><option value="cowholesale">Co-wholesale partner</option>
            <option value="realtor">Realtor</option></select></div>
          <div class="field"><label>Market</label><input id="c-market" value="Dallas, TX"></div>
          <div class="field"><label>Area / ZIP</label><input id="c-area" value="75215"></div>
          <button class="btn primary" onclick="addContact()">Queue for Riley</button>
        </div>
      </div>
    </section>

    <!-- DISPOSITION -->
    <section class="view" id="v-dispo">
      <h2 class="title">Disposition — sell the contract</h2>
      <p class="desc">Where to assign/sell and how you get paid.</p>
      <div class="grid cards3" id="dispo"></div>
    </section>

    <!-- LEGAL -->
    <section class="view" id="v-legal">
      <h2 class="title">Legal — Counsel-AI</h2>
      <p class="desc">Informational triage, <b>not legal advice</b>. Prepares questions for your attorney.</p>
      <div class="grid cards2">
        <div class="panel"><h3>Run a triage</h3>
          <div class="field"><label>State</label><select id="l-state">
            <option>TX</option><option>FL</option><option>GA</option><option>OH</option><option>NC</option></select></div>
          <div class="field"><label><input type="checkbox" id="l-pf" style="width:auto"> Pre-foreclosure seller</label></div>
          <button class="btn primary" onclick="runLegal()">Analyze</button>
          <div id="legalout" style="margin-top:14px"></div>
        </div>
        <div class="panel"><h3>Deal legal memos</h3><div id="legalmemos" class="feed"></div></div>
      </div>
    </section>

    <!-- CONTRACTS -->
    <section class="view" id="v-contracts">
      <h2 class="title">Contracts</h2>
      <p class="desc">Generate the attorney-review packet for a deal: mail letter, disclosure, PSA, assignment.</p>
      <div class="panel">
        <div class="row"><select id="ct-deal" style="max-width:420px"></select>
          <button class="btn primary" onclick="loadContract()">Generate packet</button></div>
        <div class="tabs" id="ct-tabs" style="margin-top:14px"></div>
        <textarea id="ct-body" readonly placeholder="Pick a deal and generate…"></textarea>
      </div>
    </section>

    <!-- IMPORT LEADS -->
    <section class="view" id="v-import">
      <h2 class="title">Import Leads</h2>
      <p class="desc">Add real properties to the pipeline. The system processes them autonomously until it needs your decision.</p>
      <div class="tabs" style="margin-bottom:16px">
        <button class="active" id="tab-csv" onclick="showImportTab('csv',this)">📥 CSV Upload</button>
        <button id="tab-manual" onclick="showImportTab('manual',this)">✏️ Single Property</button>
        <button onclick="showImportTab('format',this)">📋 CSV Format</button>
      </div>

      <div id="import-csv">
        <div class="panel">
          <h3>Paste or upload CSV</h3>
          <p class="muted" style="font-size:12px;margin-bottom:12px">
            Required columns: address, city, state, zip, metro, beds, baths, sqft, year_built,
            property_type, distress, est_market_value, seller_name, motivation, asking_price, reachable_via
          </p>
          <div class="field">
            <label>CSV file</label>
            <input type="file" id="csv-file" accept=".csv" onchange="readCsvFile(this)" style="padding:7px">
          </div>
          <div class="field">
            <label>Or paste CSV text</label>
            <textarea id="csv-text" style="min-height:160px;font-size:11px"
              placeholder="address,city,state,zip,metro,beds,baths,sqft,year_built,property_type,distress,est_market_value,seller_name,motivation,asking_price,reachable_via&#10;123 Oak St,Dallas,TX,75201,TX-Dallas,3,2,1400,1985,SFR,pre-foreclosure,280000,John Smith,distress,310000,mail"></textarea>
          </div>
          <button class="btn primary" onclick="importCsv()">Import leads</button>
          <div id="import-result" style="margin-top:12px"></div>
        </div>
      </div>

      <div id="import-manual" style="display:none">
        <div class="panel">
          <h3>Add a single property</h3>
          <div class="grid cards2" style="gap:12px">
            <div>
              <div class="field"><label>Street address</label><input id="m-addr" placeholder="123 Elm St"></div>
              <div class="field"><label>City</label><input id="m-city" placeholder="Dallas"></div>
              <div class="field"><label>State</label><select id="m-state">
                <option>TX</option><option>FL</option><option>GA</option><option>OH</option><option>NC</option></select></div>
              <div class="field"><label>ZIP</label><input id="m-zip" placeholder="75201"></div>
            </div>
            <div>
              <div class="field"><label>Est. market value ($)</label><input id="m-arv" type="number" placeholder="300000"></div>
              <div class="field"><label>Seller asking price ($)</label><input id="m-ask" type="number" placeholder="330000"></div>
              <div class="field"><label>Seller name</label><input id="m-seller" placeholder="Jane Doe"></div>
              <div class="field"><label>Motivation</label><select id="m-motivation">
                <option value="distress">Distress / pre-foreclosure</option>
                <option value="tax-delinquent">Tax delinquent</option>
                <option value="tired-landlord">Tired landlord</option>
                <option value="inherited">Inherited / probate</option>
                <option value="divorce">Divorce</option>
                <option value="job-relocation">Job relocation</option></select></div>
              <div class="field"><label>Reachable via</label><select id="m-reach">
                <option value="mail">Mail</option><option value="phone">Phone</option><option value="email">Email</option></select></div>
            </div>
          </div>
          <button class="btn primary" onclick="addLead()">Add to pipeline</button>
          <div id="add-result" style="margin-top:12px"></div>
        </div>
      </div>

      <div id="import-format" style="display:none">
        <div class="panel">
          <h3>CSV column reference</h3>
          <table><thead><tr><th>Column</th><th>Required</th><th>Example</th><th>Notes</th></tr></thead>
          <tbody>
            <tr><td>address</td><td>✓</td><td>123 Oak St</td><td>Street address</td></tr>
            <tr><td>city</td><td>✓</td><td>Dallas</td><td></td></tr>
            <tr><td>state</td><td>✓</td><td>TX</td><td>2-letter code</td></tr>
            <tr><td>zip</td><td>✓</td><td>75201</td><td>ZIP code</td></tr>
            <tr><td>metro</td><td>✓</td><td>TX-Dallas</td><td>TX-Dallas, FL-Tampa, OH-Columbus…</td></tr>
            <tr><td>beds</td><td></td><td>3</td><td>Defaults to 3</td></tr>
            <tr><td>baths</td><td></td><td>2</td><td>Defaults to 2</td></tr>
            <tr><td>sqft</td><td></td><td>1400</td><td>Square feet</td></tr>
            <tr><td>year_built</td><td></td><td>1985</td><td></td></tr>
            <tr><td>property_type</td><td></td><td>SFR</td><td>SFR, MFR, Condo, Land</td></tr>
            <tr><td>distress</td><td></td><td>pre-foreclosure</td><td>pre-foreclosure, tax-delinquent, tired-landlord…</td></tr>
            <tr><td>est_market_value</td><td>✓</td><td>280000</td><td>ARV / as-repaired value</td></tr>
            <tr><td>seller_name</td><td>✓</td><td>John Smith</td><td></td></tr>
            <tr><td>motivation</td><td></td><td>distress</td><td>distress, relocation, probate…</td></tr>
            <tr><td>asking_price</td><td>✓</td><td>310000</td><td>Seller's asking price</td></tr>
            <tr><td>reachable_via</td><td></td><td>mail</td><td>mail, phone, email</td></tr>
            <tr><td>flexibility</td><td></td><td>0.35</td><td>0=rigid, 1=very flexible</td></tr>
            <tr><td>walk_floor</td><td></td><td>168000</td><td>Minimum the seller will accept</td></tr>
          </tbody></table>
        </div>
      </div>
    </section>

    <!-- SETTINGS -->
    <section class="view" id="v-settings">
      <h2 class="title">Settings & safety</h2><p class="desc">Safe-by-default controls.</p>
      <div class="panel"><h3>Lead sourcing mode</h3>
        <p id="set-mode" class="muted"></p>
        <p class="muted">Set <code>WS_LEAD_SOURCE=demo</code> to enable synthetic auto-generation for testing.
          Default is <code>live</code> — pipeline starts empty and only fills from imported data.</p>
      </div>
      <div class="panel"><h3>Integrations</h3>
        <p id="set-armed" class="muted"></p>
        <p class="muted">Outbound CRM/email/Slack are <b>dry-run</b> until armed
          (<code>WS_INTEGRATIONS_ARMED=1</code>) <i>and</i> a transport is wired. Nothing
          leaves the machine before you authorize it.</p>
      </div>
      <div class="panel"><h3>Compliance gate</h3>
        <p class="muted">Real homeowner outreach is blocked per-state until you attest:
          entity, licensing/exempt, attorney engaged, contracts reviewed, foreclosure-law
          cleared, TCPA/DNC process, compliant data. The buyer / co-wholesale side needs none
          of this — start there.</p></div>
      <div class="panel"><h3>LLM / AI reasoning</h3>
        <p class="muted" id="set-llm"></p>
        <p class="muted">Priority: <code>ANTHROPIC_API_KEY</code> → Claude subscription CLI → heuristic fallback.
          Run with <code>--subscription</code> to use your Claude.ai plan without an API key.</p>
      </div>
    </section>
  </main>

<script>
const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
const fmt=n=>'$'+(n||0).toLocaleString('en-US');
let STATE={};
let SELECTED_AGENT=null;
let CHAT_LOGS={};  // code -> [{role,text}]

async function api(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  return r.json().catch(()=>({}));
}
const decide=(id,d)=>api('/api/decide',{deal_id:id,decision:d}).then(load);
const act=(id,d)=>api('/api/action',{action_id:id,decision:d}).then(load);

/* nav */
$$('.nav a').forEach(a=>a.onclick=()=>nav(a.dataset.v));
function nav(v){
  $$('.nav a').forEach(x=>x.classList.toggle('active',x.dataset.v===v));
  $$('.view').forEach(x=>x.classList.remove('active'));
  $('#v-'+v).classList.add('active');
  if(v==='contracts') fillDealSelect();
}

function toggleRun(){
  const paused=STATE.paused;
  api('/api/control',{cmd:paused?'resume':'pause'}).then(load);
}

function kpi(v,l,c=''){return `<div class="kpi ${c}"><div class="v">${v}</div><div class="l">${l}</div></div>`}

const ACT_ICON={approve:'🟣',sign:'✍️',list:'📣',send:'✉️'};
function actionRow(a){
  const v=a.value?` · <span class="f">${fmt(a.value)}</span>`:'';
  let btn;
  if(a.kind==='approve') btn=`<button class="btn approve sm" onclick="act('${a.id}','approved')">Approve</button>
    <button class="btn reject sm" onclick="act('${a.id}','rejected')">Reject</button>`;
  else if(a.kind==='send') btn=`<button class="btn approve sm" onclick="act('${a.id}')">Mark sent</button>`;
  else btn=`<button class="btn sm" onclick="act('${a.id}')">Done</button>`;
  return `<div class="card esc"><div class="h">${ACT_ICON[a.kind]||'•'} ${a.title}${v}</div>
    <div class="muted">${a.detail}</div><div class="row">${btn}</div></div>`;
}

const LANES=['sourced','underwriting','outreach','negotiation','under_contract','disposition','assigned','closing'];
const LL={sourced:'Sourced',underwriting:'Underwrite',outreach:'Outreach',negotiation:'Negotiate',
  under_contract:'Contract',disposition:'Dispo',assigned:'Assigned',closing:'Closing'};

function renderAtlas(s){
  const ov=s.oversight||{};
  const health=ov.health||'green';
  const dot=$('#atlas-dot');
  dot.className='atlas-dot '+health;
  $('#atlas-note').textContent=ov.note||'Monitoring the organisation…';
  const teams=ov.teams||{};
  $('#atlas-teams').innerHTML=Object.entries(teams).map(([t,d])=>
    `<div class="atlas-team">${t} <b style="color:var(--txt)">${d.active||0}</b> active</div>`).join('');
  const qa=ov.qa_failures||0, pol=ov.policy_blocked||0, hum=ov.needs_human||0;
  $('#atlas-stats').innerHTML=
    `<div class="atlas-stat">QA failures: <span>${qa}</span></div>`+
    `<div class="atlas-stat">Policy blocks: <span>${pol}</span></div>`+
    `<div class="atlas-stat">Needs CEO: <span>${hum||'0'}</span></div>`;
}

function renderAgentList(s){
  $('#agent-list').innerHTML=s.agents.map(a=>`
    <div class="agent-row${SELECTED_AGENT===a.code?' sel':''}" onclick="selectAgent('${a.code}')"
         style="border-left-color:${a.color}">
      <div class="ar-dot ${a.status}"></div>
      <div style="flex:1;min-width:0">
        <div class="ar-name">${a.name.split('(')[0].trim()}</div>
        <div class="ar-meta">${a.team} · ${a.status}</div>
      </div>
      <div class="ar-count">${a.handled}</div>
    </div>`).join('');
}

function selectAgent(code){
  SELECTED_AGENT=code;
  renderAgentList(STATE);
  const a=STATE.agents.find(x=>x.code===code);
  if(!a) return;
  const isChief=code==='CHIEF';
  const ov=STATE.oversight||{};
  const recentActivity=(STATE.activity||[]).filter(e=>e.agent===a.name).slice(-10).reverse();

  // Atlas oversight block
  let chiefBlock='';
  if(isChief && Object.keys(ov).length){
    const teams=ov.teams||{};
    chiefBlock=`<div class="ad-section">
      <div class="ad-section-title">Org health report</div>
      <div style="margin-bottom:8px">
        <span class="atlas-dot ${ov.health||'green'}" style="display:inline-block;margin-right:8px;vertical-align:middle"></span>
        <b style="text-transform:uppercase">${ov.health||'green'}</b> — ${ov.note||''}
      </div>
      ${Object.entries(teams).map(([t,d])=>`<div class="o"><b>${t}</b> · ${d.agents?.length||0} agents · ${d.active||0} active</div>`).join('')}
      <div class="atlas-stats" style="margin-top:10px">
        <div class="atlas-stat">QA failures: <span>${ov.qa_failures||0}</span></div>
        <div class="atlas-stat">Policy blocks: <span>${ov.policy_blocked||0}</span></div>
        <div class="atlas-stat">Needs CEO: <span>${ov.needs_human||0}</span></div>
      </div>
    </div>`;
  }

  if(!CHAT_LOGS[code]) CHAT_LOGS[code]=[];
  const chatHtml=CHAT_LOGS[code].map(m=>
    `<div class="msg-${m.role}"><b>${m.role==='ceo'?'You':''+a.name.split('(')[0].trim()}:</b> ${m.text}</div>`).join('');

  $('#agent-detail').innerHTML=`
    <div class="ad-header">
      <div style="display:flex;align-items:center;gap:12px">
        <div style="width:14px;height:36px;border-radius:4px;background:${a.color};flex-shrink:0"></div>
        <div>
          <div class="ad-name">${a.name}</div>
          <div class="ad-role">${a.role}</div>
        </div>
      </div>
      <div class="ad-badges" style="margin-top:12px">
        <div class="ad-badge">${a.team}</div>
        <div class="ad-badge ${a.status==='idle'?'':'ad-badge-active'}">${a.status}</div>
        <div class="ad-badge">${a.handled} deals handled</div>
      </div>
    </div>
    <div class="ad-body">
      <div class="ad-section">
        <div class="ad-section-title">Description</div>
        <div style="color:var(--dim);font-size:13px">${a.desc}</div>
      </div>
      <div class="ad-section">
        <div class="ad-section-title">Current state</div>
        <div class="ad-stat-row">
          <div class="ad-stat"><span>Status</span><br><b>${a.status}</b></div>
          <div class="ad-stat"><span>Task</span><br><b>${a.task}</b></div>
          ${a.deal?`<div class="ad-stat"><span>Deal</span><br><b>#${a.deal}</b></div>`:''}
        </div>
      </div>
      ${chiefBlock}
      <div class="ad-section">
        <div class="ad-section-title">Recent activity (${recentActivity.length})</div>
        <div class="ad-activity">
          ${recentActivity.length?recentActivity.map(e=>{
            const t=new Date(e.ts*1000).toLocaleTimeString('en-US',{hour12:false});
            return `<div class="ev ${e.level}"><span class="t">${t}</span><span class="msg">${e.message}</span></div>`;
          }).join(''):'<div class="empty">No recent activity.</div>'}
        </div>
      </div>
    </div>
    <div class="ad-chat">
      <div class="ad-section-title" style="margin-bottom:8px">Chat with ${a.name.split('(')[0].trim()}</div>
      <div class="ad-chat-log" id="chat-log-${code}">${chatHtml||'<div class="empty">Send a message to get a reply from this agent.</div>'}</div>
      <div class="ad-chat-input">
        <input id="chat-input-${code}" placeholder="Message ${a.name.split('(')[0].trim()}…"
          onkeydown="if(event.key==='Enter')sendMsg('${code}')">
        <button class="btn primary sm" onclick="sendMsg('${code}')">Send</button>
      </div>
    </div>`;
}

async function sendMsg(code){
  const inp=$('#chat-input-'+code);
  const msg=inp.value.trim(); if(!msg) return;
  inp.value=''; inp.disabled=true;
  if(!CHAT_LOGS[code]) CHAT_LOGS[code]=[];
  CHAT_LOGS[code].push({role:'ceo',text:msg});
  selectAgent(code);
  const r=await api('/api/agent/message',{code,message:msg});
  inp.disabled=false;
  if(r.ok && r.reply){
    CHAT_LOGS[code].push({role:'agent',text:r.reply});
    selectAgent(code);
    const log=$('#chat-log-'+code);
    if(log) log.scrollTop=log.scrollHeight;
  }
}

function render(s){
  STATE=s;
  $('#co').textContent=s.company; $('#brand').textContent=s.company;
  $('#sub').textContent='CEO: '+s.ceo+' · up '+Math.floor(s.uptime_s)+'s';
  $('#tick').textContent='tick '+s.tick;
  $('#llm').textContent='LLM '+(s.llm.available?s.llm.model:'heuristic');
  $('#llm').className='pill '+(s.llm.available?'on':'off');
  $('#armed').textContent=s.armed?'ARMED':'dry-run';
  $('#armed').className='pill '+(s.armed?'off':'on');
  $('#pausebtn').textContent=s.paused?'▶ Resume':'⏸ Pause';

  // Mode badge
  const isDemo=(s.lead_source||'live')==='demo';
  $('#mode-badge').textContent=isDemo?'DEMO MODE':'LIVE';
  $('#mode-badge').className='pill '+(isDemo?'demo':'live');

  // Onboarding card
  const emptyLive=!isDemo&&(!s.deals||s.deals.length===0);
  $('#onboard-card').style.display=emptyLive?'block':'none';

  const f=s.financials;
  $('#kpis').innerHTML=kpi(fmt(f.revenue),'Revenue','win')+kpi(f.closed,'Closed won')+
    kpi(f.active,'Active')+kpi(fmt(f.pipeline_fee_value),'Pipeline value')+
    kpi(fmt(f.avg_fee),'Avg fee')+kpi(f.dead,'Dead');

  const aq=s.action_queue||[];
  const nb=$('#nav-actions'); nb.style.display=aq.length?'inline-block':'none'; nb.textContent=aq.length;
  $('#actions').innerHTML=aq.length?aq.map(actionRow).join(''):'<div class="empty">Nothing needs you — agents are working.</div>';

  $('#feed').innerHTML=s.activity.slice().reverse().map(e=>{
    const t=new Date(e.ts*1000).toLocaleTimeString('en-US',{hour12:false});
    return `<div class="ev ${e.level}"><span class="t">${t}</span><span class="who">${e.agent}</span><span class="msg">${e.message}</span></div>`;
  }).join('');
  $('#outbox').innerHTML=(s.outbox||[]).length?s.outbox.slice().reverse().map(o=>`<div class="o"><span class="tag">${o.channel}</span>${o.summary}</div>`).join(''):'<div class="empty">Nothing queued.</div>';

  // Atlas panel
  renderAtlas(s);

  // Agents
  renderAgentList(s);
  if(SELECTED_AGENT) selectAgent(SELECTED_AGENT);

  const byStage={}; LANES.forEach(l=>byStage[l]=[]);
  s.deals.forEach(d=>{if(byStage[d.stage])byStage[d.stage].push(d)});
  $('#board').innerHTML=LANES.map(l=>`<div class="lane"><h4>${LL[l]}<b>${s.counts[l]||0}</b></h4>
    ${byStage[l].slice(-8).map(d=>`<div class="chip"><div>${d.property.address}</div>
      <div class="a">${(d.property.metro||'').split('-')[1]||d.property.city}${d.assignment_fee?' · <span class="f">'+fmt(d.assignment_fee)+'</span>':''}</div></div>`).join('')||'<div class="empty">—</div>'}</div>`).join('');

  $('#dealrows').innerHTML=s.deals.slice().reverse().map(d=>`<tr class="clk" onclick="showDeal(${d.id})">
    <td>#${d.id}</td><td>${d.property.address}, ${d.property.city}</td><td><span class="stg">${d.stage}</span></td>
    <td>${d.contract_price?fmt(d.contract_price):'—'}</td><td>${d.assignment_fee?'<span class="f">'+fmt(d.assignment_fee)+'</span>':'—'}</td>
    <td>${d.buyer?d.buyer.name:'—'}</td></tr>`).join('');

  $('#contacts').innerHTML=(s.contacts||[]).slice().reverse().map(c=>`<div class="o">
    <span class="tag">${c.kind}</span><b>${c.name}</b> <span class="muted">[${c.status}]</span>
    ${c.url?'<br><span class="muted">'+c.url+'</span>':''}<br>${c.drafted||'queued'}</div>`).join('')||'<div class="empty">No contacts.</div>';

  $('#dispo').innerHTML=(s.dispo_platforms||[]).map(p=>`<div class="card"><div class="h">${p.name}
    <span class="tag" style="float:right">${p.payout}</span></div>
    <div class="muted">${p.best_for}</div><div class="muted" style="margin-top:6px">${p.cost}</div>
    <div class="row"><a class="btn sm" href="${/^http/.test(p.url)?p.url:'#'}" target="_blank">Open</a></div></div>`).join('');

  const memos=(s.deals||[]).filter(d=>d.legal_memo&&d.legal_memo.analysis);
  $('#legalmemos').innerHTML=memos.length?memos.slice(-8).reverse().map(d=>`<div class="o">
    <b>${d.legal_memo.state} · #${d.id}</b> <span class="tag">${d.legal_memo.source}</span><br>
    <span class="muted">${(d.legal_memo.analysis.split('\n')[0]||'')}</span></div>`).join(''):'<div class="empty">Memos appear at contract stage.</div>';

  $('#set-mode').textContent=(s.lead_source||'live')==='demo'
    ?'DEMO MODE — synthetic leads auto-generated every tick. Set WS_LEAD_SOURCE=live to use real data.'
    :'LIVE MODE — pipeline starts empty. Import leads via the "Import Leads" section.';
  $('#set-armed').textContent=s.armed?'ARMED — live sends possible if wired.':'Dry-run — nothing is sent.';
  $('#set-llm').textContent=(s.llm.available?s.llm.describe+' ('+s.llm.calls+' calls)':'Heuristic fallback — no Claude available. Run with --subscription or set ANTHROPIC_API_KEY.');
}

function showDeal(id){
  const d=(STATE.deals||[]).find(x=>x.id===id); if(!d)return;
  const u=d.uw||{}; const m=d.legal_memo||{};
  $('#dealdetail').innerHTML=`<div class="panel"><h3>${d.label}</h3>
    <div class="grid cards3">
      <div><div class="muted">ARV</div><b>${fmt(u.arv)}</b></div>
      <div><div class="muted">Repairs</div><b>${fmt(u.repair_estimate)}</b></div>
      <div><div class="muted">MAO</div><b>${fmt(u.mao)}</b></div>
      <div><div class="muted">Contract</div><b>${fmt(d.contract_price)}</b></div>
      <div><div class="muted">Assignment fee</div><b class="f">${fmt(d.assignment_fee)}</b></div>
      <div><div class="muted">Stage</div><b>${d.stage}</b></div>
    </div>
    ${d.flags&&d.flags.length?'<p class="muted" style="margin-top:10px">⚑ '+d.flags.join(' · ')+'</p>':''}
    ${m.analysis?'<p class="muted" style="margin-top:10px"><b>Legal ('+m.state+'):</b> '+m.analysis.split('\n')[0]+'</p>':''}
    <div style="margin-top:10px">${(d.history||[]).map(h=>'<div class="o">'+h+'</div>').join('')}</div>
    <div class="row"><button class="btn sm" onclick="$('#dealdetail').innerHTML=''">Close</button></div></div>`;
  $('#dealdetail').scrollIntoView({behavior:'smooth',block:'nearest'});
}

function addContact(){
  api('/api/contact',{name:$('#c-name').value,kind:$('#c-kind').value,market:$('#c-market').value,area:$('#c-area').value})
    .then(()=>{$('#c-name').value='';load()});
}
function runLegal(){
  $('#legalout').innerHTML='<span class="muted">Analyzing…</span>';
  api('/api/legal',{state:$('#l-state').value,pre_foreclosure:$('#l-pf').checked}).then(m=>{
    $('#legalout').innerHTML=`<div class="card"><div class="h">${m.state} triage <span class="tag">${m.source}</span></div>
      <div class="muted" style="white-space:pre-wrap">${(m.analysis||'').replace(/</g,'&lt;')}</div>
      <p class="muted" style="margin-top:8px"><i>${m.disclaimer||''}</i></p></div>`;
  });
}

/* Import leads */
function showImportTab(tab, btn){
  ['csv','manual','format'].forEach(t=>{
    const el=$('#import-'+t); if(el) el.style.display=t===tab?'block':'none';
  });
  $$('#v-import .tabs button').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
}

function readCsvFile(inp){
  const f=inp.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=e=>$('#csv-text').value=e.target.result;
  r.readAsText(f);
}

async function importCsv(){
  const csv=$('#csv-text').value.trim();
  if(!csv){$('#import-result').innerHTML='<span class="muted">Paste CSV text or upload a file first.</span>';return;}
  $('#import-result').innerHTML='<span class="muted">Importing…</span>';
  const r=await api('/api/leads/import',{csv});
  if(r.ok){
    $('#import-result').innerHTML=`<span style="color:var(--win)">✓ Imported ${r.imported} lead(s) into the pipeline.</span>`;
    $('#csv-text').value=''; load();
  } else {
    $('#import-result').innerHTML=`<span style="color:var(--danger)">Error: ${r.error||'unknown'}</span>`;
  }
}

async function addLead(){
  const addr=$('#m-addr').value.trim();
  const city=$('#m-city').value.trim();
  const arv=parseInt($('#m-arv').value)||0;
  const ask=parseInt($('#m-ask').value)||0;
  const seller=$('#m-seller').value.trim();
  if(!addr||!city||!arv||!seller){
    $('#add-result').innerHTML='<span style="color:var(--danger)">Fill in address, city, market value, and seller name.</span>';
    return;
  }
  const r=await api('/api/leads/add',{
    address:addr, city, state:$('#m-state').value,
    zip:$('#m-zip').value||'00000',
    market_value:arv, asking_price:ask||arv,
    seller_name:seller,
    motivation:$('#m-motivation').value,
    reachable_via:$('#m-reach').value,
  });
  if(r.ok){
    $('#add-result').innerHTML='<span style="color:var(--win)">✓ Lead added to pipeline.</span>';
    [$('#m-addr'),$('#m-city'),$('#m-arv'),$('#m-ask'),$('#m-seller'),$('#m-zip')].forEach(e=>{if(e)e.value='';});
    load();
  } else {
    $('#add-result').innerHTML=`<span style="color:var(--danger)">Error: ${r.error||'unknown'}</span>`;
  }
}

let CT_PACKET={};
function fillDealSelect(){
  const sel=$('#ct-deal'); const cur=sel.value;
  sel.innerHTML=(STATE.deals||[]).slice().reverse().map(d=>`<option value="${d.id}">#${d.id} — ${d.property.address}, ${d.property.city} ${d.property.state}</option>`).join('');
  if(cur)sel.value=cur;
}
function loadContract(){
  const id=$('#ct-deal').value; if(!id)return;
  $('#ct-body').value='Generating…';
  fetch('/api/contract?deal_id='+id).then(r=>r.json()).then(p=>{
    CT_PACKET=p;
    const keys=Object.keys(p);
    $('#ct-tabs').innerHTML=keys.map((k,i)=>`<button class="${i===0?'active':''}" onclick="showCt('${k}',this)">${k.replace(/_/g,' ')}</button>`).join('');
    $('#ct-body').value=p[keys[0]]||'';
  });
}
function showCt(k,btn){$$('#ct-tabs button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');$('#ct-body').value=CT_PACKET[k]||'';}

async function load(){ try{const r=await fetch('/api/state');render(await r.json());}catch(e){} }
load(); setInterval(load,1800);
</script>
</body>
</html>"""
