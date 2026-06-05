"""The control-center UI (single-page app, vanilla JS, no build step)."""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Wholesale Ops — Control Center</title>
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

  /* Sidebar */
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

  /* Top bar */
  .top{grid-area:top;border-bottom:1px solid var(--line);background:var(--bg2);
    display:flex;align-items:center;gap:14px;padding:0 20px}
  .top h1{font-size:16px;margin:0}
  .top .sub{color:var(--dim);font-size:12px}
  .grow{flex:1}
  .pill{font-size:11px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);color:var(--dim)}
  .pill.on{color:var(--win);border-color:#1d5b39} .pill.off{color:var(--warn);border-color:#5b421d}
  .btn{font:inherit;border:1px solid var(--line);background:#16212e;color:var(--txt);
    padding:7px 13px;border-radius:9px;cursor:pointer;font-weight:600}
  .btn:hover{filter:brightness(1.2)}
  .btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#06101a;border:0}
  .btn.approve{background:#143a26;border-color:#1d5b39;color:var(--win)}
  .btn.reject{background:#3a1a1a;border-color:#5b2424;color:#ff8b8b}
  .btn.sm{padding:5px 10px;font-size:12px}

  /* Main */
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

  .agent{background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:13px;border-left:4px solid #456}
  .agent .top2{display:flex;justify-content:space-between;align-items:center}
  .agent .nm{font-weight:650} .agent .role{color:var(--dim);font-size:11px;margin:2px 0 8px}
  .agent .task{font-size:12.5px;min-height:34px} .agent .meta{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;margin-top:8px}
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

  .card{background:var(--panel2);border:1px solid #33405200;border:1px solid var(--line);border-radius:11px;padding:12px;margin-bottom:10px}
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
    <div class="brand"><div class="logo">K</div><div id="brand">Wholesale Ops</div></div>
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
      <a data-v="settings"><span class="ico">⚙️</span>Settings</a>
    </nav>
  </aside>

  <header class="top">
    <div><h1 id="co">Wholesale Ops</h1><div class="sub" id="sub"></div></div>
    <span class="grow"></span>
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
      <div class="grid kpis" id="kpis"></div>
      <div class="panel" style="margin-top:16px">
        <h3>⚡ Action Queue — needs you</h3><div id="actions"></div>
      </div>
      <div class="grid cards2">
        <div class="panel"><h3>Live activity</h3><div class="feed" id="feed"></div></div>
        <div class="panel"><h3>Integration outbox (dry-run)</h3><div id="outbox" class="feed"></div></div>
      </div>
    </section>

    <!-- AGENTS -->
    <section class="view" id="v-agents">
      <h2 class="title">Agents</h2><p class="desc">Your autonomous workforce, live.</p>
      <div class="grid cards3" id="agents"></div>
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

    <!-- SETTINGS -->
    <section class="view" id="v-settings">
      <h2 class="title">Settings & safety</h2><p class="desc">Safe-by-default controls.</p>
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
      <div class="panel"><h3>LLM</h3><p class="muted" id="set-llm"></p>
        <p class="muted">Set <code>ANTHROPIC_API_KEY</code> to switch agents from heuristics to Claude.</p></div>
    </section>
  </main>

<script>
const $=s=>document.querySelector(s), $$=s=>document.querySelectorAll(s);
const fmt=n=>'$'+(n||0).toLocaleString('en-US');
let STATE={};

async function api(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  return r.json().catch(()=>({}));
}
const decide=(id,d)=>api('/api/decide',{deal_id:id,decision:d}).then(load);
const act=(id,d)=>api('/api/action',{action_id:id,decision:d}).then(load);

/* nav */
$('#nav');
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

function render(s){
  STATE=s;
  $('#co').textContent=s.company; $('#brand').textContent=s.company;
  $('#sub').textContent='CEO: '+s.ceo+' · up '+Math.floor(s.uptime_s)+'s';
  $('#tick').textContent='tick '+s.tick;
  $('#llm').textContent='LLM '+(s.llm.available?s.llm.model:'heuristic'); $('#llm').className='pill '+(s.llm.available?'on':'off');
  $('#armed').textContent=s.armed?'ARMED':'dry-run'; $('#armed').className='pill '+(s.armed?'off':'on');
  $('#pausebtn').textContent=s.paused?'▶ Resume':'⏸ Pause';

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

  $('#agents').innerHTML=s.agents.map(a=>`<div class="agent" style="border-left-color:${a.color}">
    <div class="top2"><span class="nm">${a.name}</span><span class="dot ${a.status}"></span></div>
    <div class="role">${a.role}</div><div class="task">${a.task}${a.deal?' <span class="muted">#'+a.deal+'</span>':''}</div>
    <div class="meta"><span>${a.status}</span><span>${a.handled} actions</span></div></div>`).join('');

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

  $('#set-armed').textContent=s.armed?'ARMED — live sends possible if wired.':'Dry-run — nothing is sent.';
  $('#set-llm').textContent=(s.llm.available?'Claude active ('+s.llm.model+'), '+s.llm.calls+' calls.':'Heuristic fallback (no API key).');
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
</html>
"""
