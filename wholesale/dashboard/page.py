"""The dashboard HTML/CSS/JS, served as one static string.

Vanilla JS, no build step, no external assets — matches the repo's
`brain_web.py` philosophy. Polls /api/state and re-renders.
"""

from __future__ import annotations

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Wholesale Ops — CEO Dashboard</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#131a23; --panel2:#0f151d; --line:#22303d;
    --txt:#e6edf3; --dim:#8499ab; --accent:#4cc9f0;
    --win:#43d17a; --warn:#f4a259; --esc:#c77dff; --info:#5a708a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font:14px/1.45 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
  a{color:var(--accent)}
  header{display:flex;align-items:center;gap:18px;padding:14px 20px;
    border-bottom:1px solid var(--line);background:linear-gradient(180deg,#101722,#0b0f14)}
  header h1{font-size:17px;margin:0;letter-spacing:.3px}
  header .sub{color:var(--dim);font-size:12px}
  .pill{font-size:11px;padding:3px 9px;border-radius:999px;border:1px solid var(--line);
    color:var(--dim);white-space:nowrap}
  .pill.on{color:var(--win);border-color:#1d5b39}
  .pill.off{color:var(--warn);border-color:#5b421d}
  .grow{flex:1}
  main{padding:16px 20px;display:grid;gap:16px;
    grid-template-columns:1fr;max-width:1500px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .kpi .v{font-size:22px;font-weight:650}
  .kpi .l{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:3px}
  .kpi.win .v{color:var(--win)}
  section h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--dim);
    margin:0 0 8px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  .cols{display:grid;grid-template-columns:1.35fr .9fr;gap:16px}
  .agents{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .agent{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:11px 12px;
    border-left:4px solid var(--info)}
  .agent .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
  .agent .nm{font-weight:600;font-size:13px}
  .agent .role{color:var(--dim);font-size:11px;margin:1px 0 7px}
  .agent .task{font-size:12px;min-height:32px}
  .agent .meta{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;margin-top:7px}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--info)}
  .dot.thinking{background:var(--warn);box-shadow:0 0 8px var(--warn);animation:pulse 1s infinite}
  .dot.acting{background:var(--win);box-shadow:0 0 8px var(--win);animation:pulse 1s infinite}
  .dot.idle{background:#3a4a5a}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .board{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;overflow-x:auto}
  .lane{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px;min-height:80px}
  .lane h3{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);
    margin:0 0 6px;display:flex;justify-content:space-between}
  .lane h3 .c{color:var(--txt)}
  .chip{background:#15202b;border:1px solid var(--line);border-radius:7px;padding:5px 7px;
    margin-bottom:6px;font-size:11px;cursor:default}
  .chip .a{color:var(--dim);font-size:10px}
  .chip .f{color:var(--win);font-weight:600}
  .ceo .card{background:var(--panel2);border:1px solid #4a2d6b;border-radius:10px;padding:11px;margin-bottom:9px}
  .ceo .card .h{font-weight:600;margin-bottom:4px}
  .ceo .row{display:flex;gap:8px;margin-top:8px}
  button{font:inherit;border:1px solid var(--line);background:#1a2530;color:var(--txt);
    padding:6px 12px;border-radius:8px;cursor:pointer}
  button.approve{background:#143a26;border-color:#1d5b39;color:var(--win)}
  button.reject{background:#3a1a1a;border-color:#5b2424;color:#ff8b8b}
  button:hover{filter:brightness(1.2)}
  .feed{max-height:340px;overflow:auto;font-size:12px}
  .ev{display:flex;gap:8px;padding:4px 0;border-bottom:1px solid #18222c}
  .ev .who{color:var(--dim);min-width:120px}
  .ev.win .msg{color:var(--win)} .ev.warn .msg{color:var(--warn)}
  .ev.escalate .msg{color:var(--esc)} .ev .t{color:#475c70;min-width:54px}
  .outbox{max-height:200px;overflow:auto;font-size:11.5px;color:var(--dim)}
  .outbox .o{padding:3px 0;border-bottom:1px solid #18222c}
  .tag{font-size:9px;padding:1px 5px;border-radius:5px;border:1px solid var(--line);margin-right:6px}
  .muted{color:var(--dim)}
  .empty{color:#3a4a5a;font-style:italic;font-size:12px}
</style>
</head>
<body>
<header>
  <h1 id="co">Wholesale Ops</h1>
  <span class="sub" id="sub"></span>
  <span class="grow"></span>
  <span class="pill" id="llm">LLM —</span>
  <span class="pill" id="armed">integrations —</span>
  <span class="pill" id="tick">tick 0</span>
</header>
<main>
  <div class="kpis" id="kpis"></div>

  <section>
    <h2>Agents — who's doing what</h2>
    <div class="panel"><div class="agents" id="agents"></div></div>
  </section>

  <section>
    <h2>Deal pipeline</h2>
    <div class="panel"><div class="board" id="board"></div></div>
  </section>

  <div class="cols">
    <section>
      <h2>Activity feed</h2>
      <div class="panel"><div class="feed" id="feed"></div></div>
    </section>
    <section>
      <h2>CEO approval queue</h2>
      <div class="panel ceo" id="ceo"></div>
      <h2 style="margin-top:14px">Integration outbox (dry-run)</h2>
      <div class="panel"><div class="outbox" id="outbox"></div></div>
    </section>
  </div>

  <div class="cols">
    <section>
      <h2>BizDev outreach — Riley (B2B, dry-run)</h2>
      <div class="panel"><div class="outbox" id="bizdev"></div></div>
    </section>
    <section>
      <h2>Sell the contract — disposition channels</h2>
      <div class="panel"><div class="outbox" id="dispo"></div></div>
    </section>
  </div>

  <section>
    <h2>Legal triage — Counsel-AI (informational, not legal advice)</h2>
    <div class="panel"><div class="feed" id="legal"></div></div>
  </section>
</main>

<script>
const $ = s => document.querySelector(s);
const fmt = n => '$' + (n||0).toLocaleString('en-US');
const LANES = ['sourced','underwriting','outreach','negotiation','under_contract','disposition','assigned','closing'];
const LANE_LABEL = {sourced:'Sourced',underwriting:'Underwriting',outreach:'Outreach',
  negotiation:'Negotiation',under_contract:'Under contract',disposition:'Disposition',
  assigned:'Assigned',closing:'Closing'};

async function decide(id, decision){
  await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({deal_id:id,decision})});
  tick();
}

function kpi(v,l,cls=''){return `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${l}</div></div>`}

function render(s){
  $('#co').textContent = s.company;
  $('#sub').textContent = 'CEO: ' + s.ceo + ' · up ' + Math.floor(s.uptime_s) + 's';
  $('#tick').textContent = 'tick ' + s.tick;
  const llm = s.llm;
  $('#llm').textContent = 'LLM ' + (llm.available ? llm.model : 'heuristic') +
    (llm.calls ? ' · ' + llm.calls + ' calls' : '');
  $('#llm').className = 'pill ' + (llm.available ? 'on':'off');
  $('#armed').textContent = s.armed ? 'integrations ARMED' : 'integrations dry-run';
  $('#armed').className = 'pill ' + (s.armed ? 'off':'on');

  const f = s.financials;
  $('#kpis').innerHTML =
    kpi(fmt(f.revenue),'Revenue (fees)','win') +
    kpi(f.closed,'Closed won') +
    kpi(f.active,'Active deals') +
    kpi(fmt(f.pipeline_fee_value),'Pipeline value') +
    kpi(fmt(f.avg_fee),'Avg fee / deal') +
    kpi(f.dead,'Dead');

  $('#agents').innerHTML = s.agents.map(a => `
    <div class="agent" style="border-left-color:${a.color}">
      <div class="top"><span class="nm">${a.name}</span>
        <span class="dot ${a.status}"></span></div>
      <div class="role">${a.role}</div>
      <div class="task">${a.task}${a.deal? ' <span class="muted">#'+a.deal+'</span>':''}</div>
      <div class="meta"><span>${a.status}</span><span>${a.handled} actions</span></div>
    </div>`).join('');

  const byStage = {};
  LANES.forEach(l => byStage[l]=[]);
  s.deals.forEach(d => { if(byStage[d.stage]) byStage[d.stage].push(d); });
  $('#board').innerHTML = LANES.map(l => `
    <div class="lane"><h3>${LANE_LABEL[l]}<span class="c">${s.counts[l]||0}</span></h3>
      ${byStage[l].slice(-8).map(d => `
        <div class="chip" title="${d.label}">
          <div>${d.property.address}</div>
          <div class="a">${d.property.metro.split('-')[1]||d.property.city}
            ${d.assignment_fee? '· <span class="f">'+fmt(d.assignment_fee)+'</span>':''}</div>
        </div>`).join('') || '<div class="empty">—</div>'}
    </div>`).join('');

  $('#ceo').innerHTML = s.ceo_queue.length ? s.ceo_queue.map(d => `
    <div class="card">
      <div class="h">${d.label}</div>
      <div class="muted">Contract ${fmt(d.contract_price)} · fee ${fmt(d.assignment_fee)}</div>
      <div class="muted" style="margin-top:4px">${(d.flags||[]).join(' · ')}</div>
      <div class="row">
        <button class="approve" onclick="decide(${d.id},'approved')">Approve</button>
        <button class="reject" onclick="decide(${d.id},'rejected')">Reject</button>
      </div>
    </div>`).join('') : '<div class="empty">No deals awaiting sign-off.</div>';

  $('#feed').innerHTML = s.activity.slice().reverse().map(e => {
    const t = new Date(e.ts*1000).toLocaleTimeString('en-US',{hour12:false});
    return `<div class="ev ${e.level}"><span class="t">${t}</span>
      <span class="who">${e.agent}</span><span class="msg">${e.message}</span></div>`;
  }).join('');

  $('#outbox').innerHTML = s.outbox.length ? s.outbox.slice().reverse().map(o =>
    `<div class="o"><span class="tag">${o.channel}</span>${o.summary}</div>`).join('')
    : '<div class="empty">Nothing queued.</div>';

  $('#bizdev').innerHTML = (s.contacts||[]).length ? s.contacts.slice().reverse().map(c =>
    `<div class="o"><span class="tag">${c.kind}</span><b>${c.name}</b>
      <span class="muted">[${c.status}]</span> — ${c.drafted||'queued'}</div>`).join('')
    : '<div class="empty">No contacts queued.</div>';

  $('#dispo').innerHTML = (s.dispo_platforms||[]).map(p =>
    `<div class="o"><span class="tag">${p.payout}</span>
      <b>${p.name}</b> <span class="muted">${p.cost}</span><br>
      <span class="muted">${p.best_for} — ${p.url}</span></div>`).join('');

  const memoDeals = (s.deals||[]).filter(d => d.legal_memo && d.legal_memo.analysis);
  $('#legal').innerHTML = memoDeals.length ? memoDeals.slice(-4).reverse().map(d => {
    const m = d.legal_memo;
    return `<div class="ev"><span class="who">${m.state} · #${d.id}</span>
      <span class="msg"><span class="muted">[${m.source}]</span> `
      + (m.analysis.split('\n')[0]||'').replace(/</g,'&lt;')
      + ` <span class="muted">— ${m.disclaimer}</span></span></div>`;
  }).join('') : '<div class="empty">No legal memos yet (generated at contract stage).</div>';
}

async function tick(){
  try{ const r = await fetch('/api/state'); render(await r.json()); }
  catch(e){ /* transient */ }
}
tick(); setInterval(tick, 1500);
</script>
</body>
</html>
"""
