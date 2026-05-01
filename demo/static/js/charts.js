// IndiaFinBench — Charts & Interactions
(function(){
const MODELS = window.IFB_MODELS;
const HUMAN = window.IFB_HUMAN;
const CLAUDE = window.IFB_CLAUDE;
const DIFF_DATA = window.IFB_DIFF;
const TASKS_DEF = window.IFB_TASKS;

const TASK_META={
  overall:{title:'Overall Accuracy',desc:'All 406 items · zero-shot · no fine-tuning'},
  reg:{title:'REG — Regulatory Interpretation',desc:'174 items · extract rules, thresholds and compliance deadlines'},
  num:{title:'NUM — Numerical Reasoning',desc:'92 items · arithmetic over regulatory figures'},
  con:{title:'CON — Contradiction Detection',desc:'62 items · identify contradictions between passages'},
  tmp:{title:'TMP — Temporal Reasoning',desc:'78 items · sequence amendments and effective dates'},
};

function scClass(v){return v>=85?'sc-hi':v>=70?'sc-md':'sc-lo'}

/* ═════ BAR CHART ═════ */
window.buildBarChart = function(task){
  const meta=TASK_META[task], key=task==='overall'?'overall':task;
  document.getElementById('chartTitle').textContent=meta.title;
  document.getElementById('chartDesc').textContent=meta.desc;
  const rows=[...MODELS.map(m=>({...m})),{...HUMAN}];
  rows.sort((a,b)=>(b[key]||0)-(a[key]||0));
  const container=document.getElementById('barChart');
  container.innerHTML='';

  rows.forEach((m,ri)=>{
    const score=m[key]||0;
    const row=document.createElement('div');
    row.className='bar-row';
    row.innerHTML=`
      <div class="bar-label">
        <span class="bar-rank">${m.is_human?'—':'#'+(ri+1)}</span>
        <span class="bar-name ${m.is_human?'human':''}">${m.label}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${m.is_human?'bar-fill-human':''}" data-pct="${score}" style="width:0">
          <div class="bar-shimmer"></div>
          <span class="bar-pct">${score.toFixed(1)}%</span>
        </div>
        <div class="bar-icon-wrap" style="background:#fff;border:2.5px solid ${m.color};box-shadow:0 2px 10px rgba(0,0,0,0.18)">
          ${m.logo
            ? `<img src="${m.logo}" style="width:24px;height:24px;object-fit:contain;" alt="${m.label}">`
            : `<span style="color:${m.color};font-size:10px;font-weight:800">${m.icon}</span>`}
        </div>
        <div class="bar-tooltip">
          <div class="tt-name" style="color:${m.color}">${m.label}</div>
          <div class="tt-grid">
            <span class="tt-k">Overall</span><span class="tt-v">${(m.overall||0).toFixed(1)}%</span>
            <span class="tt-k">REG</span><span class="tt-v">${(m.reg||0).toFixed(1)}%</span>
            <span class="tt-k">NUM</span><span class="tt-v">${(m.num||0).toFixed(1)}%</span>
            <span class="tt-k">CON</span><span class="tt-v">${(m.con||0).toFixed(1)}%</span>
            <span class="tt-k">TMP</span><span class="tt-v">${(m.tmp||0).toFixed(1)}%</span>
            <span class="tt-k">n</span><span class="tt-v">${m.n_items}</span>
          </div>
          <div class="tt-ci">95% CI: ${m.ci||'—'}</div>
        </div>
      </div>`;
    container.appendChild(row);
  });

  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    container.querySelectorAll('.bar-fill').forEach((b,i)=>{
      b.style.transitionDelay=`${i*70}ms`;
      b.style.width=b.dataset.pct+'%';
      setTimeout(()=>b.classList.add('shown'),900+i*70);
    });
  }));
};

document.getElementById('taskTabs').addEventListener('click',e=>{
  const btn=e.target.closest('.task-tab');if(!btn)return;
  document.querySelectorAll('.task-tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  buildBarChart(btn.dataset.t);
});

/* ═════ FULL TABLE ═════ */
let _sc='overall',_sd=-1;
function iconHtml(m,i){
  return`<div class="model-icon" style="background:${m.color}20;color:${m.color};border-color:${m.color}40">${m.icon||m.label.slice(0,2)}</div>`;
}
function typePill(t){
  const m={'Frontier API':'tb-frontier','Frontier API†':'tb-subset','Open-weight API':'tb-open','Local (Ollama)':'tb-local','Reasoning API':'tb-reasoning','Human Baseline':'tb-human'};
  return`<span class="type-badge ${m[t]||'tb-open'}">${t}</span>`;
}

window.buildTable = function(col,dir){
  const sorted=[...MODELS].sort((a,b)=>{
    const av=a[col],bv=b[col];
    return typeof av==='number'?(av-bv)*dir:String(av).localeCompare(String(bv))*dir;
  });
  const tbody=document.getElementById('tBody');tbody.innerHTML='';

  sorted.forEach((m,i)=>{
    const dr=col==='rank'?m.rank:i+1;
    const rbc=dr===1?'rb1':dr===2?'rb2':dr===3?'rb3':'rbn';
    const hfShort=m.hf_id.length>36?m.hf_id.slice(0,34)+'…':m.hf_id;
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td class="c"><span class="rank-b ${rbc}">${dr}</span></td>
      <td><div class="model-cell">${iconHtml(m,i)}<div><div class="model-name">${m.label}</div><div class="model-hfid">${hfShort}</div></div></div></td>
      <td><span style="font-family:var(--mono);font-size:12px">${m.params||'—'}</span></td>
      <td>${typePill(m.type)}</td>
      <td class="c"><span class="sc ${scClass(m.reg)}">${m.reg.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(m.num)}">${m.num.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(m.con)}">${m.con.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(m.tmp)}">${m.tmp.toFixed(1)}%</span></td>
      <td class="c"><span class="ov ${scClass(m.overall)}">${m.overall.toFixed(1)}%</span></td>
      <td class="c" style="font-size:11px;color:var(--text4);font-family:var(--mono)">${m.ci}</td>`;
    tbody.appendChild(tr);
  });

  // Claude subset row
  if(CLAUDE){
    const ct=document.createElement('tr');ct.className='tr-subset';
    const csHf=CLAUDE.hf_id.length>36?CLAUDE.hf_id.slice(0,34)+'…':CLAUDE.hf_id;
    ct.innerHTML=`
      <td class="c"><span class="rank-b rbh">†</span></td>
      <td><div class="model-cell"><div class="model-icon" style="background:#C17B4220;color:#C17B42;border-color:#C17B4240">C</div><div><div class="model-name" style="opacity:0.7">†Claude 3 Haiku</div><div class="model-note">150-item subset only</div></div></div></td>
      <td><span style="font-family:var(--mono);font-size:12px">—</span></td>
      <td>${typePill(CLAUDE.type)}</td>
      <td class="c"><span class="sc ${scClass(CLAUDE.reg)}">${CLAUDE.reg.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(CLAUDE.num)}">${CLAUDE.num.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(CLAUDE.con)}">${CLAUDE.con.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(CLAUDE.tmp)}">${CLAUDE.tmp.toFixed(1)}%</span></td>
      <td class="c"><span class="ov ${scClass(CLAUDE.overall)}">${CLAUDE.overall.toFixed(1)}%</span></td>
      <td class="c" style="font-size:11px;color:var(--text4);font-family:var(--mono)">${CLAUDE.ci}</td>`;
    tbody.appendChild(ct);
  }

  // Human
  const h=HUMAN,tr=document.createElement('tr');tr.className='tr-human';
  tr.innerHTML=`
    <td class="c"><span class="rank-b rbh">—</span></td>
    <td><div class="model-cell"><div class="model-icon" style="background:#f1f5f9;color:#94a3b8;border-color:#e2e8f0">H</div><div><div class="model-name" style="font-style:italic;font-weight:400">Human Expert <span style="font-size:10px">(n=100)</span></div><div class="model-hfid">Expert annotators</div></div></div></td>
    <td>—</td><td>${typePill('Human Baseline')}</td>
    <td class="c"><span style="color:var(--text4);font-family:var(--mono);font-size:12px">—</span></td>
    <td class="c"><span style="color:var(--text4);font-family:var(--mono);font-size:12px">—</span></td>
    <td class="c"><span class="sc ${scClass(h.con)}">${h.con.toFixed(1)}%</span></td>
    <td class="c"><span style="color:var(--text4);font-family:var(--mono);font-size:12px">—</span></td>
    <td class="c"><span class="ov sc-lo" style="background:#f1f5f9;color:#94a3b8">${h.overall.toFixed(1)}%</span></td>
    <td class="c" style="font-size:11px;color:var(--text4)">—</td>`;
  tbody.appendChild(tr);

  document.querySelectorAll('th[onclick]').forEach(th=>{th.classList.remove('sorted');const a=th.querySelector('.sort-icon');if(a)a.textContent='↕'});
  const active=document.querySelector(`th[onclick="tSort('${col}')"]`);
  if(active){active.classList.add('sorted');const a=active.querySelector('.sort-icon');if(a)a.textContent=dir===-1?'↓':'↑'}
};
window.tSort = function(col){if(_sc===col)_sd=-_sd;else{_sc=col;_sd=-1}buildTable(_sc,_sd)};

/* ═════ BREAKDOWN ═════ */
window.buildBreakdown = function(){
  const grid=document.getElementById('breakdownGrid');grid.innerHTML='';
  TASKS_DEF.forEach(task=>{
    const ranked=[...MODELS].sort((a,b)=>(b[task.key]||0)-(a[task.key]||0)).slice(0,5);
    const card=document.createElement('div');card.className='bc glass-card';
    card.innerHTML=`
      <div class="bc-top"><span class="task-pill" style="background:${task.color}18;color:${task.color}">${task.code}</span><span class="bc-n">n = ${task.n}</span></div>
      <div class="bc-title">${task.label}</div>
      <div class="bc-desc">${task.desc}</div>
      ${ranked.map((m,i)=>`
        <div class="bc-row">
          <div style="display:flex;align-items:center;gap:7px">
            <span class="bc-pos">#${i+1}</span>
            <div class="bc-dot" style="background:${m.color}"></div>
            <span class="bc-model-name">${m.label}</span>
          </div>
          <span class="bc-score" style="color:${m.color}">${(m[task.key]||0).toFixed(1)}%</span>
        </div>`).join('')}
      <div class="bc-sep"></div>
      <div class="bc-row" style="opacity:0.5">
        <div style="display:flex;align-items:center;gap:7px"><span class="bc-pos">—</span><div class="bc-dot" style="background:#94a3b8"></div><span style="font-style:italic;color:var(--text3)">Human Expert</span></div>
        <span class="bc-score" style="color:var(--text3)">${HUMAN[task.key].toFixed(1)}%</span>
      </div>`;
    grid.appendChild(card);
  });
};

/* ═════ DIFFICULTY ═════ */
window.buildDiffTable = function(){
  const tbody=document.getElementById('diffBody');tbody.innerHTML='';
  DIFF_DATA.forEach((m,i)=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><div style="display:flex;align-items:center;gap:8px"><div class="model-icon" style="background:${m.color}18;color:${m.color};border-color:${m.color}40;width:28px;height:28px;font-size:10px">${MODELS[i]?.icon||''}</div><span style="font-weight:600;font-size:13px">${m.label}</span></div></td>
      <td class="c"><span class="sc ${scClass(m.easy)}">${m.easy.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(m.med)}">${m.med.toFixed(1)}%</span></td>
      <td class="c"><span class="sc ${scClass(m.hard)}">${m.hard.toFixed(1)}%</span></td>`;
    tbody.appendChild(tr);
  });
};

/* ═════ SUBMIT ═════ */
window.doSubmit = function(){
  const hfId=document.getElementById('hfId').value.trim();
  if(!hfId){showStatus('st-e','Please enter a HuggingFace model ID.');return}
  const label=document.getElementById('dispName').value.trim();
  const btn=document.getElementById('subBtn');
  btn.disabled=true;
  showStatus('st-q','Preparing submission…');
  fetch('/api/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({hf_id:hfId,label,params:document.getElementById('modelParams').value.trim(),model_type:document.getElementById('mtype').value})})
    .then(r=>r.json()).then(data=>{
      if(data.error){showStatus('st-e',data.error);btn.disabled=false;return}
      window.open(data.issue_url,'_blank');
      showStatus('st-d','Submission opened as a GitHub issue. We will run the evaluation and update the leaderboard within a few days.');
      btn.disabled=false;
    }).catch(e=>{showStatus('st-e','Error: '+e.message);btn.disabled=false});
};
function showStatus(cls,msg){const b=document.getElementById('statusBox');b.className='status-box '+cls;b.style.display='block';b.textContent=msg}

/* ═════ CITATION COPY ═════ */
window.copyCite = function(){
  navigator.clipboard.writeText(document.getElementById('citeText').textContent).then(()=>{
    const btn=document.getElementById('copyBtn');btn.textContent='Copied!';setTimeout(()=>btn.textContent='Copy',2200);
  });
};
})();
