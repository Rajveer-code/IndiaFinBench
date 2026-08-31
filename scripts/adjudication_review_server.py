"""Local one-item-at-a-time review tool for annotation/judge_audit/adjudication_sheet.csv.

Serves a single page at http://127.0.0.1:5051. Shows question/reference/
prediction for the next un-adjudicated row; you click CORRECT/INCORRECT (or
press C/I). Writes straight back to the real CSV after every single item, so
progress survives closing the browser or stopping this server.

Deliberately blind: gemini_verdict/phi4_verdict/strict_correct are withheld
until AFTER you submit a verdict, so your judgment isn't anchored by what the
automated judges said. They're revealed immediately after, for your interest.

Usage: python scripts/adjudication_review_server.py
"""
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent.parent
SHEET_PATH = ROOT / "annotation" / "judge_audit" / "adjudication_sheet.csv"
PORT = 5051


def load_rows():
    with open(SHEET_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def save_rows(fieldnames, rows):
    tmp = SHEET_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, SHEET_PATH)


def find_next(rows):
    for i, r in enumerate(rows):
        if not r["human_verdict_CORRECT_or_INCORRECT"].strip():
            return i
    return None


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Human Adjudication</title>
<style>
:root{--bg:#faf9f7;--card:#fff;--ink:#1a1a1a;--sub:#6b6b6b;--line:#e3e0da;
--correct:#1e7d4f;--incorrect:#b3382c;--accent:#3a5a8c;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:Georgia,'Times New Roman',serif;padding:32px 16px;}
.wrap{max-width:760px;margin:0 auto}
.progress{display:flex;justify-content:space-between;align-items:center;
font-family:Arial,sans-serif;font-size:13px;color:var(--sub);margin-bottom:6px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:28px}
.bar>div{height:100%;background:var(--accent);transition:width .3s}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:28px 32px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;font-family:Arial,sans-serif}
.badge{font-size:11px;text-transform:uppercase;letter-spacing:.04em;
padding:3px 9px;border-radius:20px;background:#eef0f4;color:#444;font-weight:600}
.label{font-family:Arial,sans-serif;font-size:11px;text-transform:uppercase;
letter-spacing:.06em;color:var(--sub);margin:18px 0 6px;font-weight:600}
.label:first-of-type{margin-top:0}
.text{font-size:16px;line-height:1.6;white-space:pre-wrap}
.pred{background:#fbf7ec;border:1px solid #ecdfb8;border-radius:8px;padding:14px 16px}
.actions{display:flex;gap:12px;margin-top:28px}
button{flex:1;font-family:Arial,sans-serif;font-size:15px;font-weight:700;
padding:14px;border-radius:8px;border:none;cursor:pointer;color:#fff;
transition:transform .08s}
button:active{transform:scale(.97)}
.btn-correct{background:var(--correct)}
.btn-incorrect{background:var(--incorrect)}
.kbd{opacity:.7;font-weight:400;font-size:12px}
textarea{width:100%;margin-top:14px;font-family:Arial,sans-serif;font-size:13px;
border:1px solid var(--line);border-radius:6px;padding:8px;resize:vertical;min-height:44px}
.reveal{font-family:Arial,sans-serif;font-size:13px;background:#eef4ee;
border:1px solid #cfe3cf;border-radius:8px;padding:12px 16px;margin-top:16px;color:#2a4a34}
.reveal b{color:var(--ink)}
.warn{font-family:Arial,sans-serif;font-size:12.5px;background:#fdecea;
border:1px solid #f3b8b0;border-radius:6px;padding:8px 12px;margin-top:8px;color:#8a2e22}
.done{text-align:center;padding:60px 20px;font-family:Arial,sans-serif}
.next-hint{text-align:center;margin-top:14px;font-family:Arial,sans-serif;
font-size:13px;color:var(--sub)}
</style></head><body><div class="wrap">
<div class="progress"><span id="pos"></span><span id="stratum"></span></div>
<div class="bar"><div id="barfill" style="width:0%"></div></div>
<div id="card"></div>
</div>
<script>
let state = null;
const el = id => document.getElementById(id);

async function loadNext(){
  const r = await fetch('/api/state');
  state = await r.json();
  render();
}

function render(){
  if(state.done){
    el('card').innerHTML = `<div class="card done"><h2>All ${state.total} items adjudicated.</h2>
      <p>Sheet is complete: annotation/judge_audit/adjudication_sheet.csv</p></div>`;
    el('pos').textContent = `${state.total} / ${state.total} done`;
    el('barfill').style.width = '100%';
    el('stratum').textContent = '';
    return;
  }
  const c = state.current;
  el('pos').textContent = `${state.index + 1} / ${state.total}`;
  el('barfill').style.width = (100*state.index/state.total) + '%';
  el('stratum').textContent = c.stratum || c.sample_group || '';
  el('card').innerHTML = `
    <div class="card">
      <div class="meta">
        <span class="badge">${c.model}</span>
        <span class="badge">${c.task_type.replace('_',' ')}</span>
      </div>
      <div class="label">Question</div><div class="text">${esc(c.question)}</div>
      <div class="label">Reference Answer</div><div class="text">${esc(c.reference_answer)}</div>
      <div class="label">Model Prediction</div><div class="text pred">${esc(c.model_prediction)}</div>
      ${c.prediction_truncated === 'TRUE' ? '<div class="warn">&#9888; This prediction was cut off at 200 characters when originally logged and the rest was never saved. Judge it as INCORRECT only if it is wrong within what is shown — do not penalise it for being incomplete.</div>' : ''}
      <div class="actions">
        <button class="btn-incorrect" onclick="submit('INCORRECT')">INCORRECT <span class="kbd">(I)</span></button>
        <button class="btn-correct" onclick="submit('CORRECT')">CORRECT <span class="kbd">(C)</span></button>
      </div>
      <textarea id="notes" placeholder="Optional note..."></textarea>
      <div id="revealBox"></div>
    </div>
    <div class="next-hint" id="nextHint" style="display:none">Press any key or click to continue &rarr;</div>
  `;
}

function esc(s){
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function submit(verdict){
  if(!state || state.done) return;
  const notes = el('notes') ? el('notes').value : '';
  const idx = state.index;
  const r = await fetch('/api/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: idx, item_id: state.current.item_id, model: state.current.model,
      verdict, notes})});
  const reveal = await r.json();
  el('revealBox').innerHTML = `<div class="reveal">
    Recorded: <b>${verdict}</b><br>
    Gemini judge said: <b>${labelVerdict(reveal.gemini_verdict)}</b> &middot;
    phi4-mini judge said: <b>${labelVerdict(reveal.phi4_verdict)}</b> &middot;
    strict scoring said: <b>${reveal.strict_correct==='1'?'CORRECT':'INCORRECT'}</b>
  </div>`;
  el('nextHint').style.display = 'block';
  document.onkeydown = null;
  const advance = () => { document.onclick=null; document.onkeydown=null; loadNext(); };
  document.onclick = advance;
  document.addEventListener('keydown', advance, {once:true});
}

function labelVerdict(v){
  if(v === '' || v === undefined) return 'n/a (never judged)';
  return v === '1' ? 'CORRECT' : 'INCORRECT';
}

document.addEventListener('keydown', e => {
  if(!state || state.done) return;
  if(e.key === 'c' || e.key === 'C') submit('CORRECT');
  if(e.key === 'i' || e.key === 'I') submit('INCORRECT');
});

loadNext();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            fieldnames, rows = load_rows()
            idx = find_next(rows)
            if idx is None:
                self._json({"done": True, "total": len(rows)})
                return
            r = rows[idx]
            self._json({
                "done": False, "index": idx, "total": len(rows),
                "current": {k: r[k] for k in
                            ("item_id", "model", "task_type", "question",
                             "reference_answer", "model_prediction",
                             "prediction_truncated", "sample_group", "stratum")},
            })
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/verdict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        verdict = payload.get("verdict")
        if verdict not in ("CORRECT", "INCORRECT"):
            self._json({"error": "verdict must be CORRECT or INCORRECT"}, 400)
            return

        fieldnames, rows = load_rows()
        idx = payload["index"]
        if not (0 <= idx < len(rows)) or rows[idx]["item_id"] != payload.get("item_id") \
                or rows[idx]["model"] != payload.get("model"):
            self._json({"error": "row mismatch, reload and retry"}, 409)
            return

        row = rows[idx]
        row["human_verdict_CORRECT_or_INCORRECT"] = verdict
        row["human_notes"] = payload.get("notes", "")
        save_rows(fieldnames, rows)

        self._json({
            "gemini_verdict": row["gemini_verdict"],
            "phi4_verdict": row["phi4_verdict"],
            "strict_correct": row["strict_correct"],
        })


if __name__ == "__main__":
    _, rows = load_rows()
    done = sum(1 for r in rows if r["human_verdict_CORRECT_or_INCORRECT"].strip())
    print(f"Adjudication sheet: {done}/{len(rows)} already done.")
    print(f"Serving at http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    server.serve_forever()
