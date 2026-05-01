// IndiaFinBench — Hero Canvas + Animations + Observers
(function(){

/* ═════ HERO PARTICLE MESH ═════ */
const c=document.getElementById('heroCanvas');
if(c){
  const ctx=c.getContext('2d');
  let W,H,pts=[],mouse={x:-1000,y:-1000};
  const COLORS=['rgba(74,144,255,0.25)','rgba(74,144,255,0.2)','rgba(100,160,255,0.2)','rgba(130,180,255,0.18)','rgba(74,144,255,0.15)'];

  function resize(){
    W=c.width=c.parentElement.offsetWidth;
    H=c.height=c.parentElement.offsetHeight;
    pts=[];
    const n=Math.min(Math.floor(W*H/18000),50);
    for(let i=0;i<n;i++)pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-0.5)*0.5,vy:(Math.random()-0.5)*0.5,r:Math.random()*2+1,color:COLORS[i%COLORS.length],phase:Math.random()*Math.PI*2});
  }
  window.addEventListener('resize',resize);resize();

  c.parentElement.addEventListener('mousemove',e=>{
    const r=c.parentElement.getBoundingClientRect();
    mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;
  });
  c.parentElement.addEventListener('mouseleave',()=>{mouse.x=-1000;mouse.y=-1000});

  let time=0;
  function draw(){
    time+=0.005;
    ctx.clearRect(0,0,W,H);

    // Draw connection lines
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<110){
          const alpha=(110-d)/110*0.08;
          ctx.strokeStyle=`rgba(255,255,255,${alpha})`;
          ctx.lineWidth=0.8;
          ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();
        }
      }
    }

    // Draw + animate particles
    for(const p of pts){
      // Mouse repulsion
      const mdx=p.x-mouse.x, mdy=p.y-mouse.y, md=Math.sqrt(mdx*mdx+mdy*mdy);
      if(md<150){
        const force=(150-md)/150*0.8;
        p.vx+=mdx/md*force*0.3;
        p.vy+=mdy/md*force*0.3;
      }
      // Gentle oscillation
      p.x+=p.vx+Math.sin(time+p.phase)*0.15;
      p.y+=p.vy+Math.cos(time+p.phase)*0.15;
      // Damping
      p.vx*=0.98;p.vy*=0.98;
      // Bounds
      if(p.x<0||p.x>W)p.vx*=-1;
      if(p.y<0||p.y>H)p.vy*=-1;
      // Glow
      const r=p.r*(1+Math.sin(time*2+p.phase)*0.3);
      const grad=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,r*4);
      grad.addColorStop(0,p.color);
      grad.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=grad;
      ctx.beginPath();ctx.arc(p.x,p.y,r*4,0,Math.PI*2);ctx.fill();
      // Core dot
      ctx.fillStyle='rgba(255,255,255,0.6)';
      ctx.beginPath();ctx.arc(p.x,p.y,r*0.5,0,Math.PI*2);ctx.fill();
    }
    requestAnimationFrame(draw);
  }
  draw();
}

/* ═════ FLOATING ORBS ═════ */
const orbs=document.querySelectorAll('.hero-orb');
orbs.forEach((orb,i)=>{
  const speed=0.3+Math.random()*0.2;
  const amp=15+Math.random()*20;
  const phase=Math.random()*Math.PI*2;
  let t=0;
  function float(){
    t+=0.01*speed;
    orb.style.transform=`translate(${Math.sin(t+phase)*amp}px, ${Math.cos(t*0.7+phase)*amp}px) rotate(${t*20}deg)`;
    requestAnimationFrame(float);
  }
  float();
});

/* ═════ ANIMATED COUNTERS ═════ */
function animateCounter(el,target){
  const isPercent=String(target).includes('%');
  const num=parseFloat(target);
  if(isNaN(num))return;
  let start=0,dur=1800,t0=null;
  function step(ts){
    if(!t0)t0=ts;
    let p=Math.min((ts-t0)/dur,1);
    // Spring easing with slight overshoot
    p=1-Math.pow(1-p,4);
    if(p>0.85)p=1-(1-p)*Math.cos((1-p)*Math.PI*6);
    const v=start+p*(num-start);
    el.textContent=Number.isInteger(num)?Math.round(v).toString():v.toFixed(1)+'%';
    if(Math.abs(v-num)>0.1)requestAnimationFrame(step);
    else el.textContent=Number.isInteger(num)?num.toString():num.toFixed(1)+'%';
  }
  requestAnimationFrame(step);
}

/* ═════ INTERSECTION OBSERVER ═════ */
const io=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.classList.add('visible');
      io.unobserve(e.target);
    }
  });
},{threshold:0.08,rootMargin:'0px 0px -40px 0px'});
document.querySelectorAll('.anim').forEach(el=>io.observe(el));

// Counter observer
const counterObs=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.querySelectorAll('[data-count]').forEach(n=>animateCounter(n,n.dataset.count));
      counterObs.unobserve(e.target);
    }
  });
},{threshold:0.3});
document.querySelectorAll('.stats-row').forEach(el=>counterObs.observe(el));

// Diff bars observer
const diffObs=new IntersectionObserver((entries)=>{
  entries.forEach(e=>{
    if(e.isIntersecting){
      e.target.querySelectorAll('.diff-bar-fill').forEach(b=>{b.style.width=b.dataset.w+'%'});
      diffObs.unobserve(e.target);
    }
  });
},{threshold:0.2});
document.querySelectorAll('.diff-grid').forEach(el=>diffObs.observe(el));

/* ═════ 3D TILT ON GLASS CARDS ═════ */
document.querySelectorAll('.glass-card').forEach(card=>{
  card.addEventListener('mousemove',e=>{
    const r=card.getBoundingClientRect();
    const x=(e.clientX-r.left)/r.width-0.5;
    const y=(e.clientY-r.top)/r.height-0.5;
    card.style.transform=`perspective(1200px) rotateY(${x*2}deg) rotateX(${-y*2}deg) translateY(-2px)`;
    const shine=card.querySelector('.card-shine');
    if(shine){shine.style.opacity='1';shine.style.background=`radial-gradient(circle at ${e.clientX-r.left}px ${e.clientY-r.top}px, rgba(255,255,255,0.08) 0%, transparent 60%)`}
  });
  card.addEventListener('mouseleave',()=>{
    card.style.transform='';
    const shine=card.querySelector('.card-shine');
    if(shine)shine.style.opacity='0';
  });
});

/* ═════ NAV SCROLL ACTIVE ═════ */
const _ns=['leaderboard','tasks','difficulty','submit','rag','about'];
window.addEventListener('scroll',()=>{
  let cur='';_ns.forEach(id=>{const el=document.getElementById(id);if(el&&window.scrollY>=el.offsetTop-120)cur=id});
  document.querySelectorAll('.nav-link').forEach(a=>{
    a.classList.toggle('active',a.getAttribute('href').replace('#','')===cur||(!cur&&a.getAttribute('href').includes('leaderboard')));
  });
  // Nav background opacity on scroll
  const nav=document.querySelector('nav');
  if(nav){
    const scrolled=window.scrollY>50;
    nav.style.background=scrolled?'rgba(6,9,23,0.95)':'rgba(6,9,23,0.75)';
    nav.style.borderColor=scrolled?'rgba(255,255,255,0.15)':'rgba(255,255,255,0.08)';
  }
},{passive:true});

/* ═════ TWEAKS ═════ */
window.addEventListener('message',e=>{
  if(e.data?.type==='__activate_edit_mode')document.getElementById('tweaksPanel').style.display='block';
  if(e.data?.type==='__deactivate_edit_mode')document.getElementById('tweaksPanel').style.display='none';
});
window.parent.postMessage({type:'__edit_mode_available'},'*');

window.setHeroBg=function(color,btn){
  document.querySelector('.hero').style.background=color;
  document.querySelectorAll('[onclick^="setHeroBg"]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');
  window.parent.postMessage({type:'__edit_mode_set_keys',edits:{heroBg:color}},'*');
};
window.setAccent=function(val,btn){
  document.documentElement.style.setProperty('--accent',val);
  document.querySelectorAll('[onclick^="setAccent"]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');
  window.parent.postMessage({type:'__edit_mode_set_keys',edits:{accent:val}},'*');
};
window.setBarStyle=function(val,btn){
  document.querySelectorAll('.bar-fill').forEach(b=>{
    b.style.borderRadius=val==='rounded'?'10px':'3px';
  });
  document.querySelectorAll('[onclick^="setBarStyle"]').forEach(b=>b.classList.remove('active'));btn.classList.add('active');
  window.parent.postMessage({type:'__edit_mode_set_keys',edits:{barStyle:val}},'*');
};

})();
