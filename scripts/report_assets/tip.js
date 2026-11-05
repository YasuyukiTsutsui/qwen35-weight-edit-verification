(function(){
var tip=document.getElementById('tip');
function fill(t){while(tip.firstChild)tip.removeChild(tip.firstChild);var p=t.split('||');
 var b=document.createElement('b');b.textContent=p[0];tip.appendChild(b);
 if(p[1]){var s=document.createElement('span');s.textContent=p[1];tip.appendChild(s);}tip.hidden=false;}
function place(x,y){var w=tip.offsetWidth,h=tip.offsetHeight;var l=x+14,t=y+14;
 if(l+w>window.innerWidth-8)l=Math.max(8,x-w-14);if(t+h>window.innerHeight-8)t=Math.max(8,y-h-14);
 tip.style.left=l+'px';tip.style.top=t+'px';}
document.addEventListener('pointermove',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
 if(!el){tip.hidden=true;return;}fill(el.getAttribute('data-tip'));place(e.clientX,e.clientY);});
document.addEventListener('pointerleave',function(){tip.hidden=true;});
document.addEventListener('focusin',function(e){var el=e.target.closest&&e.target.closest('[data-tip]');
 if(!el)return;fill(el.getAttribute('data-tip'));var r=el.getBoundingClientRect();place(r.left+r.width/2,r.top+r.height/2);});
document.addEventListener('focusout',function(){tip.hidden=true;});
})();
