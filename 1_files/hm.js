(function(){var h={},mt={},c={id:"8e2a116daf0104a78d601f40a45c75b4",dm:["w3cschool.cc","runoob.com"],js:"tongji.baidu.com/hm-web/js/",etrk:[{id:"%23sidebar-right-ads",eventType:"onclick"}],icon:'',ctrk:false,align:-1,nv:1,vdur:1800000,age:31536000000,rec:1,rp:[],trust:0,vcard:0,qiao:0,lxb:0,conv:0,med:0,cvcc:'',cvcf:[],apps:''};var q=void 0,r=!0,u=null,v=!1;mt.cookie={};mt.cookie.set=function(a,b,d){var e;d.N&&(e=new Date,e.setTime(e.getTime()+d.N));document.cookie=a+"="+b+(d.domain?"; domain="+d.domain:"")+(d.path?"; path="+d.path:"")+(e?"; expires="+e.toGMTString():"")+(d.Bb?"; secure":"")};mt.cookie.get=function(a){return(a=RegExp("(^| )"+a+"=([^;]*)(;|$)").exec(document.cookie))?a[2]:u};mt.g={};mt.g.O=function(a){return document.getElementById(a)};
mt.g.Q=function(a,b){var d=[],e=[];if(!a)return e;for(;a.parentNode!=u;){for(var k=0,m=0,g=a.parentNode.childNodes.length,p=0;p<g;p++){var f=a.parentNode.childNodes[p];if(f.nodeName===a.nodeName&&(k++,f===a&&(m=k),0<m&&1<k))break}if((g=""!==a.id)&&b){d.unshift("#"+encodeURIComponent(a.id));break}else g&&(g="#"+encodeURIComponent(a.id),g=0<d.length?g+">"+d.join(">"):g,e.push(g)),d.unshift(encodeURIComponent(String(a.nodeName).toLowerCase())+(1<k?"["+m+"]":""));a=a.parentNode}e.push(d.join(">"));return e};
mt.g.Oa=function(a){return(a=mt.g.Q(a,r))&&a.length?String(a[0]):""};mt.g.Na=function(a){return mt.g.Q(a,v)};mt.g.Fa=function(a){var b;for(b="A";(a=a.parentNode)&&1==a.nodeType;)if(a.tagName==b)return a;return u};mt.g.Ha=function(a){return 9===a.nodeType?a:a.ownerDocument||a.document};
mt.g.La=function(a){var b={top:0,left:0};if(!a)return b;var d=mt.g.Ha(a).documentElement;"undefined"!==typeof a.getBoundingClientRect&&(b=a.getBoundingClientRect());return{top:b.top+(window.pageYOffset||d.scrollTop)-(d.clientTop||0),left:b.left+(window.pageXOffset||d.scrollLeft)-(d.clientLeft||0)}};
(mt.g.ga=function(){function a(){if(!a.G){a.G=r;for(var b=0,d=e.length;b<d;b++)e[b]()}}function b(){try{document.documentElement.doScroll("left")}catch(e){setTimeout(b,1);return}a()}var d=v,e=[],k;document.addEventListener?k=function(){document.removeEventListener("DOMContentLoaded",k,v);a()}:document.attachEvent&&(k=function(){"complete"===document.readyState&&(document.detachEvent("onreadystatechange",k),a())});(function(){if(!d)if(d=r,"complete"===document.readyState)a.G=r;else if(document.addEventListener)document.addEventListener("DOMContentLoaded",
k,v),window.addEventListener("load",a,v);else if(document.attachEvent){document.attachEvent("onreadystatechange",k);window.attachEvent("onload",a);var e=v;try{e=window.frameElement==u}catch(g){}document.documentElement.doScroll&&e&&b()}})();return function(b){a.G?b():e.push(b)}}()).G=v;mt.event={};mt.event.c=function(a,b,d){a.attachEvent?a.attachEvent("on"+b,function(e){d.call(a,e)}):a.addEventListener&&a.addEventListener(b,d,v)};
mt.event.preventDefault=function(a){a.preventDefault?a.preventDefault():a.returnValue=v};
(function(){var a=mt.event;mt.f={};mt.f.da=/msie (\d+\.\d+)/i.test(navigator.userAgent);mt.f.Za=/msie (\d+\.\d+)/i.test(navigator.userAgent)?document.documentMode||+RegExp.$1:q;mt.f.cookieEnabled=navigator.cookieEnabled;mt.f.javaEnabled=navigator.javaEnabled();mt.f.language=navigator.language||navigator.browserLanguage||navigator.systemLanguage||navigator.userLanguage||"";mt.f.gb=(window.screen.width||0)+"x"+(window.screen.height||0);mt.f.colorDepth=window.screen.colorDepth||0;mt.f.C=function(){var a;
a=a||document;return parseInt(window.pageYOffset||a.documentElement.scrollTop||a.body&&a.body.scrollTop||0,10)};mt.f.D=function(){var a=document;return parseInt(window.innerHeight||a.documentElement.clientHeight||a.body&&a.body.clientHeight||0,10)};mt.f.orientation=0;(function(){function b(){var a=0;window.orientation!==q&&(a=window.orientation);screen&&(screen.orientation&&screen.orientation.angle!==q)&&(a=screen.orientation.angle);mt.f.orientation=a}b();a.c(window,"orientationchange",b)})();return mt.f})();
mt.m={};mt.m.parse=function(){return(new Function('return (" + source + ")'))()};
mt.m.stringify=function(){function a(a){/["\\\x00-\x1f]/.test(a)&&(a=a.replace(/["\\\x00-\x1f]/g,function(a){var e=d[a];if(e)return e;e=a.charCodeAt();return"\\u00"+Math.floor(e/16).toString(16)+(e%16).toString(16)}));return'"'+a+'"'}function b(a){return 10>a?"0"+a:a}var d={"\b":"\\b","\t":"\\t","\n":"\\n","\f":"\\f","\r":"\\r",'"':'\\"',"\\":"\\\\"};return function(e){switch(typeof e){case "undefined":return"undefined";case "number":return isFinite(e)?String(e):"null";case "string":return a(e);case "boolean":return String(e);
default:if(e===u)return"null";if(e instanceof Array){var d=["["],m=e.length,g,p,f;for(p=0;p<m;p++)switch(f=e[p],typeof f){case "undefined":case "function":case "unknown":break;default:g&&d.push(","),d.push(mt.m.stringify(f)),g=1}d.push("]");return d.join("")}if(e instanceof Date)return'"'+e.getFullYear()+"-"+b(e.getMonth()+1)+"-"+b(e.getDate())+"T"+b(e.getHours())+":"+b(e.getMinutes())+":"+b(e.getSeconds())+'"';g=["{"];p=mt.m.stringify;for(m in e)if(Object.prototype.hasOwnProperty.call(e,m))switch(f=
e[m],typeof f){case "undefined":case "unknown":case "function":break;default:d&&g.push(","),d=1,g.push(p(m)+":"+p(f))}g.push("}");return g.join("")}}}();mt.lang={};mt.lang.d=function(a,b){return"[object "+b+"]"==={}.toString.call(a)};mt.lang.yb=function(a){return mt.lang.d(a,"Number")&&isFinite(a)};mt.lang.Ab=function(a){return mt.lang.d(a,"String")};mt.lang.h=function(a){return a.replace?a.replace(/'/g,"'0").replace(/\*/g,"'1").replace(/!/g,"'2"):a};mt.localStorage={};
mt.localStorage.K=function(){if(!mt.localStorage.i)try{mt.localStorage.i=document.createElement("input"),mt.localStorage.i.type="hidden",mt.localStorage.i.style.display="none",mt.localStorage.i.addBehavior("#default#userData"),document.getElementsByTagName("head")[0].appendChild(mt.localStorage.i)}catch(a){return v}return r};
mt.localStorage.set=function(a,b,d){var e=new Date;e.setTime(e.getTime()+d||31536E6);try{window.localStorage?(b=e.getTime()+"|"+b,window.localStorage.setItem(a,b)):mt.localStorage.K()&&(mt.localStorage.i.expires=e.toUTCString(),mt.localStorage.i.load(document.location.hostname),mt.localStorage.i.setAttribute(a,b),mt.localStorage.i.save(document.location.hostname))}catch(k){}};
mt.localStorage.get=function(a){if(window.localStorage){if(a=window.localStorage.getItem(a)){var b=a.indexOf("|"),d=a.substring(0,b)-0;if(d&&d>(new Date).getTime())return a.substring(b+1)}}else if(mt.localStorage.K())try{return mt.localStorage.i.load(document.location.hostname),mt.localStorage.i.getAttribute(a)}catch(e){}return u};
mt.localStorage.remove=function(a){if(window.localStorage)window.localStorage.removeItem(a);else if(mt.localStorage.K())try{mt.localStorage.i.load(document.location.hostname),mt.localStorage.i.removeAttribute(a),mt.localStorage.i.save(document.location.hostname)}catch(b){}};mt.sessionStorage={};mt.sessionStorage.set=function(a,b){if(window.sessionStorage)try{window.sessionStorage.setItem(a,b)}catch(d){}};
mt.sessionStorage.get=function(a){return window.sessionStorage?window.sessionStorage.getItem(a):u};mt.sessionStorage.remove=function(a){window.sessionStorage&&window.sessionStorage.removeItem(a)};mt.la={};mt.la.log=function(a,b){var d=new Image,e="mini_tangram_log_"+Math.floor(2147483648*Math.random()).toString(36);window[e]=d;d.onload=d.onerror=d.onabort=function(){d.onload=d.onerror=d.onabort=u;d=window[e]=u;b&&b(a)};d.src=a};mt.W={};
mt.W.Qa=function(){var a="";if(navigator.plugins&&navigator.mimeTypes.length){var b=navigator.plugins["Shockwave Flash"];b&&b.description&&(a=b.description.replace(/^.*\s+(\S+)\s+\S+$/,"$1"))}else if(window.ActiveXObject)try{if(b=new ActiveXObject("ShockwaveFlash.ShockwaveFlash"))(a=b.GetVariable("$version"))&&(a=a.replace(/^.*\s+(\d+),(\d+).*$/,"$1.$2"))}catch(d){}return a};
mt.W.wb=function(a,b,d,e,k){return'<object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" id="'+a+'" width="'+d+'" height="'+e+'"><param name="movie" value="'+b+'" /><param name="flashvars" value="'+(k||"")+'" /><param name="allowscriptaccess" value="always" /><embed type="application/x-shockwave-flash" name="'+a+'" width="'+d+'" height="'+e+'" src="'+b+'" flashvars="'+(k||"")+'" allowscriptaccess="always" /></object>'};mt.url={};
mt.url.k=function(a,b){var d=a.match(RegExp("(^|&|\\?|#)("+b+")=([^&#]*)(&|$|#)",""));return d?d[3]:u};mt.url.xb=function(a){return(a=a.match(/^(https?:)\/\//))?a[1]:u};mt.url.Ja=function(a){return(a=a.match(/^(https?:\/\/)?([^\/\?#]*)/))?a[2].replace(/.*@/,""):u};mt.url.ba=function(a){return(a=mt.url.Ja(a))?a.replace(/:\d+$/,""):a};mt.url.Q=function(a){return(a=a.match(/^(https?:\/\/)?[^\/]*(.*)/))?a[2].replace(/[\?#].*/,"").replace(/^$/,"/"):u};
(function(){function a(){for(var a=v,d=document.getElementsByTagName("script"),e=d.length,e=100<e?100:e,k=0;k<e;k++){var m=d[k].src;if(m&&0===m.indexOf("https://hm.baidu.com/h")){a=r;break}}return a}return h.Ea=a})();var x=h.Ea;
h.l={Ya:"http://tongji.baidu.com/hm-web/welcome/ico",U:"hm.baidu.com/hm.gif",sa:"baidu.com",Va:"hmmd",Wa:"hmpl",rb:"utm_medium",Ua:"hmkw",ub:"utm_term",Sa:"hmci",qb:"utm_content",Xa:"hmsr",tb:"utm_source",Ta:"hmcu",pb:"utm_campaign",z:0,p:Math.round(+new Date/1E3),Z:Math.round(+new Date/1E3)%65535,protocol:"https:"===document.location.protocol?"https:":"http:",H:x()||"https:"===document.location.protocol?"https:":"http:",zb:0,xa:6E5,hb:5E3,ya:5,Y:1024,wa:1,w:2147483647,ma:"cc cf ci ck cl cm cp cu cw ds vl ep et fl ja ln lo lt rnd si su v cv lv api sn ct u tt".split(" ")};
(function(){var a={s:{},c:function(a,d){this.s[a]=this.s[a]||[];this.s[a].push(d)},B:function(a,d){this.s[a]=this.s[a]||[];for(var e=this.s[a].length,k=0;k<e;k++)this.s[a][k](d)}};return h.o=a})();
(function(){function a(a,e){var k=document.createElement("script");k.charset="utf-8";b.d(e,"Function")&&(k.readyState?k.onreadystatechange=function(){if("loaded"===k.readyState||"complete"===k.readyState)k.onreadystatechange=u,e()}:k.onload=function(){e()});k.src=a;var m=document.getElementsByTagName("script")[0];m.parentNode.insertBefore(k,m)}var b=mt.lang;return h.load=a})();
(function(){function a(){var a="";h.b.a.nv?(a=encodeURIComponent(document.referrer),window.sessionStorage?d.set("Hm_from_"+c.id,a):b.set("Hm_from_"+c.id,a,864E5)):a=(window.sessionStorage?d.get("Hm_from_"+c.id):b.get("Hm_from_"+c.id))||"";return a}var b=mt.localStorage,d=mt.sessionStorage;return h.$=a})();
(function(){var a=h.l,b={init:function(){if(""!==c.icon){var d=c.icon.split("|"),e=a.Ya+"?s="+c.id,b="https://hmcdn.baidu.com/static"+d[0]+".gif";document.write("swf"===d[1]||"gif"===d[1]?'<a href="'+e+'" target="_blank"><img border="0" src="'+b+'" width="'+d[2]+'" height="'+d[3]+'"></a>':'<a href="'+e+'" target="_blank">'+d[0]+"</a>")}}};h.o.c("pv-b",b.init);return b})();
(function(){var a=mt.g,b=mt.event,d={oa:function(){b.c(document,"click",d.Ba());for(var e=c.etrk.length,k=0;k<e;k++){var m=c.etrk[k],g=decodeURIComponent(String(m.id));-1===g.indexOf(">")&&(0===g.indexOf("#")&&(g=g.substring(1)),(g=a.O(decodeURIComponent(g)))&&b.c(g,m.eventType,d.M()))}},M:function(){return function(a){(a.target||a.srcElement).setAttribute("HM_fix",a.clientX+":"+a.clientY);d.ha("#"+encodeURIComponent(this.id),a.type)}},Ba:function(){return function(e){var b=e.target||e.srcElement;
if(b){var m=b.getAttribute("HM_fix");e=e.clientX+":"+e.clientY;if(m&&m==e)b.removeAttribute("HM_fix");else if(0<c.etrk.length&&(b=a.Na(b))&&b.length)if(m=b.length,e=b[b.length-1],1E4>m*e.split(">").length)for(e=0;e<m;e++)d.ka(b[e]);else d.ka(e)}}},ka:function(a){for(var b={},m=String(a).split(">").length,g=0;g<m;g++)b[a]="",a=a.substring(0,a.lastIndexOf(">"));a=c.etrk.length;for(m=0;m<a;m++)g=decodeURIComponent(String(c.etrk[m].id)),b.hasOwnProperty(g)&&d.ha(g)},ha:function(a,b){h.b.a.et=1;h.b.a.ep=
"{id:"+a+",eventType:"+(b||"click")+"}";h.b.j()}};h.o.c("pv-b",d.oa);return d})();
(function(){var a=mt.g,b=mt.lang,d=mt.event,e=mt.f,k=h.l,m=h.o,g=[],p={na:function(){c.ctrk&&(d.c(document,"mouseup",p.va()),d.c(window,"unload",function(){p.I()}),setInterval(function(){p.I()},k.xa))},va:function(){return function(a){a=p.Ga(a);if(""!==a){var b=(k.H+"//"+k.U+"?"+h.b.ia().replace(/ep=[^&]*/,"ep="+encodeURIComponent(a))).length;b+(k.w+"").length>k.Y||(b+encodeURIComponent(g.join("!")+(g.length?"!":"")).length+(k.w+"").length>k.Y&&p.I(),g.push(a),(g.length>=k.ya||/\*a\*/.test(a))&&p.I())}}},
Ga:function(l){var d=l.target||l.srcElement;if(0===k.wa){var n=(d.tagName||"").toLowerCase();if("embed"==n||"object"==n)return""}var s;e.da?(s=Math.max(document.documentElement.scrollTop,document.body.scrollTop),n=Math.max(document.documentElement.scrollLeft,document.body.scrollLeft),n=l.clientX+n,s=l.clientY+s):(n=l.pageX,s=l.pageY);l=p.Ka(l,d,n,s);var f=window.innerWidth||document.documentElement.clientWidth||document.body.offsetWidth;switch(c.align){case 1:n-=f/2;break;case 2:n-=f}f=[];f.push(n);
f.push(s);f.push(l.ab);f.push(l.bb);f.push(l.eb);f.push(b.h(l.cb));f.push(l.vb);f.push(l.Ra);(d="a"===(d.tagName||"").toLowerCase()?d:a.Fa(d))?(f.push("a"),f.push(b.h(encodeURIComponent(d.href)))):f.push("b");return f.join("*")},Ka:function(l,d,n,s){l=a.Oa(d);var f=0,t=0,g=0,k=0;if(d&&(f=d.offsetWidth||d.clientWidth,t=d.offsetHeight||d.clientHeight,k=a.La(d),g=k.left,k=k.top,b.d(d.getBBox,"Function")&&(t=d.getBBox(),f=t.width,t=t.height),"html"===(d.tagName||"").toLowerCase()))f=Math.max(f,d.clientWidth),
t=Math.max(t,d.clientHeight);return{ab:Math.round(100*((n-g)/f)),bb:Math.round(100*((s-k)/t)),eb:e.orientation,cb:l,vb:f,Ra:t}},I:function(){0!==g.length&&(h.b.a.et=2,h.b.a.ep=g.join("!"),h.b.j(),g=[])}},f={qa:function(){c.ctrk&&setInterval(f.ib,k.hb)},ib:function(){var a=e.C()+e.D();0<a-h.b.a.vl&&(h.b.a.vl=a)}};m.c("pv-b",p.na);m.c("pv-b",f.qa);return p})();
(function(){var a=mt.g,b=h.l,d=h.load,e=h.$;h.o.c("pv-b",function(){var k=b.protocol+"//crs.baidu.com/";c.rec&&a.ga(function(){for(var m=0,g=c.rp.length;m<g;m++){var p=c.rp[m][0],f=c.rp[m][1],l=a.O("hm_t_"+p);if(f&&!(2==f&&!l||l&&""!==l.innerHTML))l="",l=Math.round(Math.random()*b.w),l=4==f?k+"hl.js?"+["siteId="+c.id,"planId="+p,"rnd="+l].join("&"):k+"t.js?"+["siteId="+c.id,"planId="+p,"from="+e(),"referer="+encodeURIComponent(document.referrer),"title="+encodeURIComponent(document.title),"rnd="+
l].join("&"),d(l)}})})})();(function(){var a=h.l,b=h.load,d=h.$;h.o.c("pv-b",function(){if(c.trust&&c.vcard){var e="https://tag.baidu.com/vcard/v.js?"+["siteid="+c.vcard,"url="+encodeURIComponent(document.location.href),"source="+d(),"rnd="+Math.round(Math.random()*a.w),"hm=1"].join("&");b(e)}})})();
(function(){function a(){return function(){h.b.a.nv=0;h.b.a.st=4;h.b.a.et=3;h.b.a.ep=h.L.Ma()+","+h.L.Ia();h.b.j()}}function b(){clearTimeout(C);var a;t&&(a="visible"==document[t]);B&&(a=!document[B]);p="undefined"==typeof a?r:a;if((!g||!f)&&p&&l)y=r,n=+new Date;else if(g&&f&&(!p||!l))y=v,s+=+new Date-n;g=p;f=l;C=setTimeout(b,100)}function d(a){var l=document,d="";if(a in l)d=a;else for(var n=["webkit","ms","moz","o"],b=0;b<n.length;b++){var e=n[b]+a.charAt(0).toUpperCase()+a.slice(1);if(e in l){d=
e;break}}return d}function e(a){if(!("focus"==a.type||"blur"==a.type)||!(a.target&&a.target!=window))l="focus"==a.type||"focusin"==a.type?r:v,b()}var k=mt.event,m=h.o,g=r,p=r,f=r,l=r,w=+new Date,n=w,s=0,y=r,t=d("visibilityState"),B=d("hidden"),C;b();(function(){var a=t.replace(/[vV]isibilityState/,"visibilitychange");k.c(document,a,b);k.c(window,"pageshow",b);k.c(window,"pagehide",b);"object"==typeof document.onfocusin?(k.c(document,"focusin",e),k.c(document,"focusout",e)):(k.c(window,"focus",e),
k.c(window,"blur",e))})();h.L={Ma:function(){return+new Date-w},Ia:function(){return y?+new Date-n+s:s}};m.c("pv-b",function(){k.c(window,"unload",a())});return h.L})();
(function(){var a=mt.lang,b=h.l,d=h.load,e={$a:function(e){if((window._dxt===q||a.d(window._dxt,"Array"))&&"undefined"!==typeof h.b){var m=h.b.P();d([b.protocol,"//datax.baidu.com/x.js?si=",c.id,"&dm=",encodeURIComponent(m)].join(""),e)}},ob:function(d){if(a.d(d,"String")||a.d(d,"Number"))window._dxt=window._dxt||[],window._dxt.push(["_setUserId",d])}};return h.za=e})();
(function(){function a(a,d,n,b){if(!(a===q||d===q||b===q)){if(""===a)return[d,n,b].join("*");a=String(a).split("!");for(var e,f=v,g=0;g<a.length;g++)if(e=a[g].split("*"),String(d)===e[0]){e[1]=n;e[2]=b;a[g]=e.join("*");f=r;break}f||a.push([d,n,b].join("*"));return a.join("!")}}function b(a){for(var e in a)if({}.hasOwnProperty.call(a,e)){var n=a[e];d.d(n,"Object")||d.d(n,"Array")?b(n):a[e]=String(n)}}var d=mt.lang,e=mt.m,k=mt.f,m=h.l,g=h.o,p=h.za,f={A:[],J:0,ea:v,r:{X:"",page:""},init:function(){f.e=
0;g.c("pv-b",function(){f.Aa();f.Ca()});g.c("pv-d",function(){f.Da();f.r.page=""});g.c("stag-b",function(){h.b.a.api=f.e||f.J?f.e+"_"+f.J:"";h.b.a.ct=[decodeURIComponent(h.b.getData("Hm_ct_"+c.id)||""),f.r.X,f.r.page].join("!")});g.c("stag-d",function(){h.b.a.api=0;f.e=0;f.J=0})},Aa:function(){var a=window._hmt||[];if(!a||d.d(a,"Array"))window._hmt={id:c.id,cmd:{},push:function(){for(var a=window._hmt,l=0;l<arguments.length;l++){var b=arguments[l];d.d(b,"Array")&&(a.cmd[a.id].push(b),"_setAccount"===
b[0]&&(1<b.length&&/^[0-9a-f]{32}$/.test(b[1]))&&(b=b[1],a.id=b,a.cmd[b]=a.cmd[b]||[]))}}},window._hmt.cmd[c.id]=[],window._hmt.push.apply(window._hmt,a)},Ca:function(){var a=window._hmt;if(a&&a.cmd&&a.cmd[c.id])for(var d=a.cmd[c.id],b=/^_track(Event|MobConv|Order|RTEvent)$/,e=0,g=d.length;e<g;e++){var t=d[e];b.test(t[0])?f.A.push(t):f.T(t)}a.cmd[c.id]={push:f.T}},Da:function(){if(0<f.A.length)for(var a=0,d=f.A.length;a<d;a++)f.T(f.A[a]);f.A=u},T:function(a){var b=a[0];if(f.hasOwnProperty(b)&&d.d(f[b],
"Function"))f[b](a)},_setAccount:function(a){1<a.length&&/^[0-9a-f]{32}$/.test(a[1])&&(f.e|=1)},_setAutoPageview:function(a){if(1<a.length&&(a=a[1],v===a||r===a))f.e|=2,h.b.ca=a},_trackPageview:function(a){if(1<a.length&&a[1].charAt&&"/"===a[1].charAt(0)){f.e|=4;h.b.a.et=0;h.b.a.ep="";h.b.a.vl=k.C()+k.D();h.b.R?(h.b.a.nv=0,h.b.a.st=4):h.b.R=r;var d=h.b.a.u,b=h.b.a.su;h.b.a.u=m.protocol+"//"+document.location.host+a[1];f.ea||(h.b.a.su=document.location.href);h.b.j();h.b.a.u=d;h.b.a.su=b}},_trackEvent:function(a){2<
a.length&&(f.e|=8,h.b.a.nv=0,h.b.a.st=4,h.b.a.et=4,h.b.a.ep=d.h(a[1])+"*"+d.h(a[2])+(a[3]?"*"+d.h(a[3]):"")+(a[4]?"*"+d.h(a[4]):""),h.b.j())},_setCustomVar:function(a){if(!(4>a.length)){var b=a[1],n=a[4]||3;if(0<b&&6>b&&0<n&&4>n){f.J++;for(var e=(h.b.a.cv||"*").split("!"),g=e.length;g<b-1;g++)e.push("*");e[b-1]=n+"*"+d.h(a[2])+"*"+d.h(a[3]);h.b.a.cv=e.join("!");a=h.b.a.cv.replace(/[^1](\*[^!]*){2}/g,"*").replace(/((^|!)\*)+$/g,"");""!==a?h.b.setData("Hm_cv_"+c.id,encodeURIComponent(a),c.age):h.b.fb("Hm_cv_"+
c.id)}}},_setUserTag:function(b){if(!(3>b.length)){var e=d.h(b[1]);b=d.h(b[2]);if(e!==q&&b!==q){var n=decodeURIComponent(h.b.getData("Hm_ct_"+c.id)||""),n=a(n,e,1,b);h.b.setData("Hm_ct_"+c.id,encodeURIComponent(n),c.age)}}},_setVisitTag:function(b){if(!(3>b.length)){var e=d.h(b[1]);b=d.h(b[2]);if(e!==q&&b!==q){var n=f.r.X,n=a(n,e,2,b);f.r.X=n}}},_setPageTag:function(b){if(!(3>b.length)){var e=d.h(b[1]);b=d.h(b[2]);if(e!==q&&b!==q){var n=f.r.page,n=a(n,e,3,b);f.r.page=n}}},_setReferrerOverride:function(a){1<
a.length&&(h.b.a.su=a[1].charAt&&"/"===a[1].charAt(0)?m.protocol+"//"+window.location.host+a[1]:a[1],f.ea=r)},_trackOrder:function(a){a=a[1];d.d(a,"Object")&&(b(a),f.e|=16,h.b.a.nv=0,h.b.a.st=4,h.b.a.et=94,h.b.a.ep=e.stringify(a),h.b.j())},_trackMobConv:function(a){if(a={webim:1,tel:2,map:3,sms:4,callback:5,share:6}[a[1]])f.e|=32,h.b.a.et=93,h.b.a.ep=a,h.b.j()},_trackRTPageview:function(a){a=a[1];d.d(a,"Object")&&(b(a),a=e.stringify(a),512>=encodeURIComponent(a).length&&(f.e|=64,h.b.a.rt=a))},_trackRTEvent:function(a){a=
a[1];if(d.d(a,"Object")){b(a);a=encodeURIComponent(e.stringify(a));var g=function(a){var b=h.b.a.rt;f.e|=128;h.b.a.et=90;h.b.a.rt=a;h.b.j();h.b.a.rt=b},n=a.length;if(900>=n)g.call(this,a);else for(var n=Math.ceil(n/900),s="block|"+Math.round(Math.random()*m.w).toString(16)+"|"+n+"|",k=[],t=0;t<n;t++)k.push(t),k.push(a.substring(900*t,900*t+900)),g.call(this,s+k.join("|")),k=[]}},_setUserId:function(a){a=a[1];p.$a();p.ob(a)}};f.init();h.ta=f;return h.ta})();
(function(){function a(){"undefined"===typeof window["_bdhm_loaded_"+c.id]&&(window["_bdhm_loaded_"+c.id]=r,this.a={},this.ca=r,this.R=v,this.init())}var b=mt.url,d=mt.la,e=mt.W,k=mt.lang,m=mt.cookie,g=mt.f,p=mt.localStorage,f=mt.sessionStorage,l=h.l,w=h.o;a.prototype={S:function(a,b){a="."+a.replace(/:\d+/,"");b="."+b.replace(/:\d+/,"");var d=a.indexOf(b);return-1<d&&d+b.length===a.length},fa:function(a,b){a=a.replace(/^https?:\/\//,"");return 0===a.indexOf(b)},F:function(a){for(var d=0;d<c.dm.length;d++)if(-1<
c.dm[d].indexOf("/")){if(this.fa(a,c.dm[d]))return r}else{var e=b.ba(a);if(e&&this.S(e,c.dm[d]))return r}return v},P:function(){for(var a=document.location.hostname,b=0,d=c.dm.length;b<d;b++)if(this.S(a,c.dm[b]))return c.dm[b].replace(/(:\d+)?[\/\?#].*/,"");return a},aa:function(){for(var a=0,b=c.dm.length;a<b;a++){var d=c.dm[a];if(-1<d.indexOf("/")&&this.fa(document.location.href,d))return d.replace(/^[^\/]+(\/.*)/,"$1")+"/"}return"/"},Pa:function(){if(!document.referrer)return l.p-l.z>c.vdur?1:
4;var a=v;this.F(document.referrer)&&this.F(document.location.href)?a=r:(a=b.ba(document.referrer),a=this.S(a||"",document.location.hostname));return a?l.p-l.z>c.vdur?1:4:3},getData:function(a){try{return m.get(a)||f.get(a)||p.get(a)}catch(b){}},setData:function(a,b,d){try{m.set(a,b,{domain:this.P(),path:this.aa(),N:d}),d?p.set(a,b,d):f.set(a,b)}catch(e){}},fb:function(a){try{m.set(a,"",{domain:this.P(),path:this.aa(),N:-1}),f.remove(a),p.remove(a)}catch(b){}},mb:function(){var a,b,d,e,f;l.z=this.getData("Hm_lpvt_"+
c.id)||0;13===l.z.length&&(l.z=Math.round(l.z/1E3));b=this.Pa();a=4!==b?1:0;if(d=this.getData("Hm_lvt_"+c.id)){e=d.split(",");for(f=e.length-1;0<=f;f--)13===e[f].length&&(e[f]=""+Math.round(e[f]/1E3));for(;2592E3<l.p-e[0];)e.shift();f=4>e.length?2:3;for(1===a&&e.push(l.p);4<e.length;)e.shift();d=e.join(",");e=e[e.length-1]}else d=l.p,e="",f=1;this.setData("Hm_lvt_"+c.id,d,c.age);this.setData("Hm_lpvt_"+c.id,l.p);d=l.p===this.getData("Hm_lpvt_"+c.id)?"1":"0";if(0===c.nv&&this.F(document.location.href)&&
(""===document.referrer||this.F(document.referrer)))a=0,b=4;this.a.nv=a;this.a.st=b;this.a.cc=d;this.a.lt=e;this.a.lv=f},ia:function(){for(var a=[],b=this.a.et,d=0,e=l.ma.length;d<e;d++){var f=l.ma[d],g=this.a[f];"undefined"!==typeof g&&""!==g&&("tt"!==f||"tt"===f&&0===b)&&("ct"!==f||"ct"===f&&0===b)&&a.push(f+"="+encodeURIComponent(g))}switch(b){case 0:a.push("sn="+l.Z);this.a.rt&&a.push("rt="+encodeURIComponent(this.a.rt));break;case 3:a.push("sn="+l.Z);break;case 90:this.a.rt&&a.push("rt="+this.a.rt)}return a.join("&")},
nb:function(){this.mb();this.a.si=c.id;this.a.su=document.referrer;this.a.ds=g.gb;this.a.cl=g.colorDepth+"-bit";this.a.ln=String(g.language).toLowerCase();this.a.ja=g.javaEnabled?1:0;this.a.ck=g.cookieEnabled?1:0;this.a.lo="number"===typeof _bdhm_top?1:0;this.a.fl=e.Qa();this.a.v="1.2.30";this.a.cv=decodeURIComponent(this.getData("Hm_cv_"+c.id)||"");this.a.tt=document.title||"";this.a.vl=g.C()+g.D();var a=document.location.href;this.a.cm=b.k(a,l.Va)||"";this.a.cp=b.k(a,l.Wa)||b.k(a,l.rb)||"";this.a.cw=
b.k(a,l.Ua)||b.k(a,l.ub)||"";this.a.ci=b.k(a,l.Sa)||b.k(a,l.qb)||"";this.a.cf=b.k(a,l.Xa)||b.k(a,l.tb)||"";this.a.cu=b.k(a,l.Ta)||b.k(a,l.pb)||""},init:function(){try{this.nb(),0===this.a.nv?this.lb():this.V(".*"),h.b=this,this.ua(),w.B("pv-b"),this.jb()}catch(a){var b=[];b.push("si="+c.id);b.push("n="+encodeURIComponent(a.name));b.push("m="+encodeURIComponent(a.message));b.push("r="+encodeURIComponent(document.referrer));d.log(l.H+"//"+l.U+"?"+b.join("&"))}},jb:function(){function a(){w.B("pv-d")}
this.ca?(this.R=r,this.a.et=0,this.a.ep="",this.a.vl=g.C()+g.D(),this.j(a)):a()},j:function(a){var b=this;b.a.rnd=Math.round(Math.random()*l.w);w.B("stag-b");var e=l.H+"//"+l.U+"?"+b.ia();w.B("stag-d");b.ra(e);d.log(e,function(d){b.V(d);k.d(a,"Function")&&a.call(b)})},ua:function(){var a=document.location.hash.substring(1),d=RegExp(c.id),e=-1<document.referrer.indexOf(l.sa),f=b.k(a,"jn"),g=/^heatlink$|^select$|^pageclick$/.test(f);a&&(d.test(a)&&e&&g)&&(this.a.rnd=Math.round(Math.random()*l.w),a=
document.createElement("script"),a.setAttribute("type","text/javascript"),a.setAttribute("charset","utf-8"),a.setAttribute("src",l.protocol+"//"+c.js+f+".js?"+this.a.rnd),f=document.getElementsByTagName("script")[0],f.parentNode.insertBefore(a,f))},ra:function(a){var b=f.get("Hm_unsent_"+c.id)||"",d=this.a.u?"":"&u="+encodeURIComponent(document.location.href),b=encodeURIComponent(a.replace(/^https?:\/\//,"")+d)+(b?","+b:"");f.set("Hm_unsent_"+c.id,b)},V:function(a){var b=f.get("Hm_unsent_"+c.id)||
"";b&&(a=encodeURIComponent(a.replace(/^https?:\/\//,"")),a=RegExp(a.replace(/([\*\(\)])/g,"\\$1")+"(%26u%3D[^,]*)?,?","g"),(b=b.replace(a,"").replace(/,$/,""))?f.set("Hm_unsent_"+c.id,b):f.remove("Hm_unsent_"+c.id))},lb:function(){var a=this,b=f.get("Hm_unsent_"+c.id);if(b)for(var b=b.split(","),e=function(b){d.log(l.H+"//"+decodeURIComponent(b),function(b){a.V(b)})},g=0,k=b.length;g<k;g++)e(b[g])}};return new a})();var z=h.l,A=h.load;
if(c.apps){var D=[z.protocol,"//ers.baidu.com/app/s.js?"];D.push(c.apps);A(D.join(""))}var E=h.l,F=h.load;c.lxb&&F([E.protocol,"//lxbjs.baidu.com/lxb.js?sid=",c.lxb].join(""));var G=h.load,H=h.l.protocol;if(c.qiao){for(var I=[H+"//goutong.baidu.com/site/"],J=c.id,K=5381,L=J.length,M=0;M<L;M++)K=(33*K+Number(J.charCodeAt(M)))%4294967296;2147483648<K&&(K-=2147483648);I.push(K%1E3+"/");I.push(c.id+"/b.js");I.push("?siteId="+c.qiao);G(I.join(""))}
(function(){var a=mt.event,b=mt.m;try{if(window.performance&&performance.timing&&"undefined"!==typeof h.b){var d=function(a){var b=performance.timing,d=b[a+"Start"]?b[a+"Start"]:0;a=b[a+"End"]?b[a+"End"]:0;return{start:d,end:a,value:0<a-d?a-d:0}},e=function(){var a;a=d("navigation");var e=d("request");a={netAll:e.start-a.start,netDns:d("domainLookup").value,netTcp:d("connect").value,srv:d("response").start-e.start,dom:performance.timing.domInteractive-performance.timing.fetchStart,loadEvent:d("loadEvent").end-
a.start};h.b.a.et=87;h.b.a.ep=b.stringify(a);h.b.j()};a.c(window,"load",function(){setTimeout(e,500)})}}catch(k){}})();
(function(){var a=mt.f,b=mt.lang,d=mt.event,e=mt.m;if("undefined"!==typeof h.b&&(c.med||(!a.da||7<a.Za)&&c.cvcc)){var k,m,g,p,f=function(a){if(a.item){for(var b=a.length,d=Array(b);b--;)d[b]=a[b];return d}return[].slice.call(a)},l=function(a,b){for(var d in a)if(a.hasOwnProperty(d)&&b.call(a,d,a[d])===v)return v},w=function(a,d){var f={};f.n=k;f.t="clk";f.v=a;if(d){var l=d.getAttribute("href"),m=d.getAttribute("onclick")?""+d.getAttribute("onclick"):u,n=d.getAttribute("id")||"";g.test(l)?(f.sn="mediate",
f.snv=l):b.d(m,"String")&&g.test(m)&&(f.sn="wrap",f.snv=m);f.id=n}h.b.a.et=86;h.b.a.ep=e.stringify(f);h.b.j();for(f=+new Date;400>=+new Date-f;);};if(c.med)m="/zoosnet",k="swt",g=/swt|zixun|call|chat|zoos|business|talk|kefu|openkf|online|\/LR\/Chatpre\.aspx/i,p={click:function(){for(var a=[],b=f(document.getElementsByTagName("a")),b=[].concat.apply(b,f(document.getElementsByTagName("area"))),b=[].concat.apply(b,f(document.getElementsByTagName("img"))),d,e,k=0,l=b.length;k<l;k++)d=b[k],e=d.getAttribute("onclick"),
d=d.getAttribute("href"),(g.test(e)||g.test(d))&&a.push(b[k]);return a}};else if(c.cvcc){m="/other-comm";k="other";g=c.cvcc.q||q;var n=c.cvcc.id||q;p={click:function(){for(var a=[],b=f(document.getElementsByTagName("a")),b=[].concat.apply(b,f(document.getElementsByTagName("area"))),b=[].concat.apply(b,f(document.getElementsByTagName("img"))),d,e,k,l=0,m=b.length;l<m;l++)d=b[l],g!==q?(e=d.getAttribute("onclick"),k=d.getAttribute("href"),n?(d=d.getAttribute("id"),(g.test(e)||g.test(k)||n.test(d))&&
a.push(b[l])):(g.test(e)||g.test(k))&&a.push(b[l])):n!==q&&(d=d.getAttribute("id"),n.test(d)&&a.push(b[l]));return a}}}if("undefined"!==typeof p&&"undefined"!==typeof g){var s;m+=/\/$/.test(m)?"":"/";var y=function(a,d){if(s===d)return w(m+a,d),v;if(b.d(d,"Array")||b.d(d,"NodeList"))for(var e=0,f=d.length;e<f;e++)if(s===d[e])return w(m+a+"/"+(e+1),d[e]),v};d.c(document,"mousedown",function(a){a=a||window.event;s=a.target||a.srcElement;var d={};for(l(p,function(a,e){d[a]=b.d(e,"Function")?e():document.getElementById(e)});s&&
s!==document&&l(d,y)!==v;)s=s.parentNode})}}})();(function(){var a=mt.g,b=mt.lang,d=mt.event,e=mt.m;if("undefined"!==typeof h.b&&b.d(c.cvcf,"Array")&&0<c.cvcf.length){var k={pa:function(){for(var b=c.cvcf.length,e,p=0;p<b;p++)(e=a.O(decodeURIComponent(c.cvcf[p])))&&d.c(e,"click",k.M())},M:function(){return function(){h.b.a.et=86;var a={n:"form",t:"clk"};a.id=this.id;h.b.a.ep=e.stringify(a);h.b.j()}}};a.ga(function(){k.pa()})}})();
(function(){var a=mt.event,b=mt.m;if(c.med&&"undefined"!==typeof h.b){var d=+new Date,e={n:"anti",sb:0,kb:0,clk:0},k=function(){h.b.a.et=86;h.b.a.ep=b.stringify(e);h.b.j()};a.c(document,"click",function(){e.clk++});a.c(document,"keyup",function(){e.kb=1});a.c(window,"scroll",function(){e.sb++});a.c(window,"unload",function(){e.t=+new Date-d;k()});a.c(window,"load",function(){setTimeout(k,5E3)})}})();})();