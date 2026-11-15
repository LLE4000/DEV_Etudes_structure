// Le PLAN QUI SE MANIPULE (le « Générateur SAF », api_web/dessin_saf.py) : on déplace un objet du dessin, on le tourne, on
// tape une cote, on ajoute un sommet — au doigt comme à la souris. Ce fichier ne sait RIEN de ce qui est dessiné : le
// moteur écrit dans le dessin ce que chaque geste règle (un chemin de l'état, les valeurs de départ, l'échelle, la grille,
// les accroches) ; au lâcher, la valeur part au moteur (« modifier »), qui la relit, recalcule et redessine.
//
//   data-plan-glisser = {chemin, x, y, ech, pas, ox, oy, ax, ay, noms, unite, min, max, suit, double}
//       un objet qui se déplace. « x » et « y » : les valeurs de départ (une seule : le geste n'a qu'un sens) ; « ech » :
//       unités du dessin par unité de la valeur ; « pas » : la grille (son origine « ox », « oy ») ; « ax », « ay » : les
//       accroches, plus fortes que la grille ; « suit » = {sel, i} : les polygones dont le sommet i suit le geste ;
//       « double » : le chemin d'un double-clic (retirer ce sommet).
//   data-plan-tourner = {chemin, cx, cy, angle, accroches, cible} : une poignée de rotation autour de (cx, cy).
//   data-plan-cote    = {chemin, valeur, unite} : une cote qui se tape (un clic ouvre un champ sur elle).
//   data-plan-sec     = la rubrique des données qu'un clic sur l'objet ouvre.
//   data-plan-clic    = {chemin, valeur} : ce qu'un appui bref sur l'objet RÈGLE (un bord devient le repère des cotes).
//   data-plan-bord    = le rang d'un bord : un double appui y ajoute un sommet ({apres, x, y}, au chemin de data-plan-fond).
//   data-plan-fond    = {chemin, ech, pas} : la feuille — un clic dans le vide ajoute un sommet ({x, y}).
//   <svg data-plan>   = {ech, origine: {x, y}} : l'origine du repère, en unités du dessin (y du modèle vers le haut).
//
// … et, sur toute la page de l'établi (hors du cadre des données, que js/etabli.js écoute déjà) :
//   data-agir         : un bouton (data-chemin + data-valeur), un menu, une case ou un CHAMP (data-chemin) du résultat, du
//                       dessin ou d'une fenêtre qui MODIFIE la note ;
//   data-plan-volet   : un <details> du résultat qui reste ouvert d'un calcul à l'autre, une fois déplié ;
//   data-suite        = {role, etapes, fin, sortie} : une SUITE de lectures (« selecteur », rôle « role:etape ») dont
//                       chaque réponse se pose dans « sortie », sous une barre de progression ; au bout, « fin » (un
//                       chemin) part en « modifier ». Un bouton data-suite-arret l'arrête ;
//   data-reprendre    = {module, chemin, titre} : choisir une NOTE d'un autre module, dont le nom et l'état partent au
//                       chemin donné.
import { SEUIL } from "./compact.js";

const SVG = "http://www.w3.org/2000/svg";
const TOLERANCE_PX = 9;          // une accroche prend à cette distance du pointeur
const TOLERANCE_ANGLE = 4;       // … et une accroche d'angle à 4°
const arrondi = (v) => Math.round(v * 1e4) / 1e4;
const fr = (v, nd = 2) => arrondi(v).toFixed(nd).replace(".", ",").replace(/^-(0,?0*)$/, "$1").replace("-", "−");
const lire = (el, cle) => { try { return JSON.parse(el.dataset[cle]); } catch (e) { return null; } };

// la valeur accrochée : une accroche proche l'emporte, sinon la grille (origine o), dans les bornes
export function accrocher(brut, { pas = 0, o = 0, accroches = [], tol = 0, min = null, max = null } = {}) {
  let v = pas ? Math.round((brut + o) / pas) * pas - o : brut;
  let pres = null;
  for (const a of accroches) if (Math.abs(brut - a) <= tol && (pres === null || Math.abs(brut - a) < Math.abs(brut - pres))) pres = a;
  if (pres !== null) v = pres;
  if (min !== null && min !== undefined) v = Math.max(min, v);
  if (max !== null && max !== undefined) v = Math.min(max, v);
  return arrondi(v);
}

// l'angle accroché (degrés, sens trigonométrique), ramené dans ]−180, 180]
export function accrocherAngle(brut, accroches = [], tol = TOLERANCE_ANGLE) {
  const norme = (a) => { let x = ((a % 360) + 360) % 360; if (x > 180) x -= 360; return x; };
  let v = Math.round(brut), ecart = tol;
  for (const a of accroches) {
    const e = Math.abs(norme(brut - a));
    if (e <= ecart) { ecart = e; v = a; }
  }
  return arrondi(norme(v));
}

export function cablerPlan({ bench, stage, envoyer, ouvrirRubrique, notes, toast, signal }) {
  const fini = () => signal && signal.aborted;
  const chemin = (c) => String(c).split("/");
  let geste = null, appui = null, avalerClic = false;
  // le DOUBLE appui : la toile capture le pointeur dès l'appui (elle déplace la vue), et l'événement « dblclick » tombe alors
  // sur le fond du dessin, jamais sur l'objet — on reconnaît donc ici deux appuis brefs sur le même objet
  let dernierAppui = null, doubleFait = -1e9;
  const double = (cle) => {
    const t = performance.now();
    const oui = !!dernierAppui && dernierAppui.cle === cle && t - dernierAppui.t < 450;
    dernierAppui = oui ? null : { cle, t };
    if (oui) doubleFait = t;
    return oui;
  };

  const etiquette = (svg) => {
    const t = document.createElementNS(SVG, "text");
    t.setAttribute("class", "sf-etiq");
    svg.appendChild(t);
    return t;
  };
  const poserEtiquette = (g, ev, txt) => {
    const m = g.svg.getScreenCTM();
    if (!m) return;
    const p = g.svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(m.inverse());
    const t = 13 / m.a;                       // 13 px à l'écran, quel que soit le zoom
    g.etiq.textContent = txt;
    g.etiq.setAttribute("font-size", t);
    g.etiq.setAttribute("stroke-width", t * 0.3);
    g.etiq.setAttribute("x", q.x);
    g.etiq.setAttribute("y", q.y - t * 1.6);
  };

  // ------------------------------------------------------------------ déplacer, tourner
  function valeurs(g, ev) {
    const { d, kx, ky } = g;
    const tol = TOLERANCE_PX * kx / d.ech;
    const out = {};
    if (d.x !== undefined) out.x = accrocher(d.x + (ev.clientX - g.x0) * kx / d.ech, { pas: d.pas, o: d.ox || 0, accroches: d.ax || [], tol, min: d.min, max: d.max });
    if (d.y !== undefined) out.y = accrocher(d.y - (ev.clientY - g.y0) * ky / d.ech, { pas: d.pas, o: d.oy || 0, accroches: d.ay || [], tol, min: d.min, max: d.max });
    return out;
  }
  function angle(g, ev) {
    const m = g.svg.getScreenCTM();
    const p = g.svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(m.inverse());
    return accrocherAngle(Math.atan2(-(q.y - g.d.cy), q.x - g.d.cx) * 180 / Math.PI, g.d.accroches || []);
  }
  function suivre(g, dx, dy) {                // les polygones dont un sommet suit le geste
    if (!g.d.suit) return;
    g.svg.querySelectorAll(g.d.suit.sel).forEach((poly) => {
      if (!poly.planBase) poly.planBase = poly.getAttribute("points");
      const pts = poly.planBase.trim().split(/\s+/).map((c) => c.split(",").map(Number));
      for (const i of [].concat(g.d.suit.i)) if (pts[i]) { pts[i][0] += dx; pts[i][1] += dy; }
      poly.setAttribute("points", pts.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join(" "));
    });
  }
  function relacherSuite(g) {
    if (!g.d.suit) return;
    g.svg.querySelectorAll(g.d.suit.sel).forEach((poly) => { if (poly.planBase) { poly.setAttribute("points", poly.planBase); delete poly.planBase; } });
  }

  stage.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest || (ev.pointerType === "mouse" && ev.button !== 0)) return;
    const el = ev.target.closest(".view.on [data-plan-tourner], .view.on [data-plan-glisser]");
    const svg = el && el.ownerSVGElement;
    const m = svg && svg.getScreenCTM();
    if (!el || !m) {
      // un objet qui ne se déplace pas : un appui bref ouvre sa rubrique (la toile garde le déplacement de la vue)
      const s = ev.target.closest(".view.on [data-plan-sec], .view.on [data-plan-cote], .view.on [data-plan-fond], .view.on [data-plan-clic]");
      appui = s ? { el: s, x: ev.clientX, y: ev.clientY, id: ev.pointerId, seuil: SEUIL[ev.pointerType] || 5 } : null;
      return;
    }
    const tourne = el.hasAttribute("data-plan-tourner");
    const d = lire(el, tourne ? "planTourner" : "planGlisser");
    if (!d) return;
    ev.stopImmediatePropagation();
    ev.preventDefault();
    const cible = tourne ? el.closest(d.cible) || el : el;
    geste = { el, cible, svg, d, tourne, x0: ev.clientX, y0: ev.clientY, kx: 1 / m.a, ky: 1 / m.d, id: ev.pointerId, bouge: false, etiq: null,
      base: cible.getAttribute("transform") || "", seuil: SEUIL[ev.pointerType] || 5, v: null };
    try { el.setPointerCapture(ev.pointerId); } catch (e) { /* pointeur déjà relâché */ }
  }, { capture: true, signal });
  // au doigt, un appui qui bouge fait défiler la page : celui qui commence sur un objet déplaçable est à nous
  stage.addEventListener("touchstart", (ev) => {
    if (ev.target.closest && ev.target.closest(".view.on [data-plan-tourner], .view.on [data-plan-glisser]")) ev.preventDefault();
  }, { passive: false, signal });

  window.addEventListener("pointermove", (ev) => {
    const g = geste;
    if (!g || ev.pointerId !== g.id) return;
    if (!g.bouge && Math.hypot(ev.clientX - g.x0, ev.clientY - g.y0) <= g.seuil) return;
    ev.preventDefault();
    if (!g.bouge) {
      g.bouge = true;
      g.svg.classList.add("sf-en-geste");
      g.cible.classList.add("sf-glisse");
      g.etiq = etiquette(g.svg);
    }
    const noms = g.d.noms || [];
    if (g.tourne) {
      g.v = angle(g, ev);
      g.cible.setAttribute("transform", `${g.base} rotate(${-(g.v - g.d.angle)} ${g.d.cx} ${g.d.cy})`.trim());
      poserEtiquette(g, ev, `β = ${fr(g.v, Number.isInteger(g.v) ? 0 : 2)}°`);
      return;
    }
    g.v = valeurs(g, ev);
    const dx = g.v.x !== undefined ? (g.v.x - g.d.x) * g.d.ech : 0, dy = g.v.y !== undefined ? -(g.v.y - g.d.y) * g.d.ech : 0;
    g.cible.setAttribute("transform", `${g.base} translate(${dx} ${dy})`.trim());
    suivre(g, dx, dy);
    const u = g.d.unite ? ` ${g.d.unite}` : "";
    const mots = [];
    if (g.v.x !== undefined) mots.push(`${(noms[0] || "x").replace("_", "")} = ${fr(g.v.x)}${u}`);
    if (g.v.y !== undefined) mots.push(`${(noms[g.v.x !== undefined ? 1 : 0] || "y").replace("_", "")} = ${fr(g.v.y)}${u}`);
    poserEtiquette(g, ev, mots.join("   "));
  }, { passive: false, signal });

  function finir(ev, annule) {
    const g = geste;
    if (!g || ev.pointerId !== g.id) return;
    geste = null;
    g.svg.classList.remove("sf-en-geste");
    g.cible.classList.remove("sf-glisse");
    if (g.etiq) g.etiq.remove();
    const remettre = () => { if (g.base) g.cible.setAttribute("transform", g.base); else g.cible.removeAttribute("transform"); relacherSuite(g); };
    if (!g.bouge) {
      // un appui sans mouvement : un clic sur l'objet — sa rubrique s'ouvre ; deux de suite : son double-clic (retirer un sommet)
      if (annule) return;
      if (g.d.double && double(`geste:${g.d.chemin}`)) { envoyer("modifier", { chemin: chemin(g.d.double), valeur: "1" }); return; }
      const c = g.el.hasAttribute("data-plan-clic") && lire(g.el, "planClic");
      if (c) { envoyer("modifier", { chemin: chemin(c.chemin), valeur: c.valeur }); return; }
      const s = g.el.closest("[data-plan-sec]");
      if (s) ouvrirRubrique(s.dataset.planSec);
      return;
    }
    avalerClic = true;
    setTimeout(() => { avalerClic = false; }, 0);
    if (annule || g.v === null) { remettre(); return; }
    let valeur, change;
    if (g.tourne) { valeur = g.v; change = Math.abs(g.v - g.d.angle) > 1e-9; }
    else if (g.v.x !== undefined && g.v.y !== undefined) { valeur = [g.v.x, g.v.y]; change = g.v.x !== g.d.x || g.v.y !== g.d.y; }
    else { valeur = g.v.x !== undefined ? g.v.x : g.v.y; change = valeur !== (g.v.x !== undefined ? g.d.x : g.d.y); }
    if (!change) { remettre(); return; }
    // (le dessin garde la pose du geste jusqu'à la réponse du moteur, qui le refait ; un refus le remet)
    envoyer("modifier", { chemin: chemin(g.d.chemin), valeur }).then((r) => { if (!r && g.cible.isConnected) remettre(); });
  }
  window.addEventListener("pointerup", (ev) => finir(ev, false), { capture: true, signal });
  window.addEventListener("pointercancel", (ev) => finir(ev, true), { capture: true, signal });

  // ------------------------------------------------------------------ un appui bref : une rubrique, une cote, un sommet
  const pointModele = (svg, ev, pas) => {
    const p0 = lire(svg, "plan"), m = svg.getScreenCTM();
    if (!p0 || !m) return null;
    const p = svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(m.inverse());
    const grille = (v) => arrondi(pas ? Math.round(v / pas) * pas : v);
    return { x: grille((q.x - p0.origine.x) / p0.ech), y: grille((p0.origine.y - q.y) / p0.ech) };
  };
  window.addEventListener("pointerup", (ev) => {
    const a = appui;
    appui = null;
    if (!a || ev.pointerId !== a.id || fini()) return;
    if (Math.hypot(ev.clientX - a.x, ev.clientY - a.y) > a.seuil) return;
    if (a.el.hasAttribute("data-plan-cote")) { ouvrirCote(a.el); return; }
    if (a.el.hasAttribute("data-plan-bord") && double(`bord:${a.el.dataset.planBord}`)) {
      // deux appuis sur un bord : un sommet de plus, à cet endroit
      const svg = a.el.ownerSVGElement, fond = svg.querySelector("[data-plan-fond]"), d = fond && lire(fond, "planFond"), pt = d && pointModele(svg, ev, d.pas);
      if (pt) envoyer("modifier", { chemin: chemin(d.chemin), valeur: { ...pt, apres: +a.el.dataset.planBord } });
      return;
    }
    if (a.el.hasAttribute("data-plan-fond")) {
      const d = lire(a.el, "planFond"), pt = d && pointModele(a.el.ownerSVGElement, ev, d.pas);
      if (pt) envoyer("modifier", { chemin: chemin(d.chemin), valeur: pt });
      return;
    }
    const c = a.el.hasAttribute("data-plan-clic") && lire(a.el, "planClic");
    if (c) { envoyer("modifier", { chemin: chemin(c.chemin), valeur: c.valeur }); return; }
    if (a.el.dataset.planSec) ouvrirRubrique(a.el.dataset.planSec);
  }, { capture: true, signal });
  window.addEventListener("pointercancel", () => { appui = null; }, { capture: true, signal });

  // un double appui que le plan vient de prendre (un sommet ajouté, retiré) n'est pas aussi celui de la toile (qui remet
  // le dessin agrandi à sa feuille)
  stage.addEventListener("dblclick", (ev) => { if (performance.now() - doubleFait < 700) ev.stopImmediatePropagation(); }, { capture: true, signal });

  // ------------------------------------------------------------------ une cote qui se tape
  let champ = null;
  function fermerCote() { if (champ) { champ.remove(); champ = null; } }
  function ouvrirCote(el) {
    const d = lire(el, "planCote");
    if (!d) return;
    fermerCote();
    const r = el.getBoundingClientRect();
    const i = document.createElement("input");
    i.className = "sf-saisie";
    i.setAttribute("inputmode", "decimal");
    i.setAttribute("aria-label", `Valeur de la cote${d.unite ? ` (${d.unite})` : ""}`);
    i.value = String(arrondi(d.valeur)).replace(".", ",");
    const l = Math.max(64, r.width + 26);
    i.style.left = `${Math.round(Math.min(Math.max(4, r.left + r.width / 2 - l / 2), innerWidth - l - 4))}px`;
    i.style.top = `${Math.round(Math.max(4, r.top + r.height / 2 - 15))}px`;
    i.style.width = `${Math.round(l)}px`;
    document.body.appendChild(i);
    champ = i;
    i.focus();
    i.select();
    let fait = false;
    const valider = () => {
      if (fait) return;
      fait = true;
      const txt = i.value.trim();
      fermerCote();
      if (txt && txt !== String(arrondi(d.valeur)).replace(".", ",")) envoyer("modifier", { chemin: chemin(d.chemin), valeur: txt });
    };
    i.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); valider(); }
      else if (e.key === "Escape") { fait = true; fermerCote(); }
    });
    i.addEventListener("blur", valider);
  }
  if (signal) signal.addEventListener("abort", fermerCote, { once: true });
  stage.addEventListener("wheel", fermerCote, { passive: true, signal });

  // ------------------------------------------------------------------ data-agir : un contrôle du résultat ou du dessin
  const horsDonnees = (el) => el && !el.closest("#donnees") && !el.closest(".panel.left .phead");
  bench.addEventListener("click", (ev) => {
    if (avalerClic) { ev.stopPropagation(); return; }
    // (une case « data-agir » : son clic ne remonte pas — la ligne qui la porte n'ouvre pas sa fenêtre de calcul)
    if (ev.target.closest && ev.target.closest("label.sf-scia, input[data-agir], select[data-agir]") && horsDonnees(ev.target)) { ev.stopPropagation(); return; }
    const b = ev.target.closest && ev.target.closest("button[data-agir][data-chemin]");
    if (b && horsDonnees(b) && !b.disabled) {
      ev.stopPropagation();
      envoyer("modifier", { chemin: chemin(b.dataset.chemin), valeur: b.dataset.valeur ?? "1" });
      return;
    }
    const s = ev.target.closest && ev.target.closest("[data-suite]");
    if (s && !s.disabled) { ev.stopPropagation(); lancerSuite(s); return; }
    if (ev.target.closest && ev.target.closest("[data-suite-arret]")) { suiteEnCours = null; return; }
    const rp = ev.target.closest && ev.target.closest("[data-reprendre]");
    if (rp && !rp.disabled) { ev.stopPropagation(); choisirNote(rp); }
  }, { capture: true, signal });
  bench.addEventListener("change", (ev) => {
    const el = ev.target.closest && ev.target.closest("select[data-agir][data-chemin], input[type=checkbox][data-agir][data-chemin]");
    if (!el || !horsDonnees(el)) return;
    envoyer("modifier", { chemin: chemin(el.dataset.chemin), valeur: el.type === "checkbox" ? el.checked : el.value });
  }, { signal });
  // un FICHIER choisi hors du cadre des données (« Importer le fichier reçu » d'un loueur, Prix des grues) : il part au
  // moteur lu en texte ({nom, texte}) ou, s'il porte data-binaire (un classeur Excel), en base64 ({nom, base64}) ; 2 Mo au
  // plus. Le champ est vidé ensuite : choisir deux fois le même fichier le relit.
  bench.addEventListener("change", async (ev) => {
    const f = ev.target.closest && ev.target.closest("input[type=file][data-agir][data-chemin]");
    if (!f || !horsDonnees(f)) return;
    const fichier = f.files && f.files[0];
    if (!fichier) return;
    if (fichier.size > 2 * 1024 * 1024) { if (toast) toast("Fichier trop gros (2 Mo au plus)", true); f.value = ""; return; }
    let valeur;
    if (f.dataset.binaire) {
      const o = new Uint8Array(await fichier.arrayBuffer());
      let bin = "";
      for (let i = 0; i < o.length; i += 0x8000) bin += String.fromCharCode.apply(null, o.subarray(i, i + 0x8000));
      valeur = { nom: fichier.name, base64: btoa(bin) };
    } else {
      valeur = { nom: fichier.name, texte: await fichier.text() };
    }
    f.value = "";
    if (signal && signal.aborted) return;
    envoyer("modifier", { chemin: chemin(f.dataset.chemin), valeur });
  }, { signal });

  // ------------------------------------------------------------------ data-agir : un CHAMP du résultat, du dessin ou d'une fenêtre
  // (le « Prix des grues » : une quantité, un prix forcé, une justification, une case de la base des prix) : il part en quittant
  // le champ, seulement si sa valeur a changé — Tab passe au champ suivant, Entrée aussi (au dernier, il quitte le champ),
  // Échap rend la valeur d'avant. Le cadre est refait à chaque réponse : le champ où l'on est arrivé garde le curseur et ce
  // qu'on y tape (js/etabli.js, saisieEnCours). Une COLONNE collée depuis Excel (plusieurs lignes) dans un champ qui porte
  // data-coller part en une fois, au chemin data-coller.
  const champAgir = (el) => !!el && !!el.matches && horsDonnees(el)
    && el.matches("input[data-agir][data-chemin]:not([type=checkbox]):not([type=file]), textarea[data-agir][data-chemin]");
  bench.addEventListener("focusout", (ev) => {
    const el = ev.target;
    if (!champAgir(el) || el.value === el.defaultValue) return;
    envoyer("modifier", { chemin: chemin(el.dataset.chemin), valeur: el.value });
  }, { signal });
  bench.addEventListener("keydown", (ev) => {
    const el = ev.target;
    if (!champAgir(el) || el.tagName === "TEXTAREA") return;
    if (ev.key === "Enter") {
      ev.preventDefault();
      const tous = [...bench.querySelectorAll("input[data-agir][data-chemin]:not([type=checkbox]):not([type=file])")]
        .filter((x) => horsDonnees(x) && x.offsetParent !== null && !x.disabled);
      const suivant = tous[tous.indexOf(el) + 1];
      if (suivant) { suivant.focus(); if (suivant.select) suivant.select(); } else el.blur();
    } else if (ev.key === "Escape") { el.value = el.defaultValue; el.blur(); }
  }, { signal });
  bench.addEventListener("paste", (ev) => {
    const el = ev.target.closest && ev.target.closest("input[data-agir][data-coller]");
    const texte = ev.clipboardData && ev.clipboardData.getData("text");
    if (!el || !horsDonnees(el) || !texte || !/[\t\n]/.test(texte.trim())) return;
    ev.preventDefault();
    envoyer("modifier", { chemin: chemin(el.dataset.coller), valeur: texte });
  }, { signal });

  // ------------------------------------------------------------------ data-plan-volet : une liste qu'on a dépliée le reste
  // (le cadre des résultats est refait à chaque réponse du moteur : un <details data-plan-volet> qu'on a ouvert se rouvre)
  const volets = new Set();
  bench.addEventListener("toggle", (ev) => {
    const d = ev.target;
    if (!d.matches || !d.matches("details[data-plan-volet]")) return;
    if (d.open) volets.add(d.dataset.planVolet); else volets.delete(d.dataset.planVolet);
  }, { capture: true, signal });
  const rouvrir = new MutationObserver(() => {
    bench.querySelectorAll("details[data-plan-volet]").forEach((d) => { if (volets.has(d.dataset.planVolet) && !d.open) d.open = true; });
  });
  const resultat = bench.querySelector("#resultat");
  if (resultat) rouvrir.observe(resultat, { childList: true });
  if (signal) signal.addEventListener("abort", () => rouvrir.disconnect(), { once: true });

  // ------------------------------------------------------------------ data-courts : le libellé court d'un choix qui ne tient pas
  // (la barre du centre — cahier du lot 4, § 2.3 : « sur tablette, tout tient sur une ligne, avec des libellés courts si
  // besoin ») : le choix AFFICHÉ prend son libellé court quand le long dépasse la place que le menu lui laisse ; la liste
  // dépliée garde les longs. Mesuré dans la police du menu ; refait à chaque réponse du moteur et à chaque redimension.
  const regle = document.createElement("canvas").getContext("2d");
  const deborde = (s) => {                                        // le texte affiché dépasse-t-il la place du menu ?
    const o = s.options[s.selectedIndex];
    if (!o || !s.clientWidth) return false;
    const cs = getComputedStyle(s);
    regle.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    return regle.measureText(o.textContent).width > s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) + 0.5;
  };
  const passe = (menus) => menus.forEach((s) => {
    let courts;
    try { courts = JSON.parse(s.dataset.courts); } catch (e) { return; }
    [...s.options].forEach((o) => { if (o.dataset.long !== undefined) { o.textContent = o.dataset.long; delete o.dataset.long; } });
    const o = s.options[s.selectedIndex];
    if (o && o.value in courts && deborde(s)) { o.dataset.long = o.textContent; o.textContent = courts[o.value]; }
  });
  const raccourcir = () => {
    stage.querySelectorAll(".sf-barre").forEach((barre) => {
      const menus = [...barre.querySelectorAll("select[data-courts]")];
      barre.classList.remove("serre");
      passe(menus);
      if (menus.some(deborde)) { barre.classList.add("serre"); passe(menus); }   // (même le court ne tient pas : deux lignes)
    });
  };
  let attente = 0;
  const bientot = () => { cancelAnimationFrame(attente); attente = requestAnimationFrame(() => { veille.disconnect(); raccourcir(); veille.takeRecords(); veille.observe(stage, { childList: true, subtree: true }); }); };
  const veille = new MutationObserver(bientot);                 // (déconnecté le temps de raccourcir : ses propres changements ne le relancent pas)
  veille.observe(stage, { childList: true, subtree: true });
  const taille = typeof ResizeObserver === "function" ? new ResizeObserver(bientot) : null;
  if (taille) taille.observe(stage);
  bientot();
  if (signal) signal.addEventListener("abort", () => { veille.disconnect(); if (taille) taille.disconnect(); cancelAnimationFrame(attente); }, { once: true });

  // ------------------------------------------------------------------ data-cherche : la recherche d'un sélecteur
  // les lignes du sélecteur qui ne portent pas tous les mots tapés se cachent (sans tenir compte des accents ni de la casse)
  const simple = (t) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  document.addEventListener("input", (ev) => {
    const c = ev.target.closest && ev.target.closest(".picker [data-cherche]");
    if (!c) return;
    const mots = simple(c.value).split(/\s+/).filter(Boolean);
    c.closest(".picker").querySelectorAll(".pk-row").forEach((r) => { const t = simple(r.textContent); r.hidden = !mots.every((m) => t.includes(m)); });
  }, { signal });

  // ------------------------------------------------------------------ data-suite : une suite de lectures, sa progression
  let suiteEnCours = null;
  async function lancerSuite(b) {
    const d = lire(b, "suite");
    const hote = b.closest("[data-suite-hote]") || b.parentElement;
    if (!d || !Array.isArray(d.etapes) || !d.etapes.length || suiteEnCours) return;
    const jeton = {};
    suiteEnCours = jeton;
    const barre = document.createElement("div");
    barre.className = "sf-prog";
    barre.innerHTML = '<div class="sf-prog-b" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><i></i></div>'
      + '<span class="sf-prog-t"></span><button type="button" class="btn sm ghost" data-suite-arret="1">Arrêter</button>';
    b.hidden = true;
    hote.appendChild(barre);
    const avance = (i) => {
      const pc = Math.round(100 * i / d.etapes.length);
      barre.querySelector("i").style.width = `${pc}%`;
      barre.querySelector(".sf-prog-b").setAttribute("aria-valuenow", String(pc));
      barre.querySelector(".sf-prog-t").textContent = `${i} / ${d.etapes.length}`;
    };
    avance(0);
    let complet = true;
    for (let i = 0; i < d.etapes.length; i++) {
      const r = await envoyer("selecteur", { role: `${d.role}:${d.etapes[i]}` });
      // (le résultat a été refait pendant la suite — une autre saisie — ou l'utilisateur l'a arrêtée : on n'écrit rien)
      if (fini() || suiteEnCours !== jeton || !barre.isConnected || !r) { complet = false; break; }
      const sortie = d.sortie && document.querySelector(d.sortie);
      if (sortie) sortie.innerHTML = r.html;
      avance(i + 1);
    }
    if (suiteEnCours === jeton) suiteEnCours = null;
    if (barre.isConnected) { barre.remove(); b.hidden = false; }
    if (complet && d.fin) envoyer("modifier", { chemin: chemin(d.fin), valeur: "1" });
  }

  // ------------------------------------------------------------------ data-reprendre : la note d'un autre module
  function choisirNote(b) {
    const d = lire(b, "reprendre");
    if (!d || !notes) return;
    document.querySelectorAll(".sf-notes").forEach((x) => x.remove());
    const liste = Object.values(notes.toutes()).filter((n) => n && notes.moduleServi(n.module) === d.module && n.etat)
      .sort((p, q) => (q.modifiee || q.creee || 0) - (p.modifiee || p.creee || 0));
    const boite = document.createElement("div");
    boite.className = "sf-notes";
    boite.setAttribute("role", "dialog");
    boite.setAttribute("aria-label", d.titre || "Choisir une note");
    const echapper = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    boite.innerHTML = `<div class="sf-notes-h"><b>${echapper(d.titre || "Choisir une note")}</b><button type="button" class="pcoll" data-fermer-notes="1" title="Fermer">✕</button></div>`
      + (liste.length ? `<div class="sf-notes-l">${liste.map((n, i) => `<button type="button" data-note="${i}"><b>${echapper(n.nom || "Sans nom")}</b>`
        + `<span>${echapper(n.chantier || "")}</span></button>`).join("")}</div>`
        : `<p class="sf-notes-v">${echapper(d.vide || "Aucune note de ce module dans ce navigateur.")}</p>`);
    const r = b.getBoundingClientRect();
    document.body.appendChild(boite);
    boite.style.left = `${Math.round(Math.min(Math.max(8, r.left), innerWidth - boite.offsetWidth - 8))}px`;
    boite.style.top = `${Math.round(Math.min(r.bottom + 6, innerHeight - boite.offsetHeight - 8))}px`;
    const fermer = () => { boite.remove(); document.removeEventListener("pointerdown", dehors, true); };
    const dehors = (e) => { if (!boite.contains(e.target)) fermer(); };
    setTimeout(() => document.addEventListener("pointerdown", dehors, true));
    boite.addEventListener("click", (e) => {
      if (e.target.closest("[data-fermer-notes]")) { fermer(); return; }
      const c = e.target.closest("[data-note]");
      if (!c) return;
      const n = liste[+c.dataset.note];
      fermer();
      envoyer("modifier", { chemin: chemin(d.chemin), valeur: { nom: n.nom || "", chantier: n.chantier || "", etat: n.etat } })
        .then((rep) => { if (rep && !rep.message && toast) toast(`« ${n.nom || "Sans nom"} » reprise`); });
    });
    boite.addEventListener("keydown", (e) => { if (e.key === "Escape") fermer(); });
    if (signal) signal.addEventListener("abort", fermer, { once: true });
    const premier = boite.querySelector("[data-note]") || boite.querySelector("[data-fermer-notes]");
    if (premier) premier.focus();
  }
}
