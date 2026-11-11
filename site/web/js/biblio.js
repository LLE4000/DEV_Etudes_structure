// La BIBLIOTHÈQUE câblée dans son hôte (docs/site/BIBLIOTHEQUES.md § 2.2 et § 2.4) : la page bibliotheque-<famille>
// (js/bibliotheque.js, hôte « page ») et le panneau « Bibliothèque » d'une note (js/etabli.js, hôte « note »). Tout le
// HTML vient du moteur (api_web/biblio_<famille>.consulter, par le routeur) : ce fichier pose les fragments, relit les
// paramètres, allume la rubrique du menu et tient le volet « Pourquoi ? ». Il ne calcule rien et n'écrit aucune note.
import { toast } from "./commun.js";

// racine : l'élément qui contient le gabarit (document pour la page, le panneau pour une note) ; demander(rubrique,
// params) rend la réponse du moteur ; apres(r) : ce que l'hôte fait de la réponse (l'adresse et le titre de la page,
// les boutons de la note) ; rubrique : celle à ouvrir d'abord (sinon celle que le menu allume) ; signal : l'AbortSignal
// du montage de l'écran (js/app.js) — à la fermeture, ses écouteurs tombent, et une réponse arrivée après n'est plus
// posée (le corps est déjà celui de l'écran suivant) ; une requête refusée parce que l'écran est fermé (EcranFerme,
// js/moteur.js) se tait
export function cablerBibliotheque({ racine, demander, apres = () => {}, rubrique = null, signal = null }) {
  const $ = (s) => racine.querySelector(s);
  const corps = $("#bib-corps"), pourquoi = $("#bib-pourquoi");
  const page = racine === document;           // la page bloque le défilement derrière le volet ; le panneau est déjà modal
  let courante = rubrique || $(".bib-m.on")?.dataset.rubrique || $(".bib-m")?.dataset.rubrique;
  const params = {};        // les paramètres de chaque rubrique visitée (gardés quand on en change)
  let derniere = null;      // la dernière réponse du moteur (sa fiche, ses cibles)
  let affichee = null;      // la rubrique dont le tableau est à l'écran
  let enCours = 0;

  function occupe(plus) {
    enCours += plus;
    corps.setAttribute("aria-busy", enCours > 0 ? "true" : "false");
  }

  async function consulter() {
    // (sans rubrique courante — le panneau d'une note qui vient d'oublier —, le moteur choisit celle qu'il ouvre d'abord)
    const p = courante ? (params[courante] || (params[courante] = {})) : {};
    occupe(1);
    try {
      let r;
      try { r = await demander(courante, p); } catch (e) { if (e && e.name === "EcranFerme") return null; throw e; }
      if (signal && signal.aborted) return null;
      if (!r || !r.ok) {
        toast(`Le moteur a refusé : ${r ? r.erreur : "pas de réponse"}`, true);
        if (r && r.trace) console.error(r.trace);
        return null;
      }
      appliquer(r);
      apres(r);
      return r;
    } finally { occupe(-1); }
  }

  function appliquer(r) {
    // dans la même rubrique, le tableau garde son défilement (un grand tableau — les profilés — ne remonte pas en haut
    // à chaque clic) ; la case allumée est ensuite ramenée à l'écran si elle en est sortie
    const cadre = $("#bib-tableau");
    const defile = r.rubrique === affichee ? [cadre.scrollTop, cadre.scrollLeft] : null;
    courante = affichee = r.rubrique;
    params[courante] = r.params;                           // les paramètres relus par le moteur (défauts compris)
    derniere = r;
    racine.querySelectorAll(".bib-m").forEach((b) => {
      const on = b.dataset.rubrique === courante;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const titre = $("#bib-titre");
    if (titre) titre.textContent = r.titre;
    // un menu déroulant de la barre (« Colonnes » de l'acier : <details data-garde>) ouvert le reste quand la barre est
    // réécrite — on y coche plusieurs choses de suite
    const ouverts = [...racine.querySelectorAll("#bib-barre details[data-garde][open]")].map((d) => d.dataset.garde);
    // le champ en cours de saisie garde le FOCUS quand la barre et le cadre sont réécrits (on passe d'un champ au suivant
    // avec Tab : la réponse du champ quitté ne doit pas faire perdre la main sur le champ atteint), et ce qu'on y a déjà
    // tapé y reste
    const actif = document.activeElement;
    const garde = actif && actif.dataset && actif.dataset.param && actif.closest("#bib-barre, #bib-tableau") && racine.contains(actif)
      ? { param: actif.dataset.param, zone: actif.closest("#bib-barre") ? "#bib-barre" : "#bib-tableau",
        valeur: actif.value !== actif.defaultValue ? actif.value : null, debut: actif.selectionStart, fin: actif.selectionEnd } : null;
    $("#bib-barre").innerHTML = r.barre;
    for (const g of ouverts) { const d = racine.querySelector(`#bib-barre details[data-garde="${CSS.escape(g)}"]`); if (d) d.open = true; }
    cadre.innerHTML = r.tableau;
    if (defile) [cadre.scrollTop, cadre.scrollLeft] = defile;
    $("#bib-bande").innerHTML = r.bande;
    $("#bib-sources").innerHTML = r.sources;
    $("#bib-pq-corps").innerHTML = r.pourquoi;
    if (garde) {
      const el = racine.querySelector(`${garde.zone} [data-param="${CSS.escape(garde.param)}"]`);
      if (el && typeof el.focus === "function" && el.tagName !== "DIV") {
        el.focus({ preventScroll: true });
        if (garde.valeur !== null && el.tagName === "INPUT" && el.type !== "range" && el.type !== "checkbox") {
          el.value = garde.valeur;
          try { el.setSelectionRange(garde.debut, garde.fin); } catch (e) { /* un champ sans sélection */ }
        } else if (typeof el.select === "function" && el.type !== "range") el.select();
      }
    }
    if (r.message) toast(r.message, true);
    // la case allumée se voit (le tableau défile dans son cadre au téléphone)
    const cur = $("#bib-tableau .cur");
    if (cur) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function regler(cle, valeur) {
    (params[courante] || (params[courante] = {}))[cle] = valeur;
    return consulter();
  }

  function ouvrirPourquoi(on) {
    pourquoi.hidden = !on;
    if (page) document.documentElement.classList.toggle("volet-ouvert", on);
    if (on) $("#bib-pourquoi [data-fermer-pourquoi]")?.focus();
  }

  // ------------------------------------------------------------------ les gestes
  // ------------------------------------------------------------------ glisser un élément du dessin
  // Un élément du dessin qui porte ``data-glisser`` se déplace au doigt ou à la souris. Le MOTEUR écrit dans cet attribut ce
  // que le geste règle : {x: {param: valeur de départ…}, y: {…}, ech (unités du dessin par unité du paramètre), pas, min,
  // max, sy (−1 si l'axe y du paramètre monte), unite}. Pendant le geste l'élément suit (transform) et une étiquette dit la
  // valeur ; au lâcher, les paramètres sont réglés et la rubrique relue. Ce fichier ne sait rien de ce qui est dessiné.
  let glisse = null, avalerClic = false;
  function deplacement(ev) {
    const { d, kx, ky } = glisse;
    const pas = d.pas || 0.01;
    const borne = (vals, delta) => {
      if (!vals.length) return 0;
      if (d.min !== null && d.min !== undefined) delta = Math.max(delta, d.min - Math.min(...vals));
      if (d.max !== null && d.max !== undefined) delta = Math.min(delta, d.max - Math.max(...vals));
      return delta;
    };
    const dx = borne(Object.values(d.x || {}), Math.round(((ev.clientX - glisse.x0) * kx / d.ech) / pas) * pas);
    const dy = borne(Object.values(d.y || {}), Math.round(((ev.clientY - glisse.y0) * ky / d.ech) * (d.sy || 1) / pas) * pas);
    return { dx: d.x ? dx : 0, dy: d.y ? dy : 0 };
  }
  const arrondi = (v) => Math.round(v * 1e6) / 1e6;
  racine.addEventListener("pointerdown", (ev) => {
    const g = ev.target.closest && ev.target.closest("#bib-tableau [data-glisser]");
    if (!g || ev.button > 0) return;
    let d = null;
    try { d = JSON.parse(g.dataset.glisser); } catch (e) { d = null; }
    const svg = g.ownerSVGElement;
    const m = svg && svg.getScreenCTM();
    if (!d || !m || !d.ech) return;
    glisse = { g, d, svg, x0: ev.clientX, y0: ev.clientY, kx: 1 / m.a, ky: 1 / m.d, id: ev.pointerId, bouge: false, etiquette: null, base: g.getAttribute("transform") || "" };
    try { g.setPointerCapture(ev.pointerId); } catch (e) { /* pointeur déjà relâché */ }
    ev.preventDefault();
  }, { signal });
  racine.addEventListener("pointermove", (ev) => {
    if (!glisse || ev.pointerId !== glisse.id) return;
    const { dx, dy } = deplacement(ev);
    if (!glisse.bouge && dx === 0 && dy === 0) return;
    const { g, d, svg } = glisse;
    if (!glisse.bouge) {
      glisse.bouge = true;
      g.classList.add("glisse");
      glisse.etiquette = document.createElementNS("http://www.w3.org/2000/svg", "text");
      glisse.etiquette.setAttribute("class", "glis-t");
      svg.appendChild(glisse.etiquette);
    }
    g.setAttribute("transform", `${glisse.base} translate(${dx * d.ech} ${dy * d.ech * (d.sy || 1)})`.trim());
    const fr = (v) => String(arrondi(v)).replace(".", ",");
    const txt = [...Object.values(d.x || {}).map((v) => fr(v + dx)), ...Object.values(d.y || {}).map((v) => fr(v + dy))].join(" ; ") + (d.unite ? ` ${d.unite}` : "");
    const p = svg.createSVGPoint();
    p.x = ev.clientX; p.y = ev.clientY;
    const q = p.matrixTransform(svg.getScreenCTM().inverse());
    glisse.etiquette.textContent = txt;
    glisse.etiquette.setAttribute("x", q.x);
    glisse.etiquette.setAttribute("y", q.y - 14);
  }, { signal });
  function finGlisser(ev, annule) {
    if (!glisse || ev.pointerId !== glisse.id) return;
    const { g, d, bouge, etiquette, base } = glisse;
    const { dx, dy } = annule ? { dx: 0, dy: 0 } : deplacement(ev);
    glisse = null;
    g.classList.remove("glisse");
    if (etiquette) etiquette.remove();
    if (!bouge) return;
    avalerClic = true;
    setTimeout(() => { avalerClic = false; }, 0);
    if (dx === 0 && dy === 0) { if (base) g.setAttribute("transform", base); else g.removeAttribute("transform"); return; }
    const p = params[courante] || (params[courante] = {});
    for (const [k, v] of Object.entries(d.x || {})) p[k] = arrondi(v + dx);
    for (const [k, v] of Object.entries(d.y || {})) p[k] = arrondi(v + dy);
    if (d.sel && typeof d.sel === "object") Object.assign(p, d.sel);
    consulter();
  }
  racine.addEventListener("pointerup", (ev) => finGlisser(ev, false), { signal });
  racine.addEventListener("pointercancel", (ev) => finGlisser(ev, true), { signal });

  racine.addEventListener("click", (ev) => {
    if (avalerClic) return;                    // le clic qui clôt un glisser n'est pas un clic
    if (ev.target.closest("[data-imprimer]")) { window.print(); return; }
    // un clic hors d'un menu déroulant de la barre le referme
    racine.querySelectorAll("#bib-barre details[data-garde][open]").forEach((d) => { if (!d.contains(ev.target)) d.open = false; });
    const m = ev.target.closest(".bib-m[data-rubrique]");
    if (m) { courante = m.dataset.rubrique; consulter(); return; }
    // un bouton qui porte une valeur (segments, pastilles) : le paramètre est celui du bloc qui le contient
    const seg = ev.target.closest(".bib-barre [data-param] button[data-valeur], #bib-tableau [data-param] button[data-valeur]");
    if (seg) { regler(seg.closest("[data-param]").dataset.param, seg.dataset.valeur); return; }
    // ce qui porte des paramètres à régler (data-sel) : une case du tableau, son en-tête (le tri de l'acier), sa ligne
    // (un profilé), un bouton de la bande (« Ouvrir » le plus léger)
    // … et, dans le cadre d'un outil (Bibliothèque Autres), un bouton ou une pièce du dessin
    const td = ev.target.closest("#bib-tableau [data-sel], #bib-bande [data-sel]");
    if (td) {
      let sel = null;
      try { sel = JSON.parse(td.dataset.sel); } catch (e) { sel = null; }
      if (sel && typeof sel === "object") { Object.assign(params[courante] || (params[courante] = {}), sel); consulter(); }
      return;
    }
    if (ev.target.closest("[data-pourquoi]")) { ouvrirPourquoi(true); return; }
    if (ev.target.closest("[data-fermer-pourquoi]") || ev.target === pourquoi) ouvrirPourquoi(false);
  }, { signal });
  racine.addEventListener("change", (ev) => {
    // (un champ du CADRE règle son paramètre comme un champ de la barre : les charges d'une poutre, les éléments d'une section)
    const el = ev.target.closest(".bib-barre [data-param], #bib-tableau [data-param]");
    if (!el) return;
    if (el.type === "checkbox") regler(el.dataset.param, el.checked ? "1" : "0");
    else regler(el.dataset.param, el.value);
  }, { signal });
  racine.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target.matches(".bib-barre input[data-param], #bib-tableau input[data-param]")) { ev.target.blur(); return; }
    // une ligne ou un bouton du cadre qui se clique se valide aussi au clavier
    if ((ev.key === "Enter" || ev.key === " ") && ev.target.matches("#bib-tableau tr[data-sel][tabindex]")) { ev.preventDefault(); ev.target.click(); return; }
    // Échap referme « Pourquoi ? » d'abord ; l'hôte ne voit pas cet Échap-là (il refermerait son panneau en même temps)
    if (ev.key === "Escape" && !pourquoi.hidden) { ev.stopPropagation(); ouvrirPourquoi(false); }
  }, { signal });
  // un champ reçoit le focus : son cadre le montre (comme les champs de l'établi)
  racine.addEventListener("focusin", (ev) => { const f = ev.target.closest(".bib-barre .field, #bib-tableau .field"); if (f) f.classList.add("focus"); }, { signal });
  racine.addEventListener("focusout", (ev) => { const f = ev.target.closest(".bib-barre .field, #bib-tableau .field"); if (f) f.classList.remove("focus"); }, { signal });

  return {
    consulter,
    // ouvrir une rubrique avec des paramètres donnés (l'adresse de la page) ; sans rubrique, la courante
    ouvrir(k, p) { if (k) courante = k; if (p) params[courante] = { ...p }; return consulter(); },
    // oublier les paramètres visités ET la rubrique : la prochaine consultation repart de ce que l'hôte décide (le
    // préremplissage d'une note, et la rubrique que son module ouvre d'abord)
    oublier() { for (const k of Object.keys(params)) delete params[k]; derniere = null; courante = null; },
    rubrique: () => courante,
    derniere: () => derniere,
    pourquoi: ouvrirPourquoi,
    pourquoiOuvert: () => !pourquoi.hidden,
  };
}
