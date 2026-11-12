// L'établi COMPACT des trois modules béton armé — poutres, dalles, prédalles (css/etabli.css, site/web/construire.py
// compact() ; leurs fragments : api_web/etabli_compact.py, api_web/plaque_compact.py).
//
// Trois colonnes : les DONNÉES à gauche (l'arborescence de la note : chaque élément, et sous l'élément ouvert ses
// rubriques), le DESSIN au centre, les RÉSULTATS à droite. Ce fichier ne calcule rien et ne recopie rien du moteur : il
// relie ce que le moteur a rendu.
//   - la SÉLECTION : une pièce du dessin porte data-sel (« s0.lit-inf-1 », « s0.inf_y-0 »), la ligne des données qui la
//     règle porte data-cible (la même clé ; « beton » pour le béton), une observation des résultats data-piece. Un clic
//     sur l'une choisit les autres : la rubrique s'ouvre, la ligne se déplie et s'éclaire, la pièce s'éclaire sur le
//     dessin. Échap ou un clic dans le vide la relâche.
//   - la SECTION COURANTE : celle qu'on règle à gauche (.et-arm[data-section]) est celle du dessin (ses vues :
//     « s<i> » pour une poutre, « c<i> » et « p<i> » — coupes, plan — pour une dalle) et celle des résultats
//     (.et-bloc[data-section] : les autres blocs se cachent ; la vue « Toutes » les montre tous).
//   - les CADRES s'éditent au dessin : les poignées du cadre choisi glissent d'une barre à l'autre du lit 1 inférieur
//     (aimantées : jamais entre deux barres), et dans les données (les barres entourées, un clic par bout).
//   - la LISTE DE CONTRÔLE des dispositions : une observation « à examiner » se marque examinée, avec une justification
//     facultative, gardée dans la note ; le moteur refuse de le faire pour une non-conformité.
//   - les COLONNES se redimensionnent (poignées .et-poignee, PC), la largeur est gardée dans ce navigateur.
import { toast } from "./commun.js";

const memoire = {
  lire(k, d) { try { return JSON.parse(sessionStorage.getItem(k) || "null") ?? d; } catch (e) { return d; } },
  ecrire(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } },
};
const reglages = {
  lire() { try { return JSON.parse(localStorage.getItem("stab.etabli.v1") || "{}") || {}; } catch (e) { return {}; } },
  ecrire(v) { try { localStorage.setItem("stab.etabli.v1", JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } },
};

// le seuil au-delà duquel un appui sur le dessin est un glisser (on déplace la vue) et non un choix ; plus large au doigt
// (une source : js/app.js le lit aussi — au doigt, un appui qui bouge de plus que SEUIL.touch fait défiler l'accueil, il
// ne préchauffe rien)
export const SEUIL = Object.freeze({ mouse: 5, pen: 8, touch: 10 });
// les largeurs permises des colonnes (px), et celles d'origine (la feuille de style)
const LARGEURS = { g: [264, 560], d: [280, 560] };
const JUSTIF_MAX = 600;   // la longueur d'une justification (beton/poutre/entree.py, JUSTIF_MAX)

// ``signal`` : l'AbortSignal du montage de l'écran (js/app.js) — à la fermeture, les écouteurs posés sur le document, la
// fenêtre et la requête média tombent, la minuterie de la barre du zoom s'arrête
export function cablerCompact({ cle, note, ouvrirRubrique, vue, envoyer, signal = null }) {
  const fini = () => !!signal && signal.aborted;
  const image = (f) => requestAnimationFrame(() => { if (!fini()) f(); });
  const donnees = document.getElementById("donnees");
  const resultat = document.getElementById("resultat");
  const stage = document.getElementById("stage");
  const bench = document.querySelector(".bench");
  // (colonne compacte) la ligne de l'élément ouvert, posée par js/etabli.js dans l'en-tête du panneau des données : son
  // menu (la liste des éléments) s'ouvre et se referme comme ceux des données
  const elt = document.querySelector(".panel.left .phead .et-elt");
  // les menus ⋯ des données et de la ligne de l'élément, et le menu « Exporter » du pied des résultats
  const zonesMenus = [donnees, elt, resultat].filter(Boolean);
  const memo = () => `stab.compact.${cle}.${note().id}`;
  const lu = memoire.lire(memo(), {});
  let sel = lu.sel || "";
  let courante = +lu.courante || 0;
  // l'état de l'ÉCRAN (pas de la note) : la poutre ouverte repliée, les blocs de résultats repliés, les listes de
  // contrôle dépliées — gardés d'un calcul à l'autre, et pour la session
  let poutreRepliee = !!lu.poutreRepliee;
  const replies = new Set(lu.replies || []);
  const listes = new Set(lu.listes || []);
  const garder = () => memoire.ecrire(memo(), { sel, courante, poutreRepliee, replies: [...replies], listes: [...listes] });

  // ------------------------------------------------------------ la poutre ouverte : repliée ou dépliée
  function poserPoutre() {
    const p = donnees.querySelector(".et-pt.on");
    if (!p) return;
    p.classList.toggle("replie", poutreRepliee);
    const t = p.querySelector(".et-pt-t");
    if (t) t.setAttribute("aria-expanded", poutreRepliee ? "false" : "true");
  }
  function deplierPoutre() { if (poutreRepliee) { poutreRepliee = false; garder(); poserPoutre(); } }

  // ------------------------------------------------------------ la section courante (données, dessin, résultats)
  function poserCourante() {
    const blocs = [...donnees.querySelectorAll(".et-arm[data-section]")];
    const n = blocs.length || resultat.querySelectorAll(".et-bloc[data-section]").length;
    if (n && courante >= n) courante = 0;
    blocs.forEach((b) => b.classList.toggle("on", +b.dataset.section === courante));
    donnees.querySelectorAll("[data-section-courante]").forEach((b) => {
      const on = +b.dataset.sectionCourante === courante;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    donnees.querySelectorAll(".et-sec[data-section]").forEach((c) => c.classList.toggle("et-courante", +c.dataset.section === courante));
    donnees.querySelectorAll(".et-r3[data-section]").forEach((c) => c.classList.toggle("on", +c.dataset.section === courante));
    // à droite : le bloc de la section courante seul ; la vue « Toutes » les montre tous
    const toutes = vue.courante() === "toutes";
    resultat.querySelectorAll(".et-bloc[data-section]").forEach((b) => {
      const i = +b.dataset.section;
      b.hidden = !toutes && i !== courante;
      b.classList.toggle("replie", replies.has(i));
      const t = b.querySelector(".et-bloc-t");
      if (t) t.setAttribute("aria-expanded", replies.has(i) ? "false" : "true");
    });
    // un bloc THÉMATIQUE (data-theme : les cornières — Sections, Ancrages, Comparaison ; etabli_compact.bloc(theme=True))
    // n'est pas celui d'une section : toujours visible, il se replie comme les autres
    resultat.querySelectorAll(".et-bloc[data-theme]").forEach((b) => {
      const i = +b.dataset.theme;
      b.classList.toggle("replie", replies.has(i));
      const t = b.querySelector(".et-bloc-t");
      if (t) t.setAttribute("aria-expanded", replies.has(i) ? "false" : "true");
    });
    resultat.querySelectorAll(".et-dispo[data-dispo]").forEach((d) => {
      const on = listes.has(+d.dataset.dispo);
      d.classList.toggle("ouvert", on);
      d.setAttribute("aria-expanded", on ? "true" : "false");
      const l = resultat.querySelector(`.et-obs-l[data-dispo-l="${d.dataset.dispo}"]`);
      if (l) l.classList.toggle("ouvert", on);
    });
  }
  // la section d'une vue : « s1 » (une poutre), « c1 » ou « p1 » (les coupes, le plan d'une dalle) → 1 ; « toutes » → null
  const indiceVue = (k) => { const m = /^[a-z]+(\d+)$/.exec(k || ""); return m ? +m[1] : null; };
  // la vue d'une section : du même GENRE que la vue ouverte (des coupes aux coupes, du plan au plan), sinon sa première
  function vueSection(i) {
    const cles = [...document.querySelectorAll("#vues [data-vue]")].map((b) => b.dataset.vue).filter((k) => indiceVue(k) === i);
    const genre = (/^([a-z]+)\d+$/.exec(vue.courante() || "") || [])[1];
    return cles.find((k) => k.replace(/\d+$/, "") === genre) || cles[0] || `s${i}`;
  }
  // la section choisie à gauche est celle du dessin et des résultats (même depuis la vue « Toutes »)
  function choisirSection(i) {
    courante = i;
    garder();
    vue.choisir(vueSection(i));
    poserCourante();
    surligner();
  }

  // ------------------------------------------------------------ la sélection (données ↔ dessin ↔ résultats)
  const indice = (k) => { const m = /^s(\d+)\./.exec(k || ""); return m ? +m[1] : null; };
  const cibleDonnees = (k) => {
    if (!k) return null;
    if (/\.beton$/.test(k)) return donnees.querySelector('[data-cible="beton"]');
    return donnees.querySelector(`[data-cible="${CSS.escape(k)}"]`);
  };
  function surligner() {
    stage.querySelectorAll(".pc[data-sel]").forEach((g) => {
      const k = g.dataset.sel;
      const on = !!sel && (k === sel || (sel === "beton" && /\.beton$/.test(k)));
      g.classList.toggle("et-sel", on);
      // le cadre choisi passe au-dessus des autres (ses poignées ne sont pas sous un cadre voisin)
      if (on && g.parentNode && g.parentNode.classList.contains("couche-cadres") && g.parentNode.lastElementChild !== g) g.parentNode.appendChild(g);
    });
    donnees.querySelectorAll("[data-cible]").forEach((e) => e.classList.toggle("et-cible", !!sel && e.dataset.cible === (/\.beton$/.test(sel) ? "beton" : sel)));
    resultat.querySelectorAll(".et-obs[data-piece]").forEach((e) => e.classList.toggle("et-cible", !!sel && e.dataset.piece === sel));
  }
  // depuis le DESSIN (ou une observation des résultats) : la poutre se déplie, la rubrique s'ouvre, la ligne se déplie
  // et vient sous les yeux
  function choisirPiece(k) {
    sel = k || "";
    const i = indice(sel);
    if (i !== null && i !== courante) courante = i;
    garder();
    poserCourante();
    const c = cibleDonnees(sel);
    if (c) {
      deplierPoutre();
      const rub = c.closest(".sec[data-sec]");
      if (rub && !rub.classList.contains("open")) ouvrirRubrique(rub.dataset.sec);
      c.classList.remove("et-replie");
      surligner();
      image(() => c.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    } else surligner();
  }
  // depuis les DONNÉES : la pièce s'éclaire sur le dessin (rien ne défile) ; une ligne d'une autre section rend celle-ci
  // courante — la vue « Toutes », choisie exprès pour comparer, reste ouverte
  function choisirLigne(k) {
    if (k === sel) return;
    sel = k;
    const i = indice(sel);
    if (i !== null && i !== courante) {
      courante = i;
      if (vue.courante() !== "toutes") vue.choisir(vueSection(i));
      poserCourante();
    }
    garder();
    surligner();
  }
  function relacher() { if (!sel) return; sel = ""; garder(); surligner(); }

  // ------------------------------------------------------------ les menus ⋯ (une poutre, une section)
  function fermerMenus(sauf) {
    zonesMenus.forEach((z) => z.querySelectorAll(".et-mn-l:not([hidden])").forEach((l) => {
      if (l === sauf) return;
      l.hidden = true;
      const b = l.parentElement.querySelector("[data-menu]");
      if (b) b.setAttribute("aria-expanded", "false");
    }));
  }
  const menuOuvert = () => zonesMenus.some((z) => z.querySelector(".et-mn-l:not([hidden])"));
  function basculerMenu(b) {
    const l = b.parentElement.querySelector(".et-mn-l");
    if (!l) return;
    fermerMenus(l);
    l.hidden = !l.hidden;
    b.setAttribute("aria-expanded", l.hidden ? "false" : "true");
    if (l.hidden) return;
    // Le menu reste DANS SA COLONNE (le panneau le coupe) — mesuré par le harnais « exporter » (12/11/2026 : le menu
    // « Exporter », plus large que la colonne des résultats, y perdait ses icônes au PC ; en tablette portrait, la colonne
    // trop courte le coupait en bas). En largeur : décalé, ou rétréci s'il est plus large qu'elle. En hauteur : vers le
    // bas ; trop près du bas de la colonne (ou de la barre du téléphone), du côté où il tient le mieux ; s'il ne tient
    // d'aucun côté, il défile.
    l.classList.remove("haut");
    for (const p of ["left", "right", "width", "minWidth", "maxHeight"]) l.style[p] = "";
    const zone = resultat && resultat.contains(b) ? resultat : donnees;
    const z = zone.getBoundingClientRect(), barre = document.querySelector(".mbar");
    const g = Math.max(z.left, 0) + 6, d = Math.min(z.right, innerWidth) - 6;
    let r = l.getBoundingClientRect();
    if (r.width > d - g) { l.style.minWidth = "0px"; l.style.width = `${Math.floor(d - g)}px`; r = l.getBoundingClientRect(); }
    if (r.left < g - 0.5 || r.right > d + 0.5) {
      const m = (l.offsetParent || l.parentElement).getBoundingClientRect();
      l.style.left = `${Math.round(Math.min(Math.max(r.left, g), d - r.width) - m.left)}px`;
      l.style.right = "auto";
      r = l.getBoundingClientRect();
    }
    const bas = Math.min(z.bottom, innerHeight - (barre && barre.offsetHeight ? barre.offsetHeight : 0)) - 6;
    const tb = b.getBoundingClientRect(), dessous = bas - tb.bottom - 2, dessus = tb.top - (Math.max(z.top, 0) + 6) - 2;
    if (r.height > dessous) {
      if (dessus > dessous) l.classList.add("haut");
      const place = Math.max(dessus, dessous);
      if (r.height > place) l.style.maxHeight = `${Math.max(Math.floor(place), 96)}px`;
    }
    const premier = l.querySelector("button:not(:disabled)");
    if (premier) premier.focus({ preventScroll: true });
  }
  document.addEventListener("click", (ev) => { if (!ev.target.closest(".et-mn")) fermerMenus(); }, { signal });
  // le menu « Exporter » des résultats : il s'ouvre et se ferme ici ; un choix le referme, l'action suit (js/etabli.js :
  // un fichier à télécharger, une fenêtre de calcul)
  if (resultat) resultat.addEventListener("click", (ev) => {
    const mb = ev.target.closest("[data-menu]");
    if (mb) { ev.stopImmediatePropagation(); basculerMenu(mb); return; }
    if (ev.target.closest(".et-mn-l")) fermerMenus();
  }, { signal });
  // … et un fichier à télécharger le referme aussi : js/etabli.js fait ce téléchargement à la CAPTURE du clic et l'arrête
  // là — l'écoute ci-dessus ne l'entend jamais (le harnais l'a trouvé : le menu restait ouvert). Posée sur le document, à
  // la capture, celle-ci passe avant lui ; elle ne fait que refermer, l'action suit.
  document.addEventListener("click", (ev) => {
    if (resultat && resultat.contains(ev.target) && ev.target.closest(".et-mn-l button")) fermerMenus();
  }, { capture: true, signal });
  // la ligne de l'élément (en-tête du panneau) : son menu, la liste des éléments ; ouvrir un autre élément lâche la
  // pièce choisie (écouté AVANT js/etabli.js, qui fait l'action : ouvrir, ajouter, copier, supprimer)
  if (elt) elt.addEventListener("click", (ev) => {
    const mb = ev.target.closest("[data-menu]");
    if (mb) { ev.stopImmediatePropagation(); basculerMenu(mb); return; }
    if (ev.target.closest(".et-mn-l")) fermerMenus();
    if (ev.target.closest("[data-activer]:not(.on)")) { sel = ""; garder(); }
  }, { signal });

  // ------------------------------------------------------------ les barres qu'un cadre entoure (données)
  // Un clic HORS de la plage l'élargit jusqu'à la barre ; un clic sur un BOUT le retire (deux barres au moins) ; un clic
  // à l'INTÉRIEUR y amène le bout le plus proche. Une épingle : la barre touchée. Le moteur relit et refuse l'impossible.
  function choisirBarre(b) {
    const z = b.closest(".et-barres");
    const k = +b.dataset.barre;
    let de = +z.dataset.de, a = +z.dataset.a;
    if (z.dataset.type === "epingle") { if (k !== de) poserPlage(z.dataset.plage, `${k}`); return; }
    if (k < de) de = k;
    else if (k > a) a = k;
    else if (k === de || k === a) {
      if (a - de < 2) { toast("Un cadre entoure au moins deux barres (une seule : une épingle)", true); return; }
      if (k === de) de += 1; else a -= 1;
    } else if (k - de <= a - k) de = k;
    else a = k;
    poserPlage(z.dataset.plage, `${de}-${a}`);
  }
  const poserPlage = (chemin, valeur) => envoyer("modifier", { chemin: chemin.split("/"), valeur });

  // la sélection AVANT l'appui : un clic sur un bouton le focalise, et le « focusin » ci-dessous choisit sa ligne avant
  // que le « click » n'arrive — sans ce relevé, le chevron d'une ligne non choisie la choisissait PUIS la repliait
  let avantAppui = "";
  donnees.addEventListener("pointerdown", () => { avantAppui = sel; }, { capture: true, signal });
  // (écouté AVANT celui de js/etabli.js, qui agit sur les mêmes clics : ce qui est traité ici s'arrête ici)
  donnees.addEventListener("click", (ev) => {
    const mb = ev.target.closest("[data-menu]");
    if (mb) { ev.stopImmediatePropagation(); basculerMenu(mb); return; }
    if (ev.target.closest(".et-mn-l")) fermerMenus();   // un choix du menu : il se referme, l'action suit (etabli.js)
    // le titre de la poutre OUVERTE la replie ou la déplie ; celui d'une autre l'ouvre (etabli.js, « activer »), dépliée
    const pt = ev.target.closest(".et-pt-t[data-activer]");
    if (pt && pt.classList.contains("on")) { ev.stopImmediatePropagation(); poutreRepliee = !poutreRepliee; garder(); poserPoutre(); return; }
    if (pt) { poutreRepliee = false; sel = ""; garder(); return; }
    const sc = ev.target.closest("[data-section-courante]");
    if (sc) { ev.stopImmediatePropagation(); choisirSection(+sc.dataset.sectionCourante); return; }
    const barre = ev.target.closest(".et-barres [data-barre]");
    if (barre) { ev.stopImmediatePropagation(); choisirBarre(barre); return; }
    const large = ev.target.closest("[data-plage-tout]");
    if (large) {
      ev.stopImmediatePropagation();
      const z = large.parentElement.querySelector(".et-barres");
      if (z && !large.classList.contains("on")) poserPlage(z.dataset.plage, "");
      return;
    }
    const dep = ev.target.closest("[data-deplier]");
    if (dep) {
      const p = dep.closest(".et-piece");
      if (p) {
        // le chevron d'une ligne DÉJÀ choisie la replie ou la déplie ; sinon, elle devient la sélection, dépliée
        const dejaChoisie = p.dataset.cible === avantAppui && p.classList.contains("et-cible");
        if (dejaChoisie) { p.classList.toggle("et-replie"); avantAppui = sel; return; }
        p.classList.remove("et-replie");
        choisirLigne(p.dataset.cible);
        avantAppui = sel;
      }
      return;
    }
    // un clic sur la carte d'une section (hors de ses champs et boutons) la rend courante
    const carte = ev.target.closest(".et-sec[data-section]");
    if (carte && !ev.target.closest("button, input, select, textarea")) choisirSection(+carte.dataset.section);
  }, { signal });
  donnees.addEventListener("focusin", (ev) => {
    const p = ev.target.closest("[data-cible]");
    if (p) choisirLigne(p.dataset.cible);
  }, { signal });

  // ------------------------------------------------------------ le dessin : glisser une poignée de cadre
  // L'appui sur une poignée du cadre choisi est à nous seul (ni déplacement de la vue, ni choix d'une pièce) : la
  // poignée suit le pointeur d'une barre à l'autre du lit 1 inférieur, la barre visée s'éclaire ; au relâcher, le
  // moteur reçoit les barres entourées (« de-à »). Un cadre garde deux barres au moins ; une épingle en a une.
  let glisse = null;
  const lireTranslation = (e) => { const m = /translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)/.exec(e.getAttribute("transform") || ""); return m ? [+m[1], +m[2]] : [0, 0]; };
  stage.addEventListener("pointerdown", (ev) => {
    const h = ev.target.closest && ev.target.closest(".view.on .pc.cadre.et-sel .poignee");
    if (!h || (ev.pointerType === "mouse" && ev.button !== 0)) return;
    const g = h.closest(".pc.cadre"), coupe = h.closest(".coupe");
    const xs = (coupe.dataset.barres || "").split(",").map(Number).filter(Number.isFinite);
    const bord = coupe.querySelector(".pc.beton rect.bord");
    if (!xs.length || !bord) return;
    ev.stopImmediatePropagation();
    ev.preventDefault();
    const [x0, y0] = lireTranslation(h);
    glisse = { h, g, coupe, xs, bord, x0, y0, id: ev.pointerId, k: null };
    try { h.setPointerCapture(ev.pointerId); } catch (e) { /* pointeur déjà relâché */ }
    g.classList.add("et-glisse");
  }, { capture: true, signal });
  // AU DOIGT, le navigateur prend un appui qui bouge pour faire défiler la page (la feuille du dessin l'autorise :
  // touch-action « pan-x pan-y » ; celui d'une poignée, élément SVG, n'est pas appliqué) et ANNULE le glisser
  // (pointercancel : la poignée revenait à sa place). Un appui qui commence sur une poignée du cadre choisi lui est
  // refusé ; ailleurs sur le dessin, le doigt fait défiler comme avant. (Écouteur non passif : posé sur la toile, pas
  // sur le document, que le navigateur rend passif.)
  stage.addEventListener("touchstart", (ev) => {
    if (ev.target.closest && ev.target.closest(".view.on .pc.cadre.et-sel .poignee")) ev.preventDefault();
  }, { passive: false, signal });
  window.addEventListener("pointermove", (ev) => {
    if (!glisse || ev.pointerId !== glisse.id) return;
    ev.preventDefault();
    const { h, g, coupe, xs, bord } = glisse;
    const r = bord.getBoundingClientRect();
    const x = (ev.clientX - r.left) / r.width * +coupe.dataset.b;   // en mm, dans le repère de la coupe
    let k = 0;
    xs.forEach((xb, j) => { if (Math.abs(xb - x) < Math.abs(xs[k] - x)) k = j; });
    // jamais un cadre sur une seule barre : le bout gauche reste à gauche du droit, et inversement
    const de = +g.dataset.de, a = +g.dataset.a;
    if (g.dataset.type !== "epingle") k = h.dataset.bout === "de" ? Math.min(k, a - 2) : Math.max(k, de);
    if (k === glisse.k) return;
    glisse.k = k;
    h.setAttribute("transform", `translate(${xs[k]} ${glisse.y0})`);
    coupe.querySelectorAll(".et-visee").forEach((c) => c.classList.remove("et-visee"));
    coupe.querySelectorAll('.pc.lit[data-sel$=".lit-inf-0"] > circle').forEach((c) => { if (Math.abs(+c.getAttribute("cx") - xs[k]) < 0.6) c.classList.add("et-visee"); });
  }, { passive: false, signal });
  const finirGlisse = (ev, annule) => {
    if (!glisse || ev.pointerId !== glisse.id) return;
    const { h, g, coupe, x0, y0, k } = glisse;
    glisse = null;
    g.classList.remove("et-glisse");
    coupe.querySelectorAll(".et-visee").forEach((c) => c.classList.remove("et-visee"));
    if (annule || k === null) { h.setAttribute("transform", `translate(${x0} ${y0})`); return; }
    const b = k + 1, de = +g.dataset.de, a = +g.dataset.a;
    let v = null;
    if (g.dataset.type === "epingle") { if (b !== de) v = `${b}`; }
    else if (h.dataset.bout === "de") { if (b !== de) v = `${b}-${a}`; }
    else if (b !== a) v = `${de}-${b}`;
    if (!v) { h.setAttribute("transform", `translate(${x0} ${y0})`); return; }
    envoyer("modifier", { chemin: ["sections", coupe.dataset.section, "etriers", g.dataset.cadre, "plage"], valeur: v });
  };
  window.addEventListener("pointerup", (ev) => finirGlisse(ev, false), { capture: true, signal });
  window.addEventListener("pointercancel", (ev) => finirGlisse(ev, true), { capture: true, signal });

  // ------------------------------------------------------------ le dessin : choisir une pièce
  // On lit l'APPUI (pointerdown) : la toile capture ensuite le pointeur pour déplacer la vue, et le « click » tomberait
  // sur le fond. Au relâcher, un pointeur qui a peu bougé choisit la pièce appuyée (ou relâche la sélection, dans le
  // vide) ; au-delà du seuil, c'était un glisser.
  let appui = null;
  stage.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest || !ev.target.closest(".draw")) { appui = null; return; }
    const g = ev.target.closest(".draw [data-sel]");
    appui = { g, x: ev.clientX, y: ev.clientY, id: ev.pointerId, seuil: SEUIL[ev.pointerType] || 5 };
  }, { capture: true, signal });
  window.addEventListener("pointerup", (ev) => {
    const a = appui;
    appui = null;
    if (!a || ev.pointerId !== a.id) return;
    if (Math.hypot(ev.clientX - a.x, ev.clientY - a.y) > a.seuil) return;
    if (a.g) choisirPiece(a.g.dataset.sel); else relacher();
  }, { capture: true, signal });
  window.addEventListener("pointercancel", () => { appui = null; }, { capture: true, signal });
  // au doigt, la barre du zoom (discrète) se montre quelques secondes après un appui sur la toile
  let vuMinuterie = 0;
  stage.addEventListener("pointerdown", (ev) => {
    if (ev.pointerType === "mouse") return;
    bench.classList.add("et-zoom-vu");
    clearTimeout(vuMinuterie);
    vuMinuterie = setTimeout(() => bench.classList.remove("et-zoom-vu"), 4000);
  }, { capture: true, signal });
  if (signal) signal.addEventListener("abort", () => clearTimeout(vuMinuterie), { once: true });
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    if (menuOuvert()) { fermerMenus(); return; }
    if (ev.target.closest && ev.target.closest(".et-jf")) { brouillon = null; poserBrouillon(); return; }
    if (sel && !document.querySelector(".flyout.on") && !ev.target.closest("input, select, textarea")) relacher();
  }, { signal });

  // ------------------------------------------------------------ les résultats : blocs, liste de contrôle, justification
  // Le brouillon d'une justification en cours : il survit aux calculs (le cadre des résultats est refait à chaque
  // réponse du moteur) tant que l'observation reste « à examiner » avec la même empreinte.
  let brouillon = null;   // { i, id, empreinte, texte }
  function poserBrouillon(focus = false) {
    resultat.querySelectorAll(".et-jf").forEach((f) => f.remove());
    resultat.querySelectorAll(".et-obs.et-jouv").forEach((o) => o.classList.remove("et-jouv"));
    if (!brouillon) return;
    const bt = [...resultat.querySelectorAll(`.et-bloc[data-section="${brouillon.i}"] [data-justifier]`)]
      .find((b) => b.dataset.obs === brouillon.id && b.dataset.empreinte === brouillon.empreinte);
    if (!bt) { brouillon = null; return; }
    const obs = bt.closest(".et-obs");
    const f = document.createElement("div");
    f.className = "et-jf";
    f.innerHTML = '<label><span>Justification de l\'ingénieur <i>(facultative)</i></span>'
      + `<textarea rows="2" maxlength="${JUSTIF_MAX}" placeholder="ex. barres de montage Ø8 prévues au plan d'armatures"></textarea></label>`
      + '<div class="et-jf-b"><button type="button" class="btn sm primary" data-just-valider="1">Marquer examinée</button>'
      + '<button type="button" class="btn sm ghost" data-just-annuler="1">Annuler</button></div>';
    const t = f.querySelector("textarea");
    t.value = brouillon.texte;
    t.addEventListener("input", () => { if (brouillon) brouillon.texte = t.value; }, { signal });
    obs.querySelector(".ob").appendChild(f);
    obs.classList.add("et-jouv");
    if (focus) t.focus();
  }
  resultat.addEventListener("click", (ev) => {
    const bt = ev.target.closest(".et-bloc-t[data-bloc]");
    if (bt) {
      const i = +bt.dataset.bloc;
      if (replies.has(i)) replies.delete(i); else replies.add(i);
      garder(); poserCourante();
      return;
    }
    const dispo = ev.target.closest(".et-dispo[data-dispo]");
    if (dispo) {
      const i = +dispo.dataset.dispo;
      if (listes.has(i)) listes.delete(i); else listes.add(i);
      garder(); poserCourante();
      return;
    }
    const j = ev.target.closest("[data-justifier]");
    if (j) {
      brouillon = { i: +j.dataset.justifier, id: j.dataset.obs, empreinte: j.dataset.empreinte, texte: "" };
      poserBrouillon(true);
      return;
    }
    if (ev.target.closest("[data-just-annuler]")) { brouillon = null; poserBrouillon(); return; }
    if (ev.target.closest("[data-just-valider]")) {
      const b = brouillon;
      if (!b) return;
      brouillon = null;
      envoyer("modifier", { chemin: ["sections", String(b.i), "justifs", b.id], valeur: { texte: b.texte.trim().slice(0, JUSTIF_MAX), empreinte: b.empreinte } })
        .then((r) => { if (r && !r.message) toast("Observation marquée examinée"); });
      return;
    }
    const dj = ev.target.closest("[data-dejustifier]");
    if (dj) {
      envoyer("modifier", { chemin: ["sections", dj.dataset.dejustifier, "justifs", dj.dataset.obs], valeur: null })
        .then((r) => { if (r && !r.message) toast("Examen annulé : l'observation est de nouveau à examiner"); });
      return;
    }
    // une observation montre sa pièce (dessin et données), hors de ses boutons et de son formulaire
    const obs = ev.target.closest(".et-obs[data-piece]");
    if (obs && !ev.target.closest("button, textarea, .et-jf")) choisirPiece(obs.dataset.piece);
  }, { signal });

  // ------------------------------------------------------------ les colonnes : largeur réglable (PC)
  const pc = matchMedia("(min-width: 1200px)");
  function poserLargeurs() {
    const l = reglages.lire().largeurs || {};
    for (const k of ["g", "d"]) {
      const v = pc.matches && l[k] ? Math.min(LARGEURS[k][1], Math.max(LARGEURS[k][0], l[k])) : null;
      if (v) bench.style.setProperty(k === "g" ? "--pl-r" : "--pr-r", `${v}px`);
      else bench.style.removeProperty(k === "g" ? "--pl-r" : "--pr-r");
    }
  }
  poserLargeurs();
  pc.addEventListener("change", () => { poserLargeurs(); vue.ajuster(); }, { signal });
  document.querySelectorAll(".et-poignee[data-poignee]").forEach((p) => {
    const k = p.dataset.poignee;
    p.addEventListener("pointerdown", (ev) => {
      if (!pc.matches || ev.button !== 0) return;
      ev.preventDefault();
      p.setPointerCapture(ev.pointerId);
      const panneau = document.querySelector(k === "g" ? ".panel.left" : ".panel.right");
      const x0 = ev.clientX, w0 = panneau.getBoundingClientRect().width;
      bench.classList.add("et-redim");
      const bouger = (e) => {
        const w = Math.round(Math.min(LARGEURS[k][1], Math.max(LARGEURS[k][0], w0 + (k === "g" ? 1 : -1) * (e.clientX - x0))));
        bench.style.setProperty(k === "g" ? "--pl-r" : "--pr-r", `${w}px`);
      };
      const finir = () => {
        p.removeEventListener("pointermove", bouger);
        bench.classList.remove("et-redim");
        const r = reglages.lire();
        r.largeurs = r.largeurs || {};
        r.largeurs[k] = Math.round(panneau.getBoundingClientRect().width);
        reglages.ecrire(r);
        vue.ajuster();
      };
      p.addEventListener("pointermove", bouger, { signal });
      p.addEventListener("pointerup", finir, { once: true, signal });
      p.addEventListener("pointercancel", finir, { once: true, signal });
    }, { signal });
    p.addEventListener("dblclick", () => {
      const r = reglages.lire();
      if (r.largeurs) delete r.largeurs[k];
      reglages.ecrire(r);
      poserLargeurs();
      vue.ajuster();
    }, { signal });
  });

  // ------------------------------------------------------------ après chaque réponse du moteur
  // le NOM de l'élément ouvert (un champ : sur sa ligne de l'arborescence, ou dans la ligne de l'élément en tête du
  // panneau quand la colonne de gauche est compacte) a la largeur de son texte : ce qui le suit reste à côté, sans vide
  // (l'attribut size, en caractères moyens, laissait un blanc)
  const regle = document.createElement("canvas").getContext("2d");
  function ajusterNom(n) {
    if (!n) return;
    const st = getComputedStyle(n);
    regle.font = `${st.fontWeight} ${st.fontSize} ${st.fontFamily}`;
    const bords = parseFloat(st.paddingLeft) + parseFloat(st.paddingRight) + parseFloat(st.borderLeftWidth) + parseFloat(st.borderRightWidth);
    n.style.width = `${Math.ceil(regle.measureText(n.value || n.placeholder || "").width + bords + 4)}px`;
  }
  const ajusterNoms = () => { ajusterNom(donnees.querySelector(".et-pt-nom")); if (elt) ajusterNom(elt.querySelector("input.et-elt-nom")); };
  const surNom = (ev) => { if (ev.target.classList && (ev.target.classList.contains("et-pt-nom") || ev.target.classList.contains("et-elt-nom"))) ajusterNom(ev.target); };
  donnees.addEventListener("input", surNom, { signal });
  if (elt) elt.addEventListener("input", surNom, { signal });
  function rafraichir() {
    poserPoutre();
    ajusterNoms();
    // la section courante est celle du dessin (une vue « Toutes » reste choisie)
    const nSec = donnees.querySelectorAll(".et-arm[data-section]").length;
    if (nSec && courante >= nSec) courante = 0;
    // (une vue qui n'est pas celle d'une section — une fiche de la bibliothèque, « fiche-0 » — reste ouverte)
    const iv = indiceVue(vue.courante());
    if (vue.courante() !== "toutes" && iv !== null && iv !== courante) vue.choisir(vueSection(courante));
    poserCourante();
    // la pièce choisie a pu disparaître (un lit retiré) : la sélection tombe
    if (sel && !cibleDonnees(sel) && !stage.querySelector(`[data-sel="${CSS.escape(sel)}"]`)) sel = "";
    surligner();
    poserBrouillon();
  }
  // l'onglet du dessin choisit la section courante (« Toutes » la garde, et montre tous les blocs de résultats)
  function vueChoisie(k) {
    const i = indiceVue(k);
    if (i !== null) courante = i;
    garder();
    poserCourante();
    surligner();
  }

  return { rafraichir, surligner, vueChoisie, deplierPoutre, majNote() {}, courante: () => courante };
}
