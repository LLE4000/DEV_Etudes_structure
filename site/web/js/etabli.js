// L'établi d'un module (garde-corps.html, poutre.html…) : il envoie au moteur ce que l'utilisateur saisit et affiche les
// fragments qu'il reçoit (données, résultat, fenêtres de calcul, sélecteurs, dessins). Il ne calcule rien.
//
// Une seule page pour les trois formats : les fonctions sont les mêmes au PC, à la tablette et au téléphone, seule la
// disposition change (css/site.css). Au téléphone, les onglets Données · Dessin · Résultat choisissent le cadre visible,
// les actions de la barre du haut passent dans un menu, et les volets montent du bas.
import { Moteur } from "./moteur.js";
import * as N from "./notes.js";
import { cablerRail, toast, esc, telecharger, garderGrec, texteTaux } from "./commun.js";
import { cablerToile } from "./toile.js";
import { recupererMeteo } from "./meteo.js";
import { SONDE, marquer, marquerUneFois } from "./sonde.js";
import { cablerCompact } from "./compact.js";
import { cablerRangement } from "./rangement.js";

marquer("etabli_debut");   // la sonde du démarrage (js/sonde.js) : rien sans « ?sonde=1 »
const MODULE = document.body.dataset.module;
const $ = (s, r = document) => r.querySelector(s);
const bench = $(".bench");
const etapes = $("#demarrage-etape");
const telephone = matchMedia("(max-width: 700px)");
// l'établi COMPACT (pilote : « Poutres béton armé », construire.ETABLI_COMPACT) : la liste à gauche, les propriétés de
// l'étape choisie et le résultat à droite (js/compact.js) ; les autres modules gardent l'établi d'avant
const COMPACT = document.body.dataset.etabli === "compact";
const memoire = { lire(k, d) { try { return JSON.parse(sessionStorage.getItem(k) || "null") ?? d; } catch (e) { return d; } },
  ecrire(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } } };
cablerRail();

// ------------------------------------------------------------------ la note
const q = new URLSearchParams(location.search);
let note = q.get("note") && N.lire(q.get("note"));
const relue = !!note && note.module === MODULE;   // une note déjà enregistrée, rouverte
// PROVISOIRE : le calcul ouvert depuis sa carte (établi compact, sans nom ni chantier demandés). La note n'existe qu'en
// mémoire : RIEN n'est écrit tant que l'utilisateur ne modifie rien (ouvrir pour regarder ne remplit pas la liste des
// notes). La première saisie, un renommage ou « Enregistrer comme note… » l'écrivent, et l'adresse prend alors ?note=.
let provisoire = false;
if (!note || note.module !== MODULE) {
  note = N.nouvelle({ module: MODULE, nom: q.get("nom") || "", chantier: q.get("chantier") || "", dossier: q.get("dossier") || "" });
  provisoire = COMPACT && !["nom", "chantier", "dossier"].some((k) => q.get(k));
  if (!provisoire) history.replaceState(null, "", `?note=${note.id}`);
}
// les actions qui MODIFIENT la note (une note provisoire s'écrit à la première) ; « nouveau », « calculer »,
// « selecteur » ne font que la lire
const MODIFIENT = new Set(["modifier", "choisir", "ajouter", "dupliquer", "supprimer", "activer"]);
// le nom de la note : le champ de la barre du haut (PC, tablette) et celui de l'en-tête du téléphone
const noms = [...document.querySelectorAll("[data-nom-note]")];
function afficherEntete() {
  noms.forEach((n) => { if (n !== document.activeElement) n.value = note.nom || ""; });
  document.title = `${note.nom || N.MODULES[MODULE].nom} · Stab Méthodes`;
  const fil = [note.chantier, note.dossier].filter(Boolean).join(" › ");
  $("#fil-chantier").textContent = fil || "Sans chantier";
  $("#fil-m").textContent = `${fil || "Sans chantier"} · ${N.MODULES[MODULE].nom}`;
}
afficherEntete();
noms.forEach((n) => n.addEventListener("change", () => { note.nom = n.value.trim() || N.MODULES[MODULE].nom; enregistrer(); apresEntete(); }));
// renommée ou classée : l'en-tête, la liste (établi compact) et le cartouche du dessin le disent tout de suite
function apresEntete() {
  afficherEntete();
  if (compact) compact.majNote();
  document.querySelectorAll('#stage .cart [data-cart="nom"]').forEach((b) => { b.textContent = note.nom || ""; });
  document.querySelectorAll('#stage .cart [data-cart="chantier"]').forEach((b) => { b.textContent = note.chantier || "—"; });
}
noms.forEach((n) => n.addEventListener("keydown", (ev) => { if (ev.key === "Enter") n.blur(); }));

// ------------------------------------------------------------------ l'enregistrement : visible aux trois formats
// Chaque indicateur .et-enr (la barre du haut au PC et à la tablette, la 2ᵉ ligne de l'en-tête au téléphone : le même
// composant, construire.enregistrement) dit l'état de la DERNIÈRE écriture. Un échec y reste, en rouge, jusqu'à la
// prochaine écriture réussie, et ouvre une fois le message #et-alerte, qui offre « Exporter .stab ». Aucune règle de la
// feuille ne les cache : l'ancien texte (.saved .txt) l'était sous 1 360 px par la charte, et l'échec y était muet
// (verifier_site.js, filtre « enregistrement »).
// Le texte d'un indicateur : l'ÉTAT (.et-enr-etat), puis, en échec, la RAISON (.et-enr-raison, « : stockage plein »). Au
// téléphone, la 2ᵉ ligne n'écrit que l'état (la raison n'y tient pas à côté de la pastille de la note) : l'entrée du
// menu ⋯ (revoir) dit l'échec en entier et rouvre le message.
const indicateurs = [...document.querySelectorAll(".et-enr")];
const alerte = $("#et-alerte");
const revoir = $("[data-revoir-alerte]");
let alerteDite = "";   // la raison que le message a déjà dite : il ne revient pas à chaque saisie qui échoue
let alerteEchec = "";  // le texte du message du dernier échec : « revoir » le rouvre
// l'heure seule aujourd'hui, la date sinon (l'indicateur est court ; la date entière est au survol)
const heure = (t) => (new Date(t).toDateString() === new Date().toDateString()
  ? new Date(t).toLocaleTimeString("fr-BE", { hour: "2-digit", minute: "2-digit" })
  : new Date(t).toLocaleDateString("fr-BE", { day: "numeric", month: "short" }));
function direEnregistrement(etat, mot, texte, raison, titre) {
  indicateurs.forEach((e) => {
    e.dataset.enr = etat;
    e.querySelector(".et-enr-mot").textContent = mot;
    e.querySelector(".et-enr-etat").textContent = texte;
    e.querySelector(".et-enr-raison").textContent = raison;
    e.title = titre;
  });
  revoir.hidden = etat !== "echec";
  revoir.querySelector(".et-mi-enr-txt").textContent = etat === "echec" ? texte + raison : "";
}
// SUITE : ce que l'utilisateur peut faire, selon la raison du refus (notes.RAISONS dit la raison)
const SUITE = {
  plein: "Cette note n'est gardée nulle part : exportez-la en fichier .stab pour ne pas la perdre.",
  illisible: "Sa copie de sauvegarde n'a pas pu être faite : rien n'y est écrit, pour ne pas effacer ce qu'elle contient. "
    + "Exportez cette note en fichier .stab pour la garder.",
  indisponible: "Cette note n'est gardée nulle part : exportez-la en fichier .stab pour ne pas la perdre.",
};
function direAlerte(sorte, texte) {
  alerte.classList.toggle("et-info", sorte === "info");
  alerte.querySelector(".et-alerte-txt").textContent = texte;
  alerte.querySelector(".et-alerte-stab").hidden = sorte === "info";
  alerte.hidden = false;
}
function enregistrer() {
  if (provisoire) { provisoire = false; history.replaceState(null, "", `?note=${note.id}`); rangement.libeller(); }
  const ok = N.ecrire(note), b = N.bilan();
  if (ok) {
    direEnregistrement("ok", "Enregistré · ", heure(note.modifie), "", `Enregistré dans ce navigateur · ${N.quand(note.modifie)}`);
    if (alerteDite) { alerteDite = ""; alerteEchec = ""; alerte.hidden = true; }
    // la mémoire des notes était illisible : son texte vient d'être gardé à part (notes.js), on le dit une fois
    if (b.copie) {
      direAlerte("info", "La mémoire des notes de ce navigateur était illisible : son texte est gardé à part, tel quel (l'accueil permet "
        + "de le télécharger). Cette note est enregistrée dans une mémoire neuve.");
    }
    return true;
  }
  const r = N.RAISONS[b.raison] || N.RAISONS.indisponible;
  direEnregistrement("echec", "", "Non enregistré", ` : ${r.court}`, `Non enregistré : ${r.long}`);
  alerteEchec = `Non enregistré : ${r.long}. ${SUITE[b.raison] || SUITE.indisponible}`;
  if (alerteDite !== b.raison) {
    alerteDite = b.raison;
    direAlerte("echec", alerteEchec);
  }
  return false;
}
if (relue) direEnregistrement("ok", "Enregistré · ", heure(note.modifie), "", `Enregistré dans ce navigateur · ${N.quand(note.modifie)}`);
else if (provisoire) {
  direEnregistrement("neuf", "", "Pas encore enregistré", "",
    "Rien n'est écrit tant que vous ne modifiez rien : la note s'enregistre dans ce navigateur à la première saisie");
} else direEnregistrement("neuf", "", "Pas encore enregistré", "", "La note s'enregistre dans ce navigateur dès le premier calcul");
alerte.addEventListener("click", (ev) => { if (ev.target.closest("[data-fermer-alerte]")) alerte.hidden = true; });
// le menu ⋯ (téléphone) : l'échec en entier ; le toucher rouvre son message (le menu se referme seul, comme pour ses
// autres entrées)
revoir.addEventListener("click", () => { if (alerteEchec) direAlerte("echec", alerteEchec); });
// « Enregistrer comme note… » (l'établi compact) : nommer, ranger, écrire la note provisoire
const rangement = cablerRangement({ note: () => note, provisoire: () => provisoire,
  apres: () => { const ok = enregistrer(); apresEntete(); return ok; } });

// ------------------------------------------------------------------ le moteur
const moteur = new Moteur((t) => { if (etapes) etapes.textContent = t; }, MODULE);
let enCours = 0;
let meteoEnCours = null;   // le chemin du bouton météo dont la récupération court (une à la fois)
function occupe(plus) {
  enCours += plus;
  const on = enCours > 0 || !!meteoEnCours;
  document.querySelectorAll(".calcule").forEach((i) => {
    i.classList.toggle("on", on);
    i.textContent = on && !i.closest(".mtop") ? (meteoEnCours ? "Récupération de la météo…" : "Calcul…") : "";
  });
}

async function envoyer(action, extra = {}, { sansHistorique = false } = {}) {
  occupe(1);
  try {
    marquerUneFois("requete1_envoyee");
    let avant = null;
    const r = await moteur.demander(() => { avant = note.etat; return { module: MODULE, action, etat: note.etat, ...extra }; });
    marquerUneFois("reponse1_recue");
    if (!r.ok) { toast(`Le moteur a refusé : ${r.erreur}`, true); console.error(r.trace || r.erreur); return null; }
    if (r.etat) {
      if (!sansHistorique) retenir(action, avant, r.etat);
      appliquer(r, action);
    }
    return r;
  } finally { occupe(-1); }
}

// ------------------------------------------------------------------ annuler, rétablir (établi compact)
// Chaque modification RÉUSSIE garde l'état d'avant (50 au plus) ; « Annuler » le rend au moteur, qui recalcule, et la
// note s'enregistre. Une nouvelle modification efface ce qu'on pouvait rétablir. Ouvrir un autre élément n'est pas une
// modification. Dans un champ, Ctrl+Z reste celui du champ.
const hist = { avant: [], apres: [] };
const HIST_MAX = 50;
function retenir(action, avant, apres) {
  if (!COMPACT || !avant || !MODIFIENT.has(action) || action === "activer") return;
  if (JSON.stringify(avant) === JSON.stringify(apres)) return;
  hist.avant.push(avant);
  if (hist.avant.length > HIST_MAX) hist.avant.shift();
  hist.apres.length = 0;
  majHistorique();
}
function majHistorique() {
  document.querySelectorAll("[data-historique]").forEach((b) => {
    b.disabled = !(b.dataset.historique === "annuler" ? hist.avant : hist.apres).length;
  });
}
let enHistorique = false;
async function historiser(sens) {
  const de = sens === "annuler" ? hist.avant : hist.apres, vers = sens === "annuler" ? hist.apres : hist.avant;
  if (!de.length || enHistorique) return;
  enHistorique = true;
  try {
    vers.push(note.etat);
    note.etat = de.pop();
    majHistorique();
    const r = await envoyer("calculer", {}, { sansHistorique: true });
    if (r) toast(sens === "annuler" ? "Modification annulée" : "Modification rétablie");
  } finally { enHistorique = false; }
}
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-historique]");
  if (b && !b.disabled) historiser(b.dataset.historique);
});
document.addEventListener("keydown", (ev) => {
  if (!COMPACT || !(ev.ctrlKey || ev.metaKey) || ev.altKey) return;
  if (ev.target.closest && ev.target.closest("input, textarea, select, [contenteditable]")) return;
  const k = ev.key.toLowerCase();
  if (k === "z" && !ev.shiftKey) { ev.preventDefault(); historiser("annuler"); }
  else if ((k === "z" && ev.shiftKey) || k === "y") { ev.preventDefault(); historiser("retablir"); }
});

// ------------------------------------------------------------------ l'affichage
const ouvertes = new Set(memoire.lire(`stab.sec.${MODULE}`, []));
let premier = true;
let vueCourante = null;
const toile = cablerToile({ bench, stage: $("#stage"), barre: $("#zoom") });
// l'établi compact : la sélection données ↔ dessin, la section courante, les colonnes réglables (js/compact.js)
const compact = COMPACT ? cablerCompact({
  cle: MODULE, note: () => note, ouvrirRubrique,
  vue: { courante: () => vueCourante, choisir: choisirVue, ajuster: () => requestAnimationFrame(() => toile.vue(vueCourante)) },
}) : null;

function appliquer(r, action) {
  note.etat = r.etat;
  note.resume = r.resume;
  // la vignette de l'accueil : le premier DESSIN (une vue peut être un tableau, illisible en vignette)
  const v = (r.vues || []).find((x) => /^\s*<svg/.test(x.svg || ""));
  note.vignette = v ? v.svg : "";
  if (!provisoire || MODIFIENT.has(action)) enregistrer();
  $("#statut").innerHTML = r.statut_html || "";
  afficherResume(r.resume || {});
  $("#resultat").innerHTML = r.resultat;
  const ouvert = $(".flyout.on.calcul");
  $("#volets").innerHTML = r.volets || "";
  if (ouvert) ouvrir(ouvert.id, $(`#resultat [data-calcul="${ouvert.id.slice(3)}"]`), true);
  afficherVues(r.vues || [], r.phrase || "");
  afficherDonnees(r.elements || "", r.donnees);
  [$("#resultat"), $("#volets"), $("#donnees")].forEach(garderGrec);
  if (r.message) toast(r.message, true);
}

// le taux de la note dans l'en-tête du téléphone (le moteur l'a déjà arrondi : on l'écrit à la française)
function afficherResume(res) {
  const c = $("#chip-m");
  const avecTaux = res.taux !== null && res.taux !== undefined;
  c.className = `mchip chip ${avecTaux && ["p", "f", "v", "r"].includes(res.niveau) ? res.niveau : "c"}`;
  // un module de consultation (niveau « i ») n'a pas de taux : son statut n'apprend rien, la place va au nom ; un taux
  // de plus de 999 % s'écrit « > 999 % » (la valeur exacte au survol), comme les pastilles du moteur ; un statut trop
  // long pour l'en-tête a sa forme courte (res.court, commun.STATUTS_COURTS), le statut entier au survol
  const tx = texteTaux(res.taux);
  c.textContent = avecTaux ? tx.texte : (res.niveau === "i" ? "" : res.court || res.statut || "");
  c.title = tx.titre || (res.court ? `${res.statut} · voir le résultat` : "Voir le résultat");
  c.hidden = !c.textContent;
}

function afficherDonnees(elements, html) {
  const zone = $("#donnees");
  // le champ en cours de frappe survit au rafraîchissement : un calcul qui revient pendant qu'on tape dans un autre
  // champ ne doit ni effacer ce qu'on tape, ni le valider à notre place
  const el0 = document.activeElement && zone.contains(document.activeElement) ? document.activeElement : null;
  const actif = el0 ? el0.dataset.chemin : null;
  const brouillon = el0 && (el0.tagName === "INPUT" || el0.tagName === "TEXTAREA") && el0.type !== "checkbox" && el0.type !== "file" && el0.value !== el0.defaultValue
    ? { valeur: el0.value, debut: el0.selectionStart, fin: el0.selectionEnd } : null;
  const defile = zone.scrollTop;
  zone.innerHTML = elements + html;
  separerNoms(zone);
  // les sections gardent leur état (ouvert, fermé) d'un calcul à l'autre
  if (COMPACT) {
    // une rubrique à la fois : celle gardée de la session, sinon la première que le moteur ouvre
    const secs = [...zone.querySelectorAll(".sec[data-sec]")];
    let k = [...ouvertes].find((c) => secs.some((s) => s.dataset.sec === c));
    if (premier && !k) k = (secs.find((s) => s.classList.contains("open")) || secs[0] || {}).dataset?.sec;
    secs.forEach((s) => s.classList.toggle("open", s.dataset.sec === k));
    ouvertes.clear();
    if (k) ouvertes.add(k);
  } else {
    zone.querySelectorAll(".sec[data-sec]").forEach((s) => {
      const cle = s.dataset.sec;
      if (premier) { if (s.classList.contains("open")) ouvertes.add(cle); }
      else s.classList.toggle("open", ouvertes.has(cle));
    });
  }
  premier = false;
  zone.scrollTop = defile;
  marquerMeteo();
  if (actif) {
    const el = zone.querySelector(`[data-chemin="${CSS.escape(actif)}"]:not(button)`);
    if (el) {
      if (brouillon) el.value = brouillon.valeur;
      el.focus({ preventScroll: true });
      if (brouillon) { try { el.setSelectionRange(brouillon.debut, brouillon.fin); } catch (e) { /* champ sans curseur */ } }
    }
  }
  if (compact) compact.rafraichir();
}

// « Dans cette note » : chaque élément s'écrit « nom · dimensions ». Le nom passe dans son propre <span class="nom">
// pour que ce soit LUI qui se coupe (points de suspension) quand la place manque, jamais les dimensions qui le suivent.
function separerNoms(zone) {
  zone.querySelectorAll(".ni .t").forEach((t) => {
    if (t.querySelector(":scope > .nom")) return;
    const avant = [];
    for (const n of t.childNodes) { if (n.nodeType === 1 && n.tagName === "SPAN") break; avant.push(n); }
    if (!avant.length) return;
    const nom = document.createElement("span");
    nom.className = "nom";
    avant.forEach((n) => nom.append(n));
    nom.normalize();
    if (nom.lastChild && nom.lastChild.nodeType === 3) nom.lastChild.data = nom.lastChild.data.replace(/\s+$/, "");
    t.prepend(nom);
  });
}

// « mm · à l'échelle » : sur un dessin seulement — pas sur un tableau (vue HTML), ni sur une vue que le module dit hors
// échelle (une courbe, un diagramme : « echelle »: false)
const estDessin = (v) => /^\s*<svg/.test(v.svg || "");
const echelle = (v) => v.echelle !== false && estDessin(v);

function afficherVues(vues, phrase) {
  const onglets = $("#vues"), stage = $("#stage");
  bench.classList.toggle("sans-vue", !vues.length);
  if (!vues.length) {
    // un module de consultation peut n'avoir aucune vue pour une note : rien d'anormal, on le dit sans alarmer
    onglets.innerHTML = "";
    stage.innerHTML = '<div class="sansvue">Pas de dessin pour cette note.</div>';
    toile.vue(null);
    return;
  }
  if (!vues.some((v) => v.cle === vueCourante)) vueCourante = vues[0].cle;
  onglets.innerHTML = vues.map((v) => `<button type="button" data-vue="${esc(v.cle)}" class="${v.cle === vueCourante ? "on" : ""}">${esc(v.court)}</button>`).join("");
  requestAnimationFrame(marquerDebord);
  const date = new Date().toLocaleDateString("fr-BE");
  const cart = `<div class="cart"><div><span>Note</span><b data-cart="nom">${esc(note.nom || "")}</b></div><div><span>Chantier</span><b data-cart="chantier">${esc(note.chantier || "—")}</b></div>`
    + `<div class="ind"><span>Indice</span><b>${esc(note.indice || "A")}</b></div><div class="dt"><span>Date</span><b>${date}</b></div>`
    + `<div class="mark"><i></i><span>Stab Méthodes</span></div></div>`;
  // chaque vue porte SA légende ; la phrase du module (la note en une ligne) se lit sous le titre de la première
  stage.innerHTML = vues.map((v, i) => `<div class="view${v.cle === vueCourante ? " on" : ""}" data-vue="${esc(v.cle)}"><div class="sheet"><div class="stitle">`
    + `<b>${esc(v.titre)}</b>${v.legende ? `<span class="leg">${esc(v.legende)}</span>` : ""}${echelle(v) ? '<span class="scale">mm · à l\'échelle</span>' : ""}</div>`
    + `${i === 0 && phrase ? `<div class="sphrase">${phrase}</div>` : ""}`
    + `<div class="draw${estDessin(v) ? " dessin" : " bloc"}">${v.svg}</div>${cart}</div></div>`).join("");
  garderGrec(stage);
  toile.vue(vueCourante);
}

// trop de vues pour la largeur : la bande défile, son bord droit s'estompe (on devine la suite)
function marquerDebord() {
  const o = $("#vues");
  o.classList.toggle("defile", o.scrollWidth > o.clientWidth + 1);
}
// (à chaque changement de taille de la bande, transitions comprises : la toile anime sa marge quand un cadre se replie)
new ResizeObserver(marquerDebord).observe($("#vues"));

function choisirVue(k) {
  if (!bench.querySelector(`#vues [data-vue="${CSS.escape(k)}"]`)) return;
  vueCourante = k;
  bench.querySelectorAll("[data-vue]").forEach((e) => e.classList.toggle("on", e.dataset.vue === vueCourante));
  toile.vue(vueCourante);
}
$("#vues").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-vue]");
  if (!b) return;
  choisirVue(b.dataset.vue);
  if (compact) compact.vueChoisie(vueCourante);
});

// ------------------------------------------------------------------ le téléphone : onglets, menu
function onglet(t) {
  bench.dataset.tab = t;
  document.querySelectorAll(".mtabs [data-onglet]").forEach((b) => {
    b.classList.toggle("on", b.dataset.onglet === t);
    b.setAttribute("aria-pressed", b.dataset.onglet === t ? "true" : "false");
  });
  fermer();
  if (telephone.matches) scrollTo({ top: 0 });
  if (t === "dessin") requestAnimationFrame(() => toile.vue(vueCourante));
}
document.addEventListener("click", (ev) => {
  const o = ev.target.closest("[data-onglet], [data-vers-onglet]");
  if (o) onglet(o.dataset.onglet || o.dataset.versOnglet);
});

const menu = $("#mmenu"), menuBouton = $("#menu-m");
function menuOuvert(on) {
  menu.hidden = !on;
  menuBouton.setAttribute("aria-expanded", on ? "true" : "false");
}
menuBouton.addEventListener("click", (ev) => { ev.stopPropagation(); menuOuvert(menu.hidden); });
document.addEventListener("click", (ev) => {
  if (menu.hidden || ev.target.closest("#menu-m")) return;
  // un geste ailleurs referme le menu ; un choix du menu aussi, après avoir agi (une étiquette « Ouvrir » ouvre le
  // choix du fichier APRÈS le clic : le menu ne disparaît qu'ensuite)
  if (!ev.target.closest("#mmenu")) menuOuvert(false);
  else if (ev.target.closest(".mi")) setTimeout(() => menuOuvert(false));
});

// ------------------------------------------------------------------ la saisie
function lireChamp(el) {
  if (el.type === "checkbox") return el.checked;
  return el.value;
}

// les menus et les cases partent à la sélection ; les champs de texte, en quittant le champ (ou Entrée) — et
// seulement si leur valeur diffère de celle du moteur (defaultValue), même après un rafraîchissement
function valider(el) {
  if (el.dataset.sorte === "nombre" || el.dataset.sorte === "entier") {
    const t = el.value.replace(/[\s  ]/g, "").replace(",", ".");
    const x = Number(t);
    const ko = t === "" || !Number.isFinite(x) || (el.dataset.min !== undefined && x < +el.dataset.min) || (el.dataset.max !== undefined && x > +el.dataset.max);
    el.closest(".field").classList.toggle("erreur", ko);
    if (ko) { toast(`Valeur refusée : « ${el.value} »`, true); return; }
  }
  envoyer("modifier", { chemin: el.dataset.chemin.split("/"), valeur: lireChamp(el) });
}
$("#donnees").addEventListener("change", (ev) => {
  const f = ev.target.closest("input[type=file][data-chemin]");
  if (f) { lireFichier(f); return; }
  const el = ev.target.closest("select[data-chemin], input[type=checkbox][data-chemin]");
  if (el) valider(el);
});
$("#donnees").addEventListener("focusout", (ev) => {
  const el = ev.target.closest("input[data-chemin]:not([type=checkbox]):not([type=file]), textarea[data-chemin]");
  if (el && el.value !== el.defaultValue) valider(el);
});

// un fichier choisi (essais de sol, relevés…) part au moteur comme TEXTE : { nom, texte } ; le moteur le lit
async function lireFichier(f) {
  const fichier = f.files && f.files[0];
  if (!fichier) return;
  if (fichier.size > 2 * 1024 * 1024) { toast("Fichier trop gros (2 Mo au plus)", true); return; }
  envoyer("modifier", { chemin: f.dataset.chemin.split("/"), valeur: { nom: fichier.name, texte: await fichier.text() } });
}

$("#donnees").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && ev.target.matches("input[data-chemin]")) ev.target.blur();
});

// « Dans cette note » : supprimer, ouvrir, ajouter, copier un élément. Le bloc vient du cadre Données ; l'établi compact
// le déplace dans la liste de gauche (js/compact.js) : les deux cadres répondent par cette même fonction.
function agirElement(ev) {
  const sup = ev.target.closest("[data-supprimer]");
  if (sup) {
    ev.stopPropagation();
    if (confirm("Supprimer cet élément de la note ?")) envoyer("supprimer", { index: +sup.dataset.supprimer });
    return true;
  }
  const act = ev.target.closest("[data-activer]");
  if (act) { if (!act.classList.contains("on")) envoyer("activer", { index: +act.dataset.activer }); return true; }
  if (ev.target.closest("[data-ajouter]")) { envoyer("ajouter"); return true; }
  if (ev.target.closest("[data-dupliquer]")) { envoyer("dupliquer"); return true; }
  return false;
}
$("#donnees").addEventListener("click", (ev) => {
  const seg = ev.target.closest("button[data-chemin][data-valeur]");
  if (seg) { if (!seg.classList.contains("on")) envoyer("modifier", { chemin: seg.dataset.chemin.split("/"), valeur: seg.dataset.valeur }); return; }
  const hd = ev.target.closest(".sechd[data-replier]");
  if (hd) {
    const s = hd.parentElement;
    if (COMPACT && !s.classList.contains("open")) { ouvrirRubrique(s.dataset.sec); return; }
    s.classList.toggle("open");
    if (s.classList.contains("open")) ouvertes.add(s.dataset.sec); else ouvertes.delete(s.dataset.sec);
    memoire.ecrire(`stab.sec.${MODULE}`, [...ouvertes]);
    return;
  }
  // le sélecteur d'armatures (« Proposer ») et « Renommer » d'un élément, posés dans les données
  const pk = ev.target.closest("[data-pk]");
  if (pk) { ev.stopPropagation(); ouvrirSelecteur(pk.dataset.pk, pk); return; }
  if (ev.target.closest("[data-renommer]")) { ev.stopPropagation(); renommer(); return; }
  agirElement(ev);
});

// l'établi compact : ouvrir UNE rubrique (les autres se replient) — un clic sur son en-tête, ou une pièce du dessin
function ouvrirRubrique(cle) {
  const zone = $("#donnees");
  zone.querySelectorAll(".sec[data-sec]").forEach((x) => x.classList.toggle("open", x.dataset.sec === cle));
  ouvertes.clear();
  ouvertes.add(cle);
  memoire.ecrire(`stab.sec.${MODULE}`, [...ouvertes]);
  const s = zone.querySelector(`.sec[data-sec="${CSS.escape(cle)}"]`);
  if (s) requestAnimationFrame(() => s.scrollIntoView({ block: "nearest" }));
}
// « Renommer » l'élément ouvert : son champ « Nom » (rubrique 1), choisi
function renommer() {
  const champ = $('#donnees input[data-chemin="nom"]');
  if (!champ) return;
  const rub = champ.closest(".sec[data-sec]");
  if (rub && COMPACT) ouvrirRubrique(rub.dataset.sec); else if (rub) rub.classList.add("open");
  champ.focus();
  champ.select();
}

// ------------------------------------------------------------------ les volets : calcul détaillé et sélecteur
function fermer() {
  bench.querySelectorAll(".flyout.on").forEach((f) => f.classList.remove("on"));
  bench.querySelectorAll(".el.sel").forEach((e) => e.classList.remove("sel"));
  bench.classList.remove("fly");
  document.documentElement.classList.remove("volet-ouvert");
}

function ouvrir(id, ancre, garder = false) {
  if (!garder) fermer();
  const f = document.getElementById(id);
  if (!f) return;
  f.classList.add("on");
  // au téléphone, le volet monte du bas, par-dessus l'onglet ouvert ; le fond s'assombrit et ne défile plus
  bench.classList.add("fly");
  document.documentElement.classList.toggle("volet-ouvert", telephone.matches);
  if (ancre && ancre.classList.contains("el")) ancre.classList.add("sel");
  if (ancre && !f.classList.contains("calcul") && !telephone.matches) {
    const b = bench.getBoundingClientRect(), a = ancre.getBoundingClientRect();
    f.style.top = Math.max(10, Math.min(a.top - b.top - 12, b.height - f.offsetHeight - 10)) + "px";
  }
  const l = f.querySelector(".pk-list");
  if (l) {
    const r = l.querySelector(".pk-row");
    l.scrollTop = (r ? r.offsetHeight : 34) * (+f.dataset.debut || 0);
  }
}

const filtres = {};   // les familles cochées de chaque sélecteur, gardées pendant la session
async function ouvrirSelecteur(role, ancre) {
  const f = filtres[role] || {};
  const r = await envoyer("selecteur", { role, familles: f.familles || null, froid: f.froid ?? true });
  if (!r) return;
  const ancien = document.getElementById(`pk-${role}`);
  if (ancien) ancien.remove();
  $("#volets").insertAdjacentHTML("beforeend", r.html);
  garderGrec(document.getElementById(`pk-${role}`));
  ouvrir(`pk-${role}`, ancre);
}

$("#resultat").addEventListener("click", (ev) => {
  const pk = ev.target.closest(".prof[data-pk]");
  if (pk) { ev.stopPropagation(); ouvrirSelecteur(pk.dataset.pk, pk.closest(".el") || pk); return; }
  const c = ev.target.closest("[data-calcul]");
  if (c) { ev.stopPropagation(); ouvrir(`ck-${c.dataset.calcul}`, c); }
});

$("#volets").addEventListener("click", (ev) => {
  ev.stopPropagation();
  if (ev.target.closest("[data-fermer]")) { fermer(); return; }
  const picker = ev.target.closest(".picker");
  const ligne = ev.target.closest(".pk-row[data-choix]");
  if (ligne && picker) {
    const role = picker.dataset.pk;
    fermer();
    if (!ligne.classList.contains("cur")) envoyer("choisir", { role, nom: ligne.dataset.choix }).then((r) => r && toast(`${ligne.dataset.choix} retenu`));
    return;
  }
  const chip = ev.target.closest(".fchip");
  if (chip && picker) {
    const role = picker.dataset.pk;
    const f = filtres[role] || (filtres[role] = {});
    const on = [...picker.querySelectorAll(".fchip[data-famille].on")].map((c) => c.dataset.famille);
    if (chip.dataset.famille) {
      const k = chip.dataset.famille;
      f.familles = on.includes(k) ? on.filter((x) => x !== k) : [...on, k];
      if (!f.familles.length) { toast("Garder au moins une famille", true); f.familles = on; return; }
    } else if (chip.dataset.froid) {
      f.familles = on;
      f.froid = !chip.classList.contains("on");
    }
    ouvrirSelecteur(role, $(`#resultat .prof[data-pk="${CSS.escape(role)}"]`)?.closest(".el"));
  }
});

$("#volets").addEventListener("mouseover", (ev) => {
  const r = ev.target.closest(".pk-row");
  if (!r) return;
  const p = r.closest(".picker").querySelector(".pk-prop");
  if (p) p.innerHTML = r.dataset.prop || "";
});

// au toucher, pas de survol : le premier appui sur une ligne du sélecteur montre ses propriétés… et la choisit ; on
// montre donc les propriétés dès que le doigt se pose
$("#volets").addEventListener("pointerdown", (ev) => {
  const r = ev.target.closest(".pk-row");
  if (!r || ev.pointerType === "mouse") return;
  const p = r.closest(".picker").querySelector(".pk-prop");
  if (p) p.innerHTML = r.dataset.prop || "";
});

document.addEventListener("click", (ev) => { if (!ev.target.closest(".flyout,[data-pk],[data-calcul]")) fermer(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { fermer(); menuOuvert(false); } });
bench.querySelectorAll("[data-coll]").forEach((b) => b.addEventListener("click", () => {
  bench.classList.toggle(b.dataset.coll === "g" ? "sans-g" : "sans-d");
  requestAnimationFrame(() => toile.vue(vueCourante));
}));

// ------------------------------------------------------------------ un fichier propre au module (export JSON…)
// Le module pose dans ses fragments <button data-telecharger="<quoi>"> ; le moteur rend le fichier
// ({"fichier": {"nom", "type", "texte"}}) et le navigateur l'enregistre sous ce nom, nettoyé (commun.js).
async function telechargerFichier(quoi) {
  occupe(1);
  // la note PDF : la première se prépare quelques secondes (le paquet de la note se charge), on le dit
  if (quoi === "note_pdf") toast("Préparation de la note de calcul PDF…");
  try {
    // le nom, le chantier, l'indice de la note : la page de garde d'une note PDF (un module qui ne les attend pas les ignore)
    const infos = { nom: note.nom || "", chantier: note.chantier || "", indice: note.indice || "A", date: new Date().toLocaleDateString("fr-BE") };
    const r = await moteur.demander(() => ({ module: MODULE, action: "telecharger", etat: note.etat, quoi, infos }), { note: quoi === "note_pdf" });
    const f = r && r.ok ? r.fichier : null;
    if (!f || typeof f.nom !== "string" || (typeof f.texte !== "string" && typeof f.base64 !== "string")) {
      toast(`Téléchargement refusé : ${(r && !r.ok && r.erreur) || "le moteur n'a pas rendu de fichier"}`, true);
      if (r && r.trace) console.error(r.trace);
      return;
    }
    toast(`${telecharger(f.nom, f.base64 ? { base64: f.base64 } : f.texte, f.type || "application/octet-stream")} téléchargé`);
  } finally { occupe(-1); }
}
// à la capture : les volets arrêtent la propagation de leurs clics
bench.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-telecharger]");
  if (!b) return;
  ev.preventDefault();
  ev.stopPropagation();
  telechargerFichier(b.dataset.telecharger);
}, true);

// ------------------------------------------------------------------ la météo du lieu (Open-Meteo), à la demande
// Le module pose <button data-meteo="<plan>" data-meteo-chemin="…"> (commun.bouton_meteo) : le PLAN vient du moteur ;
// js/meteo.js fait les appels vers Open-Meteo et rend les réponses BRUTES, qui partent au module (« modifier » sur le
// chemin du bouton). Le module les lit, les garde dans la note, et le calcul se fait hors ligne. Une seule récupération à
// la fois ; pendant ce temps, le bouton dit « Récupération de la météo… ».
function marquerMeteo() {
  document.querySelectorAll("#donnees [data-meteo]").forEach((b) => {
    const attend = !!meteoEnCours && b.dataset.meteoChemin === meteoEnCours;
    b.classList.toggle("attente", attend);
    if (attend) {
      if (b.dataset.etaitGrise === undefined) b.dataset.etaitGrise = b.disabled ? "1" : "";
      b.disabled = true;
      b.setAttribute("aria-busy", "true");
      const t = b.querySelector("span");
      if (t) { t.dataset.libelle = t.dataset.libelle || t.textContent; t.textContent = "Récupération de la météo…"; }
    }
  });
}

async function importerMeteo(b) {
  if (meteoEnCours) return;
  let plan = null;
  try { plan = JSON.parse(b.dataset.meteo); } catch (e) { /* le plan illisible est refusé par recupererMeteo */ }
  const chemin = b.dataset.meteoChemin || "";
  meteoEnCours = chemin;
  marquerMeteo();
  occupe(0);
  try {
    const r = await recupererMeteo(plan);
    if (!r.ok) { toast(r.erreur, true); return; }
    const rep = await envoyer("modifier", { chemin: chemin.split("/"), valeur: r.valeur });
    if (rep && !rep.message) toast("Météo récupérée : la note garde les températures, le calcul se fait hors ligne.");
  } catch (e) {
    toast(`La météo n'a pas pu être récupérée : ${e.message}`, true);
  } finally {
    meteoEnCours = null;
    // le bouton revient à lui-même (le cadre Données a pu être refait entre-temps : chaque bouton retrouve son libellé)
    document.querySelectorAll("#donnees [data-meteo].attente").forEach((x) => {
      x.classList.remove("attente");
      x.removeAttribute("aria-busy");
      x.disabled = x.dataset.etaitGrise === "1";
      delete x.dataset.etaitGrise;
      const t = x.querySelector("span");
      if (t && t.dataset.libelle) t.textContent = t.dataset.libelle;
    });
    occupe(0);
  }
}
// à la capture, comme « télécharger » : rien d'autre ne réagit à ce clic
bench.addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-meteo]");
  if (!b) return;
  ev.preventDefault();
  ev.stopPropagation();
  if (!b.disabled) importerMeteo(b);
}, true);

// ------------------------------------------------------------------ fichier, impression
document.addEventListener("click", (ev) => {
  const a = ev.target.closest("[data-action]");
  if (!a) return;
  if (a.dataset.action === "exporter") N.exporter(note);
  else if (a.dataset.action === "imprimer") window.print();
  else if (a.dataset.action === "pdf") {
    if (COMPACT) telechargerFichier("note_pdf");
    else toast("La note PDF arrive bientôt sur le site ; elle reste dans Streamlit.");
  }
});
$("#ouvrir").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    const n = await N.importer(f);
    location.href = `${N.MODULES[n.module].page}?note=${n.id}`;
  } catch (e) { toast(`Import impossible : ${e.message}`, true); }
});

// ------------------------------------------------------------------ démarrage
(async () => {
  try {
    await moteur.pret;
    const r = await envoyer(note.etat ? "calculer" : "nouveau");
    marquer("reponse1_appliquee");
    if (!r) throw new Error("le premier calcul a échoué");
    $("#demarrage").hidden = true;
    // la sonde : deux images plus tard, le navigateur a peint le premier résultat
    if (SONDE) requestAnimationFrame(() => requestAnimationFrame(() => marquer("premier_rendu")));
  } catch (e) {
    const d = $("#demarrage");
    d.classList.add("erreur");
    etapes.textContent = `Le moteur n'a pas pu démarrer : ${e.message}\nRecharger la page ; si cela persiste, utiliser Streamlit.`;
  }
})();
