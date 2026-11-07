// L'établi d'un module (garde-corps.html, poutre.html) : il envoie au moteur ce que l'utilisateur saisit et affiche les
// fragments qu'il reçoit (données, résultat, fenêtres de calcul, sélecteurs, dessins). Il ne calcule rien.
import { Moteur } from "./moteur.js";
import * as N from "./notes.js";
import { cablerRail, toast, esc } from "./commun.js";

const MODULE = document.body.dataset.module;
const $ = (s, r = document) => r.querySelector(s);
const bench = $(".bench");
const etapes = $("#demarrage-etape");
cablerRail();

// ------------------------------------------------------------------ la note
const q = new URLSearchParams(location.search);
let note = q.get("note") && N.lire(q.get("note"));
if (!note || note.module !== MODULE) {
  note = N.nouvelle({ module: MODULE, nom: q.get("nom") || "", chantier: q.get("chantier") || "", dossier: q.get("dossier") || "" });
  history.replaceState(null, "", `?note=${note.id}`);
}
const nom = $("#nom-note");
function afficherEntete() {
  nom.value = note.nom || "";
  document.title = `${note.nom || N.MODULES[MODULE].nom} · Stab Méthodes`;
  const fil = [note.chantier, note.dossier].filter(Boolean).join(" › ");
  $("#fil-chantier").textContent = fil || "Sans chantier";
}
afficherEntete();
nom.addEventListener("change", () => { note.nom = nom.value.trim() || N.MODULES[MODULE].nom; enregistrer(); afficherEntete(); });

function enregistrer() {
  const ok = N.ecrire(note);
  const s = $("#enregistre");
  s.classList.toggle("attente", !ok);
  s.querySelector(".txt").textContent = ok ? `Enregistré dans ce navigateur · ${N.quand(note.modifie)}` : "Non enregistré (stockage plein ?)";
}

// ------------------------------------------------------------------ le moteur
const moteur = new Moteur((t) => { if (etapes) etapes.textContent = t; });
const indic = $("#calcule");
let enCours = 0;
function occupe(plus) {
  enCours += plus;
  indic.classList.toggle("on", enCours > 0);
  indic.textContent = enCours > 0 ? "Calcul…" : "";
}

async function envoyer(action, extra = {}) {
  occupe(1);
  try {
    const r = await moteur.demander(() => ({ module: MODULE, action, etat: note.etat, ...extra }));
    if (!r.ok) { toast(`Le moteur a refusé : ${r.erreur}`, true); console.error(r.trace || r.erreur); return null; }
    if (r.etat) appliquer(r);
    return r;
  } finally { occupe(-1); }
}

// ------------------------------------------------------------------ l'affichage
const ouvertes = new Set(JSON.parse(sessionStorage.getItem(`stab.sec.${MODULE}`) || "null") || []);
let premier = true;
let vueCourante = null;

function appliquer(r) {
  note.etat = r.etat;
  note.resume = r.resume;
  // la vignette de l'accueil : le premier DESSIN (une vue peut être un tableau, illisible en vignette)
  const v = (r.vues || []).find((x) => /^\s*<svg/.test(x.svg || ""));
  note.vignette = v ? v.svg : "";
  enregistrer();
  $("#statut").innerHTML = r.statut_html || "";
  $("#resultat").innerHTML = r.resultat;
  const ouvert = $(".flyout.on.calcul");
  $("#volets").innerHTML = r.volets || "";
  if (ouvert) ouvrir(ouvert.id, $(`#resultat [data-calcul="${ouvert.id.slice(3)}"]`), true);
  afficherVues(r.vues || [], r.phrase || "");
  afficherDonnees(r.elements || "", r.donnees);
  if (r.message) toast(r.message, true);
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
  // les sections gardent leur état (ouvert, fermé) d'un calcul à l'autre
  zone.querySelectorAll(".sec[data-sec]").forEach((s) => {
    const cle = s.dataset.sec;
    if (premier) { if (s.classList.contains("open")) ouvertes.add(cle); }
    else s.classList.toggle("open", ouvertes.has(cle));
  });
  premier = false;
  zone.scrollTop = defile;
  if (actif) {
    const el = zone.querySelector(`[data-chemin="${CSS.escape(actif)}"]:not(button)`);
    if (el) {
      if (brouillon) el.value = brouillon.valeur;
      el.focus({ preventScroll: true });
      if (brouillon) { try { el.setSelectionRange(brouillon.debut, brouillon.fin); } catch (e) { /* champ sans curseur */ } }
    }
  }
}

// « mm · à l'échelle » : sur un dessin seulement — pas sur un tableau (vue HTML), ni sur une vue que le module dit hors
// échelle (une courbe, un diagramme : « echelle »: false)
const echelle = (v) => v.echelle !== false && /^\s*<svg/.test(v.svg || "");

function afficherVues(vues, phrase) {
  const onglets = $("#vues"), stage = $("#stage");
  if (!vues.length) { onglets.innerHTML = ""; stage.innerHTML = '<div class="attente">Pas de dessin : le calcul est impossible.</div>'; return; }
  if (!vues.some((v) => v.cle === vueCourante)) vueCourante = vues[0].cle;
  onglets.innerHTML = vues.map((v) => `<button data-vue="${esc(v.cle)}" class="${v.cle === vueCourante ? "on" : ""}">${esc(v.court)}</button>`).join("");
  const date = new Date().toLocaleDateString("fr-BE");
  const cart = `<div class="cart"><div><span>Note</span><b>${esc(note.nom || "")}</b></div><div><span>Chantier</span><b>${esc(note.chantier || "—")}</b></div>`
    + `<div><span>Indice</span><b>${esc(note.indice || "A")}</b></div><div><span>Date</span><b>${date}</b></div><div class="mark"><i></i>Stab Méthodes</div></div>`;
  stage.innerHTML = vues.map((v) => `<div class="view${v.cle === vueCourante ? " on" : ""}" data-vue="${esc(v.cle)}"><div class="sheet"><div class="stitle">`
    + `<b>${esc(v.titre.toUpperCase())}</b><span>${v.cle === vues[0].cle ? phrase : esc(v.legende || "")}</span>${echelle(v) ? '<span class="scale">mm · à l\'échelle</span>' : ""}</div>`
    + `<div class="draw">${v.svg}</div>${cart}</div></div>`).join("");
}

$("#vues").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-vue]");
  if (!b) return;
  vueCourante = b.dataset.vue;
  bench.querySelectorAll("[data-vue]").forEach((e) => e.classList.toggle("on", e.dataset.vue === vueCourante));
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
    const t = el.value.replace(/[\s  ]/g, "").replace(",", ".");
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

$("#donnees").addEventListener("click", (ev) => {
  const seg = ev.target.closest("button[data-chemin][data-valeur]");
  if (seg) { if (!seg.classList.contains("on")) envoyer("modifier", { chemin: seg.dataset.chemin.split("/"), valeur: seg.dataset.valeur }); return; }
  const hd = ev.target.closest(".sechd[data-replier]");
  if (hd) {
    const s = hd.parentElement;
    s.classList.toggle("open");
    if (s.classList.contains("open")) ouvertes.add(s.dataset.sec); else ouvertes.delete(s.dataset.sec);
    sessionStorage.setItem(`stab.sec.${MODULE}`, JSON.stringify([...ouvertes]));
    return;
  }
  const sup = ev.target.closest("[data-supprimer]");
  if (sup) {
    ev.stopPropagation();
    if (confirm("Supprimer cet élément de la note ?")) envoyer("supprimer", { index: +sup.dataset.supprimer });
    return;
  }
  const act = ev.target.closest("[data-activer]");
  if (act && !act.classList.contains("on")) { envoyer("activer", { index: +act.dataset.activer }); return; }
  if (ev.target.closest("[data-ajouter]")) { envoyer("ajouter"); return; }
  if (ev.target.closest("[data-dupliquer]")) envoyer("dupliquer");
});

// ------------------------------------------------------------------ les volets : calcul détaillé et sélecteur
function fermer() {
  bench.querySelectorAll(".flyout.on").forEach((f) => f.classList.remove("on"));
  bench.querySelectorAll(".el.sel").forEach((e) => e.classList.remove("sel"));
}

function ouvrir(id, ancre, garder = false) {
  if (!garder) fermer();
  const f = document.getElementById(id);
  if (!f) return;
  f.classList.add("on");
  if (ancre && ancre.classList.contains("el")) ancre.classList.add("sel");
  if (ancre && !f.classList.contains("calcul")) {
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

document.addEventListener("click", (ev) => { if (!ev.target.closest(".flyout,[data-pk],[data-calcul]")) fermer(); });
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") fermer(); });
bench.querySelectorAll("[data-coll]").forEach((b) => b.addEventListener("click", () => bench.classList.toggle(b.dataset.coll === "g" ? "sans-g" : "sans-d")));

// ------------------------------------------------------------------ fichier, impression
$("#exporter").addEventListener("click", () => N.exporter(note));
$("#imprimer").addEventListener("click", () => window.print());
$("#ouvrir").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  try {
    const n = await N.importer(f);
    location.href = `${N.MODULES[n.module].page}?note=${n.id}`;
  } catch (e) { toast(`Fichier refusé : ${e.message}`, true); }
});

// ------------------------------------------------------------------ démarrage
(async () => {
  try {
    await moteur.pret;
    const r = await envoyer(note.etat ? "calculer" : "nouveau");
    if (!r) throw new Error("le premier calcul a échoué");
    $("#demarrage").hidden = true;
  } catch (e) {
    const d = $("#demarrage");
    d.classList.add("erreur");
    etapes.textContent = `Le moteur n'a pas pu démarrer : ${e.message}\nRecharger la page ; si cela persiste, utiliser Streamlit.`;
  }
})();
