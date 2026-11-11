// Ce que les écrans du BUREAU — l'accueil, les projets, l'équipe — AFFICHENT de la même façon (R1 : une fonctionnalité,
// un composant partagé) : une ligne du journal, la liste groupée par jour, le formulaire « Qui êtes-vous ? », le champ
// qui crée à la touche Entrée, la ligne d'une note. Les données viennent de js/equipe.js, js/journal.js, js/projets.js ;
// ici, on ne fait que les montrer. Tout texte venu de l'utilisateur passe par esc().
import { esc, toast, texteTaux } from "./commun.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import * as N from "./notes.js";

// les icônes déposées dans la page par construire.icones_bureau (des <template> inertes)
export function icone(nom) {
  const t = document.querySelector(`#icones template[data-icone="${nom}"]`);
  return t ? t.innerHTML : "";
}
export function iconeModule(cle) {
  const t = document.querySelector(`#icones template[data-icone-module="${cle}"]`);
  return t ? t.innerHTML : "";
}

export const prenom = (nom) => String(nom || "").trim().split(/\s+/)[0] || "";
const pluriel = (n, mot, mots = `${mot}s`) => `${n} ${n > 1 ? mots : mot}`;
export const compter = pluriel;

// ------------------------------------------------------------------ le journal
// Une ligne : la pastille de l'auteur, la phrase (« Vous avez créé la note … », « Marie a rangé la note … »), le lieu
// (projet › sous-projet), l'heure. Le nom de la note mène à la note tant qu'elle est dans ce navigateur.
export function ligneJournal(e, { moi = null, notes = {}, maintenant = Date.now(), lieu = true } = {}) {
  const p = J.phrase(e);
  const estMoi = !!(moi && e.q && e.q.id === moi.id);
  const qui = estMoi ? "Vous" : e.q ? prenom(p.qui) : "Quelqu'un";
  const verbe = estMoi ? p.verbe.replace(/^a /, "avez ") : p.verbe;
  let objet = p.objet ? `<b>${esc(p.objet)}</b>` : "";
  const n = e.n && e.k !== "note.supprimee" ? notes[e.n.id] : null;
  if (n && Object.hasOwn(N.MODULES, n.module)) objet = `<a class="bu-lien" href="${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}"><b>${esc(p.objet)}</b></a>`;
  // le lieu : « projet › sous-projet » ; dans la page d'un projet (lieu = "sous"), le sous-projet seul
  const texteLieu = lieu === "sous" ? (e.k.startsWith("note.") ? e.s : "") : lieu ? p.lieu : "";
  const ou = texteLieu ? ` <span class="bu-lieu">· ${esc(texteLieu)}</span>` : "";
  const quand = new Date(e.t).toLocaleString("fr-BE", { dateStyle: "medium", timeStyle: "short" });
  return `<div class="feed">${E.pastille(e.q, "sm")}<div>${esc(qui)} ${esc(verbe)} ${objet}${p.suite ? ` ${esc(p.suite)}` : ""}${ou}`
    + `<span class="when" title="${esc(quand)}">${esc(J.ilYA(e.t, maintenant))}</span></div></div>`;
}

// Le journal : ses lignes ; ``jours`` : groupées sous « Aujourd'hui », « Hier », la date ; sans ligne : ``vide``.
export function journalHTML(lignes, { moi = null, notes = {}, jours = false, vide = "Rien pour l'instant.", lieu = true } = {}) {
  if (!lignes.length) return `<p class="muted bu-vide">${esc(vide)}</p>`;
  const maintenant = Date.now();
  let dernier = null, out = "";
  for (const e of lignes) {
    if (jours) {
      const j = J.jourDe(e.t, maintenant);
      if (j !== dernier) { out += `<div class="bu-jour">${esc(j)}</div>`; dernier = j; }
    }
    out += ligneJournal(e, { moi, notes, maintenant, lieu });
  }
  return out;
}

// ------------------------------------------------------------------ un champ qui crée à la touche Entrée
// (construire.champ_creer écrit le gabarit : <form id="X-form"><input id="X"><button id="X-creer" hidden>.)
// ``valider(nom)`` : crée ; rend vrai si le champ peut se vider. ``saisie(texte)`` : appelé à chaque frappe (un filtre).
export function cablerCreer(id, { signal, valider, saisie = null }) {
  const form = document.getElementById(`${id}-form`), champ = document.getElementById(id), bouton = document.getElementById(`${id}-creer`);
  if (!form || !champ) return { vider() {} };
  const voir = () => { if (bouton) bouton.hidden = !champ.value.trim(); };
  champ.addEventListener("input", () => { voir(); if (saisie) saisie(champ.value); }, { signal });
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const nom = champ.value.trim();
    if (!nom) return;
    if (valider(nom)) { champ.value = ""; voir(); if (saisie) saisie(""); }
  }, { signal });
  voir();
  return { vider() { champ.value = ""; voir(); } };
}

// ------------------------------------------------------------------ « Qui êtes-vous ? »
// Se présenter : le prénom et le nom ; les initiales sont proposées à mesure (LLE pour Louis Lejeune) et se corrigent ;
// l'e-mail est facultatif — prérempli par l'adresse de connexion quand le site la connaît.
export function formulaireQui(titre = "Qui êtes-vous ?", aide = "Vos initiales signeront vos notes et vos gestes dans le journal.") {
  return `<form class="bu-qui" data-qui="1" autocomplete="off"><div class="bu-qui-t"><b>${esc(titre)}</b><span>${esc(aide)}</span></div>`
    + `<div class="bu-qui-c"><span class="avatar inconnu lg" data-qui-pastille="1">?</span>`
    + `<div class="field text bu-qui-nom"><input name="nom" maxlength="80" placeholder="Prénom et nom" aria-label="Votre prénom et votre nom" autocomplete="name"></div>`
    + `<div class="field text bu-ini"><input name="initiales" maxlength="5" placeholder="Initiales" aria-label="Vos initiales (1 à 5)" autocapitalize="characters"></div></div>`
    + `<div class="bu-qui-c"><div class="field text bu-qui-mail">${icone("mail")}<input name="email" type="email" maxlength="120" placeholder="e-mail (facultatif)" aria-label="Votre adresse e-mail (facultatif)" autocomplete="email"></div>`
    + `<button type="submit" class="btn primary">C'est moi</button></div>`
    + `<p class="hint bu-erreur" role="alert" hidden></p></form>`;
}

// Câbler un formulaire de fiche (se présenter, ajouter un membre) : les initiales suivent le nom tant qu'on ne les a pas
// écrites soi-même ; ``pastille`` : l'aperçu du rond. Rend { lire(), erreur(texte, champ), vider() }.
export function cablerFiche({ nom, initiales, email = null, pastille = null, erreur = null, signal, prises = () => [] }) {
  let libre = !initiales.value.trim();
  const apercu = () => {
    if (!pastille) return;
    const ini = E.initialesValides(initiales.value);
    pastille.textContent = ini || "?";
    pastille.className = `avatar lg ${ini ? `c0 n${ini.length}` : "inconnu"}`;
  };
  nom.addEventListener("input", () => { if (libre) initiales.value = E.proposerInitiales(nom.value, prises()); apercu(); }, { signal });
  initiales.addEventListener("input", () => { libre = !initiales.value.trim(); initiales.value = initiales.value.toLocaleUpperCase("fr"); apercu(); }, { signal });
  apercu();
  return {
    lire: () => ({ nom: nom.value, initiales: initiales.value, email: email ? email.value : "" }),
    erreur(texte = "", champ = "") {
      if (erreur) { erreur.textContent = texte; erreur.hidden = !texte; }
      for (const [k, c] of Object.entries({ nom, initiales, email })) if (c) c.closest(".field").classList.toggle("refus", !!texte && k === champ);
      const c = { nom, initiales, email }[champ];
      if (texte && c) c.focus();
    },
    vider() { nom.value = ""; initiales.value = ""; if (email) email.value = ""; libre = true; apercu(); },
  };
}

// Câbler « Qui êtes-vous ? » dans ``racine`` : à la validation, le membre est créé, il devient la personne du poste, le
// journal le note, puis ``apres(membre)``.
export function cablerQui(racine, { signal, apres }) {
  const form = racine.querySelector("form[data-qui]");
  if (!form) return;
  const c = (n) => form.querySelector(`[name="${n}"]`);
  const fiche = cablerFiche({ nom: c("nom"), initiales: c("initiales"), email: c("email"), pastille: form.querySelector("[data-qui-pastille]"),
                              erreur: form.querySelector(".bu-erreur"), signal, prises: () => E.equipe().membres.map((m) => m.initiales) });
  // l'adresse avec laquelle on est entré sur le site, si le site la connaît : elle préremplit le champ resté vide. Elle
  // n'est demandée qu'au moment où l'on COMMENCE à se présenter (le premier champ touché), une seule fois — pas à
  // chaque ouverture de l'accueil : hors de Cloudflare, cette adresse n'existe pas, et la demander pour rien laissait
  // une erreur 404 dans la console de chaque page (le harnais l'a trouvée, « aucune erreur JavaScript »).
  let demandee = false;
  form.addEventListener("focusin", () => {
    if (demandee) return;
    demandee = true;
    E.emailDeConnexion().then((a) => { if (a && !signal.aborted && form.isConnected && !c("email").value) c("email").value = a; });
  }, { signal });
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const r = E.ajouter(fiche.lire(), { estMoi: true });
    if (!r.ok) { fiche.erreur(r.erreur, r.champ); return; }
    J.noter("membre.arrive", {});
    toast(`Bienvenue, ${prenom(r.membre.nom)} : vos initiales sont ${r.membre.initiales}.`);
    apres(r.membre);
  }, { signal });
}

// ------------------------------------------------------------------ la ligne d'une note (la page des projets)
// ``ranger`` : le HTML d'un menu « ranger dans… » (ou rien).
export function ligneNote(n, { ranger = "" } = {}) {
  const m = N.MODULES[n.module], r = n.resume || {}, tx = texteTaux(r.taux);
  const chip = tx.texte ? `<span class="chip ${esc(r.niveau)}"${tx.titre ? ` title="${esc(tx.titre)}"` : ""}>${esc(tx.texte)}</span>` : '<span class="muted">—</span>';
  const par = n.par || n.auteur || null;
  return `<tr data-note="${esc(n.id)}"><td class="bu-n"><a class="mod bu-lien" href="${N.pageDe(n.module)}?note=${encodeURIComponent(n.id)}">`
    + `<span class="modic ${esc(m.famille)}">${iconeModule(N.moduleServi(n.module))}</span><span><b>${esc(n.nom || m.nom)}</b><span class="muted">${esc(m.nom)}</span></span></a></td>`
    + `<td class="bu-r">${chip}</td><td class="bu-i"><span class="indice">${esc(n.indice || "A")}</span></td>`
    + `<td class="muted bu-q" title="${esc(N.quand(n.modifie))}">${esc(J.ilYA(n.modifie))}</td><td class="bu-p">${E.pastille(par, "sm")}</td>`
    + `<td class="n bu-g">${ranger}</td></tr>`;
}

export function tableNotes(notes, opts = {}) {
  if (!notes.length) return "";
  return `<div class="card tablecard bu-table"><table class="table"><thead><tr><th>Note</th><th>Résultat</th><th>Indice</th><th>Modifiée</th><th>Par</th><th class="n"></th></tr></thead>`
    + `<tbody>${notes.map((n) => ligneNote(n, { ranger: opts.ranger ? opts.ranger(n) : "" })).join("")}</tbody></table></div>`;
}

// les pastilles de plusieurs personnes, empilées (trois au plus, puis « +n »)
export function pile(signatures, max = 3) {
  if (!signatures.length) return "";
  const vues = signatures.slice(0, max).map((s) => E.pastille(s, "sm")).join("");
  return `<span class="avstack">${vues}${signatures.length > max ? `<span class="avatar sm plus">+${signatures.length - max}</span>` : ""}</span>`;
}
