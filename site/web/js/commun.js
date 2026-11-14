// Ce que toutes les pages partagent : le thème (clair, sombre), le volet des modules, les messages éphémères, le
// bandeau du démarrage du moteur et l'état « lecture seule » des contrôles (lot S4), le manifeste du moteur.
//
// cablerRail n'est appelé qu'UNE fois, par js/app.js (l'application) : le rail survit aux écrans qui se succèdent ; câblé
// deux fois, le thème basculait deux fois (donc pas du tout).
export function cablerRail() {
  const racine = document.documentElement;
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-theme-toggle]");
    if (t) {
      const sombre = racine.dataset.theme === "dark" || (!racine.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      poserTheme(sombre ? "light" : "dark");
      return;
    }
    const m = ev.target.closest("[data-modules]");
    if (m) { document.body.classList.toggle("show-modules"); return; }
    if (!ev.target.closest(".modpanel")) document.body.classList.remove("show-modules");
  });
}

// Le THÈME (une source : le bouton du rail, l'en-tête du téléphone et la page « Paramètres » le posent ici) : « light »,
// « dark », ou « » — celui de l'appareil (la clé retirée : js/theme.js n'impose rien). themeChoisi() le relit.
export function poserTheme(choix) {
  const racine = document.documentElement;
  if (choix === "light" || choix === "dark") racine.dataset.theme = choix;
  else delete racine.dataset.theme;
  try {
    if (choix === "light" || choix === "dark") localStorage.setItem("stab.theme", choix);
    else localStorage.removeItem("stab.theme");
  } catch (e) { /* navigation privée : le thème vaut pour cette page */ }
}
export function themeChoisi() {
  const t = document.documentElement.dataset.theme;
  return t === "light" || t === "dark" ? t : "";
}

// LA VALEUR D'UNE SAISIE — un champ, un menu, une case que le moteur a écrits (api_web/commun.py : data-chemin,
// data-sorte, data-min, data-max) — telle qu'elle part au moteur. Une source : js/etabli.js (les notes) et
// js/ecran_parametres.js (le gabarit du bureau) la lisent. Un nombre illisible ou hors de ses bornes : { ok: false }.
export function valeurSaisie(el) {
  if (el.type === "checkbox") return { ok: true, valeur: el.checked };
  if (el.dataset.sorte === "nombre" || el.dataset.sorte === "entier") {
    const t = el.value.replace(/[\s\u00a0\u202f]/g, "").replace(",", ".");
    const x = Number(t);
    const ko = t === "" || !Number.isFinite(x) || (el.dataset.min !== undefined && x < +el.dataset.min) || (el.dataset.max !== undefined && x > +el.dataset.max);
    if (ko) return { ok: false, valeur: el.value };
  }
  return { ok: true, valeur: el.value };
}
// un SEGMENT part s'il n'est pas déjà choisi ; une PUCE à bascule (data-bascule : les espacements admis des dalles)
// part toujours — allumée, elle s'éteint
export function segmentAEnvoyer(b) {
  return !b.classList.contains("on") || !!b.dataset.bascule;
}
// (13/11/2026, le propriétaire : « si je clique sur un 0, je dois l'effacer — je préfère que la case soit vide ») un champ
// NUMÉRIQUE de ``racine`` se VIDE quand on y entre : sa valeur reste écrite en gris (l'indication), on tape directement la
// nouvelle ; quitté sans rien taper, il reprend sa valeur et rien n'est envoyé. L'écouteur de sortie passe AVANT celui qui
// valide (phase de capture) ; une saisie en cours reprise après un calcul n'est pas touchée. Une source : les notes
// (js/etabli.js) et la page « Paramètres ».
export const NUMERIQUE = 'input[data-sorte="nombre"]:not([readonly]):not(:disabled), input[data-sorte="entier"]:not([readonly]):not(:disabled)';
export function viderAuFocus(racine, { signal } = {}) {
  racine.addEventListener("focusin", (ev) => {
    const el = ev.target.closest(NUMERIQUE);
    if (!el || el.dataset.indication !== undefined || el.defaultValue === "") return;
    if (el.value !== el.defaultValue && el.value !== "") return;
    el.dataset.indication = el.placeholder;
    el.placeholder = el.defaultValue;
    el.value = "";
  }, { signal });
  racine.addEventListener("focusout", (ev) => {
    const el = ev.target.closest(NUMERIQUE);
    if (!el || el.dataset.indication === undefined) return;
    if (el.value.trim() === "") el.value = el.defaultValue;
    el.placeholder = el.dataset.indication;
    delete el.dataset.indication;
  }, { capture: true, signal });
}

// L'entrée « bientôt » de la barre (Paramètres) : elle viendra avec les comptes du bureau ; d'ici là, un clic le dit.
// Chaque écran du bureau (l'accueil, les projets, l'équipe) la câble avec le signal de son montage.
export function cablerBientot(signal) {
  document.addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-bientot]");
    if (b) { ev.preventDefault(); toast("Bientôt : avec les comptes du bureau."); }
  }, { signal });
}

let minuterie = null;
// les messages RETENUS pendant la fermeture d'un écran (js/app.js, C1) : le champ qui perd le focus à la fermeture peut
// dire « Valeur refusée » dans un corps qui va être remplacé ; app.js les retient, puis les rend à l'écran suivant
let retenus = null;
export function retenirToasts() { retenus = []; }
export function lacherToasts() { const r = retenus || []; retenus = null; return r; }
// (duree_ms : un message qui doit rester plus longtemps à l'écran — l'échec d'un plan PDF, que l'on doit pouvoir lire en
//  entier et rapporter ; sans elle, 2,6 s, et 7 s pour une erreur. Infinity : un message TENU, qui ne s'efface pas seul —
//  l'attente d'un plan PDF, jusqu'au message suivant ; son écran le retire s'il se ferme avant : retirerToast)
export function toast(texte, erreur = false, duree_ms = 0) {
  if (retenus) { retenus.push([texte, !!erreur]); return; }
  const t = document.getElementById("toast");
  if (!t) { if (erreur) alert(texte); return; }
  t.textContent = texte;
  t.classList.toggle("erreur", !!erreur);
  t.hidden = false;
  clearTimeout(minuterie);
  minuterie = duree_ms === Infinity ? null : setTimeout(() => { t.hidden = true; }, duree_ms > 0 ? duree_ms : (erreur ? 7000 : 2600));
}
// un message tenu s'en va — s'il est encore celui qu'on voit (un autre l'a peut-être remplacé depuis)
export function retirerToast(texte) {
  const t = document.getElementById("toast");
  if (!t || t.hidden || !texte || t.textContent !== texte) return;
  clearTimeout(minuterie);
  t.hidden = true;
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------------------ fichiers à télécharger
// Un nom de fichier sûr, quel qu'il vienne (le moteur, un nom de note) : le dernier morceau d'un chemin (ni « / » ni
// « \ »), sans « .. », sans caractère que Windows refuse ni caractère INVISIBLE (contrôle, mise en forme — l'inversion
// du sens d'écriture U+202E fait lire un « .json » comme un « .exe » —, séparateur de ligne), sans point ni espace aux
// bouts ; l'extension survit à la coupe d'un nom trop long ; « fichier » s'il ne reste rien. (Les mêmes classes que
// commun.nom_fichier, côté Python : Cc, Cf, Zl, Zp.)
export function nomDeFichier(nom, defaut = "fichier") {
  let n = String(nom ?? "").split(/[\\/]/).pop()
    .replace(/[\p{Cc}\p{Cf}\p{Zl}\p{Zp}<>:"|?*]/gu, "_").replace(/\.{2,}/g, ".").replace(/^[\s.]+|[\s.]+$/g, "");
  if (n.length > 150) {
    const ext = (n.match(/\.[\w-]{1,10}$/) || [""])[0];
    n = n.slice(0, 150 - ext.length) + ext;
  }
  return n || defaut;
}

const TYPE_SUR = /^[a-z]+\/[\w.+-]+(;\s*charset=[\w-]+)?$/i;

// Un texte rendu en fichier, enregistré par le navigateur sous ce nom (un Blob : rien ne quitte la page, rien ne
// s'exécute — la politique de sécurité du site n'a rien à permettre de plus).
export function telecharger(nom, texte, type = "application/octet-stream") {
  // un fichier BINAIRE arrive en base 64 : { base64: "…" } (la note PDF, du moteur Python), ou en octets (Uint8Array : les
  // plans, du moteur de RBAR — js/plans.js) ; un texte part tel quel
  const contenu = texte instanceof Uint8Array ? texte
    : texte && typeof texte === "object" && typeof texte.base64 === "string"
      ? Uint8Array.from(atob(texte.base64), (c) => c.charCodeAt(0)) : String(texte ?? "");
  const blob = new Blob([contenu], { type: TYPE_SUR.test(type || "") ? type : "application/octet-stream" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = nomDeFichier(nom);
  a.hidden = true;
  // ce clic-là n'est pas un geste : il ne remonte pas jusqu'à la page, où il refermait la fenêtre de calcul (ou le
  // menu) d'où venait le bouton (verifier_site.js, « télécharger » : la fenêtre reste ouverte)
  a.addEventListener("click", (ev) => ev.stopPropagation());
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  return a.download;
}

// ------------------------------------------------------------------ le taux d'une note, écrit à la française
// Le moteur l'a déjà arrondi (commun.pct) ; l'écran l'écrit comme les pastilles du moteur (commun._taux_txt) : au-delà
// de PLAFOND_TAUX, « > 999 % » et la valeur exacte au survol — un taux de huit chiffres débordait de la pastille de
// l'en-tête du téléphone et de la carte de l'accueil. Le seuil est une COPIE de commun.PLAFOND_TAUX (deux langages) :
// tests/test_site_web.py les lie et joue cette fonction.
export const PLAFOND_TAUX = 999;
export function texteTaux(t) {
  const x = typeof t === "number" ? t : (typeof t === "string" && t.trim() ? Number(t) : NaN);
  if (!Number.isFinite(x)) return { texte: "", titre: "" };
  const [ent, dec] = String(x).split(".");
  const fr = (e) => `${e}${dec ? "," + dec : ""}\u00a0%`;
  if (x <= PLAFOND_TAUX) return { texte: fr(ent), titre: "" };
  return { texte: `>\u00a0${PLAFOND_TAUX}\u00a0%`, titre: fr(ent.replace(/\B(?=(\d{3})+(?!\d))/g, "\u202f")) };
}

// ------------------------------------------------------------------ les lettres grecques dans un texte en capitales
// La charte écrit en capitales les en-têtes de tableau et les titres de groupe (.vtab th, .elgrp, .kicker…) : η y
// devenait « H » et σ « Σ ». Un symbole n'est pas un mot : chaque suite de lettres grecques d'un texte en capitales
// passe dans un <span class="gr"> qui garde sa casse (css/site.css). On lit la casse CALCULÉE, ainsi une feuille de
// module qui met un titre en capitales est couverte aussi.
const GREC = /[Ͱ-Ͽἀ-῿]+/g;
export function garderGrec(racine) {
  if (!racine) return 0;
  const w = document.createTreeWalker(racine, NodeFilter.SHOW_TEXT);
  const noeuds = [];
  for (let n = w.nextNode(); n; n = w.nextNode()) {
    GREC.lastIndex = 0;
    if (GREC.test(n.data) && n.parentElement && !n.parentElement.closest(".gr")
        && getComputedStyle(n.parentElement).textTransform === "uppercase") noeuds.push(n);
  }
  for (const n of noeuds) {
    const f = document.createDocumentFragment();
    let i = 0;
    n.data.replace(GREC, (m, pos) => {
      f.append(n.data.slice(i, pos));
      const s = document.createElement("span");
      s.className = "gr";
      s.textContent = m;
      f.append(s);
      i = pos + m.length;
      return m;
    });
    f.append(n.data.slice(i));
    n.replaceWith(f);
  }
  return noeuds.length;
}

// ------------------------------------------------------------------ le manifeste du moteur (moteur/manifeste.json)
// Lu une fois par document (revalidé : « no-cache ») — la version du site, l'empreinte de chaque aperçu, la version de
// Pyodide ; ``frais`` le relit (le retour sur l'onglet, js/app.js : une mise en ligne pendant la session se voit).
let manifeste = null;
export function lireManifeste(frais = false) {
  if (frais || !manifeste) {
    const p = manifeste = fetch(new URL("../moteur/manifeste.json", import.meta.url), { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`manifeste du moteur : ${r.status}`))));
    p.catch(() => { if (manifeste === p) manifeste = null; });
  }
  return manifeste;
}

// ------------------------------------------------------------------ le démarrage du moteur, visible (lot S4, § 3.4)
// Plus de voile plein écran : l'écran est là tout de suite, en LECTURE SEULE tant que le moteur n'a pas répondu. Le
// bandeau .et-demarre (un composant, le même pour les établis et les bibliothèques, R1) dit « Le moteur de calcul
// démarre », avec un chronomètre (aria-hidden : seul l'état se lit) ; il se pose sur sa RANGÉE pleine largeur, juste sous
// la barre du haut (PC, tablette), ET sous l'en-tête du téléphone (dans .mtabs, collée sous .mtop — visible quel que soit
// l'onglet — ou sous .mhead d'une bibliothèque) : chaque format voit le sien, une seule fois (data-zone). Il était DANS la
// barre du haut, déjà pleine à toutes les largeurs : il y effaçait le nom de la note (à 1 440 px : « Poutre » pour
// « Poutres béton armé »), et en erreur sa raison débordait de l'écran et poussait « Recharger » hors de vue (revue S4,
// ecrans-bandeau-barre-tablette) ; sur sa rangée, la phrase entière se lit et la raison passe à la ligne.
export const TEXTE_DEMARRAGE = "Le moteur de calcul démarre";
// la suite du texte : l'écran est en lecture seule ; « aperçu » seulement quand c'en est un (une note rouverte n'est
// jamais un aperçu, C11 : son dernier enregistrement, montré neutre)
export const SUITE_LECTURE = "lecture seule";
export const SUITE_APERCU = "aperçu en lecture seule";
const ZONES_BANDEAU = [[".top", "afterend", "rangee"], [".mtabs", "afterbegin", "onglets"], [".mhead", "afterend", "entete"]];
// ce qui marche SANS le moteur, et redevient actif quand son démarrage a échoué (revue S4, ecrans-erreur-export) :
// exporter la note telle qu'elle est enregistrée, imprimer l'écran — jamais les champs, ni la note PDF, ni « ranger »
export const ACTIONS_SANS_MOTEUR = '[data-action="exporter"], [data-action="imprimer"]';
// le texte d'un démarrage raté, le même pour les établis et les bibliothèques : une erreur NOMMÉE de version (le site a
// été mis à jour : js/moteur.js, « perime ») se dit seule — elle se lisait « Le moteur n'a pas pu démarrer : Le site a été
// mis à jour : rechargez la page. Recharger la page ; … » — ; « Recharger » est le bouton du bandeau, le texte ne le
// répète pas
export function texteEchecDemarrage(e, moteur) {
  const m = String((e && e.message) || e || "").trim().replace(/[.\s]+$/, "");
  if (moteur && moteur.perime) return `${m}.`;
  return `Le moteur n'a pas pu démarrer : ${m || "raison inconnue"}. Si cela persiste, utilisez Streamlit.`;
}
export function bandeauDemarrage({ texte = TEXTE_DEMARRAGE, signal } = {}) {
  const noeuds = [];
  // « Le moteur de calcul démarre » se lit toujours entier ; ce qui suit (« · aperçu en lecture seule ») cède la place
  const ecrire = (b, t) => {
    const i = t.indexOf(" · ");
    b.querySelector(".et-demarre-t1").textContent = i < 0 ? t : t.slice(0, i);
    b.querySelector(".et-demarre-t2").textContent = i < 0 ? "" : `\u00a0${t.slice(i + 1)}`;
    b.title = t;
  };
  for (const [sel, ou, zone] of ZONES_BANDEAU) {
    const z = document.querySelector(sel);
    if (!z) continue;
    const b = document.createElement("div");
    b.className = "et-demarre";
    b.dataset.zone = zone;
    b.setAttribute("role", "status");
    b.innerHTML = '<span class="et-demarre-ic" aria-hidden="true"></span>'
      + '<span class="et-demarre-txt"><span class="et-demarre-t1"></span><span class="et-demarre-t2"></span></span>'
      + '<span class="et-demarre-chrono" aria-hidden="true"></span>'
      + '<button type="button" class="btn sm et-demarre-recharger" data-recharger="1" hidden>Recharger</button>';
    ecrire(b, texte);
    if (ou === "afterend") z.after(b);
    else z.prepend(b);
    noeuds.push(b);
  }
  const t0 = performance.now();
  const chrono = () => { const s = Math.floor((performance.now() - t0) / 1000); noeuds.forEach((b) => { b.querySelector(".et-demarre-chrono").textContent = `${s} s`; }); };
  chrono();
  let minuterie = setInterval(chrono, 1000);
  const arreter = () => { if (minuterie) { clearInterval(minuterie); minuterie = null; } };
  if (signal) signal.addEventListener("abort", arreter, { once: true });
  noeuds.forEach((b) => b.querySelector("[data-recharger]").addEventListener("click", () => location.reload(), { signal }));
  return {
    // le moteur a répondu : le bandeau s'en va
    retirer() { arreter(); noeuds.forEach((b) => b.remove()); },
    // le texte du démarrage change (l'aperçu peint : « · aperçu en lecture seule ») ; rien une fois en erreur
    texte(t) { noeuds.forEach((b) => { if (b.dataset.etat !== "erreur") ecrire(b, t); }); },
    // le démarrage a échoué : le bandeau reste, en rouge, avec la raison et « Recharger »
    erreur(raison) {
      arreter();
      noeuds.forEach((b) => {
        b.dataset.etat = "erreur";
        b.querySelector(".et-demarre-t1").textContent = raison;
        b.querySelector(".et-demarre-t2").textContent = "";
        b.title = raison;
        b.querySelector(".et-demarre-chrono").textContent = "";
        b.querySelector("[data-recharger]").hidden = false;
      });
    },
  };
}

// Les CONTRÔLES d'un écran en lecture seule (inert) : les champs et boutons des zones nommées — pas les panneaux, qui
// restent lisibles, sélectionnables et défilables (aria-busy) ; ni la navigation (liens, rail, thème, le bouton ⋯ du
// téléphone qui ouvre le menu où sont le thème, les modules et « Ouvrir .stab », la croix ✕ qui ferme une fenêtre de
// calcul : fermer, c'est naviguer — revue S4, ecrans-menu-telephone-inerte, ecrans-volet-croix-inerte). Rend de quoi
// les rendre : les contrôles ajoutés depuis (une réponse du moteur refait les cadres) sont relus aussi ; rendre(sel)
// rend actifs pour de bon ceux qui marchent sans le moteur (un démarrage raté : ACTIONS_SANS_MOTEUR).
const CONTROLES = ":is(input, select, textarea, button, [data-action])"
  + ":not([data-theme-toggle]):not([data-modules]):not([data-recharger]):not([aria-controls=\"mmenu\"]):not([data-fermer])";
export function lectureSeule(zones, panneaux) {
  const rendus = [];
  const poser = () => {
    for (const z of zones) {
      document.querySelectorAll(`${z} ${CONTROLES}`).forEach((e) => {
        if (!e.closest("a[href]") && !rendus.some((s) => e.matches(s))) e.setAttribute("inert", "");
      });
    }
  };
  poser();
  panneaux.forEach((s) => document.querySelectorAll(s).forEach((p) => p.setAttribute("aria-busy", "true")));
  return {
    reposer: poser,
    rendre(sel) {
      rendus.push(sel);
      for (const z of zones) document.querySelectorAll(`${z} [inert]`).forEach((e) => { if (e.matches(sel)) e.removeAttribute("inert"); });
    },
    lever() {
      for (const z of zones) document.querySelectorAll(`${z} [inert]`).forEach((e) => e.removeAttribute("inert"));
      panneaux.forEach((s) => document.querySelectorAll(s).forEach((p) => p.removeAttribute("aria-busy")));
    },
  };
}
