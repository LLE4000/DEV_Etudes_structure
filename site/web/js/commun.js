// Ce que toutes les pages partagent : le thème (clair, sombre), le volet des modules, les messages éphémères.
export function cablerRail() {
  const racine = document.documentElement;
  document.addEventListener("click", (ev) => {
    const t = ev.target.closest("[data-theme-toggle]");
    if (t) {
      const sombre = racine.dataset.theme === "dark" || (!racine.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      racine.dataset.theme = sombre ? "light" : "dark";
      try { localStorage.setItem("stab.theme", racine.dataset.theme); } catch (e) { /* navigation privée */ }
      return;
    }
    const m = ev.target.closest("[data-modules]");
    if (m) { document.body.classList.toggle("show-modules"); return; }
    if (!ev.target.closest(".modpanel")) document.body.classList.remove("show-modules");
  });
}

let minuterie = null;
export function toast(texte, erreur = false) {
  const t = document.getElementById("toast");
  if (!t) { if (erreur) alert(texte); return; }
  t.textContent = texte;
  t.classList.toggle("erreur", !!erreur);
  t.hidden = false;
  clearTimeout(minuterie);
  minuterie = setTimeout(() => { t.hidden = true; }, erreur ? 7000 : 2600);
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
  // un fichier BINAIRE (la note PDF) arrive en base 64 : { base64: "…" } ; un texte part tel quel
  const contenu = texte && typeof texte === "object" && typeof texte.base64 === "string"
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
