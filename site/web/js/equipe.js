// L'ÉQUIPE du bureau, enregistrée DANS CE NAVIGATEUR en attendant les comptes partagés (docs/site/PROJETS_EQUIPE.md § 3).
// Un membre : { id, nom, email, initiales (1 à 5 lettres ou chiffres, en capitales : « LLE »), couleur (0 à 7), cree }.
// « moi » : le membre qui travaille sur ce poste — c'est sa SIGNATURE ({ id, ini, nom, c }) que portent les notes qu'il
// crée ou modifie et les lignes du journal (js/journal.js). La signature est une COPIE prise au moment du geste : un
// membre retiré ou renommé depuis garde ses initiales sur ce qu'il a fait.
//
// Aucun effet au chargement ; pas de document ici (ces fonctions tournent aussi dans node : tests/test_site_bureau.py).
import * as M from "./memoire.js";

export const CLE = "stab.equipe.v1";
export const INITIALES_MAX = 5;
export const COULEURS = 8;                       // les teintes des pastilles : css/bureau.css, .avatar.c0 à .avatar.c7
const NOM_MAX = 80, EMAIL_MAX = 120;

const vide = () => ({ moi: null, membres: {} });
const valide = (v) => !!v && typeof v === "object" && !Array.isArray(v) && !!v.membres && typeof v.membres === "object" && !Array.isArray(v.membres);

const propre = (t, max) => String(t ?? "").replace(/\s+/g, " ").trim().slice(0, max);

// « l.le » → « LLE » ; « » si ce ne sont pas 1 à 5 lettres ou chiffres (les espaces, points et tirets sont retirés)
export function initialesValides(t) {
  const s = String(t ?? "").replace(/[\s.\-_']/g, "").toLocaleUpperCase("fr");
  return /^[\p{L}\p{N}]{1,5}$/u.test(s) ? s : "";
}

// une adresse plausible (quelque chose, une arobase, un domaine avec un point) — pas une vérification d'existence
export function emailValide(t) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(String(t ?? "").trim());
}

// Les initiales PROPOSÉES pour un nom, l'usage des bureaux d'études : l'initiale du prénom et les deux premières lettres
// du nom (« Louis Lejeune » → LLE) ; trois mots et plus : l'initiale de chacun (« Jean-Marc Dupont » → JMD) ; un seul
// mot : ses trois premières lettres. ``prises`` : celles des autres membres — on en propose une libre.
export function proposerInitiales(nom, prises = []) {
  const mots = String(nom ?? "").split(/[\s\-'’.]+/).map((m) => m.replace(/[^\p{L}\p{N}]/gu, "")).filter(Boolean);
  if (!mots.length) return "";
  const M_ = mots.map((m) => m.toLocaleUpperCase("fr"));
  const dernier = M_[M_.length - 1];
  const essais = [];
  if (M_.length === 1) essais.push(M_[0].slice(0, 3), M_[0].slice(0, 2), M_[0].slice(0, 4));
  else if (M_.length === 2) essais.push(M_[0][0] + dernier.slice(0, 2), M_[0][0] + dernier[0], M_[0][0] + dernier[0] + (dernier[2] || ""), M_[0].slice(0, 2) + dernier[0], M_[0][0] + dernier.slice(0, 3));
  else essais.push(M_.map((m) => m[0]).join("").slice(0, INITIALES_MAX), M_[0][0] + dernier.slice(0, 2));
  const deja = new Set(prises.map((p) => String(p).toLocaleUpperCase("fr")));
  const bons = essais.map(initialesValides).filter(Boolean);
  const libre = bons.find((e) => !deja.has(e));
  if (libre) return libre;
  for (let n = 2; n < 100; n++) {
    const e = initialesValides((bons[0] || "X").slice(0, INITIALES_MAX - String(n).length) + n);
    if (e && !deja.has(e)) return e;
  }
  return bons[0] || "";
}

// la teinte d'un membre neuf : la moins portée de l'équipe (les pastilles se distinguent tant qu'on est huit ou moins)
function couleurLibre(membres) {
  const n = new Array(COULEURS).fill(0);
  for (const m of Object.values(membres)) if (Number.isInteger(m.couleur) && m.couleur >= 0 && m.couleur < COULEURS) n[m.couleur]++;
  return n.indexOf(Math.min(...n));
}

function lu() {
  const m = M.lire(CLE, vide, valide);
  const e = m.valeur;
  // on ne garde que des membres bien formés (un registre réécrit à la main, un ancien format)
  const membres = {};
  for (const [id, x] of Object.entries(e.membres || {})) {
    if (!x || typeof x !== "object") continue;
    const ini = initialesValides(x.initiales), nom = propre(x.nom, NOM_MAX);
    if (!ini || !nom) continue;
    membres[id] = { id, nom, email: propre(x.email, EMAIL_MAX), initiales: ini,
                    couleur: Number.isInteger(x.couleur) && x.couleur >= 0 && x.couleur < COULEURS ? x.couleur : 0, cree: Number(x.cree) || 0 };
  }
  return { etat: m.etat, moi: Object.hasOwn(membres, e.moi) ? e.moi : null, membres };
}

// L'équipe, à LIRE : { etat, moi (le membre, ou null), membres (triés par nom) }
export function equipe() {
  const e = lu();
  const membres = Object.values(e.membres).sort((a, b) => a.nom.localeCompare(b.nom, "fr"));
  return { etat: e.etat, moi: e.moi ? e.membres[e.moi] : null, membres };
}

export function moi() { return equipe().moi; }

// la copie qui signe un geste : { id, ini, nom, c } — ou null tant que personne ne s'est présenté sur ce poste
export const signatureDe = (m) => (m ? { id: m.id, ini: m.initiales, nom: m.nom, c: m.couleur } : null);
export function signature() { return signatureDe(moi()); }

// Une signature venue d'AILLEURS (un .stab, un journal réécrit) : on n'en garde que des textes courts et un entier.
export function signaturePropre(s) {
  if (!s || typeof s !== "object") return null;
  const ini = initialesValides(s.ini);
  if (!ini) return null;
  const c = Number.isInteger(s.c) && s.c >= 0 && s.c < COULEURS ? s.c : 0;
  return { id: typeof s.id === "string" ? s.id.slice(0, 40) : "", ini, nom: propre(s.nom, NOM_MAX) || ini, c };
}

// Ce qui manque ou cloche dans une fiche : { nom, initiales, email } → messages (« » : rien à dire).
function controler(membres, { nom, initiales, email }, saufId = null) {
  const autres = Object.values(membres).filter((m) => m.id !== saufId);
  if (!nom) return { champ: "nom", erreur: "Le nom manque." };
  if (!initiales) return { champ: "initiales", erreur: `Les initiales : de 1 à ${INITIALES_MAX} lettres ou chiffres.` };
  const pris = autres.find((m) => m.initiales === initiales);
  if (pris) return { champ: "initiales", erreur: `Les initiales ${initiales} sont déjà celles de ${pris.nom}.` };
  if (email && !emailValide(email)) return { champ: "email", erreur: "L'adresse e-mail n'est pas complète." };
  const meme = email && autres.find((m) => m.email && m.email.toLocaleLowerCase("fr") === email.toLocaleLowerCase("fr"));
  if (meme) return { champ: "email", erreur: `Cette adresse est déjà celle de ${meme.nom}.` };
  return null;
}

// Ajouter un membre. ``estMoi`` : c'est la personne de ce poste (elle se présente). Rend { ok, membre } ou { ok: false,
// champ, erreur }. Des initiales absentes sont proposées d'après le nom.
export function ajouter({ nom, email = "", initiales = "", couleur = null }, { estMoi = false } = {}) {
  const e = lu();
  const f = { nom: propre(nom, NOM_MAX), email: propre(email, EMAIL_MAX), initiales: "" };
  f.initiales = String(initiales ?? "").trim() ? initialesValides(initiales) : proposerInitiales(f.nom, Object.values(e.membres).map((m) => m.initiales));
  const pb = controler(e.membres, f);
  if (pb) return { ok: false, ...pb };
  const membre = { id: M.ident(), ...f, couleur: Number.isInteger(couleur) && couleur >= 0 && couleur < COULEURS ? couleur : couleurLibre(e.membres), cree: Date.now() };
  const ok = M.changer(CLE, vide, valide, (v) => {
    v.membres[membre.id] = membre;
    if (estMoi) v.moi = membre.id;
  });
  return ok ? { ok: true, membre } : { ok: false, champ: "", erreur: "L'équipe n'a pas pu être enregistrée dans ce navigateur." };
}

// Modifier la fiche d'un membre (les champs donnés seulement).
export function modifier(id, champs) {
  const e = lu();
  const m = e.membres[id];
  if (!m) return { ok: false, champ: "", erreur: "Ce membre n'est plus dans l'équipe." };
  const f = { nom: "nom" in champs ? propre(champs.nom, NOM_MAX) : m.nom, email: "email" in champs ? propre(champs.email, EMAIL_MAX) : m.email,
              initiales: "initiales" in champs ? initialesValides(champs.initiales) : m.initiales };
  const pb = controler(e.membres, f, id);
  if (pb) return { ok: false, ...pb };
  const couleur = Number.isInteger(champs.couleur) && champs.couleur >= 0 && champs.couleur < COULEURS ? champs.couleur : m.couleur;
  const ok = M.changer(CLE, vide, valide, (v) => { v.membres[id] = { ...m, ...f, couleur }; });
  return ok ? { ok: true, membre: { ...m, ...f, couleur } } : { ok: false, champ: "", erreur: "L'équipe n'a pas pu être enregistrée dans ce navigateur." };
}

export function retirer(id) {
  return M.changer(CLE, vide, valide, (v) => { delete v.membres[id]; if (v.moi === id) v.moi = null; });
}

// « Je suis… » : la personne qui travaille sur ce poste (null : personne)
export function choisirMoi(id) {
  return M.changer(CLE, vide, valide, (v) => { v.moi = id && Object.hasOwn(v.membres, id) ? id : null; });
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// La PASTILLE d'une signature (ou d'un membre) : ses initiales dans un rond de sa teinte ; sans signature, un rond
// neutre « ? » (quelqu'un qui ne s'est pas présenté). ``cls`` : une taille (« sm », « lg », « xl ») ou rien. La classe
// « avatar » est celle de la charte (site/maquettes/charte.css) ; « n4 », « n5 » : quatre ou cinq lettres, le rond s'allonge.
export function pastille(s, cls = "") {
  const sig = s && s.initiales ? signatureDe(s) : signaturePropre(s);
  if (!sig) return `<span class="avatar inconnu${cls ? ` ${cls}` : ""}" title="Quelqu'un qui ne s'est pas présenté sur ce poste" aria-label="Auteur inconnu">?</span>`;
  return `<span class="avatar c${sig.c} n${sig.ini.length}${cls ? ` ${cls}` : ""}" title="${esc(sig.nom)}" aria-label="${esc(sig.nom)}">${esc(sig.ini)}</span>`;
}

// L'adresse avec laquelle la personne est ENTRÉE sur le site (Cloudflare Access garde le site privé et connaît
// l'adresse de qui est connecté) : elle préremplit le champ e-mail, rien de plus. Hors de Cloudflare (le serveur local,
// un essai), l'adresse n'existe pas : « », sans bruit. Jamais d'attente de plus de 3 s.
export const ADRESSE_IDENTITE = "/cdn-cgi/access/get-identity";   // même origine (connect-src 'self')
export async function emailDeConnexion() {
  try {
    const fin = new AbortController();
    const delai = setTimeout(() => fin.abort(), 3000);
    const r = await fetch(ADRESSE_IDENTITE, { credentials: "same-origin", signal: fin.signal, headers: { Accept: "application/json" } });
    clearTimeout(delai);
    if (!r.ok || !(r.headers.get("content-type") || "").includes("json")) return "";
    const d = await r.json();
    const email = d && typeof d.email === "string" ? d.email.trim().slice(0, EMAIL_MAX) : "";
    return emailValide(email) ? email : "";
  } catch (e) { return ""; }
}
