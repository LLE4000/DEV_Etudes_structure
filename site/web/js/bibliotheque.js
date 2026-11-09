// La page d'une BIBLIOTHÈQUE (site/web/construire.py, page_bibliotheque ; docs/site/BIBLIOTHEQUES.md § 2.2) : à gauche
// le menu des rubriques, à droite la barre de paramètres, le tableau, la bande de la valeur retenue, les sources. Tout
// le HTML vient du moteur (api_web/biblio_<famille>.consulter, par l'action « consulter » du routeur) ; ce script ne
// fait que poser les fragments, relire les paramètres et tenir l'adresse (?rubrique=…&…) : une adresse se partage.
// Aucune note n'est créée ni enregistrée : une consultation n'en est pas une.
import { Moteur } from "./moteur.js";
import { cablerRail, toast } from "./commun.js";

cablerRail();
const $ = (s, r = document) => r.querySelector(s);
const BIB = document.body.dataset.bibliotheque;          // « beton »
const corps = $("#bib-corps");
const etapes = $("#demarrage-etape");
const pourquoi = $("#bib-pourquoi");
const moteur = new Moteur((t) => { if (etapes) etapes.textContent = t; }, "");

// l'adresse ouvre une rubrique avec ses paramètres ; sans elle, la première rubrique du menu
const q = new URLSearchParams(location.search);
let rubrique = q.get("rubrique") || $(".bib-m.on")?.dataset.rubrique || $(".bib-m")?.dataset.rubrique;
const params = {};                                        // les paramètres de chaque rubrique visitée (gardés quand on en change)
params[rubrique] = Object.fromEntries([...q.entries()].filter(([k]) => k !== "rubrique"));

let enCours = 0;
function occupe(plus) {
  enCours += plus;
  corps.setAttribute("aria-busy", enCours > 0 ? "true" : "false");
}

async function consulter() {
  const p = params[rubrique] || (params[rubrique] = {});
  occupe(1);
  try {
    const r = await moteur.demander({ action: "consulter", bibliotheque: BIB, rubrique, params: p });
    if (!r.ok) { toast(`Le moteur a refusé : ${r.erreur}`, true); console.error(r.trace || r.erreur); return null; }
    appliquer(r);
    return r;
  } finally { occupe(-1); }
}

function appliquer(r) {
  rubrique = r.rubrique;
  params[rubrique] = r.params;                            // les paramètres relus par le moteur (défauts compris)
  document.querySelectorAll(".bib-m").forEach((b) => {
    const on = b.dataset.rubrique === rubrique;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
  $("#bib-titre").textContent = r.titre;
  $("#bib-barre").innerHTML = r.barre;
  $("#bib-tableau").innerHTML = r.tableau;
  $("#bib-bande").innerHTML = r.bande;
  $("#bib-sources").innerHTML = r.sources;
  $("#bib-pq-corps").innerHTML = r.pourquoi;
  if (r.message) toast(r.message, true);
  // l'adresse suit : la rubrique, puis chaque paramètre
  const u = new URL(location.href);
  u.search = "";
  u.searchParams.set("rubrique", rubrique);
  for (const [k, v] of Object.entries(r.params)) if (v !== null && v !== undefined && v !== "") u.searchParams.set(k, String(v));
  history.replaceState(null, "", u);
  document.title = `${r.titre} · ${document.body.dataset.titre} · Stab Méthodes`;
  // la case allumée se voit (le tableau défile dans son cadre au téléphone)
  const cur = $("#bib-tableau .cur");
  if (cur) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function regler(cle, valeur) {
  const p = params[rubrique] || (params[rubrique] = {});
  p[cle] = valeur;
  consulter();
}

// ------------------------------------------------------------------ les gestes
document.addEventListener("click", (ev) => {
  const m = ev.target.closest(".bib-m[data-rubrique]");
  if (m) { rubrique = m.dataset.rubrique; consulter(); return; }
  const seg = ev.target.closest(".bib-barre .seg[data-param] button[data-valeur]");
  if (seg) { regler(seg.parentElement.dataset.param, seg.dataset.valeur); return; }
  const td = ev.target.closest("#bib-tableau td[data-sel]");
  if (td) {
    let sel = null;
    try { sel = JSON.parse(td.dataset.sel); } catch (e) { sel = null; }
    if (sel && typeof sel === "object") { Object.assign(params[rubrique] || (params[rubrique] = {}), sel); consulter(); }
    return;
  }
  if (ev.target.closest("[data-pourquoi]")) { ouvrirPourquoi(true); return; }
  if (ev.target.closest("[data-fermer-pourquoi]") || (ev.target === pourquoi)) { ouvrirPourquoi(false); }
});
document.addEventListener("change", (ev) => {
  const el = ev.target.closest(".bib-barre [data-param]");
  if (!el) return;
  if (el.type === "checkbox") regler(el.dataset.param, el.checked ? "1" : "0");
  else regler(el.dataset.param, el.value);
});
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && ev.target.matches(".bib-barre input[data-param]")) { ev.target.blur(); return; }
  if (ev.key === "Escape" && !pourquoi.hidden) ouvrirPourquoi(false);
});
// un champ reçoit le focus : son cadre le montre (comme les champs de l'établi)
document.addEventListener("focusin", (ev) => { const f = ev.target.closest(".field"); if (f) f.classList.add("focus"); });
document.addEventListener("focusout", (ev) => { const f = ev.target.closest(".field"); if (f) f.classList.remove("focus"); });

function ouvrirPourquoi(on) {
  pourquoi.hidden = !on;
  document.documentElement.classList.toggle("volet-ouvert", on);
  if (on) $("#bib-pourquoi [data-fermer-pourquoi]")?.focus();
}

// ------------------------------------------------------------------ le démarrage
moteur.pret.then(() => {
  $("#demarrage").hidden = true;
  return consulter();
}).catch((e) => {
  const d = $("#demarrage");
  d.classList.add("erreur");
  if (etapes) etapes.textContent = `Le moteur n'a pas démarré : ${e.message}`;
});
