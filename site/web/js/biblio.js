// La BIBLIOTHÈQUE câblée dans son hôte (docs/site/BIBLIOTHEQUES.md § 2.2 et § 2.4) : la page bibliotheque-<famille>
// (js/bibliotheque.js, hôte « page ») et le panneau « Bibliothèque » d'une note (js/etabli.js, hôte « note »). Tout le
// HTML vient du moteur (api_web/biblio_<famille>.consulter, par le routeur) : ce fichier pose les fragments, relit les
// paramètres, allume la rubrique du menu et tient le volet « Pourquoi ? ». Il ne calcule rien et n'écrit aucune note.
import { toast } from "./commun.js";

// racine : l'élément qui contient le gabarit (document pour la page, le panneau pour une note) ; demander(rubrique,
// params) rend la réponse du moteur ; apres(r) : ce que l'hôte fait de la réponse (l'adresse et le titre de la page,
// les boutons de la note) ; rubrique : celle à ouvrir d'abord (sinon celle que le menu allume)
export function cablerBibliotheque({ racine, demander, apres = () => {}, rubrique = null }) {
  const $ = (s) => racine.querySelector(s);
  const corps = $("#bib-corps"), pourquoi = $("#bib-pourquoi");
  const page = racine === document;           // la page bloque le défilement derrière le volet ; le panneau est déjà modal
  let courante = rubrique || $(".bib-m.on")?.dataset.rubrique || $(".bib-m")?.dataset.rubrique;
  const params = {};        // les paramètres de chaque rubrique visitée (gardés quand on en change)
  let derniere = null;      // la dernière réponse du moteur (sa fiche, ses cibles)
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
      const r = await demander(courante, p);
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
    courante = r.rubrique;
    params[courante] = r.params;                           // les paramètres relus par le moteur (défauts compris)
    derniere = r;
    racine.querySelectorAll(".bib-m").forEach((b) => {
      const on = b.dataset.rubrique === courante;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    const titre = $("#bib-titre");
    if (titre) titre.textContent = r.titre;
    $("#bib-barre").innerHTML = r.barre;
    $("#bib-tableau").innerHTML = r.tableau;
    $("#bib-bande").innerHTML = r.bande;
    $("#bib-sources").innerHTML = r.sources;
    $("#bib-pq-corps").innerHTML = r.pourquoi;
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
  racine.addEventListener("click", (ev) => {
    const m = ev.target.closest(".bib-m[data-rubrique]");
    if (m) { courante = m.dataset.rubrique; consulter(); return; }
    const seg = ev.target.closest(".bib-barre .seg[data-param] button[data-valeur]");
    if (seg) { regler(seg.parentElement.dataset.param, seg.dataset.valeur); return; }
    const td = ev.target.closest("#bib-tableau td[data-sel]");
    if (td) {
      let sel = null;
      try { sel = JSON.parse(td.dataset.sel); } catch (e) { sel = null; }
      if (sel && typeof sel === "object") { Object.assign(params[courante] || (params[courante] = {}), sel); consulter(); }
      return;
    }
    if (ev.target.closest("[data-pourquoi]")) { ouvrirPourquoi(true); return; }
    if (ev.target.closest("[data-fermer-pourquoi]") || ev.target === pourquoi) ouvrirPourquoi(false);
  });
  racine.addEventListener("change", (ev) => {
    const el = ev.target.closest(".bib-barre [data-param]");
    if (!el) return;
    if (el.type === "checkbox") regler(el.dataset.param, el.checked ? "1" : "0");
    else regler(el.dataset.param, el.value);
  });
  racine.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target.matches(".bib-barre input[data-param]")) { ev.target.blur(); return; }
    // Échap referme « Pourquoi ? » d'abord ; l'hôte ne voit pas cet Échap-là (il refermerait son panneau en même temps)
    if (ev.key === "Escape" && !pourquoi.hidden) { ev.stopPropagation(); ouvrirPourquoi(false); }
  });
  // un champ reçoit le focus : son cadre le montre (comme les champs de l'établi)
  racine.addEventListener("focusin", (ev) => { const f = ev.target.closest(".bib-barre .field"); if (f) f.classList.add("focus"); });
  racine.addEventListener("focusout", (ev) => { const f = ev.target.closest(".bib-barre .field"); if (f) f.classList.remove("focus"); });

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
