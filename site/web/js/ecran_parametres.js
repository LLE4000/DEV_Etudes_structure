// La page PARAMÈTRES (lot 6, 14/11/2026 ; docs/site/MODELES_DIMENSIONNEMENT.md) : les options du SITE (le thème), puis le
// GABARIT DU BUREAU — les options de départ de chaque module, fixées par un administrateur, que chaque note NEUVE reçoit
// (« on prend la longueur de recouvrement et on l'arrondit tout le temps au 5 cm près ou à 10 cm près » ; une note peut
// toujours changer une valeur dans sa rubrique Options).
//
// La rubrique Options de chaque module est RENDUE PAR LE MOTEUR (l'action « gabarit », api_web/gabarit.py) : le même
// composant que dans la note (R1, R2) — mêmes champs, mêmes bornes, mêmes refus. Chaque saisie part au moteur, qui la
// contrôle et rend les ÉCARTS à l'usine : js/gabarit.js les garde, rien d'autre. Un membre lit le gabarit ; seul un
// administrateur le change (js/equipe.js). La saisie se lit comme dans une note (commun.valeurSaisie, viderAuFocus,
// segmentAEnvoyer : une source).
//
// Un ÉCRAN MONTABLE (comme js/ecran_equipe.js) : aucun effet au chargement ; js/app.js appelle ouvrirParametres(hote, ctx).
// Il attend le moteur (le seul écran du bureau qui le demande) et n'écrit jamais l'historique.
import * as GB from "./gabarit.js";
import * as E from "./equipe.js";
import * as J from "./journal.js";
import { MODULES } from "./modules.js";
import { esc, toast, poserTheme, themeChoisi, valeurSaisie, segmentAEnvoyer, viderAuFocus, texteEchecDemarrage } from "./commun.js";

export function ouvrirParametres(hote, ctx) {
  const { signal } = ctx;
  const $ = (id) => document.getElementById(id);
  const client = ctx.moteur.client();
  const mods = [...document.querySelectorAll(".pa-mod[data-gabarit]")].map((c) => c.dataset.gabarit);
  const zone = document.querySelector(".pa-mods");
  const notes = new Set();                       // les modules dont le changement est au journal (une ligne par visite)
  const admin = () => E.estAdmin(E.moi());
  hote.dataset.ecran = "demarrage";

  // ------------------------------------------------------------------ le site : le thème
  function afficherTheme() {
    const t = themeChoisi();
    document.querySelectorAll("#pa-theme [data-theme-choix]").forEach((b) => {
      const on = b.dataset.themeChoix === t;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
  }
  $("pa-theme").addEventListener("click", (ev) => {
    const b = ev.target.closest("[data-theme-choix]");
    if (!b) return;
    poserTheme(b.dataset.themeChoix);
    afficherTheme();
  }, { signal });
  // (le bouton du rail et celui de l'en-tête du téléphone basculent aussi le thème : la page suit)
  document.addEventListener("click", (ev) => { if (ev.target.closest("[data-theme-toggle]")) requestAnimationFrame(afficherTheme); }, { signal });
  afficherTheme();

  // ------------------------------------------------------------------ qui peut le changer
  function afficherRole() {
    const moi = E.moi(), d = GB.dernier();
    const quand = d ? new Date(d.t).toLocaleDateString("fr-BE", { day: "2-digit", month: "2-digit", year: "numeric" }) : "";
    $("pa-role").innerHTML = (admin() ? "Vous êtes <b>administrateur</b> : vous changez le gabarit pour ce poste."
      : moi ? "Seul un <b>administrateur</b> change le gabarit : vous le lisez."
        : 'Seul un <b>administrateur</b> change le gabarit — présentez-vous sur la page <a href="equipe">Équipe</a> (le premier inscrit du poste l\'est).')
      + (d && d.par ? ` <span class="muted">Changé en dernier par ${esc(d.par.nom)} (${esc(d.par.ini)}), le ${esc(quand)}.</span>` : "");
    $("pa-qui").textContent = d && d.par ? `Gabarit : ${d.par.ini} · ${quand}` : "";
  }

  // ------------------------------------------------------------------ un module : sa rubrique Options, par le moteur
  function peindre(k, r) {
    const z = document.querySelector(`[data-options="${k}"]`);
    z.innerHTML = r.html;
    const n = Object.keys(r.ecarts || {}).length;
    document.querySelector(`[data-ecarts="${k}"]`).textContent = n ? `${n} valeur${n > 1 ? "s" : ""} du bureau, encadrée${n > 1 ? "s" : ""}` : "valeurs d'usine";
    // une valeur du BUREAU (un écart à l'usine) : son CONTRÔLE est encadré (le champ, les segments, les puces, la case),
    // l'usine au survol
    for (const c of Object.keys(r.ecarts || {})) {
      const u = r.usine ? r.usine[c] : undefined;
      const usine = u === undefined ? "" : Array.isArray(u) ? `${u.map((x) => String(x / 10).replace(".", ",")).join(", ")} cm`
        : typeof u === "boolean" ? (u ? "oui" : "non") : String(u).replace(".", ",");
      const vus = new Set();
      z.querySelectorAll(`[data-chemin="${CSS.escape(c)}"]`).forEach((el) => {
        const boite = el.closest(".field, .seg, .et-puces, .case");
        if (!boite || vus.has(boite)) return;
        vus.add(boite);
        boite.classList.add("pa-bureau");
        if (usine) boite.title = `Valeur du bureau — l'usine : ${usine}`;
      });
    }
    const usine = document.querySelector(`[data-usine="${k}"]`);
    usine.hidden = !(admin() && n);
    // un membre LIT : aucun contrôle ne répond (la page le dit en tête)
    if (!admin()) z.querySelectorAll("input, select, button, textarea").forEach((el) => { el.disabled = true; });
    z.classList.toggle("pa-lecture", !admin());
  }

  async function rendre(k, saisie = null) {
    const q = { module: k, action: "gabarit", gabarit: GB.lire(k) || {} };
    if (saisie) Object.assign(q, saisie);
    let r;
    try { r = await client.demander(q); } catch (e) { if (e && e.name === "EcranFerme") return null; throw e; }
    if (signal.aborted) return null;
    if (!r || !r.ok) { toast(`Le moteur a refusé : ${(r && r.erreur) || "pas de réponse"}`, true); return null; }
    if (saisie && r.message) toast(r.message, true);                       // refusée : le gabarit ne change pas
    else if (saisie && JSON.stringify(r.ecarts) !== JSON.stringify(GB.lire(k) || {})) {
      if (!GB.ecrire(k, r.ecarts)) { toast("Le gabarit n'a pas pu être enregistré dans ce navigateur.", true); return null; }
      if (!notes.has(k)) { notes.add(k); J.noter("gabarit.change", { module: MODULES[k].nom }); }
      afficherRole();
    }
    // un gabarit gardé que le moteur n'admet plus (une version plus récente du module) : dit, l'usine vaut
    if (r.refus && r.refus.length) toast(r.refus.join(" "), true);
    peindre(k, r);
    return r;
  }

  // ------------------------------------------------------------------ la saisie (un administrateur)
  const moduleDe = (el) => (el.closest("[data-gabarit]") || {}).dataset?.gabarit;
  function valider(el) {
    if (!admin()) return;
    const v = valeurSaisie(el);
    if (el.dataset.sorte === "nombre" || el.dataset.sorte === "entier") el.closest(".field")?.classList.toggle("erreur", !v.ok);
    if (!v.ok) { toast(`Valeur refusée : « ${el.value} »`, true); return; }
    rendre(moduleDe(el), { chemin: el.dataset.chemin.split("/"), valeur: v.valeur });
  }
  viderAuFocus(zone, { signal });
  zone.addEventListener("change", (ev) => {
    const el = ev.target.closest("select[data-chemin], input[type=checkbox][data-chemin]");
    if (el) valider(el);
  }, { signal });
  zone.addEventListener("focusout", (ev) => {
    const el = ev.target.closest("input[data-chemin]:not([type=checkbox])");
    if (el && el.value !== el.defaultValue) valider(el);
  }, { signal });
  zone.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && ev.target.matches("input[data-chemin]")) ev.target.blur();
  }, { signal });
  zone.addEventListener("click", (ev) => {
    const u = ev.target.closest("[data-usine]");
    if (u && admin()) {
      const k = u.dataset.usine;
      if (!GB.ecrire(k, {})) { toast("Le gabarit n'a pas pu être enregistré dans ce navigateur.", true); return; }
      J.noter("gabarit.usine", { module: MODULES[k].nom });
      toast(`${MODULES[k].nom} : les valeurs d'usine pour les notes neuves.`);
      afficherRole();
      rendre(k);
      return;
    }
    const seg = ev.target.closest("button[data-chemin][data-valeur]");
    if (seg && admin() && !seg.disabled && segmentAEnvoyer(seg)) rendre(moduleDe(seg), { chemin: seg.dataset.chemin.split("/"), valeur: seg.dataset.valeur });
  }, { signal });

  // ------------------------------------------------------------------ le démarrage : le moteur, puis chaque module
  afficherRole();
  ctx.titre("Paramètres · Stab Méthodes");
  (async () => {
    try {
      await ctx.moteur.pret;
      if (signal.aborted) return;
      for (const k of mods) if (!(await rendre(k))) throw new Error(`le gabarit « ${k} » n'a pas pu être rendu`);
      if (signal.aborted) return;
      hote.dataset.ecran = "pret";
      ctx.annoncer("prêt");
      if (typeof ctx.rendreDefilement === "function") ctx.rendreDefilement();
    } catch (e) {
      if (signal.aborted) return;
      hote.dataset.ecran = "erreur";
      const raison = texteEchecDemarrage(e, ctx.moteur);
      document.querySelectorAll(".pa-options .pa-attente").forEach((p) => { p.textContent = raison; p.classList.add("bu-erreur"); });
      ctx.annoncer(raison);
    }
  })();
  return { fermer() { return { note: null, enregistree: client.fermer().then(() => ({ refus: [] })) }; } };
}
