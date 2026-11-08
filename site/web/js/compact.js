// L'établi COMPACT (pilote : « Poutres béton armé » ; css/etabli.css, site/web/construire.py compact()).
//
// La LISTE de gauche est DÉRIVÉE du cadre Données que le moteur rend (rien n'est copié ni recalculé ici) :
//   - la note (son nom, son chantier) ;
//   - ses éléments (« Dans cette note » : le bloc .elements du moteur, DÉPLACÉ dans la liste, avec ses boutons) ;
//   - les étapes de l'élément ouvert : chaque section .sec[data-sec] du cadre Données (numéro, titre, résumé) ; celles
//     que le moteur dit secondaires (data-secondaire) sous « Options », repliées.
// Choisir une étape (dans la liste, ou d'un clic sur le dessin : les pièces portent data-sel = la clé de leur étape)
// montre ses PROPRIÉTÉS à droite : le cadre Données lui-même, filtré par la feuille de style (#donnees[data-choix]).
// « Toutes les étapes » montre tout, comme l'établi d'avant. Le choix tient d'un calcul à l'autre (session).
import { esc } from "./commun.js";

const memoire = {
  lire(k, d) { try { return JSON.parse(sessionStorage.getItem(k) || "null") ?? d; } catch (e) { return d; } },
  ecrire(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } },
};
const reglages = {
  lire() { try { return JSON.parse(localStorage.getItem("stab.etabli.v1") || "{}") || {}; } catch (e) { return {}; } },
  ecrire(v) { try { localStorage.setItem("stab.etabli.v1", JSON.stringify(v)); } catch (e) { /* stockage bloqué */ } },
};

// le seuil au-delà duquel un appui sur le dessin est un glisser (on déplace la vue) et non un choix ; plus large au doigt
const SEUIL = { mouse: 5, pen: 8, touch: 10 };

export function cablerCompact({ cle, note, agir = () => false }) {
  const liste = document.getElementById("liste");
  const donnees = document.getElementById("donnees");
  const stage = document.getElementById("stage");
  const titreChoix = document.getElementById("et-choix");
  const memo = () => `stab.choix.${cle}.${note().id}`;
  let choix = memoire.lire(memo(), "tout");

  // ------------------------------------------------------------ les sous-cadres repliables (propriétés, résultat)
  const r0 = reglages.lire();
  document.querySelectorAll("[data-et-replier]").forEach((b) => {
    const cadre = b.closest(".et-props, .et-res");
    const k = b.dataset.etReplier;
    const appliquerRepli = (ferme) => {
      cadre.classList.toggle("et-ferme", ferme);
      b.setAttribute("aria-expanded", ferme ? "false" : "true");
      b.title = `${ferme ? "Déplier" : "Replier"} ${k === "props" ? "les propriétés" : "le résultat"}`;
    };
    appliquerRepli(!!(r0.replis || {})[k]);
    b.addEventListener("click", () => {
      const r = reglages.lire();
      r.replis = r.replis || {};
      r.replis[k] = !cadre.classList.contains("et-ferme");
      reglages.ecrire(r);
      appliquerRepli(r.replis[k]);
    });
  });

  // ------------------------------------------------------------ la liste, refaite après chaque réponse du moteur
  function etapes() {
    return [...donnees.querySelectorAll(":scope > .sec[data-sec]")].map((s) => {
      const hd = s.querySelector(":scope > .sechd");
      return {
        cle: s.dataset.sec,
        n: hd ? (hd.querySelector(".n") || {}).textContent || "" : "",
        titre: hd ? (hd.querySelector(".tt") || {}).innerHTML || "" : "",
        resume: hd ? (hd.querySelector(".sum") || {}).innerHTML || "" : "",
        secondaire: s.hasAttribute("data-secondaire"),
      };
    });
  }

  const ligne = (e) => `<button type="button" class="et-li" data-choix="${esc(e.cle)}" title="${esc(texte(e.titre))}">`
    + `<span class="n">${esc(e.n)}</span><span class="tt">${e.titre}</span><span class="sum">${e.resume}</span></button>`;

  // l'en-tête de la liste : le nom de la note et son chantier (refait aussi quand on la renomme ou la classe)
  function teteNote() {
    const n = note();
    const fil = [n.chantier, n.dossier].filter(Boolean).join(" › ") || "Note non classée";
    return `<div class="et-note"><div class="et-note-t"><b>${esc(n.nom || "Note sans nom")}</b><span>${esc(fil)}</span></div></div>`;
  }
  function majNote() {
    const t = liste.querySelector(".et-note");
    if (t) t.outerHTML = teteNote();
  }

  function rafraichir() {
    const tout = etapes();
    // le bloc des éléments du moteur (avec ses boutons Ajouter, Copier, ✕) passe dans la liste, tel quel ; refaite sans
    // nouvelle réponse du moteur, la liste reprend le bloc qu'elle porte déjà
    const elements = donnees.querySelector(":scope > .elements") || liste.querySelector(".et-els > .elements");
    // le nom de l'élément ouvert (« V1 ») : etabli.separerNoms l'a posé dans son <span class="nom">
    const actif = elements ? elements.querySelector(".ni.on .t") : null;
    const nomActif = actif ? (actif.querySelector(".nom") || actif).textContent.trim() : "";
    const ouvertes = !!reglages.lire().options;
    const principales = tout.filter((e) => !e.secondaire);
    const options = tout.filter((e) => e.secondaire);
    const corps = `<div class="et-g">Étapes${nomActif ? ` · <span class="et-g-el">${esc(nomActif)}</span>` : ""}</div>`
      + `<button type="button" class="et-li et-tout" data-choix="tout"><span class="ic">≡</span><span class="tt">Toutes les étapes</span>`
      + `<span class="sum">${principales.length}</span></button>`
      + principales.map(ligne).join("")
      + (options.length ? `<details class="et-options"${ouvertes ? " open" : ""}><summary class="et-li"><span class="ic">›</span>`
        + `<span class="tt">Options</span><span class="sum">${options.length}</span></summary>${options.map(ligne).join("")}</details>` : "");
    liste.innerHTML = teteNote() + '<div class="et-els"></div>' + corps;
    if (elements) liste.querySelector(".et-els").append(elements);
    const det = liste.querySelector(".et-options");
    if (det) det.addEventListener("toggle", () => { const r = reglages.lire(); r.options = det.open; reglages.ecrire(r); });
    appliquer();
  }

  function appliquer() {
    const tout = etapes();
    if (choix !== "tout" && !tout.some((e) => e.cle === choix)) choix = "tout";
    donnees.dataset.choix = choix;
    donnees.querySelectorAll(":scope > .sec[data-sec]").forEach((s) => s.classList.toggle("et-choisie", s.dataset.sec === choix));
    liste.querySelectorAll("[data-choix]").forEach((b) => {
      b.classList.toggle("on", b.dataset.choix === choix);
      b.setAttribute("aria-current", b.dataset.choix === choix ? "true" : "false");
    });
    // une étape d'Options choisie : le repli « Options » s'ouvre pour la montrer
    const det = liste.querySelector(".et-options");
    if (det && det.querySelector(`[data-choix="${CSS.escape(choix)}"]`)) det.open = true;
    const e = tout.find((x) => x.cle === choix);
    titreChoix.innerHTML = e ? `${e.titre}<span class="et-tout-voir" data-choix="tout" role="button" tabindex="0">· tout voir</span>` : "Toutes les étapes";
    surligner();
  }

  // le dessin : les pièces de l'étape choisie, surlignées
  function surligner() {
    stage.querySelectorAll("[data-sel]").forEach((g) => g.classList.toggle("et-sel", g.dataset.sel === choix));
  }

  function choisir(k) {
    choix = k || "tout";
    memoire.ecrire(memo(), choix);
    appliquer();
    donnees.scrollTop = 0;
  }

  liste.addEventListener("click", (ev) => {
    if (agir(ev)) return;   // un bouton du bloc des éléments (ouvrir, ajouter, copier, supprimer) : js/etabli.js
    const b = ev.target.closest("[data-choix]");
    if (b && !b.closest("summary")) choisir(b.dataset.choix);
  });
  titreChoix.addEventListener("click", (ev) => { if (ev.target.closest("[data-choix]")) choisir("tout"); });
  titreChoix.addEventListener("keydown", (ev) => { if (ev.key === "Enter" && ev.target.closest("[data-choix]")) choisir("tout"); });

  // ------------------------------------------------------------ le dessin choisit l'étape d'une pièce
  // On lit l'APPUI (pointerdown) : la toile capture ensuite le pointeur pour déplacer la vue, et le « click » tomberait
  // sur le fond. Au relâcher, un pointeur qui a peu bougé choisit la pièce appuyée ; au-delà du seuil, c'était un glisser.
  let appui = null;
  stage.addEventListener("pointerdown", (ev) => {
    const g = ev.target.closest && ev.target.closest(".draw [data-sel]");
    appui = g ? { g, x: ev.clientX, y: ev.clientY, id: ev.pointerId, seuil: SEUIL[ev.pointerType] || 5 } : null;
  }, true);
  window.addEventListener("pointerup", (ev) => {
    const a = appui;
    appui = null;
    if (!a || ev.pointerId !== a.id) return;
    if (Math.hypot(ev.clientX - a.x, ev.clientY - a.y) > a.seuil) return;
    choisir(a.g.dataset.sel);
  }, true);
  window.addEventListener("pointercancel", () => { appui = null; }, true);

  return { rafraichir, majNote, surligner, choisir, choix: () => choix };
}

function texte(html) {
  const d = document.createElement("div");
  d.innerHTML = html;
  return d.textContent;
}
