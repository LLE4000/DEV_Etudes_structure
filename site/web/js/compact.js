// L'établi COMPACT (pilote : « Poutres béton armé » ; css/etabli.css, site/web/construire.py compact()).
//
// Trois colonnes : les DONNÉES à gauche (les éléments de la note, puis les rubriques du module en accordéon), le DESSIN au
// centre, les RÉSULTATS à droite. Ce fichier ne calcule rien et ne recopie rien du moteur : il relie ce que le moteur a
// rendu.
//   - la SÉLECTION : une pièce du dessin porte data-sel (« s0.lit-inf-1 »), la ligne des données qui la règle porte
//     data-cible (la même clé ; « beton » pour la section de béton). Un clic sur l'une choisit l'autre : la rubrique
//     s'ouvre, la ligne se déplie et s'éclaire, la pièce s'éclaire sur le dessin. Échap ou un clic dans le vide la
//     relâche.
//   - la SECTION COURANTE : la rubrique « Armatures » montre une section à la fois (.et-arm[data-section]) ; elle suit
//     l'onglet du dessin, et inversement.
//   - les COLONNES se redimensionnent (poignées .et-poignee, PC), la largeur est gardée dans ce navigateur.
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
// les largeurs permises des colonnes (px), et celles d'origine (la feuille de style)
const LARGEURS = { g: [264, 560], d: [280, 560] };

export function cablerCompact({ cle, note, ouvrirRubrique, vue }) {
  const donnees = document.getElementById("donnees");
  const stage = document.getElementById("stage");
  const bench = document.querySelector(".bench");
  const memo = () => `stab.compact.${cle}.${note().id}`;
  const etat = memoire.lire(memo(), { sel: "", courante: 0 });
  let sel = etat.sel || "";
  let courante = +etat.courante || 0;
  const garder = () => memoire.ecrire(memo(), { sel, courante });

  // ------------------------------------------------------------ la section courante (rubrique Armatures, dessin)
  function poserCourante() {
    const blocs = [...donnees.querySelectorAll(".et-arm[data-section]")];
    if (!blocs.length) return;
    if (!blocs.some((b) => +b.dataset.section === courante)) courante = 0;
    blocs.forEach((b) => b.classList.toggle("on", +b.dataset.section === courante));
    donnees.querySelectorAll("[data-section-courante]").forEach((b) => {
      const on = +b.dataset.sectionCourante === courante;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    donnees.querySelectorAll(".secard[data-section]").forEach((c) => c.classList.toggle("et-courante", +c.dataset.section === courante));
  }
  function choisirSection(i, versDessin = true) {
    courante = i;
    garder();
    poserCourante();
    if (versDessin && vue.courante() !== "toutes") vue.choisir(`s${i}`);
  }

  // ------------------------------------------------------------ la sélection (données ↔ dessin)
  const indice = (k) => { const m = /^s(\d+)\./.exec(k || ""); return m ? +m[1] : null; };
  const cibleDonnees = (k) => {
    if (!k) return null;
    if (/\.beton$/.test(k)) return donnees.querySelector('[data-cible="beton"]');
    return donnees.querySelector(`[data-cible="${CSS.escape(k)}"]`);
  };
  function surligner() {
    stage.querySelectorAll(".pc[data-sel]").forEach((g) => {
      const k = g.dataset.sel;
      g.classList.toggle("et-sel", !!sel && (k === sel || (sel === "beton" && /\.beton$/.test(k))));
    });
    donnees.querySelectorAll("[data-cible]").forEach((e) => e.classList.toggle("et-cible", !!sel && e.dataset.cible === (/\.beton$/.test(sel) ? "beton" : sel)));
  }
  // depuis le DESSIN : la rubrique s'ouvre, la ligne se déplie et vient sous les yeux
  function choisirPiece(k) {
    sel = k || "";
    const i = indice(sel);
    if (i !== null && i !== courante) { courante = i; poserCourante(); }
    garder();
    const c = cibleDonnees(sel);
    if (c) {
      const rub = c.closest(".sec[data-sec]");
      if (rub && !rub.classList.contains("open")) ouvrirRubrique(rub.dataset.sec);
      surligner();
      requestAnimationFrame(() => c.scrollIntoView({ block: "nearest", behavior: "smooth" }));
    } else surligner();
  }
  // depuis les DONNÉES : la pièce s'éclaire sur le dessin (rien ne défile)
  function choisirLigne(k) {
    if (k === sel) return;
    sel = k;
    const i = indice(sel);
    if (i !== null && i !== courante) choisirSection(i);
    garder();
    surligner();
  }
  function relacher() { if (!sel) return; sel = ""; garder(); surligner(); }

  // la sélection AVANT l'appui : un clic sur un bouton le focalise, et le « focusin » ci-dessous choisit sa ligne avant
  // que le « click » n'arrive — sans ce relevé, le chevron d'une ligne non choisie la choisissait PUIS la repliait
  let avantAppui = "";
  donnees.addEventListener("pointerdown", () => { avantAppui = sel; }, true);
  donnees.addEventListener("click", (ev) => {
    const sc = ev.target.closest("[data-section-courante]");
    if (sc) { choisirSection(+sc.dataset.sectionCourante); return; }
    const dep = ev.target.closest("[data-deplier]");
    if (dep) {
      const p = dep.closest(".et-piece");
      if (p) {
        // le chevron d'une ligne DÉJÀ choisie la replie ou la déplie ; sinon, elle devient la sélection, dépliée
        const dejaChoisie = p.dataset.cible === avantAppui && p.classList.contains("et-cible");
        if (dejaChoisie) { p.classList.toggle("et-replie"); avantAppui = sel; return; }
        p.classList.remove("et-replie");
        choisirLigne(p.dataset.cible);
        avantAppui = sel;
      }
      return;
    }
    const carte = ev.target.closest(".secard[data-section]");
    if (carte && !ev.target.closest("button")) choisirSection(+carte.dataset.section);
  });
  donnees.addEventListener("focusin", (ev) => {
    const p = ev.target.closest("[data-cible]");
    if (p) choisirLigne(p.dataset.cible);
  });

  // On lit l'APPUI (pointerdown) : la toile capture ensuite le pointeur pour déplacer la vue, et le « click » tomberait
  // sur le fond. Au relâcher, un pointeur qui a peu bougé choisit la pièce appuyée (ou relâche la sélection, dans le
  // vide) ; au-delà du seuil, c'était un glisser.
  let appui = null;
  stage.addEventListener("pointerdown", (ev) => {
    if (!ev.target.closest || !ev.target.closest(".draw")) { appui = null; return; }
    const g = ev.target.closest(".draw [data-sel]");
    appui = { g, x: ev.clientX, y: ev.clientY, id: ev.pointerId, seuil: SEUIL[ev.pointerType] || 5 };
  }, true);
  window.addEventListener("pointerup", (ev) => {
    const a = appui;
    appui = null;
    if (!a || ev.pointerId !== a.id) return;
    if (Math.hypot(ev.clientX - a.x, ev.clientY - a.y) > a.seuil) return;
    if (a.g) choisirPiece(a.g.dataset.sel); else relacher();
  }, true);
  window.addEventListener("pointercancel", () => { appui = null; }, true);
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape" && sel && !document.querySelector(".flyout.on") && !ev.target.closest("input, select, textarea")) relacher();
  });

  // ------------------------------------------------------------ les colonnes : largeur réglable (PC)
  const pc = matchMedia("(min-width: 1200px)");
  function poserLargeurs() {
    const l = reglages.lire().largeurs || {};
    for (const k of ["g", "d"]) {
      const v = pc.matches && l[k] ? Math.min(LARGEURS[k][1], Math.max(LARGEURS[k][0], l[k])) : null;
      if (v) bench.style.setProperty(k === "g" ? "--pl-r" : "--pr-r", `${v}px`);
      else bench.style.removeProperty(k === "g" ? "--pl-r" : "--pr-r");
    }
  }
  poserLargeurs();
  pc.addEventListener("change", () => { poserLargeurs(); vue.ajuster(); });
  document.querySelectorAll(".et-poignee[data-poignee]").forEach((p) => {
    const k = p.dataset.poignee;
    p.addEventListener("pointerdown", (ev) => {
      if (!pc.matches || ev.button !== 0) return;
      ev.preventDefault();
      p.setPointerCapture(ev.pointerId);
      const panneau = document.querySelector(k === "g" ? ".panel.left" : ".panel.right");
      const x0 = ev.clientX, w0 = panneau.getBoundingClientRect().width;
      bench.classList.add("et-redim");
      const bouger = (e) => {
        const w = Math.round(Math.min(LARGEURS[k][1], Math.max(LARGEURS[k][0], w0 + (k === "g" ? 1 : -1) * (e.clientX - x0))));
        bench.style.setProperty(k === "g" ? "--pl-r" : "--pr-r", `${w}px`);
      };
      const finir = () => {
        p.removeEventListener("pointermove", bouger);
        bench.classList.remove("et-redim");
        const r = reglages.lire();
        r.largeurs = r.largeurs || {};
        r.largeurs[k] = Math.round(panneau.getBoundingClientRect().width);
        reglages.ecrire(r);
        vue.ajuster();
      };
      p.addEventListener("pointermove", bouger);
      p.addEventListener("pointerup", finir, { once: true });
      p.addEventListener("pointercancel", finir, { once: true });
    });
    p.addEventListener("dblclick", () => {
      const r = reglages.lire();
      if (r.largeurs) delete r.largeurs[k];
      reglages.ecrire(r);
      poserLargeurs();
      vue.ajuster();
    });
  });

  // ------------------------------------------------------------ après chaque réponse du moteur
  function rafraichir() {
    poserCourante();
    // la pièce choisie a pu disparaître (un lit retiré) : la sélection tombe
    if (sel && !cibleDonnees(sel) && !stage.querySelector(`[data-sel="${CSS.escape(sel)}"]`)) sel = "";
    surligner();
  }
  // l'onglet du dessin choisit la section courante (« Toutes » la laisse)
  function vueChoisie(k) {
    const m = /^s(\d+)$/.exec(k || "");
    if (m && +m[1] !== courante) choisirSection(+m[1], false);
    surligner();
  }

  return { rafraichir, surligner, vueChoisie, majNote() {}, courante: () => courante };
}
