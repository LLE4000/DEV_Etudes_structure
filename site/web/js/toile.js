// La toile de l'établi : le ZOOM du dessin (boutons, Ctrl + molette ou pincement du pavé tactile, pincer au doigt), le
// déplacement du dessin agrandi (glisser), et le PLEIN ÉCRAN de la vue. Le dessin est un SVG : on l'agrandit par une
// transformation CSS (il reste net, c'est du dessin vectoriel). Une vue qui est un tableau (bloc HTML) ne se zoome
// pas : elle défile dans sa feuille ; le plein écran lui sert aussi.
//
// Le zoom de chaque vue est gardé d'un calcul à l'autre : on agrandit un détail, on change une valeur, le détail reste
// sous les yeux.
const PAS = 1.25, MAXI = 8;
const borne = (x, a, b) => Math.min(b, Math.max(a, x));

// ``signal`` (l'AbortSignal du montage de l'écran, js/app.js) : à la fermeture, les écouteurs posés sur le document
// tombent et l'observateur de taille se déconnecte
export function cablerToile({ bench, stage, barre, signal }) {
  const zooms = new Map();      // clé de la vue → { z, x, y } (z = 1 : le dessin ajusté à sa feuille)
  let cle = null;
  const etat = () => zooms.get(cle) || { z: 1, x: 0, y: 0 };
  const dessin = () => stage.querySelector(".view.on .draw > svg");
  // le pourcentage : le bouton lui-même, ou son <span class="v"> quand le bouton porte une icône (l'établi compact)
  const pct = barre.querySelector(".zpct .v") || barre.querySelector(".zpct");
  const plein = barre.querySelector("[data-plein]");

  // la taille du SVG avant transformation, et son coin à l'écran, lus sous la transformation qu'il porte vraiment
  function boite(svg) {
    const a = svg.zoomPose || { z: 1, x: 0, y: 0 };
    const r = svg.getBoundingClientRect();
    return { g: r.left - a.x, h: r.top - a.y, l: r.width / a.z, t: r.height / a.z };
  }

  function poser() {
    const svg = dessin();
    barre.classList.toggle("sans-zoom", !svg);
    if (!svg) { pct.textContent = "100 %"; return; }
    const e = etat();
    const b = boite(svg);
    // le dessin agrandi couvre toujours sa feuille : on ne le perd pas en le faisant glisser
    e.x = borne(e.x, b.l - b.l * e.z, 0);
    e.y = borne(e.y, b.t - b.t * e.z, 0);
    svg.style.transformOrigin = "0 0";
    svg.style.transform = e.z === 1 ? "" : `translate(${e.x}px, ${e.y}px) scale(${e.z})`;
    svg.zoomPose = { z: e.z, x: e.z === 1 ? 0 : e.x, y: e.z === 1 ? 0 : e.y };
    svg.parentElement.classList.toggle("zoome", e.z > 1);
    pct.textContent = `${Math.round(e.z * 100)} %`;
  }

  // zoomer d'un facteur autour d'un point de l'écran (le centre du dessin par défaut) : ce point ne bouge pas
  function zoomer(f, cx, cy) {
    const svg = dessin();
    if (!svg) return;
    const e = { ...etat() };
    const b = boite(svg);
    const px = (cx ?? b.g + b.l / 2) - b.g, py = (cy ?? b.h + b.t / 2) - b.h;
    const z = borne(e.z * f, 1, MAXI);
    e.x = px - (px - e.x) * (z / e.z);
    e.y = py - (py - e.y) * (z / e.z);
    e.z = z;
    zooms.set(cle, e);
    poser();
  }

  function glisser(dx, dy) {
    const e = { ...etat() };
    if (e.z === 1) return;
    e.x += dx; e.y += dy;
    zooms.set(cle, e);
    poser();
  }

  function ajuster() { zooms.delete(cle); poser(); }

  function basculerPleinEcran(on = !bench.classList.contains("plein")) {
    bench.classList.toggle("plein", on);
    document.documentElement.classList.toggle("plein-ecran", on);
    plein.title = on ? "Quitter le plein écran" : "Plein écran";
    plein.setAttribute("aria-pressed", on ? "true" : "false");
    requestAnimationFrame(() => { if (!signal || !signal.aborted) poser(); });
  }

  barre.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    if (b.dataset.zoom === "plus") zoomer(PAS);
    else if (b.dataset.zoom === "moins") zoomer(1 / PAS);
    else if (b.dataset.zoom === "ajuster") ajuster();
    else if (b.dataset.plein) basculerPleinEcran();
  }, { signal });

  // Ctrl + molette, et le pincement d'un pavé tactile (Chrome l'envoie ainsi) : zoom autour du pointeur
  stage.addEventListener("wheel", (ev) => {
    if (!ev.ctrlKey || !ev.target.closest(".view.on .draw")) return;
    ev.preventDefault();
    zoomer(Math.exp(-ev.deltaY * 0.0025), ev.clientX, ev.clientY);
  }, { passive: false, signal });

  // les doigts (et la souris) : un doigt fait glisser le dessin agrandi, deux doigts le pincent. Sans zoom, un doigt
  // fait défiler la page (touch-action, css/site.css) ; pincer reste à nous.
  const doigts = new Map();
  let avant = null;
  const deux = () => { const [a, b] = [...doigts.values()]; return { d: Math.hypot(a.x - b.x, a.y - b.y), x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; };
  stage.addEventListener("pointerdown", (ev) => {
    const d = ev.target.closest(".view.on .draw");
    if (!d || !dessin() || (ev.pointerType === "mouse" && ev.button !== 0)) return;
    doigts.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    try { d.setPointerCapture(ev.pointerId); } catch (e) { /* pointeur déjà relâché */ }
    avant = doigts.size === 2 ? deux() : null;
  }, { signal });
  stage.addEventListener("pointermove", (ev) => {
    const p = doigts.get(ev.pointerId);
    if (!p) return;
    const dx = ev.clientX - p.x, dy = ev.clientY - p.y;
    p.x = ev.clientX; p.y = ev.clientY;
    if (doigts.size === 2 && avant) {
      const m = deux();
      if (avant.d > 0) zoomer(m.d / avant.d, m.x, m.y);
      glisser(m.x - avant.x, m.y - avant.y);
      avant = m;
      ev.preventDefault();
    } else if (doigts.size === 1 && etat().z > 1) {
      glisser(dx, dy);
      ev.preventDefault();
    }
  }, { signal });
  const lacher = (ev) => { doigts.delete(ev.pointerId); avant = doigts.size === 2 ? deux() : null; };
  stage.addEventListener("pointerup", lacher, { signal });
  stage.addEventListener("pointercancel", lacher, { signal });
  // un double appui (ou double clic) remet le dessin à sa feuille
  stage.addEventListener("dblclick", (ev) => { if (ev.target.closest(".view.on .draw") && etat().z > 1) ajuster(); }, { signal });

  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && bench.classList.contains("plein")) basculerPleinEcran(false); }, { signal });
  // la toile change de taille (fenêtre, cadre replié, plein écran) : le dessin agrandi reste dans sa feuille
  const taille = new ResizeObserver(() => poser());
  taille.observe(stage);
  if (signal) signal.addEventListener("abort", () => taille.disconnect(), { once: true });

  return {
    // la vue ouverte a changé, ou les vues viennent d'être reposées (un calcul) : le zoom de la vue s'applique
    vue(k) { cle = k; poser(); },
    quitterPleinEcran() { if (bench.classList.contains("plein")) basculerPleinEcran(false); },
  };
}
