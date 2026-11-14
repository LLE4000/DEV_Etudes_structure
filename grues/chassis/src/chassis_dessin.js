/* chassis_dessin.js
 * Moteur de dessin de la bibliothèque des châssis de grue à tour.
 * Aucune dépendance. Navigateur : window.ChassisDessin. Node : require('./chassis_dessin.js').
 * Entrée : un objet « châssis » de chassis_grues.json. Sortie : des chaînes SVG.
 * Toutes les longueurs sont en millimètres. Repère : origine au centre du châssis, x vers la droite, y vers le haut.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ChassisDessin = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var R2 = Math.SQRT2;

  /* ------------------------------------------------------------------ accès aux valeurs */
  function v(o) { return (o && typeof o === 'object' && !Array.isArray(o) && 'valeur' in o) ? o.valeur : (o === undefined ? null : o); }
  function origine(o) { return (o && typeof o === 'object' && o.origine) ? o.origine : null; }
  /* texte d'une cote : « ≈ » devant une valeur mesurée ou indicative */
  function txt(o, suffixe) {
    var val = v(o); if (val === null) return '';
    var s = fmt(val) + (suffixe || '');
    var og = origine(o);
    return (og === 'mesure' || og === 'indicatif') ? '≈ ' + s : s;
  }
  function fmt(x) {
    if (typeof x !== 'number') return String(x);
    var r = Math.round(x * 10) / 10;
    return (Math.abs(r - Math.round(r)) < 1e-9 ? String(Math.round(r)) : r.toFixed(1).replace('.', ','));
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function f1(x) { return (Math.round(x * 100) / 100).toString(); }

  /* ------------------------------------------------------------------ géométrie des appuis */
  /* Les 4 appuis : position, direction radiale r (vers l'extérieur, le long du bras) et tangentielle t. */
  function appuis(ch) {
    var ex = v(ch.appuis.entraxe_x), ey = v(ch.appuis.entraxe_y);
    return [[-ex / 2, -ey / 2], [ex / 2, -ey / 2], [ex / 2, ey / 2], [-ex / 2, ey / 2]].map(function (p, i) {
      var L = Math.hypot(p[0], p[1]);
      var r = [p[0] / L, p[1] / L];
      return { n: i + 1, x: p[0], y: p[1], r: r, t: [-r[1], r[0]] };
    });
  }

  /* Emprise au sol d'un appui (surface qui charge la fondation), en mm dans le repère du châssis.
   * Retourne { n, forme, centre:[x,y], sommets:[[x,y],…] } ; pour un cercle : { rayon } en plus (sommets = polygone à 48 côtés).
   * Retourne null si la source ne donne pas la dimension d'appui. */
  function emprise(ch, a) {
    var ap = ch.pied && ch.pied.appui;
    if (!ap) return null;
    var exc = v(ap.excentrement_radial) || 0;
    var c = [a.x + a.r[0] * exc, a.y + a.r[1] * exc];
    var rot = ap.rotation_deg || 0, s = [], i;
    if (ap.forme === 'rectangle') {
      var da = v(ap.a) / 2, db = v(ap.b) / 2;
      var u = rot === 45 ? a.t : [1, 0], w = rot === 45 ? a.r : [0, 1];
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (q) {
        s.push([c[0] + q[0] * da * u[0] + q[1] * db * w[0], c[1] + q[0] * da * u[1] + q[1] * db * w[1]]);
      });
      return { n: a.n, forme: 'rectangle', centre: c, sommets: s, a: v(ap.a), b: v(ap.b), rotation_deg: rot };
    }
    if (ap.forme === 'cercle') {
      var R = v(ap.diametre) / 2;
      for (i = 0; i < 48; i++) s.push([c[0] + R * Math.cos(i * Math.PI / 24), c[1] + R * Math.sin(i * Math.PI / 24)]);
      return { n: a.n, forme: 'cercle', centre: c, rayon: R, sommets: s };
    }
    if (ap.forme === 'octogone') {
      var Rc = v(ap.cote_sur_plats) / 2 / Math.cos(Math.PI / 8);
      for (i = 0; i < 8; i++) s.push([c[0] + Rc * Math.cos(Math.PI / 8 + i * Math.PI / 4), c[1] + Rc * Math.sin(Math.PI / 8 + i * Math.PI / 4)]);
      return { n: a.n, forme: 'octogone', centre: c, sommets: s, cote_sur_plats: v(ap.cote_sur_plats) };
    }
    return null;
  }
  function emprises(ch) { return appuis(ch).map(function (a) { return emprise(ch, a); }); }

  /* Hors-tout des emprises d'appui, calculé (sert au contrôle de la base). Retourne {x, y} ou null. */
  function horsToutEmprises(ch) {
    var E = emprises(ch); if (!E[0]) return null;
    var mx = 0, my = 0;
    E.forEach(function (e) { e.sommets.forEach(function (p) { mx = Math.max(mx, Math.abs(p[0])); my = Math.max(my, Math.abs(p[1])); }); });
    return { x: 2 * mx, y: 2 * my };
  }

  /* Dépassement du bras au-delà de l'axe d'appui (le long de la diagonale). */
  function depassementBras(ch) {
    var d = ch.dessin || {};
    if (d.bras_depassement) return v(d.bras_depassement);
    var w = v(d.bras_largeur) || 300, ex = v(ch.appuis.entraxe_x);
    var ht = ch.appuis.hors_tout;
    if (ht && /cadre/.test(ht.nature || '')) return Math.max(60, (v(ht) / 2 - ex / 2) * R2 - w / 2);
    var ap = ch.pied && ch.pied.appui;
    if (ap && ap.forme === 'rectangle') return v(ap.b) / 2 + (v(ap.excentrement_radial) || 0);
    return w / 2;
  }

  /* ------------------------------------------------------------------ feuille de style des dessins */
  var STYLE = [
    '.cg{font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}',
    '.cg text{font-size:12px;fill:var(--cg-encre,#1d2b3a)}',
    '.cg .tr{fill:none;stroke:var(--cg-encre,#1d2b3a);stroke-width:1.3;stroke-linejoin:round;stroke-linecap:round}',
    '.cg .ac{fill:var(--cg-acier,#e2e8ef);stroke:var(--cg-encre,#1d2b3a);stroke-width:1.2;stroke-linejoin:round}',
    '.cg .sc{fill:none;stroke:var(--cg-schema,#8793a1);stroke-width:1;stroke-linejoin:round}',
    '.cg .scf{fill:var(--cg-schema-fond,#f2f5f8);stroke:var(--cg-schema,#8793a1);stroke-width:1;stroke-linejoin:round}',
    '.cg .ti{fill:none;stroke:var(--cg-schema,#8793a1);stroke-width:1;stroke-dasharray:6 4}',
    '.cg .le{fill:var(--cg-lest-fond,rgba(120,132,146,.05));stroke:var(--cg-schema,#8793a1);stroke-width:.8;stroke-dasharray:5 4}',
    '.cg .ax{fill:none;stroke:var(--cg-axe,#b4384e);stroke-width:.8;stroke-dasharray:14 3 2 3}',
    '.cg .ap{fill:var(--cg-appui-fond,#ffd9c2);stroke:var(--cg-appui,#d9480f);stroke-width:1.8;stroke-linejoin:round}',
    '.cg .apl{fill:none;stroke:var(--cg-appui,#d9480f);stroke-width:1.2}',
    '.cg .apc{fill:var(--cg-appui-voile,rgba(255,217,194,.45));stroke:var(--cg-appui,#d9480f);stroke-width:1.8;stroke-linejoin:round}',
    '.cg .co{fill:none;stroke:var(--cg-cote,#2c6cb0);stroke-width:.9}',
    '.cg .cof{fill:var(--cg-cote,#2c6cb0);stroke:none}',
    '.cg text.ct{fill:var(--cg-cote,#2c6cb0);font-variant-numeric:tabular-nums;paint-order:stroke;stroke:var(--cg-fond,#fff);stroke-width:3.5px;stroke-linejoin:round}',
    '.cg text.cm{font-style:italic}',
    '.cg .be{fill:var(--cg-beton,#eceff2);stroke:var(--cg-schema,#8793a1);stroke-width:1}',
    '.cg .bh{fill:none;stroke:var(--cg-schema,#8793a1);stroke-width:.7}',
    '.cg .no{fill:var(--cg-encre,#1d2b3a);stroke:none}',
    '.cg text.lg{font-size:11px;fill:var(--cg-texte-2,#5d6976);paint-order:stroke;stroke:var(--cg-fond,#fff);stroke-width:3px;stroke-linejoin:round}',
    '.cg text.nu{font-size:11px;font-weight:600;fill:var(--cg-appui,#d9480f);paint-order:stroke;stroke:var(--cg-fond,#fff);stroke-width:3px}',
    '.cg text.ttl{font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;fill:var(--cg-texte-2,#5d6976)}'
  ].join('');

  /* ------------------------------------------------------------------ petit moteur SVG */
  function Dessin(W, H, titre) {
    this.W = W; this.H = H; this.titre = titre || ''; this.p = []; this.s = 1; this.ox = 0; this.oy = 0;
  }
  /* cadre le modèle [x0,y0]-[x1,y1] (mm) dans la feuille, avec des marges en pixels */
  Dessin.prototype.cadrer = function (x0, y0, x1, y1, m) {
    var aw = this.W - m.g - m.d, ah = this.H - m.h - m.b;
    this.s = Math.min(aw / (x1 - x0), ah / (y1 - y0));
    this.ox = m.g + (aw - (x1 - x0) * this.s) / 2 - x0 * this.s;
    this.oy = m.h + (ah - (y1 - y0) * this.s) / 2 + y1 * this.s;
    return this;
  };
  Dessin.prototype.T = function (q) { return [this.ox + q[0] * this.s, this.oy - q[1] * this.s]; };
  Dessin.prototype.px = function (n) { return n / this.s; };          /* n pixels exprimés en mm */
  Dessin.prototype.raw = function (s) { this.p.push(s); return this; };
  Dessin.prototype.ligne = function (a, b, cls) {
    var A = this.T(a), B = this.T(b);
    return this.raw('<line class="' + cls + '" x1="' + f1(A[0]) + '" y1="' + f1(A[1]) + '" x2="' + f1(B[0]) + '" y2="' + f1(B[1]) + '"/>');
  };
  Dessin.prototype.poly = function (pts, cls, ouvert) {
    var self = this;
    var d = pts.map(function (q) { var P = self.T(q); return f1(P[0]) + ',' + f1(P[1]); }).join(' ');
    return this.raw('<' + (ouvert ? 'polyline' : 'polygon') + ' class="' + cls + '" points="' + d + '"/>');
  };
  Dessin.prototype.rect = function (x0, y0, x1, y1, cls) { return this.poly([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], cls); };
  Dessin.prototype.cercle = function (c, r, cls) {
    var C = this.T(c);
    return this.raw('<circle class="' + cls + '" cx="' + f1(C[0]) + '" cy="' + f1(C[1]) + '" r="' + f1(r * this.s) + '"/>');
  };
  /* texte ancré en un point du modèle, décalé de (dx, dy) pixels */
  Dessin.prototype.texte = function (q, s, cls, o) {
    o = o || {};
    var P = this.T(q), x = P[0] + (o.dx || 0), y = P[1] + (o.dy || 0);
    var rot = o.rot ? ' transform="rotate(' + f1(o.rot) + ' ' + f1(x) + ' ' + f1(y) + ')"' : '';
    return this.raw('<text class="' + (cls || '') + '" x="' + f1(x) + '" y="' + f1(y) + '" text-anchor="' + (o.ancre || 'middle') + '"' + rot + '>' + esc(s) + '</text>');
  };
  /* texte en coordonnées feuille (pixels) */
  Dessin.prototype.texteF = function (x, y, s, cls, ancre) {
    return this.raw('<text class="' + (cls || '') + '" x="' + f1(x) + '" y="' + f1(y) + '" text-anchor="' + (ancre || 'start') + '">' + esc(s) + '</text>');
  };
  /* Cote alignée entre deux points du modèle.
   * dec : décalage de la ligne de cote en pixels, perpendiculairement à a→b (positif = à gauche de a→b à l'écran, soit « au-dessus » pour une cote tracée de gauche à droite).
   * o.mesure : valeur mesurée (italique). o.court : flèches à l'extérieur. o.sansRappel : pas de lignes de rappel.
   * o.cote : 'd' ou 'g', côté du texte pour une cote courte verticale. o.jeu : écart entre le point et sa ligne de rappel. */
  Dessin.prototype.cote = function (a, b, dec, s, o) {
    o = o || {};
    var A = this.T(a), B = this.T(b);
    var L = Math.hypot(B[0] - A[0], B[1] - A[1]); if (L < 0.5) return this;
    var u = [(B[0] - A[0]) / L, (B[1] - A[1]) / L];
    var n = [u[1], -u[0]];                                   /* normale « à gauche » de a→b à l'écran */
    var A2 = [A[0] + n[0] * dec, A[1] + n[1] * dec], B2 = [B[0] + n[0] * dec, B[1] + n[1] * dec];
    var sg = dec >= 0 ? 1 : -1, self = this;
    function seg(P, Q, cls) { self.raw('<line class="' + cls + '" x1="' + f1(P[0]) + '" y1="' + f1(P[1]) + '" x2="' + f1(Q[0]) + '" y2="' + f1(Q[1]) + '"/>'); }
    if (!o.sansRappel && Math.abs(dec) > 0.5) {
      var g = o.jeu === undefined ? 3 : o.jeu;
      seg([A[0] + n[0] * sg * g, A[1] + n[1] * sg * g], [A2[0] + n[0] * sg * 4, A2[1] + n[1] * sg * 4], 'co');
      seg([B[0] + n[0] * sg * g, B[1] + n[1] * sg * g], [B2[0] + n[0] * sg * 4, B2[1] + n[1] * sg * 4], 'co');
    }
    var court = o.court || L < 34;
    function fleche(P, dir) {   /* pointe en P, corps vers dir */
      var l = 8, h = 2.6;
      var Q = [P[0] + dir[0] * l, P[1] + dir[1] * l];
      self.raw('<polygon class="cof" points="' + f1(P[0]) + ',' + f1(P[1]) + ' ' + f1(Q[0] + n[0] * h) + ',' + f1(Q[1] + n[1] * h) + ' ' + f1(Q[0] - n[0] * h) + ',' + f1(Q[1] - n[1] * h) + '"/>');
    }
    if (court) {
      seg([A2[0] - u[0] * 15, A2[1] - u[1] * 15], [B2[0] + u[0] * 15, B2[1] + u[1] * 15], 'co');
      fleche(A2, [-u[0], -u[1]]); fleche(B2, u);
      if (o.sansRappel || Math.abs(dec) <= 0.5) {            /* petits traits d'attache */
        seg([A2[0] - n[0] * 5, A2[1] - n[1] * 5], [A2[0] + n[0] * 5, A2[1] + n[1] * 5], 'co');
        seg([B2[0] - n[0] * 5, B2[1] - n[1] * 5], [B2[0] + n[0] * 5, B2[1] + n[1] * 5], 'co');
      }
    } else {
      seg(A2, B2, 'co'); fleche(A2, u); fleche(B2, [-u[0], -u[1]]);
    }
    var cls = 'ct' + (o.mesure ? ' cm' : '');
    var M = [(A2[0] + B2[0]) / 2, (A2[1] + B2[1]) / 2];
    if (court && Math.abs(u[0]) < 0.2) {                     /* cote courte verticale : texte horizontal, à côté */
      var gch = o.cote === 'g';
      this.raw('<text class="' + cls + '" x="' + f1(M[0] + (gch ? -8 : 8)) + '" y="' + f1(M[1] + 4) + '" text-anchor="' + (gch ? 'end' : 'start') + '">' + esc(s) + '</text>');
      return this;
    }
    /* texte : au milieu, au-dessus de la ligne ; une cote verticale se lit de bas en haut */
    var ang = Math.atan2(u[1], u[0]) * 180 / Math.PI;
    if (ang > 89.99) ang -= 180; else if (ang < -90.01) ang += 180;
    var ar = ang * Math.PI / 180;
    var up = [Math.sin(ar), -Math.cos(ar)];                  /* « haut » du texte après rotation */
    var ty = o.dessous ? -14 : 5;
    var P = [M[0] + up[0] * ty, M[1] + up[1] * ty];
    this.raw('<text class="' + cls + '" x="' + f1(P[0]) + '" y="' + f1(P[1]) + '" text-anchor="middle" transform="rotate(' + f1(ang) + ' ' + f1(P[0]) + ' ' + f1(P[1]) + ')">' + esc(s) + '</text>');
    return this;
  };
  /* trait de repère (pixels à partir d'un point modèle) + texte */
  Dessin.prototype.repere = function (q, dx, dy, s, cls, ancre) {
    var P = this.T(q), Q = [P[0] + dx, P[1] + dy];
    this.raw('<line class="sc" x1="' + f1(P[0]) + '" y1="' + f1(P[1]) + '" x2="' + f1(Q[0]) + '" y2="' + f1(Q[1]) + '"/>');
    var a = ancre || (dx >= 0 ? 'start' : 'end');
    this.raw('<text class="' + (cls || 'lg') + '" x="' + f1(Q[0] + (a === 'start' ? 4 : a === 'end' ? -4 : 0)) + '" y="' + f1(Q[1] + (a === 'middle' ? (dy < 0 ? -5 : 13) : 4)) + '" text-anchor="' + a + '">' + esc(s) + '</text>');
    return this;
  };
  /* hachures de béton dans un rectangle (modèle) */
  Dessin.prototype.beton = function (x0, y0, x1, y1, id) {
    var A = this.T([x0, y1]), B = this.T([x1, y0]);
    var w = B[0] - A[0], h = B[1] - A[1];
    this.raw('<rect class="be" x="' + f1(A[0]) + '" y="' + f1(A[1]) + '" width="' + f1(w) + '" height="' + f1(h) + '"/>');
    var c = 'cl' + id + '_' + this.p.length;
    this.raw('<clipPath id="' + c + '"><rect x="' + f1(A[0]) + '" y="' + f1(A[1]) + '" width="' + f1(w) + '" height="' + f1(h) + '"/></clipPath>');
    var g = '<g clip-path="url(#' + c + ')">';
    for (var t = -h; t < w + 9; t += 9) g += '<line class="bh" x1="' + f1(A[0] + t) + '" y1="' + f1(A[1] + h) + '" x2="' + f1(A[0] + t + h) + '" y2="' + f1(A[1]) + '"/>';
    return this.raw(g + '</g>');
  };
  /* o.sansStyle : pas de <style> (la page fournit STYLE une fois). o.couleursFixes : couleurs en dur (fichiers SVG autonomes). */
  Dessin.prototype.svg = function (o) {
    o = Object.assign({}, this.options || {}, o || {});
    var t = this.titre ? '<text class="ttl" x="12" y="20">' + esc(this.titre) + '</text>' : '';
    var s = '<svg xmlns="http://www.w3.org/2000/svg" class="cg" viewBox="0 0 ' + this.W + ' ' + this.H + '" role="img" aria-label="' + esc(o.label || this.titre) + '">' +
      (o.sansStyle ? '' : '<style>' + STYLE + '</style>') + t + this.p.join('') + '</svg>';
    return o.couleursFixes ? couleursFixes(s) : s;
  };
  function couleursFixes(s) { return s.replace(/var\(--[\w-]+,\s*((?:rgba?\([^)]*\))|[^()]+)\)/g, '$1'); }

  /* ------------------------------------------------------------------ outils de géométrie */
  function add(p, q, k) { return [p[0] + q[0] * k, p[1] + q[1] * k]; }
  function rectOriente(c, u, w, du, dw) {   /* rectangle centré en c, demi-côtés du selon u et dw selon w */
    return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(function (q) { return [c[0] + q[0] * du * u[0] + q[1] * dw * w[0], c[1] + q[0] * du * u[1] + q[1] * dw * w[1]]; });
  }
  function bande(p, q, w) {                 /* quadrilatère de largeur w autour du segment p-q */
    var L = Math.hypot(q[0] - p[0], q[1] - p[1]), n = [-(q[1] - p[1]) / L * w / 2, (q[0] - p[0]) / L * w / 2];
    return [[p[0] + n[0], p[1] + n[1]], [q[0] + n[0], q[1] + n[1]], [q[0] - n[0], q[1] - n[1]], [p[0] - n[0], p[1] - n[1]]];
  }
  function fondationDessinee(ch) { return (ch.fondation && ch.fondation.fondation_dessinee) || null; }

  /* ================================================================== VUE EN PLAN */
  function plan(ch, o) {
    o = Object.assign({ W: 760, H: 650, lest: true, fondation: true, numeros: true, titre: 'Vue en plan' }, o || {});
    var d = new Dessin(o.W, o.H, o.titre); d.options = { sansStyle: o.sansStyle, couleursFixes: o.couleursFixes };
    var A = appuis(ch), ex = v(ch.appuis.entraxe_x), ey = v(ch.appuis.entraxe_y);
    var E = A.map(function (a) { return emprise(ch, a); });
    var des = ch.dessin || {}, wb = v(des.bras_largeur) || 300, Le = depassementBras(ch);
    var ht = v(ch.appuis.hors_tout);
    var fo = ch.fondation || {}, fd = fondationDessinee(ch), vp = fo.variante_plots, vr = fo.variante_reservation;

    /* étendue du dessin */
    var ext = Math.max(ex, ey) / 2 + (Le + wb / 2) / R2;
    E.forEach(function (e) { if (e) e.sommets.forEach(function (q) { ext = Math.max(ext, Math.abs(q[0]), Math.abs(q[1])); }); });
    if (ht) ext = Math.max(ext, ht / 2);
    var extF = ext;
    if (o.fondation && vp && vp.cote_massif) extF = Math.max(extF, ex / 2 + v(vp.cote_massif) / 2);
    if (o.fondation && fd && fd.hors_tout) extF = Math.max(extF, v(fd.hors_tout) / 2);
    d.cadrer(-extF, -extF, extF, extF, { g: 92, d: 70, h: 44, b: 88 });

    /* 1. fondation dessinée sur la source (tirets) */
    if (o.fondation) {
      if (vp && vp.cote_massif) {
        var cm = v(vp.cote_massif) / 2;
        A.forEach(function (a) { d.rect(a.x - cm, a.y - cm, a.x + cm, a.y + cm, 'ti'); });
        d.texte([A[3].x + cm, A[3].y + cm], 'massifs ' + fmt(2 * cm) + ' × ' + fmt(2 * cm) + ' dessinés sur le plan', 'lg', { ancre: 'start', dx: 8, dy: -7 });
      }
      if (fd && fd.hors_tout) {
        var fe = v(fd.hors_tout) / 2, fi = ex / 2 - v(fd.debord_interieur);
        d.rect(-fe, -fe, fe, fe, 'ti'); d.rect(-fi, -fi, fi, fi, 'ti');
        d.texte([-fe, fe], 'longrine de ' + fmt(v(fd.largeur)) + ' dessinée sur le plan (' + fmt(2 * fe) + ' hors-tout)', 'lg', { ancre: 'start', dx: 0, dy: -7 });
      }
      if (vr) {
        var rc = v(vr.cote) / 2;
        d.rect(-rc, -rc, rc, rc, 'ti');
      }
    }

    /* 2. lest central (emprises relevées sur la source) */
    if (o.lest && ch.lest_central && ch.lest_central.blocs) {
      ch.lest_central.blocs.forEach(function (b) { (b.contours || []).forEach(function (c) { d.poly(c, 'le'); }); });
    }

    /* 3. tirants entre appuis */
    if (des.tirants) {
      var ta = v(des.tirants.axe), tl = v(des.tirants.largeur) / 2, td = v(des.tirants.demi_longueur) || (ex / 2 - 400);
      var cotes = des.tirants.cotes || ['N', 'S', 'E', 'O'];
      if (cotes.indexOf('N') >= 0) d.rect(-td, ta - tl, td, ta + tl, 'scf');
      if (cotes.indexOf('S') >= 0) d.rect(-td, -ta - tl, td, -ta + tl, 'scf');
      if (cotes.indexOf('E') >= 0) d.rect(ta - tl, -td, ta + tl, td, 'scf');
      if (cotes.indexOf('O') >= 0) d.rect(-ta - tl, -td, -ta + tl, td, 'scf');
    }

    /* 4. emprises d'appui (sous les bras), puis bras diagonaux */
    E.forEach(function (e, i) {
      var a = A[i];
      if (!e) return;
      if (e.forme === 'cercle') {
        var sb = ch.pied.sabot;
        if (sb) { var es = v(sb.axe_au_bord_exterieur) - v(sb.b) / 2; d.poly(rectOriente(add([a.x, a.y], a.r, es), a.t, a.r, v(sb.a) / 2, v(sb.b) / 2), 'scf'); }
        d.cercle(e.centre, e.rayon, 'ap');
      } else d.poly(e.sommets, 'ap');
    });
    [[A[0], A[2]], [A[1], A[3]]].forEach(function (pr) {
      var p = add([pr[0].x, pr[0].y], pr[0].r, Le), q = add([pr[1].x, pr[1].y], pr[1].r, Le);
      d.poly(bande(p, q, wb), 'ac');
    });

    /* 5. mât */
    var tm = ch.mat && v(ch.mat.entraxe), mInd = false;
    if (!tm && ch.mat && ch.mat.largeur_dessin) { tm = v(ch.mat.largeur_dessin); mInd = true; }
    if (tm) {
      var lm = (ch.mat.hors_tout ? v(ch.mat.hors_tout) - tm : 110) / 2;
      d.rect(-tm / 2, -tm / 2, tm / 2, tm / 2, mInd ? 'ti' : 'tr');
      if (!mInd) [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (q) { d.rect(q[0] * tm / 2 - lm, q[1] * tm / 2 - lm, q[0] * tm / 2 + lm, q[1] * tm / 2 + lm, 'ac'); });
    }
    if (ch.mat && ch.mat.entraxes_de_liaison) ch.mat.entraxes_de_liaison.forEach(function (m2) {
      var h2 = v(m2) / 2; if (Math.abs(2 * h2 - tm) > 1) d.rect(-h2, -h2, h2, h2, 'ax');
    });

    /* 6. axes d'appui, emprises d'appui, numéros */
    var ax = extF + d.px(4);
    [ex / 2, -ex / 2].forEach(function (x) { d.ligne([x, -ext - d.px(10)], [x, ext + d.px(10)], 'ax'); });
    [ey / 2, -ey / 2].forEach(function (y) { d.ligne([-ext - d.px(10), y], [ext + d.px(10), y], 'ax'); });
    E.forEach(function (e, i) {
      var a = A[i];
      if (e) {                                   /* contour de l'emprise, visible par-dessus le bras */
        if (e.forme === 'cercle') d.cercle(e.centre, e.rayon, 'apc'); else d.poly(e.sommets, 'apc');
      } else {
        d.cercle([a.x, a.y], d.px(5), 'apl');
      }
      var k = d.px(5);
      d.ligne([a.x - k, a.y], [a.x + k, a.y], 'apl'); d.ligne([a.x, a.y - k], [a.x, a.y + k], 'apl');
      if (o.numeros) {
        var Re = 0; if (e) e.sommets.forEach(function (q) { Re = Math.max(Re, (q[0] - a.x) * a.r[0] + (q[1] - a.y) * a.r[1]); });
        Re = Math.max(Re, Le + wb / 2);
        var pn = add([a.x, a.y], a.r, Re + d.px(i === 2 ? 46 : 13));
        d.texte(pn, String(a.n), 'nu', { dy: 4 });
      }
    });

    /* 7. cotes */
    var bas = -extF, gau = -extF;
    function dec(yModel, add2) { return (yModel - bas) * d.s + add2; }
    /* entraxes */
    d.cote([A[0].x, A[0].y], [A[1].x, A[1].y], -dec(A[0].y, 24), txt(ch.appuis.entraxe_x), { jeu: 8 });
    d.cote([A[0].x, A[0].y], [A[3].x, A[3].y], (A[0].x - gau) * d.s + 26, txt(ch.appuis.entraxe_y), { jeu: 8 });
    /* hors-tout */
    if (ht) {
      var lab = txt(ch.appuis.hors_tout) + '  ' + (ch.appuis.hors_tout.nature || 'hors-tout');
      d.cote([-ht / 2, bas], [ht / 2, bas], -50, lab, { mesure: origine(ch.appuis.hors_tout) !== 'cote' });
      d.cote([gau, -ht / 2], [gau, ht / 2], 52, txt(ch.appuis.hors_tout), { mesure: origine(ch.appuis.hors_tout) !== 'cote' });
    }
    if (ch.appuis.hors_tout_acier) {
      var ha = v(ch.appuis.hors_tout_acier);
      d.cote([extF, -ha / 2], [extF, ha / 2], -30, txt(ch.appuis.hors_tout_acier) + '  croix', {});
    }
    /* massifs / longrine : hors-tout dessiné */
    if (o.fondation && vp && vp.cote_massif) {
      var cm2 = v(vp.cote_massif) / 2;
      d.cote([A[3].x - cm2, A[3].y + cm2], [A[3].x + cm2, A[3].y + cm2], 14, fmt(2 * cm2), {});
    }
    if (o.fondation && fd && fd.hors_tout) {
      var fe2 = v(fd.hors_tout) / 2;
      d.cote([-fe2, A[0].y + ey * 0.30], [A[0].x, A[0].y + ey * 0.30], 0, fmt(v(fd.debord_exterieur)), { sansRappel: true, court: true, centre: true });
      d.cote([A[0].x, A[0].y + ey * 0.30], [A[0].x + v(fd.debord_interieur), A[0].y + ey * 0.30], 0, fmt(v(fd.debord_interieur)), { sansRappel: true, court: true, centre: true, dessous: true });
    }
    if (o.fondation && vr) {
      var yr = -(v(vr.cote) / 2 + (ey / 2 - v(vr.cote) / 2) * 0.55);
      d.ligne([0, -v(vr.cote) / 2], [0, yr + d.px(12)], 'sc');
      d.texte([0, yr], 'réservation ' + fmt(v(vr.cote)) + ' × ' + fmt(v(vr.cote)) + ', prof. ' + fmt(v(vr.profondeur)) + ' (variante dalle)', 'lg', { dy: 4 });
    }
    /* mât */
    if (tm && !mInd) {
      d.cote([-tm / 2, tm / 2], [tm / 2, tm / 2], 16, (ch.mat.entraxes_de_liaison ? '□ ' : '') + txt(ch.mat.entraxe) + '  mât', { mesure: origine(ch.mat.entraxe) === 'mesure' });
    } else if (tm && mInd) {
      d.texte([0, tm / 2], 'mât (schématique)', 'lg', { dy: -6 });
    }
    if (ch.mat && ch.mat.entraxes_de_liaison) {
      var m0 = ch.mat.entraxes_de_liaison[0];
      d.cote([-v(m0) / 2, -v(m0) / 2], [v(m0) / 2, -v(m0) / 2], -14, '□ ' + String(v(m0)).replace('.', ','), {});
    }
    /* platine de l'appui 3 */
    var e3 = E[2], a3 = A[2];
    if (e3) {
      var ap = ch.pied.appui;
      if (e3.forme === 'rectangle' && e3.rotation_deg === 45) {
        var S = e3.sommets;   /* [-t-r, +t-r, +t+r, -t+r] */
        d.cote(S[2], S[3], 15 + Math.max(0, Le - v(ap.b) / 2 - (v(ap.excentrement_radial) || 0)) * d.s, txt(ap.a), { mesure: origine(ap.a) === 'mesure' });   /* bord extérieur : dimension a */
        d.cote(S[1], S[2], 15, txt(ap.b), { mesure: origine(ap.b) === 'mesure' });       /* bord latéral : dimension b */
      } else if (e3.forme === 'rectangle') {
        var S2 = e3.sommets;
        d.cote(S2[3], S2[2], 12, txt(ap.a), {});
        d.cote(S2[1], S2[2], -12, txt(ap.b), {});
      } else if (e3.forme === 'cercle') {
        d.repere(add(e3.centre, [0.7071, 0.7071], e3.rayon), 22, -22, 'Ø ' + txt(ap.diametre), 'ct');
      } else if (e3.forme === 'octogone') {
        d.cote([e3.centre[0] - e3.cote_sur_plats / 2, e3.centre[1] + e3.cote_sur_plats / 2], [e3.centre[0] + e3.cote_sur_plats / 2, e3.centre[1] + e3.cote_sur_plats / 2], 12, txt(ap.cote_sur_plats), {});
      }
    } else {
      d.repere([a3.x, a3.y], -24, -26, 'appui : dimension non indiquée', 'lg', 'end');
    }
    /* lest : étiquettes */
    if (o.lest && ch.lest_central && ch.lest_central.blocs) {
      ch.lest_central.blocs.forEach(function (b, i) {
        var c0 = b.contours && b.contours[0]; if (!c0) return;
        var cx = 0, cy = 0; c0.forEach(function (q) { cx += q[0]; cy += q[1]; }); cx /= c0.length; cy /= c0.length;
        d.texte([cx, cy], 'lest ' + b.nom.replace('Bloc ', '') + '  ' + fmt(v(b.longueur)) + ' × ' + fmt(v(b.largeur)), 'lg', { dy: 4, rot: Math.abs(cx) > Math.abs(cy) ? -90 : 0 });
      });
    }
    return d.svg({ label: 'Vue en plan – ' + ch.nom });
  }

  /* ================================================================== ÉLÉVATION */
  function elevation(ch, o) {
    o = Object.assign({ W: 760, H: 520, titre: 'Élévation' }, o || {});
    return ch.type === 'chassis_bracons' ? elevBracons(ch, o) : elevCroix(ch, o);
  }

  function largeurVuePlatine(ch) {      /* largeur de l'emprise d'appui vue de côté */
    var ap = ch.pied && ch.pied.appui; if (!ap) return null;
    if (ap.forme === 'rectangle') return ap.rotation_deg === 45 ? (v(ap.a) + v(ap.b)) / R2 : v(ap.a);
    if (ap.forme === 'cercle') return v(ap.diametre);
    if (ap.forme === 'octogone') return v(ap.cote_sur_plats);
    return null;
  }
  function matTreillis(d, t, lw, z0, z1, cls) {       /* mât : deux membrures + diagonales schématiques */
    d.rect(-t / 2 - lw / 2, z0, -t / 2 + lw / 2, z1, cls === 'ti' ? 'ti' : 'ac');
    d.rect(t / 2 - lw / 2, z0, t / 2 + lw / 2, z1, cls === 'ti' ? 'ti' : 'ac');
    var n = Math.max(2, Math.round((z1 - z0) / (t * 0.9))), h = (z1 - z0) / n, pts = [], i;
    for (i = 0; i <= n; i++) pts.push([(i % 2 ? 1 : -1) * (t / 2 - lw / 2), z0 + i * h]);
    d.poly(pts, cls === 'ti' ? 'ti' : 'sc', true);
    d.ligne([-t / 2, z1], [t / 2, z1], cls === 'ti' ? 'ti' : 'tr');
  }

  function d0(ex) { return ex * 0.2 + 300; }    /* espace entre les deux vues, en mm */
  /* --- châssis à bracons (Liebherr HC) */
  function elevBracons(ch, o) {
    var d = new Dessin(o.W, o.H, o.titre); d.options = { sansStyle: o.sansStyle, couleursFixes: o.couleursFixes };
    var el = ch.elevation, po = el.poutre, ex = v(ch.appuis.entraxe_x);
    var zb0 = v(po.sous_face), zb1 = v(po.dessus), Lb = v(po.demi_longueur), H = v(el.hauteur_totale);
    var t = v(ch.mat.entraxe), lw = v(ch.mat.hors_tout) - t;
    var bp = v(el.bracon.pied), bt = v(el.bracon.tete);
    var fo = ch.fondation || {}, vp = fo.variante_plots, vr = fo.variante_reservation;
    var csp = fo.centre_sous_plan_appui ? v(fo.centre_sous_plan_appui) : null;
    var cau = fo.centre_au_dessus_plan_appui ? v(fo.centre_au_dessus_plan_appui) : null;
    var zc = csp !== null ? -csp : (cau !== null ? cau : zb0);
    var wv = largeurVuePlatine(ch), tp = v(ch.pied.epaisseur_platine) || 30;
    var sur = vp ? v(vp.surelevation) : 0;
    var lm = vp && vp.cote_massif ? v(vp.cote_massif) : Math.max(1200, wv + 500);
    var cw = { 4600: 550, 6000: 1100, 8000: 900 }[ex] || 800; if (/200hc/.test(ch.id)) cw = 350;
    var xg = Math.max(Lb, ex / 2 + lm / 2), prof = sur + 420;

    /* seconde vue (variante dalle avec réservation) dessinée sous la première */
    var dz = vr ? -(prof + zb1 + 420 + d0(ex)) : 0;
    d.cadrer(-xg, vr ? dz - 650 : -prof, xg, H, { g: 78, d: 96, h: 44, b: vr ? 40 : 74 });

    function base(z, variante) {     /* tout ce qui est sous les poutres, à l'altitude z du plan d'appui */
      /* fondation */
      if (variante === 'plots') {
        [-1, 1].forEach(function (sg) { d.beton(sg * ex / 2 - lm / 2, z - prof, sg * ex / 2 + lm / 2, z, 'p' + sg); });
        if (sur) { d.beton(-ex / 2 + lm / 2, z - prof, ex / 2 - lm / 2, z - sur, 'm'); }
      } else if (variante === 'reservation') {
        var rc = v(vr.cote) / 2, pr = v(vr.profondeur);
        d.beton(-xg, z - 650, -rc, z, 'g'); d.beton(rc, z - 650, xg, z, 'd'); d.beton(-rc, z - 650, rc, z - pr, 'c');
      } else {
        d.beton(-xg, z - prof, xg, z, 'f');
      }
      /* pieds */
      [-1, 1].forEach(function (sg) {
        var x = sg * ex / 2, ty = ch.pied.type;
        if (ty === 'pied_a_vis') {
          var hp = v(ch.pied.hauteur_sous_poutre), D = v(ch.pied.appui.diametre);
          d.poly([[x - 120, z + tp], [x + 120, z + tp], [x + 120, z + hp * 0.60], [x + 250, z + hp * 0.86], [x + 250, z + hp], [x - 250, z + hp], [x - 250, z + hp * 0.86], [x - 120, z + hp * 0.60]], 'ac');
          d.rect(x - D / 2, z, x + D / 2, z + tp, 'ap');
        } else if (ty === 'sabot_pyramidal') {
          var hs = v(ch.pied.hauteur_sous_poutre), ds = v(ch.pied.dessus_sabot) || wv * 0.6;
          d.poly([[x - wv / 2, z + tp], [x + wv / 2, z + tp], [x + ds / 2, z + hs * 0.45], [x + ds / 2, z + hs], [x - ds / 2, z + hs], [x - ds / 2, z + hs * 0.45]], 'ac');
          d.rect(x - wv / 2, z, x + wv / 2, z + tp, 'ap');
        } else {
          d.rect(x - wv / 2, z, x + wv / 2, z + tp, 'ap');
        }
      });
      /* partie centrale puis poutres */
      d.rect(-cw / 2, z + zc, cw / 2, z + zb1 + 60, 'ac');
      d.rect(-Lb, z + zb0, Lb, z + zb1, 'ac');
      d.ligne([-ex / 2, z + (zb0 + zb1) / 2], [ex / 2, z + (zb0 + zb1) / 2], 'ax');
    }

    /* vue principale */
    base(0, vp ? 'plots' : 'dalle');
    /* bracons */
    [-1, 1].forEach(function (sg) {
      d.poly(bande([sg * bp[0], bp[1]], [sg * bt[0], bt[1]], Math.max(110, ex * 0.028)), 'ac');
      d.cercle([sg * bp[0], bp[1]], 45, 'tr'); d.cercle([sg * bt[0], bt[1]], 45, 'tr');
      d.poly([[sg * (bp[0] - 190), zb1], [sg * bp[0], bp[1] + 95], [sg * (bp[0] + 190), zb1]], 'sc', true);
    });
    /* mât de base */
    matTreillis(d, t, lw, zb1, H, 'ac');
    d.ligne([-t / 2 - lw / 2, H], [t / 2 + lw / 2, H], 'tr');
    d.ligne([0, H + d.px(8)], [0, zc - d.px(6)], 'ax');

    /* cotes */
    var xd = xg;
    d.cote([xd, 0], [xd, H], -34, txt(el.hauteur_totale), { mesure: origine(el.hauteur_totale) === 'mesure' });
    if (el.hauteur_depuis_niveau_bas) d.cote([xd, -sur], [xd, H], -64, txt(el.hauteur_depuis_niveau_bas), {});
    d.cote([-t / 2, H], [t / 2, H], 14, txt(ch.mat.entraxe) + '  mât', {});
    var zbas = vr ? -prof : -prof;
    d.cote([-ex / 2, 0], [ex / 2, 0], -(prof * d.s + 20), txt(ch.appuis.entraxe_x), { jeu: 6 });
    if (el.largeur_hors_tout && origine(el.largeur_hors_tout) === 'cote') d.cote([-v(el.largeur_hors_tout) / 2, -prof], [v(el.largeur_hors_tout) / 2, -prof], -46, txt(el.largeur_hors_tout) + '  hors-tout', {});
    /* poutres : niveau du dessus, s'il est coté sur la source */
    if (origine(po.dessus) === 'cote') d.cote([-xg, 0], [-xg, zb1], 22, txt(po.dessus), { court: true, cote: 'g' });
    /* plots et dépassement central */
    if (vp && sur) d.cote([-ex / 2 + lm / 2 + d.px(26), -sur], [-ex / 2 + lm / 2 + d.px(26), 0], 0, txt(vp.surelevation), { sansRappel: true, court: true, mesure: origine(vp.surelevation) === 'mesure' });
    if (fo.centre_sous_poutres) d.cote([cw / 2 + d.px(24), zb0 - v(fo.centre_sous_poutres)], [cw / 2 + d.px(24), zb0], 0, txt(fo.centre_sous_poutres), { sansRappel: true, court: true });
    else if (csp !== null) d.cote([cw / 2 + d.px(24), -csp], [cw / 2 + d.px(24), 0], 0, txt(fo.centre_sous_plan_appui), { sansRappel: true, court: true, mesure: true });
    /* étiquettes */
    if (ch.pied.type === 'pied_a_vis') d.repere([-ex / 2 + wv / 2, tp], 30, -26, 'platine Ø ' + fmt(v(ch.pied.appui.diametre)) + ', ép. ' + fmt(tp), 'lg', 'start');
    else d.repere([-ex / 2 - wv / 2, tp / 2], -26, 26, 'platine ép. ' + txt(ch.pied.epaisseur_platine), 'lg', 'end');
    if (ch.pied.type === 'pied_a_vis' || ch.pied.type === 'sabot_pyramidal') d.cote([-xg, 0], [-xg, v(ch.pied.hauteur_sous_poutre)], 22, txt(ch.pied.hauteur_sous_poutre), { court: true, mesure: true, cote: 'g' });
    d.texteF(12, o.H - 10, vp ? 'Vue du haut : châssis sur massifs surélevés.' + (vr ? ' Vue du bas : dalle avec réservation centrale.' : '') : 'Plan d\'appui = dessus de la fondation.', 'lg');

    /* seconde vue : dalle avec réservation */
    if (vr) {
      base(dz, 'reservation');
      d.ligne([-xg, dz + zb1 + 420], [xg, dz + zb1 + 420], 'ti');
      [-1, 1].forEach(function (sg) {   /* amorces des bracons et du mât */
        var k = (zb1 + 420 - bp[1]) / (bt[1] - bp[1]);
        d.poly(bande([sg * bp[0], dz + bp[1]], [sg * (bp[0] + (bt[0] - bp[0]) * k), dz + zb1 + 420], Math.max(110, ex * 0.028)), 'ac');
        d.rect(sg * t / 2 - lw / 2, dz + zb1, sg * t / 2 + lw / 2, dz + zb1 + 420, 'ac');
      });
      var rc2 = v(vr.cote) / 2, pr2 = v(vr.profondeur);
      d.cote([-rc2, dz - pr2], [rc2, dz - pr2], -16, '■ ' + txt(vr.cote), {});
      d.cote([-rc2 - d.px(20), dz - pr2], [-rc2 - d.px(20), dz], 0, txt(vr.profondeur), { sansRappel: true, court: true, cote: 'g' });
      d.texte([-xg, dz + zb1 + 420], 'Variante : dalle avec réservation centrale', 'lg', { ancre: 'start', dy: -7 });
    }
    return d.svg({ label: 'Élévation – ' + ch.nom });
  }

  /* --- croix (Wolff KR, croix 63 LC, base cruciforme) */
  function elevCroix(ch, o) {
    var d = new Dessin(o.W, o.H, o.titre); d.options = { sansStyle: o.sansStyle, couleursFixes: o.couleursFixes };
    var el = ch.elevation || {}, ex = v(ch.appuis.entraxe_x), fo = ch.fondation || {};
    var ht = v(ch.appuis.hors_tout), wv = largeurVuePlatine(ch);
    var py = ch.pied.type === 'pyramide';
    var sous = fo.centre_sous_plan_appui ? v(fo.centre_sous_plan_appui) : 0;
    var tp = v(ch.pied.epaisseur_platine) || (py ? 0 : 30);
    var z0, z1, Lf, tm, hm = null, zs;          /* dessous/dessus du châssis, demi-longueur, largeur et hauteur du mât */
    var mInd = false;
    if (py) {
      z1 = v(el.hauteur_totale); z0 = z1 - v(el.hauteur_poutre); Lf = v(ch.appuis.hors_tout_acier) / 2; tm = v(ch.mat.entraxe);
    } else if (el.hauteur_chassis) {            /* Wolff */
      z0 = tp; z1 = v(el.hauteur_chassis); Lf = ht / 2; tm = v(ch.mat.largeur_dessin); mInd = true;
      if (el.hauteurs_mat_de_base && el.hauteurs_mat_de_base.length) hm = v(el.hauteurs_mat_de_base[0]);
    } else {                                    /* base cruciforme */
      z0 = tp; z1 = tp + v(el.hauteur_poutre); Lf = ex / 2 + (depassementBras(ch) + v(el.largeur_semelle) / 2) / R2; tm = v(ch.mat.entraxe); mInd = false;
    }
    zs = hm ? z1 + hm : (py ? z1 + 1500 : (el.hauteur_chassis ? z1 + 800 : v(el.hauteur_totale) + 1300));
    var lm = Math.max(1100, (wv || 500) + 600), prof = Math.max(320, sous + 170);
    var xg = Math.max(Lf, ex / 2 + lm / 2);
    d.cadrer(-xg, -prof, xg, zs, { g: 74, d: 100, h: 46, b: 76 });

    /* fondation : massifs sous les pieds, niveau bas entre les deux si la partie centrale descend */
    [-1, 1].forEach(function (sg) { d.beton(sg * ex / 2 - lm / 2, -prof, sg * ex / 2 + lm / 2, 0, 'p' + sg); });
    if (sous) d.beton(-ex / 2 + lm / 2, -prof, ex / 2 - lm / 2, -prof + 120, 'm');
    else d.beton(-ex / 2 + lm / 2, -prof, ex / 2 - lm / 2, 0, 'm');

    /* pieds */
    [-1, 1].forEach(function (sg) {
      var x = sg * ex / 2;
      if (py) {
        var hp = v(ch.pied.hauteur_pyramide), b = v(ch.pied.appui.cote_sur_plats), s2 = v(ch.pied.sommet_pyramide_diametre);
        d.poly([[x - b / 2, 0], [x + b / 2, 0], [x + s2 / 2, hp], [x - s2 / 2, hp]], 'ap');
        d.rect(x - 200, hp, x + 200, hp + 60, 'ac'); d.rect(x - 60, hp + 60, x + 60, z0, 'ac');
      } else if (wv) {
        d.rect(x - wv / 2, 0, x + wv / 2, tp, 'ap');
      } else {
        d.ligne([x - 150, 0], [x + 150, 0], 'apl');
      }
    });
    /* partie centrale sous le plan d'appui */
    if (sous) d.rect(-260, -sous, 260, z0 + 40, 'ac');
    /* châssis */
    d.rect(-Lf, z0, Lf, z1, 'ac');
    if (!py && !el.hauteur_chassis) {            /* base cruciforme : liaison du mât au-dessus des poutres */
      var zt = v(el.hauteur_totale);
      [-1, 1].forEach(function (sg) { d.rect(sg * tm / 2 - 170, z1, sg * tm / 2 + 170, zt, 'ac'); });
    }
    /* mât (schématique) */
    var zm0 = (!py && !el.hauteur_chassis) ? v(el.hauteur_totale) : z1;
    if (hm || py || !el.hauteur_chassis) {
      matTreillis(d, tm, tm * 0.07, zm0, zs, 'ti');
      d.texte([0, (zm0 + zs) / 2], 'mât (schématique)', 'lg', { dy: 4 });
    } else {                                     /* KRV 10-60 : seul le niveau de liaison est donné */
      d.ligne([-tm / 2 - 300, zs], [tm / 2 + 300, zs], 'ti');
      d.texte([0, zs], 'niveau de liaison du mât', 'lg', { dy: -6 });
    }
    d.ligne([0, zs + d.px(8)], [0, -sous - d.px(6)], 'ax');
    [-1, 1].forEach(function (sg) { d.ligne([sg * ex / 2, -prof - d.px(4)], [sg * ex / 2, z1 + d.px(14)], 'ax'); });

    /* cotes */
    d.cote([-ex / 2, 0], [ex / 2, 0], -(prof * d.s + 20), txt(ch.appuis.entraxe_x), { jeu: 6 });
    if (ht) d.cote([-ht / 2, -prof], [ht / 2, -prof], -46, txt(ch.appuis.hors_tout) + '  ' + (ch.appuis.hors_tout.nature || ''), { mesure: origine(ch.appuis.hors_tout) !== 'cote' });
    var xd = xg;
    if (py) {
      d.cote([xd, 0], [xd, z1], -34, txt(el.hauteur_totale) + '  (min. ' + ch.pied.hauteur_totale.min + ' / max. ' + ch.pied.hauteur_totale.max + ')', {});
      d.cote([-xg, 0], [-xg, v(ch.pied.hauteur_pyramide)], 26, txt(ch.pied.hauteur_pyramide), {});
    } else if (el.hauteur_chassis) {
      d.cote([xd, 0], [xd, z1], -34, txt(el.hauteur_chassis), {});
      if (hm) d.cote([xd, z1], [xd, zs], -34, txt(el.hauteurs_mat_de_base[0]), {});
      else d.cote([xd, z1], [xd, zs], -34, '800', {});
      if (sous) d.cote([xd, -sous], [xd, 0], -34, txt(fo.centre_sous_plan_appui), { court: true });
    } else {
      d.cote([xd, 0], [xd, v(el.hauteur_totale)], -34, txt(el.hauteur_totale), {});
    }
    if (!py && wv && v(ch.pied.epaisseur_platine)) d.repere([-ex / 2 - wv / 2, tp / 2], -24, 24, 'platine ép. ' + txt(ch.pied.epaisseur_platine), 'lg', 'end');
    var note = '';
    if (el.hauteurs_mat_de_base && el.hauteurs_mat_de_base.length > 1) note = 'Élément de mât de base : ' + el.hauteurs_mat_de_base.map(function (h) { return h.texte; }).join(' ; ') + '.';
    else if (sous) note = 'La partie centrale descend de ' + fmt(sous) + ' mm sous le plan d\'appui.';
    else if (py) note = 'Hauteur réglable par les pieds à vis.';
    if (ch.lest_central && ch.lest_central.designations) note += (note ? ' ' : '') + 'Lest central : ' + ch.lest_central.designations.join(', ') + '.';
    if (!py && wv && !v(ch.pied.epaisseur_platine)) note += (note ? ' ' : '') + 'Épaisseur de platine non indiquée.';
    if (note) d.texteF(12, o.H - 10, note, 'lg');
    return d.svg({ label: 'Élévation – ' + ch.nom });
  }

  /* ================================================================== DÉTAIL DU PIED – VUE EN PLAN (appui 3) */
  function piedPlan(ch, o) {
    o = Object.assign({ W: 372, H: 340, titre: 'Pied – vue en plan' }, o || {});
    var d = new Dessin(o.W, o.H, o.titre); d.options = { sansStyle: o.sansStyle, couleursFixes: o.couleursFixes };
    var a = appuis(ch)[2], e = emprise(ch, a), ap = ch.pied.appui;
    var wb = v((ch.dessin || {}).bras_largeur) || 300, Le = depassementBras(ch);
    var c0 = [a.x, a.y];
    var R = e ? 0 : 320;
    if (e) e.sommets.forEach(function (q) { R = Math.max(R, Math.hypot(q[0] - c0[0], q[1] - c0[1])); });
    var sb = ch.pied.sabot;
    if (sb) R = Math.max(R, Math.hypot(v(sb.a) / 2, v(sb.axe_au_bord_exterieur)));
    R = Math.max(R, Le + wb / 2);
    var m = R * 1.28;
    d.cadrer(c0[0] - m, c0[1] - m, c0[0] + m, c0[1] + m, { g: 30, d: 30, h: 34, b: 26 });
    /* bras */
    var p0 = add(c0, a.r, -m * 1.6), p1 = add(c0, a.r, Le);
    d.raw('<clipPath id="pp' + ch.id + '"><rect x="2" y="26" width="' + (o.W - 4) + '" height="' + (o.H - 30) + '"/></clipPath><g clip-path="url(#pp' + ch.id + ')">');
    if (ch.pied.type === 'pyramide') { d.poly(e.sommets, 'ap'); d.cercle(c0, v(ch.pied.sommet_pyramide_diametre) / 2, 'apl'); }
    if (sb) { var es = v(sb.axe_au_bord_exterieur) - v(sb.b) / 2; d.poly(rectOriente(add(c0, a.r, es), a.t, a.r, v(sb.a) / 2, v(sb.b) / 2), 'scf'); }
    if (e && ch.pied.type !== 'pyramide' && e.forme !== 'cercle') d.poly(e.sommets, 'ap');
    d.poly(bande(p0, p1, wb), 'ac');
    if (e && e.forme === 'cercle') d.cercle(e.centre, e.rayon, 'ap');
    /* trous oblongs des broches */
    var br = ch.pied.broches, to = br && br.trous_oblongs ? v(br.trous_oblongs) : (sb && sb.trous_oblongs ? v(sb.trous_oblongs) : null);
    if (to && (e && e.forme === 'rectangle' || sb)) {
      var da = sb ? v(sb.a) : v(ap.a), pos = br && br.position ? v(br.position) : da / 2 - 50, cc = sb ? add(c0, a.r, v(sb.axe_au_bord_exterieur) - v(sb.b) / 2) : e.centre;
      [-1, 1].forEach(function (sg) {
        var c = add(cc, a.t, sg * pos), l = (to[1] - to[0]) / 2;
        var P = d.T(add(c, a.r, -l)), Q = d.T(add(c, a.r, l));
        d.raw('<line x1="' + f1(P[0]) + '" y1="' + f1(P[1]) + '" x2="' + f1(Q[0]) + '" y2="' + f1(Q[1]) + '" stroke="var(--cg-encre,#1d2b3a)" stroke-width="' + f1(to[0] * d.s) + '" stroke-linecap="round"/>');
        d.raw('<line x1="' + f1(P[0]) + '" y1="' + f1(P[1]) + '" x2="' + f1(Q[0]) + '" y2="' + f1(Q[1]) + '" stroke="var(--cg-fond,#fff)" stroke-width="' + f1(Math.max(0.5, to[0] * d.s - 2)) + '" stroke-linecap="round"/>');
      });
    }
    d.raw('</g>');
    /* axes */
    d.ligne([c0[0] - m, c0[1]], [c0[0] + m, c0[1]], 'ax'); d.ligne([c0[0], c0[1] - m], [c0[0], c0[1] + m], 'ax');
    /* cotes */
    if (e && e.forme === 'rectangle') {
      var S = e.sommets;
      var ma = origine(ap.a) === 'mesure', mb = origine(ap.b) === 'mesure';
      if (e.rotation_deg === 45) {
        d.cote(S[2], S[3], 18 + Math.max(0, Le - v(ap.b) / 2 - (v(ap.excentrement_radial) || 0)) * d.s, txt(ap.a), { mesure: ma });
        d.cote(S[1], S[2], 18, txt(ap.b), { mesure: mb });
        var exc = v(ap.excentrement_radial) || 0;
        if (exc) {
          var q0 = add(c0, a.t, v(ap.a) / 2), ext2 = v(ap.b) / 2 + exc;
          d.cote(add(c0, a.t, -v(ap.a) / 2), add(add(c0, a.t, -v(ap.a) / 2), a.r, ext2), -18, fmt(ext2), { dessous: true });
          d.cote(add(add(c0, a.t, -v(ap.a) / 2), a.r, -(v(ap.b) - ext2)), add(c0, a.t, -v(ap.a) / 2), -18, fmt(v(ap.b) - ext2), { dessous: true });
        }
      } else {
        d.cote(S[3], S[2], 14, txt(ap.a), { mesure: ma });
        d.cote(S[1], S[2], -14, txt(ap.b), { mesure: mb });
      }
    } else if (e && e.forme === 'cercle') {
      d.cote(add(e.centre, a.t, -e.rayon), add(e.centre, a.t, e.rayon), 0, 'Ø ' + txt(ap.diametre), { sansRappel: true });
      if (sb) {
        var cs = add(c0, a.r, v(sb.axe_au_bord_exterieur) - v(sb.b) / 2), Ss = rectOriente(cs, a.t, a.r, v(sb.a) / 2, v(sb.b) / 2);
        d.cote(Ss[2], Ss[3], 18, fmt(v(sb.a)) + '  sabot', {}); d.cote(Ss[1], Ss[2], 18, fmt(v(sb.b)), {});
      }
    } else if (e && e.forme === 'octogone') {
      var hc = e.cote_sur_plats / 2;
      d.cote([c0[0] - hc, c0[1] + hc], [c0[0] + hc, c0[1] + hc], 14, txt(ap.cote_sur_plats), {});
      d.repere(add(c0, [-0.7071, 0.7071], v(ch.pied.sommet_pyramide_diametre) / 2), -20, -18, 'sommet Ø ' + txt(ch.pied.sommet_pyramide_diametre), 'lg', 'end');
    } else {
      d.texte(c0, 'dimension d\'appui non indiquée', 'lg', { dy: -m * d.s * 0.55 });
    }
    d.texteF(o.W - 12, o.H - 8, '↙ centre du châssis', 'lg', 'end');
    return d.svg({ label: 'Pied, vue en plan – ' + ch.nom });
  }

  /* ================================================================== DÉTAIL DU PIED – COUPE PERPENDICULAIRE AU BRAS */
  function piedCoupe(ch, o) {
    o = Object.assign({ W: 372, H: 340, titre: 'Pied – coupe' }, o || {});
    var d = new Dessin(o.W, o.H, o.titre); d.options = { sansStyle: o.sansStyle, couleursFixes: o.couleursFixes };
    var ap = ch.pied.appui, ty = ch.pied.type;
    var wb = v(ch.pied.largeur_bras_sur_platine) || v((ch.dessin || {}).bras_largeur) || 300;
    var tpo = ch.pied.epaisseur_platine, tp = v(tpo) || 30;
    var W, hh;      /* largeur de l'appui dans la coupe, hauteur montrée au-dessus du plan d'appui */
    if (!ap) W = 600; else if (ap.forme === 'rectangle') W = v(ap.a); else if (ap.forme === 'cercle') W = v(ap.diametre); else W = v(ap.cote_sur_plats);
    var Wd = W;
    if (ty === 'pied_a_vis') { hh = v(ch.pied.hauteur_sous_poutre) + 260; Wd = Math.max(W, v(ch.pied.sabot.a)); }
    else if (ty === 'pyramide') hh = v(ch.pied.hauteur_totale);
    else if (ty === 'sabot_pyramidal') hh = v(ch.pied.hauteur_sous_poutre) + 300;
    else hh = Math.max(W * 0.95, 380);
    var pb = Math.max(W * 0.40, 220), mx = Wd * 0.22 + 70;
    d.cadrer(-Wd / 2 - mx, -pb, Wd / 2 + mx, hh, { g: 44, d: 74, h: 38, b: 58 });
    var xl = -Wd / 2 - mx, xr = Wd / 2 + mx, notes = [];
    d.beton(xl, -pb, xr, 0, 'c');

    function brasI(zb, zt) {       /* bras en I vu en bout, coupé en haut */
      var tf = Math.min(40, wb * 0.14), tw = Math.max(14, wb * 0.07);
      d.poly([[-wb / 2, zb], [wb / 2, zb], [wb / 2, zb + tf], [tw / 2, zb + tf], [tw / 2, zt], [-tw / 2, zt], [-tw / 2, zb + tf], [-wb / 2, zb + tf]], 'ac');
      d.poly([[-tw / 2 - d.px(9), zt + d.px(3)], [-tw / 4, zt - d.px(3)], [tw / 4, zt + d.px(3)], [tw / 2 + d.px(9), zt - d.px(3)]], 'sc', true);
    }
    if (ty === 'pyramide') {
      var hp = v(ch.pied.hauteur_pyramide), s2 = v(ch.pied.sommet_pyramide_diametre), zt = v(ch.pied.hauteur_totale), zb = zt - v(ch.elevation.hauteur_poutre);
      d.poly([[-W / 2, 0], [W / 2, 0], [s2 / 2, hp], [-s2 / 2, hp]], 'ap');
      d.rect(-200, hp, 200, hp + 60, 'ac'); d.rect(-60, hp + 60, 60, zb, 'ac');
      d.rect(-wb / 2, zb, wb / 2, zt, 'ac');
      d.cote([-W / 2, 0], [W / 2, 0], -(pb * d.s + 16), txt(ap.cote_sur_plats), { jeu: 4 });
      d.cote([xr, 0], [xr, hp], -14, txt(ch.pied.hauteur_pyramide), {});
      d.cote([xr, 0], [xr, zt], -44, txt(ch.pied.hauteur_totale), {});
      d.repere([60, (hp + zb) / 2], 30, 0, 'pied à vis', 'lg');
      d.texteF(12, o.H - 8, 'Hauteur réglable : min. ' + ch.pied.hauteur_totale.min + ' / max. ' + ch.pied.hauteur_totale.max + '.', 'lg');
    } else if (ty === 'pied_a_vis') {
      var hs = v(ch.pied.hauteur_sous_poutre), sa = v(ch.pied.sabot.a);
      d.rect(-W / 2, 0, W / 2, tp, 'ap');
      d.poly([[-120, tp], [120, tp], [120, hs * 0.60], [250, hs * 0.86], [250, hs - 30], [-250, hs - 30], [-250, hs * 0.86], [-120, hs * 0.60]], 'ac');
      d.rect(-sa / 2, hs - 30, sa / 2, hs, 'ac');
      brasI(hs, hh);
      d.cote([-W / 2, 0], [W / 2, 0], -(pb * d.s + 16), 'Ø ' + txt(ap.diametre), { jeu: 4 });
      d.cote([xr, 0], [xr, hs], -14, txt(ch.pied.hauteur_sous_poutre), { mesure: true });
      d.cote([-W / 2 - d.px(16), 0], [-W / 2 - d.px(16), tp], 0, txt(tpo), { sansRappel: true, court: true, cote: 'g' });
      d.cote([-sa / 2, hs], [sa / 2, hs], 16 + (hh - hs) * d.s * 0, fmt(sa) + '  sabot', { dessous: false });
    } else if (ty === 'sabot_pyramidal') {
      var h3 = v(ch.pied.hauteur_sous_poutre), ds = v(ch.pied.dessus_sabot);
      d.rect(-W / 2, 0, W / 2, tp, 'ap');
      d.poly([[-W / 2, tp], [W / 2, tp], [ds / 2, h3 * 0.45], [ds / 2, h3], [-ds / 2, h3], [-ds / 2, h3 * 0.45]], 'ac');
      brasI(h3, hh);
      d.cote([-W / 2, 0], [W / 2, 0], -(pb * d.s + 16), txt(ap.a), { jeu: 4 });
      d.cote([xr, 0], [xr, h3], -14, txt(ch.pied.hauteur_sous_poutre), { mesure: true });
      broches(W / 2 - 110);
    } else {
      if (ap) d.rect(-W / 2, 0, W / 2, tp, 'ap'); else d.ligne([-wb / 2 - 40, 0], [wb / 2 + 40, 0], 'apl');
      brasI(ap ? tp : 0, hh);
      if (ap) d.cote([-W / 2, 0], [W / 2, 0], -(pb * d.s + 16), txt(ap.a), { jeu: 4, mesure: origine(ap.a) === 'mesure' });
      if (ap && v(tpo)) d.cote([xr, 0], [xr, tp], -12, txt(tpo), { court: true, mesure: origine(tpo) === 'mesure' });
      if (ch.pied.butee_laterale_C) {            /* butées latérales Wolff */
        var wc = ap ? W / 2 : wb / 2 + 20;
        [-1, 1].forEach(function (sg) { d.rect(sg * wc, 0, sg * (wc + 34), 38, 'no'); });
        d.repere([-wc - 17, 38], -22, -30, 'C', 'lg', 'end');
        d.repere([wc + 17, 38], 22, -30, 'C', 'lg');
        notes.push('Butée latérale C : ' + ch.pied.butee_laterale_C.split(' (')[0] + '.');
      }
      if (ch.pied.broches && (ch.pied.broches.diametre || ch.pied.broches.trous_oblongs)) broches(ch.pied.broches.position ? v(ch.pied.broches.position) : W / 2 - 50);
      else if (ch.pied.broches) broches(W / 2 - 50);
    }
    function broches(pos) {
      var br = ch.pied.broches || {};
      [-1, 1].forEach(function (sg) { d.rect(sg * pos - 17.5, -pb * 0.72, sg * pos + 17.5, tp + 34, 'ac'); });
      d.repere([pos + 17.5, -pb * 0.36], 26, 14, br.diametre ? 'broche Ø ' + fmt(v(br.diametre)) : 'broche', 'lg');
      if (br.trous_oblongs) notes.push('Trous oblongs ' + v(br.trous_oblongs).join(' × ') + '.');
      else if (!br.diametre) notes.push('Broches non cotées sur cette feuille.');
    }
    d.ligne([0, hh + d.px(4)], [0, -pb - d.px(2)], 'ax');
    if (ty !== 'pyramide') d.texteF(12, o.H - 8, ['Coupe perpendiculaire au bras.'].concat(notes).join(' '), 'lg');
    return d.svg({ label: 'Pied, coupe – ' + ch.nom });
  }

  return {
    version: '1.0',
    valeur: v, texteCote: txt, formater: fmt,
    appuis: appuis, emprise: emprise, emprises: emprises, horsToutEmprises: horsToutEmprises, depassementBras: depassementBras,
    plan: plan, elevation: elevation, piedPlan: piedPlan, piedCoupe: piedCoupe,
    STYLE: STYLE, couleursFixes: couleursFixes
  };
});
