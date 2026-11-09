r"""
styles.py — la note de calcul et ses dix palettes.

Mise en page unique, validée : A4 paysage, une seule page.
  · colonne de gauche sur fond sobre : coupe cotée, puis dimensions,
    matériaux et sollicitations ;
  · deux colonnes de calcul à droite, numéro de vérification dans un carré ;
  · flux continu — une vérification qui ne tient pas dans la colonne se
    poursuit dans la suivante, avec un rappel « (suite) ».

Les dix variantes ne changent que la palette : teinte du panneau, couleur
d'accent, teinte du béton. Même grille, mêmes polices, même composition.
"""
from reportlab.lib.pagesizes import A4, landscape

A4P = A4

from . import mathx as M
from .kit import Doc, Frame, hx, mix, sym, formula, kv_rows, chip, table
from .section import SectionStyle, draw_section
from . import data as D

A4L = landscape(A4)


class Style:
    key = name = pitch = ""
    page = A4L
    # --- palette (seul point de variation entre les dix)
    ink, mut, rule = "#15181F", "#6E7480", "#DEE0E7"
    panel, acc, concrete = "#F3F4F7", "#33415C", "#E5E7EE"
    ok, att, ko = "#2E7D46", "#8A6B1F", "#9C3341"
    # --- polices (communes)
    f_h, f_b, f_v = "Poppins-Medium", "Carlito", "Poppins-Medium"
    m_var, m_rom, m_res = "Carlito-Italic", "Carlito", "Carlito-Bold"
    frac, cases, tight, cases_align = "inline", "rule", 0.88, True
    min_form = 6.0
    # --- corps
    s_form, s_kv, s_lab, s_verd, s_tit = 8.0, 7.8, 6.3, 7.3, 9.8
    # --- grille
    margin = 28
    coupe_w = 292
    coupe_h = 320
    gap = 24
    col_gap = 20
    n_cols = 2
    lead = 1.25
    title_gap = 13

    # ------------------------------------------------------------ outils
    def ms(self):
        return M.MathStyle(
            var=self.m_var, rom=self.m_rom, num=self.m_rom, op=self.m_rom,
            ink=hx(self.ink), accent=hx(self.acc), rule=hx(self.ink),
            res=self.m_res, res_color=hx(self.acc), op_color=hx(self.mut),
            frac=self.frac, cases=self.cases, tight=self.tight,
            cases_align=self.cases_align)

    def sec_style(self):
        # concrete_texture : convention de coupe béton (fond blanc, hachure
        # discrète, petits triangles et points) — demandé par le bureau le
        # 25/08/2026, remplace l'aplat gris de la maquette d'origine
        # dia_colors : couleur PAR DIAMÈTRE (PALETTE_DIA) — demandé par le
        # bureau le 31/08/2026, remplace le rendu « acier » du 25/08/2026
        return SectionStyle(
            mode="plein", panel=self.panel, concrete=self.concrete,
            show_hatch=False, ink=self.ink, accent=self.acc, bar=self.acc,
            muted=self.mut, rule_w=1.2, font=self.f_b, font_bold=self.f_v,
            title=None, dim_size=6.0, label_size=6.4, concrete_texture=True,
            dia_colors=True)

    # ------------------------------------------------------------- rendu
    def build(self, path, sections=None, doc=None, garde=True, fiches=None):
        """Rapport complet : page de garde portrait + une planche par section ; ``fiches`` (les fiches de la
        bibliothèque incluses dans la note, api_web/fiches.py) : après les planches, les pages « Justifications de la
        bibliothèque », annoncées au sommaire. Sans fiche, le document est celui d'avant, à l'octet près."""
        sections = sections if sections is not None else D.SECTIONS
        doc = doc or D.DOC
        d = Doc(path, A4P if garde else self.page,
                title=doc.get("titre", "Note de calcul"))
        d.warnings = []
        if garde:
            d.new_page(A4P)
            self.garde(d, doc, sections, fiches)
        for sec in sections:
            d.new_page(self.page)
            self.planche(d, doc, sec)
        if fiches:
            self.justifications(d, doc, fiches)
        return d

    # ------------------------------------------------- page de garde (portrait)
    LIGNE_SOMMAIRE = 46          # 24 + 22 points par entrée du sommaire
    LEGENDE_GARDE = ("Vert : vérifié   ·   Ocre : admissible, à la limite   ·   "
                     "Rouge : non vérifié")

    def garde(self, d, doc, sections, fiches=None):
        """Page de garde des notes Poutre / Dalle : une ligne de sommaire par
        planche (une page chacune), puis, s'il y en a, la ligne des
        justifications de la bibliothèque (``fiches``). Dessin inchangé :
        c'est ``garde_lignes`` qui porte la mise en page, pour que le rapport
        d'assemblages ait la même garde."""
        lignes = [dict(titre=f"{sec['poutre']} — {sec['section']}",
                       sous=f"{sec.get('beton', '')} / {sec.get('acier', '')}".strip(" /"),
                       etat=sec.get("etat", ""), couleur=self.coul_etat(sec), pages=1)
                  for sec in sections]
        if fiches:
            n = len(fiches)
            lignes.append(dict(titre="Justifications de la bibliothèque",
                               sous=f"{n} fiche{'s' if n > 1 else ''} — valeurs consultées, copiées dans la note",
                               etat="", couleur=self.acc, pages=self.pages_justifications(doc, fiches)))
        self.garde_lignes(d, doc, lignes)
        return d

    def garde_lignes(self, d, doc, lignes, legende=None, avant=None):
        """Garde générique : bandeau, cartouche, sommaire. Chaque ligne porte
        ``titre``, ``sous`` (facultatif), ``etat`` (libellé de la pastille,
        facultatif), ``couleur`` et ``pages`` (nombre de pages de l'entrée) ;
        ``avant`` = lignes NON numérotées placées en tête du sommaire (un
        récapitulatif, par exemple), paginées comme les autres.
        Les numéros de page tiennent compte du nombre de pages de garde —
        le sommaire CONTINUE sur une page suivante quand il déborde (au-delà
        de 11 lignes) — et du nombre de pages de chaque ligne. Retourne le
        nombre de pages de garde."""
        S = self
        W, H = d.W, d.H
        m = 52
        bas = 70                                    # au-dessus de la légende (y = 46)
        pas = self.LIGNE_SOMMAIRE
        cap1 = max(1, int((H - 261 - bas) // pas))  # sous le cartouche
        cap2 = max(1, int((H - 77 - bas) // pas))   # page de suite
        avant = list(avant or [])
        lignes = avant + list(lignes)
        n = len(lignes)
        n_garde = 1 + (0 if n <= cap1 else -(-(n - cap1) // cap2))
        debuts, page = [], n_garde + 1
        for lg in lignes:
            debuts.append(page)
            page += max(1, int(lg.get("pages", 1) or 1))
        legende = legende or self.LEGENDE_GARDE

        d.box(0, H - 132, W, 132, fill=S.panel)
        y = H - 52
        d.t(m, y, doc.get("bureau", ""), S.f_v, 8.6, S.acc, track=1.6)
        d.t(m, y - 30, doc.get("titre", "Note de calcul"), S.f_h, 26, S.ink)
        d.line(m, H - 132, W - m, H - 132, S.acc, 1.2)

        # cartouche : seuls les champs renseignés sont remplis
        fr = Frame(m, H - 132, W - 2 * m, m)
        champs = [("PROJET", doc.get("projet", "")),
                  ("PARTIE", doc.get("partie", "")),
                  ("DATE", doc.get("date", "")),
                  ("INDICE", str(doc.get("indice", "")))]
        for lab, val in champs:
            fr.down(21)
            d.t(fr.x, fr.y, lab, S.f_v, 7.0, S.mut, track=1.3)
            d.t(fr.x + 108, fr.y, val, S.f_b, 10, S.ink)
            d.line(fr.x + 108, fr.y - 5, fr.x1, fr.y - 5,
                   mix(S.rule, S.ink, 0.10), 0.5)

        # sommaire
        fr.down(40)
        d.t(fr.x, fr.y, "SOMMAIRE", S.f_v, 7.6, S.acc, track=1.6)
        fr.down(5)
        d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.9)
        cap, k = cap1, 0
        for i, lg in enumerate(lignes):
            if k >= cap:
                d.t(m, 46, legende, S.f_b, 7.4, S.mut)
                d.new_page((W, H))
                d.t(m, H - 52, doc.get("bureau", ""), S.f_v, 8.6, S.acc, track=1.6)
                d.t(m, H - 72, "SOMMAIRE (suite)", S.f_v, 7.6, S.acc, track=1.6)
                fr = Frame(m, H - 72, W - 2 * m, m)
                fr.down(5)
                d.line(fr.x, fr.y, fr.x1, fr.y, S.acc, 0.9)
                cap, k = cap2, 0
            fr.down(24)
            if i >= len(avant):
                d.t(fr.x, fr.y, f"{i - len(avant) + 1:02d}", S.f_h, 10.5, mix(S.acc, "#FFFFFF", 0.55))
            d.t(fr.x + 26, fr.y, lg["titre"], S.f_h, 11, S.ink)
            sous = lg.get("sous") or ""
            if sous:
                d.t(fr.x + 26, fr.y - 12, sous, S.f_b, 8.2, S.mut)
            e = lg.get("etat") or ""
            if e:
                chip(d, fr.x1, fr.y - 2, e.upper(), S.f_v, 6.4, "#FFFFFF",
                     lg.get("couleur") or S.ok, align="right")
            d.t(fr.x1, fr.y - 15, f"page {debuts[i]}", S.f_b, 7.6, S.mut, "right")
            fr.down(22)
            d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.06), 0.5)
            k += 1

        d.t(m, 46, legende, S.f_b, 7.4, S.mut)
        return n_garde

    def coul_etat(self, sec):
        """Couleur de la pastille d'état d'une section, d'après ses verdicts."""
        e = (sec.get("etat") or "").lower()
        if "non" in e:
            return self.ko
        etats = [k.get("etat", "ko" if not k.get("ok", False) else "ok")
                 for v in sec.get("verifs", []) for k in v.get("verdicts", [])]
        if "ko" in etats:
            return self.ko
        if "att" in etats:
            return self.att
        return self.ok

    # --------------------------------------------- une planche (paysage)
    def planche(self, d, doc, sec):
        S = self
        W, H = d.W, d.H
        m = S.margin
        ms = S.ms()

        # en-tête
        y = H - m - 2
        d.t(m, y, doc.get("bureau", ""), S.f_v, 8.4, S.ink)
        d.t(W - m, y, f"{doc.get('date', '')} · indice {doc.get('indice', '')}",
            S.f_b, 8, S.mut, "right")
        d.t(m, y - 18, f"{sec['poutre']} — {sec['section']}", S.f_h, 14, S.ink)
        wc = 0
        if sec.get("etat"):
            wc = chip(d, W - m, y - 20, sec["etat"].upper(), S.f_v, 6.6,
                      "#FFFFFF", self.coul_etat(sec), align="right")
        mat = f"{sec.get('beton', '')} / {sec.get('acier', '')}".strip(" /")
        if mat:
            d.t(W - m - wc - 8, y - 16, mat, S.f_v, 8, S.ink, "right")
        d.line(m, y - 28, W - m, y - 28, S.ink, 1.0)
        top = y - 38

        # colonne de gauche : figure puis données, sur fond sobre.
        # Extension rétro-compatible (notes acier) : une section peut fournir
        # son propre dessinateur `dessin(d, x, y, w, h, style)`, un titre de
        # panneau, une largeur et une hauteur de figure, un nombre de
        # colonnes ; sans ces clés, la coupe béton est dessinée comme avant.
        cw = sec.get("coupe_w", S.coupe_w)
        d.box(m - 10, m - 4, cw + 20, top - m + 14, fill=S.panel, r=6)
        C = Frame(m, top, cw, m)
        C.down(11)
        d.t(C.x, C.y, sec.get("titre_coupe", "COUPE DE SECTION"), S.f_v, 6.6,
            S.acc, track=1.4)
        C.down(4)
        d.line(C.x, C.y, C.x1, C.y, mix(S.rule, S.ink, 0.12), 0.5)
        blocs = sec.get("blocs", [])
        reserve = 26 + 13 * sum(1 + len(r) for _, r in blocs[:2]) if blocs else 0
        hc = min(sec.get("coupe_h", S.coupe_h), C.y - m - max(60, reserve))
        dessin = sec.get("dessin")
        if dessin is not None:
            C.down(hc)
            dessin(d, C.x, C.y, C.w, hc, S)
        elif sec.get("coupe") is not None:
            C.down(hc)
            draw_section(d.c, C.x, C.y, C.w, hc, sec["coupe"], S.sec_style(),
                         label_w=96)
        C.down(8)
        self._data(d, ms, C.x, C.y, C.w, blocs)

        # colonnes de calcul, en flux continu
        n_cols = sec.get("n_cols", S.n_cols)
        zx = m + cw + S.gap
        gw = (W - m - zx - S.col_gap * (n_cols - 1)) / n_cols
        cols = [Frame(zx + i * (gw + S.col_gap), top, gw, m)
                for i in range(n_cols)]
        for fr in cols[1:]:
            d.line(fr.x - S.col_gap / 2, top + 4, fr.x - S.col_gap / 2, m + 2,
                   S.rule, 0.45)

        fr = cols[0]
        for v in sec.get("verifs", []):
            if not fr.fits(58) and cols.index(fr) + 1 < len(cols):
                fr = cols[cols.index(fr) + 1]
            self._titre(d, fr, v)
            fr = self._items(d, fr, v, ms, cols)

        nom = f"{sec['poutre']} — {sec['section']}"
        for f in [C] + cols:
            if f.overflow() > 1.0:
                d.warnings.append(f"{nom} : {f.overflow():.0f} pt")
        return d

    # ---------------------------------------------------------- fragments
    def _data(self, d, ms, x, y, w, blocs, size=7.6, gap=14):
        """Dimensions, matériaux et sollicitations sous la coupe."""
        S = self
        parts = [0.42, 0.58]
        widths = [(w - gap) * p for p in parts]
        offs = [x, x + widths[0] + gap]
        low = y
        for i, (titre, rows) in enumerate(blocs):
            col = i % 2
            fr = Frame(offs[col], y if i < 2 else low, widths[col], 0)
            fr.down(9)
            d.t(fr.x, fr.y, titre, S.f_v, size - 1.3, S.acc, track=1.15)
            fr.down(3)
            d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.12), 0.5)
            kv_rows(d, fr, rows, ms, S.f_b, S.f_v, size, S.ink, S.mut,
                    mix(S.rule, S.ink, 0.06), mode="flat", sym_col=0.66,
                    lead=1.42)
            low = min(low, fr.y)
        return low

    def _titre(self, d, fr, v):
        """Numéro dans un carré plein, puis l'intitulé — réduit pour
        tenir dans la colonne (les intitulés longs, ex. « … — direction
        X (principale) », ne débordent jamais sur la colonne voisine)."""
        S, size = self, self.s_tit
        fr.down(S.title_gap)
        d.box(fr.x, fr.y - 2.4, size * 1.25, size * 1.15, fill=S.acc)
        d.t(fr.x + size * 0.62, fr.y + 0.4, str(v["num"]), S.f_v, size * 0.72,
            "#FFFFFF", "center")
        avail = fr.x1 - (fr.x + size * 1.25 + 8)
        tsize = size
        wt = d.w(v["titre"], S.f_h, size)
        if wt > avail > 0:
            tsize = max(7.0, size * avail / wt)
        d.t(fr.x + size * 1.25 + 8, fr.y, v["titre"], S.f_h, tsize, S.ink)
        fr.down(size * 0.55)
        return fr.y

    def _suite(self, d, fr, v):
        """Rappel en tête de colonne quand une vérification se poursuit."""
        S = self
        fr.down(9)
        d.box(fr.x, fr.y - 1.6, 6.6, 6.6, fill=mix(S.acc, "#FFFFFF", 0.45))
        d.t(fr.x + 3.3, fr.y - 0.2, str(v["num"]), S.f_v, 5.0, "#FFFFFF",
            "center")
        suite_txt = f"{v['titre']} (suite)"
        ssize = S.s_lab + 0.6
        ws = d.w(suite_txt, S.f_b, ssize)
        avail = fr.x1 - (fr.x + 12)
        if ws > avail > 0:
            ssize = max(5.4, ssize * avail / ws)
        d.t(fr.x + 12, fr.y, suite_txt, S.f_b, ssize, S.mut)
        fr.down(4)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.1), 0.5)
        fr.down(5)
        return fr.y

    ETATS = {"ok": "ok", "att": "att", "ko": "ko"}

    def coul_verdict(self, k):
        """vert = vérifié · ocre = admissible, à la limite · rouge = non vérifié."""
        e = k.get("etat")
        if e is None:                       # rétrocompatibilité booléenne
            e = "ok" if k.get("ok", False) else "ko"
        return {"ok": self.ok, "att": self.att, "ko": self.ko}.get(e, self.ko)

    def _verdict(self, d, fr, k):
        S = self
        col = S.coul_verdict(k)
        size = S.s_verd
        lines = d.wrap(k["texte"], S.f_b, size, fr.w - 15)
        h = len(lines) * size * 1.28 + 6.5
        fr.down(5)
        t = fr.y
        d.box(fr.x, t - h, fr.w, h, fill=mix(col, "#FFFFFF", 0.90))
        d.box(fr.x, t - h + 1.5, 2.0, h - 3, fill=col)
        for i, ln in enumerate(lines):
            d.t(fr.x + 9, t - size * (1.02 + i * 1.28), ln, S.f_b, size, S.ink)
        fr.down(h)
        return fr.y

    def _items(self, d, fr, v, ms, cols):
        """Flux continu : on bascule de colonne dès que la place manque."""
        S = self

        def place(besoin):
            nonlocal fr
            if not fr.fits(besoin) and cols.index(fr) + 1 < len(cols):
                fr = cols[cols.index(fr) + 1]
                self._suite(d, fr, v)
            return fr

        for it in v["items"]:
            if it[0] == "f":
                node = M.parse(it[2])
                b = M.measure(node, ms, S.s_form)
                size = S.s_form
                if b.w > fr.w:
                    size = max(S.min_form, size * fr.w / b.w)
                    b = M.measure(node, ms, size)
                place(b.a + b.d + 5 * S.lead + S.s_lab * 1.7)
                formula(d, fr, it[1], it[2], ms, S.s_form, S.mut, S.f_b,
                        S.s_lab, lead=S.lead, label_mode="above",
                        min_size=S.min_form)
            elif it[0] == "v":
                place(S.s_kv * 1.45)
                kv_rows(d, fr, [(it[1], it[2], it[3], it[4])], ms, S.f_b,
                        S.f_v, S.s_kv, S.ink, S.mut, S.rule, mode="flat",
                        sym_col=0.52, lead=1.45)
            elif it[0] == "t":
                place(S.s_kv * 1.5)
                fr.down(S.s_kv * 1.5)
                # réduit pour tenir dans la colonne (« On prend … » longs)
                tsz = S.s_kv
                wt = d.w(it[1], S.f_v, tsz)
                if wt > fr.w > 0:
                    tsz = max(6.0, tsz * fr.w / wt)
                d.t(fr.x, fr.y, it[1], S.f_v, tsz, S.ink)
            elif it[0] == "s":
                place(S.s_kv * 3.2)
                fr.down(S.s_kv * 1.75)
                d.t(fr.x, fr.y, it[1].upper(), S.f_v, S.s_lab, S.mut, track=1.1)
            elif it[0] == "p":
                # paragraphe replié (formule en clair, valeurs introduites) :
                # petit corps gris, replié sur la largeur de colonne, jamais
                # réduit — extension rétro-compatible (notes acier)
                n = len(d.wrap(it[1], S.f_b, S.s_lab, fr.w))
                place(n * S.s_lab * 1.32 + 3)
                fr.down(2)
                d.para(fr, it[1], S.f_b, S.s_lab, S.mut, lead=1.32)
            else:
                k = v["verdicts"][it[1]]
                n = len(d.wrap(k["texte"], S.f_b, S.s_verd, fr.w - 15))
                place(5 + n * S.s_verd * 1.28 + 6.5)
                self._verdict(d, fr, k)
        return fr

    # --------------------------------- les justifications de la bibliothèque (api_web/fiches.py ; A4 paysage)
    # Deux colonnes par page ; chaque fiche COMMENCE dans une colonne neuve et se POURSUIT dans la suivante (puis sur la
    # page suivante) avec un rappel « J2 (suite) » quand elle est plus longue qu'une colonne — le flux continu des
    # planches (_items), jamais une fiche tronquée.
    N_COLS_FICHES = 2
    LEAD_TAB_FICHE = 8.0

    def _ms_col(self, col):
        """Le style mathématique de la note dans une couleur donnée (symboles composés des fiches)."""
        return M.MathStyle(var=self.m_var, rom=self.m_rom, num=self.m_rom, op=self.m_rom, ink=hx(col),
                           accent=hx(self.acc), rule=hx(col), res=self.m_res, res_color=hx(self.acc),
                           tight=self.tight)

    def pages_justifications(self, doc, fiches):
        """Le nombre de pages des justifications, par un tracé à blanc (le sommaire de la garde en a besoin avant
        qu'elles soient dessinées) — le même code que le tracé, donc le même compte."""
        import io
        d2 = Doc(io.BytesIO(), self.page)
        d2.warnings = []
        self.justifications(d2, doc, fiches)
        return d2.n

    def justifications(self, d, doc, fiches):
        """Les pages « Justifications de la bibliothèque » (A4 paysage, après les planches, le même en-tête) : les
        fiches incluses dans la note, numérotées J1, J2…, en deux colonnes — titre, valeur retenue, paramètres,
        lignes de justification avec leur règle, avertissements, extrait du tableau (la ligne et la case retenues en
        gras), sources, date et moteur. Une fiche plus longue qu'une colonne continue dans la suivante (« J2 (suite) »).
        L'en-tête de chaque page nomme ses fiches ; ce qui déborderait encore est dit dans ``d.warnings``."""
        S = self
        W, H, m = d.W, d.H, S.margin
        n_cols = self.N_COLS_FICHES
        etat = {"cols": None, "k": 0, "vus": []}

        def nommer_page():
            # les fiches de la page s'écrivent dans son en-tête quand on la quitte (on les connaît alors)
            vus = etat["vus"]
            if vus:
                txt = f"fiche J{vus[0]}" if len(vus) == 1 else f"fiches J{vus[0]} à J{vus[-1]}"
                d.t(W - m, H - m - 18, f"{txt} sur {len(fiches)}", S.f_v, 8, S.ink, "right")

        def nouvelle_page():
            if etat["cols"] is not None:
                nommer_page()
            d.new_page(self.page)
            y = H - m - 2
            d.t(m, y, doc.get("bureau", ""), S.f_v, 8.4, S.ink)
            d.t(W - m, y, f"{doc.get('date', '')} · indice {doc.get('indice', '')}", S.f_b, 8, S.mut, "right")
            d.t(m, y - 18, "Justifications de la bibliothèque", S.f_h, 14, S.ink)
            d.line(m, y - 28, W - m, y - 28, S.ink, 1.0)
            top = y - 38
            gw = (W - 2 * m - S.col_gap * (n_cols - 1)) / n_cols
            etat["cols"] = [Frame(m + i * (gw + S.col_gap), top, gw, m) for i in range(n_cols)]
            for fr in etat["cols"][1:]:
                d.line(fr.x - S.col_gap / 2, top + 4, fr.x - S.col_gap / 2, m + 2, S.rule, 0.45)
            etat["k"] = 0
            etat["vus"] = []

        def colonne_suivante():
            if etat["k"] + 1 < n_cols:
                etat["k"] += 1
            else:
                nouvelle_page()
            return etat["cols"][etat["k"]]

        nouvelle_page()
        for num, f in enumerate(fiches, 1):
            if num > 1:
                colonne_suivante()          # chaque fiche commence dans une colonne neuve
            etat["vus"].append(num)
            fr = self._fiche(d, etat["cols"][etat["k"]], f, num, colonne_suivante)
            if fr.overflow() > 1.0:
                d.warnings.append(f"justification J{num} : {fr.overflow():.0f} pt")
        nommer_page()
        return d

    def _suite_fiche(self, d, fr, num):
        """Rappel en tête de colonne quand une fiche se poursuit."""
        S = self
        fr.down(9)
        d.box(fr.x, fr.y - 1.6, 6.6, 6.6, fill=mix(S.acc, "#FFFFFF", 0.45))
        d.t(fr.x + 3.3, fr.y - 0.2, "J", S.f_v, 4.6, "#FFFFFF", "center")
        d.t(fr.x + 12, fr.y, f"J{num} (suite)", S.f_b, S.s_lab + 0.6, S.mut)
        fr.down(4)
        d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.1), 0.5)
        fr.down(5)
        return fr.y

    def _fiche(self, d, fr, f, num, colonne_suivante):
        """Une fiche dans le flux : ``colonne_suivante()`` rend la colonne où continuer quand la place manque (le
        rappel « (suite) » s'y écrit). Rend la colonne où la fiche s'achève."""
        from . import riche as RI
        S = self
        ms = self._ms_col(S.ink)
        ms_mut = self._ms_col(S.mut)
        size = S.s_tit
        lead = S.s_kv * 1.45
        s_rg = S.s_lab - 0.6

        def place(besoin):
            nonlocal fr
            if not fr.fits(besoin):
                fr = colonne_suivante()
                self._suite_fiche(d, fr, num)
            return fr

        # le titre : « J1 » dans un carré plein, puis l'intitulé (comme le numéro d'une vérification) ; la valeur
        # retenue en grand, son symbole ; la phrase des paramètres
        phrase = RI.auto(f.get("phrase") or "")
        n_ph = len(RI.envelopper(phrase, S.f_b, S.s_lab, ms_mut, fr.w)) if phrase else 0
        place(S.title_gap + size * 2.1 + 2 + min(n_ph, 3) * S.s_lab * 1.3 + lead + 6)
        fr.down(S.title_gap)
        wn = size * 1.9
        d.box(fr.x, fr.y - 2.4, wn, size * 1.15, fill=S.acc)
        d.t(fr.x + wn / 2, fr.y + 0.4, f"J{num}", S.f_v, size * 0.72, "#FFFFFF", "center")
        d.t(fr.x + wn + 8, fr.y, d.fit(f.get("titre") or "", S.f_h, size, fr.w - wn - 8), S.f_h, size, S.ink)
        fr.down(size * 2.1)
        val = f"{f.get('texte') or ''} {f.get('unite') or ''}".strip() or "—"
        d.t(fr.x, fr.y, val, S.f_h, 15, S.acc)
        if f.get("symbole"):
            RI.texte(d, fr.x + d.w(val, S.f_h, 15) + 8, fr.y, RI.auto(f["symbole"]), S.f_b, 8.5, S.mut, ms_mut)
        if phrase:
            fr.down(2)
            RI.paragraphe(d, fr, phrase, S.f_b, S.s_lab, S.mut, ms_mut, lead=1.3, max_lignes=3)
        fr.down(4)
        # les lignes de justification : symbole · nom · valeur, la règle dessous en petit
        for li in f.get("lignes") or ():
            regles = d.wrap(li.get("regle") or "", S.f_b, s_rg, fr.w - 14)
            place(lead + len(regles) * s_rg * 1.25 + 3.5)
            fr.down(lead)
            maj = (li.get("nom") or "").isupper()
            if maj:
                d.box(fr.x - 3, fr.y - S.s_kv * 0.42, fr.w + 6, lead * 0.98, fill=mix(S.acc, "#FFFFFF", 0.88))
            sw = RI.texte(d, fr.x, fr.y, RI.auto(li.get("symbole") or ""), S.f_b, S.s_kv, S.ink, ms)
            v = li.get("texte") or ("—" if li.get("valeur") is None else str(li["valeur"]))
            v = f"{v} {li.get('unite') or ''}".strip()
            f_val = S.f_v if maj else S.f_b
            wv = d.w(v, f_val, S.s_kv)
            d.t(fr.x1, fr.y, v, f_val, S.s_kv, S.ink, "right")
            nom = li.get("nom") or ""
            x_nom = fr.x + max(sw + 8, 46)
            d.t(x_nom, fr.y, d.fit(nom.capitalize() if maj else nom, S.f_b, S.s_kv, fr.x1 - wv - 8 - x_nom),
                S.f_b, S.s_kv, S.ink if maj else S.mut)
            for ln in regles:
                fr.down(s_rg * 1.25)
                d.t(fr.x + 14, fr.y, ln, S.f_b, s_rg, S.mut)
            fr.down(1.5)
            d.line(fr.x, fr.y - 2, fr.x1, fr.y - 2, mix(S.rule, S.ink, 0.06), 0.4)
        for a in f.get("avertissements") or ():
            n = len(d.wrap(a, S.f_b, S.s_verd, fr.w - 15))
            place(5 + n * S.s_verd * 1.28 + 6.5)
            self._verdict(d, fr, {"texte": a, "etat": "att"})
        # l'extrait du tableau (entier dans une colonne) : la ligne et la case retenues en gras
        x = f.get("extrait")
        if x and x.get("entetes"):
            ent = x["entetes"]
            n_col = len(ent)
            lignes_x = list(x.get("lignes") or ())
            st = 6.3 if n_col <= 6 else (5.9 if n_col <= 10 else 5.4)
            n_src = len(RI.envelopper(RI.auto(x.get("source") or ""), S.f_b, st * 0.9, ms_mut, fr.w)) if x.get("source") else 0
            place(S.s_kv * 1.9 + 5 + 12 + len(lignes_x) * (self.LEAD_TAB_FICHE + 1.2) + min(n_src, 3) * st * 0.9 * 1.24 + 8)
            fr.down(S.s_kv * 1.9)
            # (le titre tel quel, en symboles composés : « γ_s » ne devient pas « Γ_S »)
            RI.texte(d, fr.x, fr.y, RI.auto(x.get("titre") or ""), S.f_v, S.s_lab, S.mut, self._ms_col(S.mut), width=fr.w)
            fr.down(5)
            w0 = fr.w / n_col if n_col <= 2 else min(70.0, fr.w * 0.24)
            wc = (fr.w - w0) / (n_col - 1) if n_col > 1 else 0.0
            cols = [(RI.auto(ent[0]), w0, "left")] + [(RI.auto(e), wc, "right") for e in ent[1:]]
            case = x.get("case")
            lignes = []
            for i, li in enumerate(lignes_x):
                sur = bool(case) and case[0] == i
                lignes.append([("lab", li[0], sur)] + [("num", c, sur and case[1] == j) for j, c in enumerate(li[1:], 1)])
            table(d, fr, cols, lignes, S, st, self.LEAD_TAB_FICHE, s_ent=5.2, ms=self._ms_col, note=x.get("source") or None)
        # les sources, puis la date et le moteur
        if f.get("sources"):
            place(S.s_lab * 1.4 + (S.s_lab - 0.4) * 1.28)
            fr.down(S.s_lab * 1.4)
            for s in f["sources"]:
                n = len(d.wrap(s, S.f_b, S.s_lab - 0.4, fr.w))
                place(n * (S.s_lab - 0.4) * 1.28)
                d.para(fr, s, S.f_b, S.s_lab - 0.4, S.mut, lead=1.28)
        date = str(f.get("date") or "")
        if len(date) == 10 and date[4] == "-" and date[7] == "-":
            date = f"{date[8:]}/{date[5:7]}/{date[:4]}"      # l'état garde la date ISO ; la note l'écrit à la française
        pied = " · ".join(t for t in (date, f.get("moteur")) if t)
        if pied:
            place(S.s_lab * 1.6 + 2)
            fr.down(S.s_lab * 1.6)
            d.t(fr.x, fr.y, d.fit(pied, S.f_b, S.s_lab - 0.6, fr.w), S.f_b, S.s_lab - 0.6, S.mut)
        return fr


# ---------------------------------------------------------------------- 01
class Encre(Style):
    key, name = "01_encre", "Encre"
    pitch = "Gris froid et bleu encre. La référence."
    panel, acc, concrete = "#F3F4F7", "#33415C", "#E5E7EE"
    ink, mut, rule = "#15181F", "#6E7480", "#DEE0E7"
    ko = "#9C3341"


# ---------------------------------------------------------------------- 02
class Ardoise(Style):
    key, name = "02_ardoise", "Ardoise"
    pitch = "Gris neutre, accent ardoise. Le plus discret."
    panel, acc, concrete = "#F1F3F4", "#37474F", "#E3E7E9"
    ink, mut, rule = "#1A1D21", "#71787F", "#DDE1E3"
    ko = "#A33A2A"


# ---------------------------------------------------------------------- 03
class Acier(Style):
    key, name = "03_acier", "Acier"
    pitch = "Gris acier légèrement bleuté."
    panel, acc, concrete = "#EFF1F2", "#4A6572", "#E2E6E8"
    ink, mut, rule = "#171B1E", "#6B747A", "#DCE0E3"
    ko = "#A6392B"


# ---------------------------------------------------------------------- 04
class Marine(Style):
    key, name = "04_marine", "Marine"
    pitch = "Accent marine soutenu sur gris bleuté."
    panel, acc, concrete = "#F1F3F6", "#1F3864", "#E2E7EE"
    ink, mut, rule = "#141A24", "#6C7580", "#DDE2E8"
    ko = "#9E3030"


# ---------------------------------------------------------------------- 05
class Graphite(Style):
    key, name = "05_graphite", "Graphite"
    pitch = "Tout en gris et noir, rouge réservé aux non-vérifiés."
    panel, acc, concrete = "#F2F2F2", "#2B2B2B", "#E6E6E6"
    ink, mut, rule = "#111111", "#767676", "#DEDEDE"
    ko = "#B3261E"


# ---------------------------------------------------------------------- 06
class Sauge(Style):
    key, name = "06_sauge", "Sauge"
    pitch = "Vert-gris très pâle, accent sauge."
    panel, acc, concrete = "#F2F5F2", "#4A6152", "#E4EAE5"
    ink, mut, rule = "#171B18", "#6C756F", "#DBE1DC"
    ko = "#8E3B23"


# ---------------------------------------------------------------------- 07
class Prusse(Style):
    key, name = "07_prusse", "Bleu de Prusse"
    pitch = "Gris froid et bleu profond."
    panel, acc, concrete = "#F0F3F5", "#1E4A5F", "#E1E8EC"
    ink, mut, rule = "#12191D", "#68737A", "#DBE1E5"
    ko = "#9B3A2C"


# ---------------------------------------------------------------------- 08
class Prune(Style):
    key, name = "08_prune", "Prune"
    pitch = "Gris lilas très pâle, accent prune."
    panel, acc, concrete = "#F5F3F6", "#4E3D57", "#E9E5EC"
    ink, mut, rule = "#1A171C", "#726C78", "#E1DCE4"
    ko = "#9B3350"


# ---------------------------------------------------------------------- 09
class Terre(Style):
    key, name = "09_terre", "Terre"
    pitch = "Gris chaud, accent terre d'ombre."
    panel, acc, concrete = "#F5F3EF", "#7A5236", "#EAE5DE"
    ink, mut, rule = "#1F1B17", "#786F66", "#E1DACF"
    ko = "#9A3B27"


# ---------------------------------------------------------------------- 10
class Bordeaux(Style):
    key, name = "10_bordeaux", "Bordeaux"
    pitch = "Gris rosé discret, accent bordeaux."
    panel, acc, concrete = "#F5F2F2", "#6E2B34", "#EBE4E4"
    ink, mut, rule = "#1D1719", "#776C6E", "#E3DADB"
    ko = "#6E2B34"


STYLES = [Encre(), Ardoise(), Acier(), Marine(), Graphite(),
          Sauge(), Prusse(), Prune(), Terre(), Bordeaux()]
