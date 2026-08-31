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
from .kit import Doc, Frame, hx, mix, sym, formula, kv_rows, chip
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
        # steel : rendu métallique des armatures (barres en dégradé, reflet
        # cylindrique sur les étriers) — demandé par le bureau le 25/08/2026
        return SectionStyle(
            mode="plein", panel=self.panel, concrete=self.concrete,
            show_hatch=False, ink=self.ink, accent=self.acc, bar=self.acc,
            muted=self.mut, rule_w=1.2, font=self.f_b, font_bold=self.f_v,
            title=None, dim_size=6.0, label_size=6.4, concrete_texture=True,
            steel=True)

    # ------------------------------------------------------------- rendu
    def build(self, path, sections=None, doc=None, garde=True):
        """Rapport complet : page de garde portrait + une planche par section."""
        sections = sections if sections is not None else D.SECTIONS
        doc = doc or D.DOC
        d = Doc(path, A4P if garde else self.page,
                title=doc.get("titre", "Note de calcul"))
        d.warnings = []
        if garde:
            d.new_page(A4P)
            self.garde(d, doc, sections)
        for sec in sections:
            d.new_page(self.page)
            self.planche(d, doc, sec)
        return d

    # ------------------------------------------------- page de garde (portrait)
    def garde(self, d, doc, sections):
        S = self
        W, H = d.W, d.H
        m = 52
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
        for i, sec in enumerate(sections):
            fr.down(24)
            n = f"{i + 1:02d}"
            d.t(fr.x, fr.y, n, S.f_h, 10.5, mix(S.acc, "#FFFFFF", 0.55))
            titre = f"{sec['poutre']} — {sec['section']}"
            d.t(fr.x + 26, fr.y, titre, S.f_h, 11, S.ink)
            sous = f"{sec.get('beton', '')} / {sec.get('acier', '')}".strip(" /")
            if sous:
                d.t(fr.x + 26, fr.y - 12, sous, S.f_b, 8.2, S.mut)
            e = sec.get("etat", "")
            if e:
                chip(d, fr.x1, fr.y - 2, e.upper(), S.f_v, 6.4, "#FFFFFF",
                     self.coul_etat(sec), align="right")
            d.t(fr.x1, fr.y - 15, f"page {i + 2}", S.f_b, 7.6, S.mut, "right")
            fr.down(22)
            d.line(fr.x, fr.y, fr.x1, fr.y, mix(S.rule, S.ink, 0.06), 0.5)

        d.t(m, 46, "Vert : vérifié   ·   Ocre : admissible, à la limite   ·   "
                   "Rouge : non vérifié", S.f_b, 7.4, S.mut)
        return d

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

        # colonne de gauche : coupe puis données, sur fond sobre
        cw = S.coupe_w
        d.box(m - 10, m - 4, cw + 20, top - m + 14, fill=S.panel, r=6)
        C = Frame(m, top, cw, m)
        C.down(11)
        d.t(C.x, C.y, "COUPE DE SECTION", S.f_v, 6.6, S.acc, track=1.4)
        C.down(4)
        d.line(C.x, C.y, C.x1, C.y, mix(S.rule, S.ink, 0.12), 0.5)
        blocs = sec.get("blocs", [])
        reserve = 26 + 13 * sum(1 + len(r) for _, r in blocs[:2]) if blocs else 0
        hc = min(S.coupe_h, C.y - m - max(60, reserve))
        C.down(hc)
        draw_section(d.c, C.x, C.y, C.w, hc, sec["coupe"], S.sec_style(),
                     label_w=96)
        C.down(8)
        self._data(d, ms, C.x, C.y, C.w, blocs)

        # colonnes de calcul, en flux continu
        zx = m + cw + S.gap
        gw = (W - m - zx - S.col_gap * (S.n_cols - 1)) / S.n_cols
        cols = [Frame(zx + i * (gw + S.col_gap), top, gw, m)
                for i in range(S.n_cols)]
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
            else:
                k = v["verdicts"][it[1]]
                n = len(d.wrap(k["texte"], S.f_b, S.s_verd, fr.w - 15))
                place(5 + n * S.s_verd * 1.28 + 6.5)
                self._verdict(d, fr, k)
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
