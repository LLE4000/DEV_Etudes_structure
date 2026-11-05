# -*- coding: utf-8 -*-
r"""riche.py — texte courant avec des symboles COMPOSÉS, et les substitutions en typographie.

Un texte riche est une chaîne dont les segments entre « $ » sont composés par ``mathx`` (indices,
exposants, italiques, romain pour les mots) à la taille et dans la couleur du texte qui les
entoure : « la portée $L$ = 900 mm », « $q_h$ — charge linéaire horizontale »,
« $σ_{adm}$ = 160 N/mm² ». Le reste est du texte ordinaire, dans la police donnée ; un segment
composé est un mot insécable, le repli se fait sur les espaces du texte.

``tex(s)`` traduit la notation des substitutions du kit des assemblages (« W_el,y », « P₁ »,
« √(…) », « t_p² », « 10³ », « ^1,5 ») vers le mini-LaTeX de mathx ; ``auto(s)`` compose, dans
un texte ordinaire, les mots qui sont des symboles indicés (« q_h », « N_Rd », « F_t,Rd/γ_F ») ;
``formule_calee`` pose une substitution « a = b = c » avec son résultat « = r » calé à droite,
repliée puis réduite si elle est trop longue.
"""
import re

from . import mathx as M
from .fonts import string_width

_SEG = re.compile(r"\$([^$]+)\$")
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_RUN = r"[^\W_]+(?:,[^\W_]+)*"
_RE_SUP = re.compile(r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_RE_SUB = re.compile(r"([₀₁₂₃₄₅₆₇₈₉]+)")
_RE_IND = re.compile(r"_(" + _RUN + r")")
_RE_EXP = re.compile(r"\^(\d+(?:,\d+)?)")
_RE_SYM = re.compile(r"^[^\W\d_]{1,3}_" + _RUN + r"(?:/[^\W\d_]{1,3}_" + _RUN + r")?$")
_PONCT = ",;:.)"


# ------------------------------------------------------------------ notation → mini-LaTeX
def _racines(s):
    """« √(…) » et « √3 » → ``\\sqrt{…}`` (parenthèses équilibrées)."""
    out, i = "", 0
    while True:
        j = s.find("√", i)
        if j < 0:
            return out + s[i:]
        out += s[i:j]
        k = j + 1
        if k < len(s) and s[k] == "(":
            depth, m = 0, k
            while m < len(s):
                if s[m] == "(":
                    depth += 1
                elif s[m] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                m += 1
            out += r"\sqrt{" + s[k + 1:m] + "}"
            i = m + 1
        else:
            m = k
            while m < len(s) and (s[m].isalnum() or s[m] in ",."):
                m += 1
            out += r"\sqrt{" + s[k:m] + "}"
            i = m


def tex(s):
    """La notation des substitutions (« W_el,y », « P₁ », « t_p² », « 10⁶ », « √(…) ») en mini-LaTeX."""
    if not s:
        return ""
    s = _racines(s)
    s = _RE_SUP.sub(lambda m: "^{" + m.group(1).translate(_SUP) + "}", s)
    s = _RE_SUB.sub(lambda m: "_{" + m.group(1).translate(_SUB) + "}", s)
    s = _RE_EXP.sub(r"^{\1}", s)
    s = _RE_IND.sub(r"_{\1}", s)
    return s.replace("π", r"\rm{π}")


_RE_AUTO = re.compile(r"(?<![\w$])([^\W\d_]{1,3}_" + _RUN + r"(?:/[^\W\d_]{1,3}_" + _RUN + r")?[⁰¹²³⁴⁵⁶⁷⁸⁹]?)(?![^\W_])")


def auto(s):
    """Compose, dans un texte ordinaire, les symboles indicés qu'il contient — un mot entier
    (« q_h », « N_Rd », « F_t,Rd/γ_F ») ou la fin d'un mot (« M/W_el », « ±e_x/2 », « b_eff·t_p² »)."""
    if not s or "_" not in s or "$" in s:
        return s
    return _RE_AUTO.sub(lambda m: "$" + tex(m.group(1)) + "$", s)


# ------------------------------------------------------------------ texte riche
_cache = {}


def noeud(t):
    n = _cache.get(t)
    if n is None:
        n = M.parse(t)
        _cache[t] = n
    return n


def segments(s):
    """[(est_composé, texte), …] : les segments « $…$ » et le texte entre eux."""
    out, i = [], 0
    for m in _SEG.finditer(s):
        if m.start() > i:
            out.append((False, s[i:m.start()]))
        out.append((True, m.group(1)))
        i = m.end()
    if i < len(s):
        out.append((False, s[i:]))
    return out


def jetons(s):
    """Les mots d'un texte riche : [[est_composé, texte, espace_après], …] ; un segment composé est un mot."""
    out = []
    for comp, t in segments(s):
        if comp:
            out.append([True, t, False])
            continue
        parts = t.split(" ")
        for k, mot in enumerate(parts):
            if mot:
                out.append([False, mot, False])
            if k < len(parts) - 1 and out:
                out[-1][2] = True
    return out


def _w(jeton, font, size, ms):
    comp, t, _ = jeton
    return M.width(noeud(t), ms, size) if comp else string_width(t, font, size)


def largeur_ligne(ligne, font, size, ms):
    sp = string_width(" ", font, size)
    w = 0.0
    for i, j in enumerate(ligne):
        w += _w(j, font, size, ms)
        if j[2] and i < len(ligne) - 1:
            w += sp
    return w


def largeur(s, font, size, ms):
    return largeur_ligne(jetons(s), font, size, ms)


def envelopper(s, font, size, ms, width):
    """Les lignes (listes de jetons) d'un texte riche replié sur ``width``."""
    sp = string_width(" ", font, size)
    lignes, cur, wcur = [], [], 0.0
    for j in jetons(s):
        w = _w(j, font, size, ms)
        if cur and wcur + w > width + 0.01:
            lignes.append(cur)
            cur, wcur = [], 0.0
        cur.append(j)
        wcur += w + (sp if j[2] else 0.0)
    if cur:
        lignes.append(cur)
    return lignes


def ajuster(s, font, size, ms, width):
    """La première ligne d'un texte riche, tronquée avec « … » si elle dépasse ``width``."""
    js = jetons(s)
    if largeur_ligne(js, font, size, ms) <= width:
        return js
    while js and largeur_ligne(js + [[False, "…", False]], font, size, ms) > width:
        js = js[:-1]
    return js + [[False, "…", False]]


def dessiner(d, x, y, ligne, font, size, col, ms, align="left"):
    """Dessine une ligne de jetons (x = bord gauche, y = ligne de base) ; renvoie sa largeur."""
    sp = string_width(" ", font, size)
    w = largeur_ligne(ligne, font, size, ms)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2.0
    for i, (comp, t, esp) in enumerate(ligne):
        if comp:
            x += M.draw(d.c, noeud(t), x, y, ms, size)
        else:
            d.t(x, y, t, font, size, col)
            x += string_width(t, font, size)
        if esp and i < len(ligne) - 1:
            x += sp
    return w


def texte(d, x, y, s, font, size, col, ms, align="left", width=None):
    """Un texte riche sur une ligne (tronqué à ``width`` s'il est donné)."""
    ligne = ajuster(s, font, size, ms, width) if width is not None else jetons(s)
    return dessiner(d, x, y, ligne, font, size, col, ms, align)


def paragraphe(d, fr, s, font, size, col, ms, lead=1.32, indent=0, max_lignes=None):
    """Un texte riche replié dans la colonne ; renvoie le nombre de lignes."""
    lignes = envelopper(s, font, size, ms, fr.w - indent)
    if max_lignes:
        lignes = lignes[:max_lignes]
    for l in lignes:
        fr.down(size * lead)
        dessiner(d, fr.x + indent, fr.y, l, font, size, col, ms)
    return len(lignes)


# ------------------------------------------------------------------ substitutions
def formule_calee(d, fr, parts, res, ms, size, ms_res=None, indent=0, retrait=8, lead=1.36, min_size=5.0, gap=6.0):
    """« a = b = c » à gauche, « = r » calé à droite. Si la ligne est trop longue, les morceaux
    passent à la ligne (« = c » en retrait) ; si la dernière ligne et son résultat ne tiennent
    toujours pas, ils se réduisent jusqu'à ``min_size``."""
    parts = [p for p in parts if p]
    if not parts:
        return fr.y
    avail = fr.w - indent
    msr = ms_res or ms

    def w(t, s, st=ms):
        return M.width(noeud(t), st, s)

    nres = "= " + res if res else None
    wres = w(nres, size, msr) if nres else 0.0
    lignes, cur = [], parts[0]
    for p in parts[1:]:
        cand = cur + " = " + p
        if w(cand, size) <= avail - (0.0 if lignes else 0.0):
            cur = cand
        else:
            lignes.append(cur)
            cur = "= " + p
    lignes.append(cur)
    last = lignes[-1]
    if nres and len(lignes) == 1 and len(parts) > 1 and w(last, size) + gap + wres > avail:
        i = last.rfind(" = ")
        lignes = [last[:i], "= " + last[i + 3:]]
        last = lignes[-1]
    s_last = size
    if nres and w(last, size) + gap + wres > avail - (retrait if len(lignes) > 1 else 0.0):
        s_last = max(min_size, size * (avail - gap - (retrait if len(lignes) > 1 else 0.0)) / (w(last, size) + wres))
    for i, l in enumerate(lignes):
        dern = i == len(lignes) - 1
        dx = retrait if i else 0.0
        sz = s_last if dern else size
        wl = w(l, sz)
        if wl > avail - dx:
            sz = max(min_size, sz * (avail - dx) / wl)
        fr.down(size * lead)
        M.draw(d.c, noeud(l), fr.x + indent + dx, fr.y, ms, sz)
        if dern and nres:
            M.draw(d.c, noeud(nres), fr.x1 - w(nres, sz, msr), fr.y, msr, sz)
    return fr.y
