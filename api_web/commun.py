# -*- coding: utf-8 -*-
"""Ce que les modules du site partagent : le code couleur, les nombres à la française et les COMPOSANTS de l'écran
(ligne de résultat, verdict, grande fenêtre de calcul, sélecteur, champs de saisie, sections repliables, bouton de
téléchargement, bouton d'importation de la météo).

Ils sont écrits ici, en Python, et tournent dans le navigateur avec le moteur (Pyodide) : l'écran ne fait qu'afficher
les fragments HTML qu'on lui rend et renvoyer ce que l'utilisateur saisit. Il ne calcule rien, ne recopie aucun seuil.
Les classes CSS sont celles de la charte (site/web/css/charte.css), reprise des maquettes validées par le bureau.

L'arrondi est celui des formules des moteurs (``acier.formats.F``, c'est-à-dire ``acier.js.to_fixed``) : DEMI VERS
L'EXTÉRIEUR, appliqué à la valeur binaire exacte du nombre. Un tableau et la formule voisine écrivent donc le même
chiffre (490,5 → « 491 » des deux côtés). La règle n'est pas recopiée ici : ``_fixe`` appelle ``to_fixed``.
"""
import base64
import decimal
import html
import json
import math
import numbers
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from acier import js as JS
from beton import position as POSITION
from acier.garde_corps import membres as ME

from .icones import ICONES

NBSP = "\u00a0"              # espace insécable : entre un nombre et son unité
FINE = "\u202f"              # espace fine insécable : entre les milliers
NIVEAUX = {ME.NOK: "r", ME.ATTENTION: "f", ME.OK: "p"}
PLAFOND_TAUX = 999           # au-delà, un taux s'écrit « > 999 % » (la valeur exacte au survol) : comme acier.formats.pct


# ------------------------------------------------------------------ code couleur et nombres
def niveau(t: Optional[float]) -> str:
    """Le code couleur du bureau : « r » au-dessus de 100 %, « f » (vert foncé) de 95 à 100 %, « p » (vert pâle) en
    dessous ; c'est le statut du moteur qui tranche (membres.statut, seuils en source unique)."""
    if t is None:
        return "c"
    return NIVEAUX.get(ME.statut(t), "p")


def _fini(x) -> bool:
    """Un nombre réel fini — un nombre de numpy aussi (numpy.int64, numpy.float32 : ce ne sont ni des ``int`` ni des
    ``float`` de Python, et ``fr`` les écrivait « — »)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)


def _fixe(x: float, nd: int) -> str:
    """``x`` écrit avec ``nd`` décimales (point décimal), arrondi DEMI VERS L'EXTÉRIEUR sur la valeur binaire exacte :
    la règle des moteurs (``acier.js.to_fixed``), appelée — pas recopiée. La précision décimale est élargie pour qu'un
    très grand nombre (1e30) s'écrive au lieu de lever. Le signe d'un zéro est ôté : jamais « -0 ». Un nombre de numpy
    passe d'abord en nombre Python (``to_fixed`` tronquait un numpy.float32 à l'entier)."""
    x = int(x) if isinstance(x, numbers.Integral) else float(x)
    with decimal.localcontext() as ctx:
        ctx.prec = 420                            # 309 chiffres pour le plus grand flottant, plus les décimales
        s = JS.to_fixed(x, int(nd))
    if s.startswith("-") and not s.strip("-0."):
        s = s[1:]
    return s


def _grouper(ent: str) -> str:
    """La partie entière en tranches de trois chiffres, séparées par l'espace fine insécable."""
    signe, chiffres = ("-", ent[1:]) if ent.startswith("-") else ("", ent)
    tranches = []
    while len(chiffres) > 3:
        chiffres, tranches = chiffres[:-3], [chiffres[-3:]] + tranches
    return signe + FINE.join([chiffres] + tranches)


def pct(t: Optional[float], nd: int = 0):
    """Un taux à afficher, arrondi au plus proche (demi vers l'extérieur, comme les formules des moteurs) SANS quitter
    la bande de sa couleur : 94,98 % s'affiche 94 % (95 % se lirait vert foncé), 100,2 % s'affiche 101 % (100 % se
    lirait admis), 63,7 % reste 64 %, 63,5 % donne 64 %. None (ou un taux non numérique) : None."""
    if t is None or (isinstance(t, float) and math.isnan(t)):
        return None
    x, n, k = t * 100, niveau(t), 10 ** nd
    if math.isinf(x):
        return x
    v = float(_fixe(x, nd))
    if n == "r":
        v = max(v, 100 + 1 / k)
    elif n == "f":
        v = min(max(v, 95.0), 100.0)
    else:
        v = min(v, 95 - 1 / k)
    return int(v) if nd == 0 else round(v, nd)


def pc(v) -> str:
    """Un taux déjà arrondi (pct), écrit à la française : « 64 » ou « 63,7 » ; « — » s'il manque."""
    if v is None:
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "∞" if v > 0 else "-∞"
    return _fixe(v, 1).replace(".", ",") if isinstance(v, float) and v != int(v) else f"{int(v)}"


def fr(x: Optional[float], nd: int = 1) -> str:
    """Un nombre de RÉSULTAT à la française : ``nd`` décimales, virgule décimale, espace fine insécable entre les
    milliers, arrondi demi vers l'extérieur comme les formules des moteurs (490,5 → « 491 », 1,25 → « 1,3 » ; 2,675
    vaut 2,67499… en binaire → « 2,67 », comme le moteur). « — » si absent ou non fini ; jamais « -0 »."""
    if x is None or not _fini(x):
        return "—"
    ent, _, dec = _fixe(x, nd).partition(".")
    return _grouper(ent) + ("," + dec if dec else "")


def nb(x: Optional[float], dec: int = 3) -> str:
    """Une valeur saisie (coefficient, dimension) écrite telle quelle, sans arrondi au-delà de ``dec`` décimales :
    1,25 reste 1,25 (un format de taux en ferait « 1,2 »), 210000 reste 210000 (seuls les zéros APRÈS la virgule
    s'ôtent), 0,5 à zéro décimale donne 1 (demi vers l'extérieur). Pas de séparateur de milliers (``num`` les pose
    dans un champ). Le chiffre écrit est relu et comparé à la valeur ; jamais « -0 » ; « — » si non fini."""
    if x is None:
        return ""
    if not _fini(float(x)):
        return "—"
    s = _fixe(float(x), dec)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    assert abs(float(s) - float(x)) <= 0.5 * 10 ** -dec * (1 + 1e-9) + 1e-12 * max(1.0, abs(float(x))), (x, s)
    return s.replace(".", ",")


# les pas d'arrondi de l'axe de calcul (beton/position.ARRONDIS) en mots — les segments des options de la poutre et des
# dalles, leurs résumés, les hypothèses de la prédalle : une source
ARRONDIS_TXT = tuple((a, "aucun" if not a else f"{nb(a)} cm") for a in POSITION.ARRONDIS)


def qte(x: Optional[float], unite: str, nd: int = 1) -> str:
    """Une QUANTITÉ : la valeur (``fr``) et son unité liées par une espace insécable — « 1 380,5 kg/m³ » ne se coupe
    jamais entre le nombre et l'unité, ni dans l'unité (« t / m³ » : ses espaces deviennent insécables). ``unite`` est
    du HTML du module (« mm<sup>2</sup> » permis ; jamais un texte saisi). « — » si la valeur manque."""
    if x is None or not _fini(x):
        return "—"
    u = _insecable(unite)
    return f"{fr(x, nd)}{NBSP}{u}" if u else fr(x, nd)


def _insecable(h: str) -> str:
    """Les espaces d'un texte HTML rendues insécables, HORS des balises (« <span class="u"> » garde la sienne)."""
    return re.sub(r"(<[^>]*>)| ", lambda m: m.group(1) or NBSP, str(h or ""))


def lire_nombre(v) -> Optional[float]:
    """Ce que l'utilisateur a tapé (« 1 100 », « 0,5 », 12) → un nombre ; None si ce n'en est pas un."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if not isinstance(v, str):
        return None
    s = v.replace(" ", "").replace(NBSP, "").replace(FINE, "").replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def lire_auto(v) -> Optional[float]:
    """La lecture d'un champ « vide = automatique » (``num(..., auto=…)``) : un champ VIDÉ rend None — le module
    prend alors la valeur automatique ; un nombre rend ce nombre ; tout autre texte lève ValueError avec un message
    à montrer tel quel (texte simple), l'état restant intact."""
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    x = lire_nombre(v)
    if x is None:
        t = str(v).strip()
        t = t if len(t) <= 40 else t[:39] + "…"
        raise ValueError(f"« {t} » n'est pas un nombre : laisser le champ vide pour la valeur automatique.")
    return x


# ------------------------------------------------------------------ texte
def esc(s) -> str:
    return html.escape(str(s), quote=True)


_GRECQUES = "Α-Ωα-ωϐϑϕϖϰϱϵ"
_CAR_INDICE = rf"A-Za-z0-9éèàùêç{_GRECQUES}⊥∥"
# (le diamètre Ø porte son indice comme une lettre — Ø_Y, Ø_étr, Ø_l : la règle de position du 13/11/2026 écrit
#  « c + Ø_Y + Ø/2 + jeu » ; avant, il restait « Ø_Y », le tiret bas à l'écran)
_SYMBOLE = re.compile(rf"([A-Za-z{_GRECQUES}Øø][\u0300-\u036f]*)_([{_CAR_INDICE}]+(?:,[{_CAR_INDICE}]+)*)")


def symb_math(s: str) -> str:
    """Un symbole du moteur écrit comme dans la note : q_h → q<sub>h</sub>, W_el,y → W<sub>el,y</sub>, χ_LT →
    χ<sub>LT</sub>, λ̄_LT → λ̄<sub>LT</sub>, σ_⊥ → σ<sub>⊥</sub>, M_Rd,θ → M<sub>Rd,θ</sub>, Ø_Y → Ø<sub>Y</sub> : toutes
    les lettres grecques, minuscules et majuscules (et leurs variantes ϕ ϑ ϵ), et le diamètre Ø portent un indice (texte
    échappé)."""
    return _SYMBOLE.sub(r"\1<sub>\2</sub>", html.escape(str(s), quote=False))


def icon(nom: str) -> str:
    return ICONES[nom]


def _taux_txt(taux) -> Tuple[str, str]:
    """(texte, attribut title) d'un taux déjà arrondi : « 64 % » ; au-delà de 999 %, « > 999 % » et la valeur exacte
    au survol (un nombre de huit chiffres casse la pastille) ; « — » s'il manque."""
    if taux is None:
        return "—", ""
    if taux > PLAFOND_TAUX:
        ent, _, dec = pc(taux).partition(",")
        exact = (ent if not ent.lstrip("-").isdigit() else _grouper(ent)) + ("," + dec if dec else "")
        return f"&gt;{NBSP}{PLAFOND_TAUX}{NBSP}%", f' title="{exact}{NBSP}%"'
    return f"{pc(taux)}{NBSP}%", ""


def chip(taux, niv: str, xl: bool = False) -> str:
    """La pastille d'un taux (déjà arrondi, ``pct``) dans sa couleur ; ``xl`` : la grande pastille du verdict (η)."""
    txt, titre = _taux_txt(taux)
    if xl:
        return f'<span class="chip xl {niv}"{titre}><small>η</small>{txt}</span>'
    return f'<span class="chip {niv}"{titre}>{txt}</span>'


STATUTS = {"p": ("p", "Conforme"), "f": ("f", "Conforme, attention"), "r": ("r", "Non conforme"), "c": ("c", "À compléter"),
           "x": ("r", "Calcul impossible"),
           "i": ("c", "Consultation"),
           # « n » : le calcul tourne mais ne dit rien (aucune sollicitation) ; « v » : conforme au calcul, une
           # disposition constructive reste à vérifier (espacement sous le minimum) — jamais « Conforme » tout court
           "n": ("c", "Données incomplètes"),
           "v": ("v", "À vérifier")}
# la pastille de l'en-tête du téléphone n'a la place que d'un mot (à 360 px, « Données incomplètes » y prenait 152 px et
# coupait l'état de l'enregistrement) : la forme courte d'un statut trop long pour elle, le statut entier au survol
STATUTS_COURTS = {"n": "Incomplet"}


def statut(niv: str) -> str:
    """La pastille d'état de la note : conforme, attention (95 à 100 %), non conforme, à compléter."""
    cls, txt = STATUTS[niv]
    return f'<span class="status {cls}"><span class="dot"></span>{txt}</span>'


def legende() -> str:
    return '<span class="chip r">&gt;{0}100{0}%</span><span class="chip f">95–100{0}%</span><span class="chip p">&lt;{0}95{0}%</span>'.format(NBSP)


# ------------------------------------------------------------------ résultat, calcul, sélecteur
def _largeur(taux) -> float:
    """La longueur de la jauge, bornée à [0, 100] % : un taux négatif (une traction comptée en η_N) laisse la jauge
    VIDE — une largeur CSS négative serait ignorée par le navigateur et la jauge s'afficherait pleine."""
    if taux is None or (isinstance(taux, float) and math.isnan(taux)):
        return 0
    return max(0, min(taux, 100))


def ligne_res(nom: str, piece: str, taux, niv: str, calcul: str = "", pk: str = "", gov: bool = False, leger: str = "",
              sans_taux: str = "à compléter") -> str:
    """Une vérification du résultat, sur DEUX lignes : le nom et la pièce, puis la jauge et le taux. La ligne entière
    ouvre le calcul détaillé ; la pièce, quand elle se choisit (profilé, armatures), ouvre le sélecteur. ``taux`` est
    déjà arrondi (pct) ; None : le texte ``sans_taux`` (« à compléter » ; un module de consultation choisit le sien,
    « consultation », « — »…). ``piece`` est déjà du HTML. La jauge est bornée à [0, 100] % ; un taux de plus de
    999 % s'écrit « > 999 % » (la valeur exacte au survol)."""
    etoile = f'<span class="leger" title="{esc(leger)}">{icon("star")}</span>' if leger else ""
    if pk:
        p = f'<button class="prof" data-pk="{esc(pk)}" title="Choisir">{etoile}<span>{piece}</span>{icon("chevron-down")}</button>'
    else:
        p = f'<span class="prof"><span>{piece}</span></span>'
    ouvre = f' data-calcul="{esc(calcul)}" title="Voir le calcul"' if calcul else ""
    txt, titre = _taux_txt(taux) if taux is not None else (esc(sans_taux), "")
    return (f'<div class="el{" gov" if gov else ""}"{ouvre}><div class="l1"><b>{esc(nom)}</b>{p}</div>'
            f'<div class="l2"><div class="bar"><b class="{niv}" style="width:{_largeur(taux)}%"></b></div>'
            f'<span class="pct {niv}"{titre}>{txt}</span></div></div>')


def ligne_verif(nom: str, comparaison: str, taux, niv: str, calcul: str = "", pk: str = "", piece: str = "", gov: bool = False,
                etat_txt: str = "", leger: str = "") -> str:
    """Une vérification en DEUX LIGNES au plus (l'établi compact) : 1) le nom, la pièce (quand elle se choisit : un
    bouton qui ouvre le sélecteur) et le taux ; 2) ce qu'on compare, en clair (« néc. 1 206 ≤ posé 1 257 mm² »), et une
    jauge fine. La ligne entière ouvre la fenêtre de détail ``calcul``.

    ``taux`` (déjà arrondi, ``pct``) est un TAUX D'UTILISATION ; une vérification qui n'en a pas (une disposition
    constructive, une vérification non effectuée) passe ``taux`` None et dit son état en mots (``etat_txt``, dans la
    couleur ``niv``) — jamais un rapport fourni / minimum déguisé en pourcentage. ``comparaison`` et ``piece`` sont du
    HTML du module ; ``nom``, ``etat_txt`` et ``leger`` du texte (échappés). ``leger`` : une disposition plus légère
    qui tient (l'étoile du sélecteur), dite par une étoile dans le bouton de la pièce et au survol ; sans lui, la sortie
    est la même, à l'octet près."""
    ouvre = f' data-calcul="{esc(calcul)}" title="Voir le détail"' if calcul else ""
    etoile = f'<span class="leger" title="{esc(leger)}">{icon("star")}</span>' if leger and pk else ""
    p = (f'<button class="prof" data-pk="{esc(pk)}" title="Choisir une disposition">{etoile}<span>{piece}</span>{icon("chevron-down")}</button>'
         if pk else (f'<span class="prof"><span>{piece}</span></span>' if piece else ""))
    if taux is not None:
        txt, titre = _taux_txt(taux)
        droite = f'<span class="pct {niv}"{titre}>{txt}</span>'
        jauge = f'<div class="bar"><b class="{niv}" style="width:{_largeur(taux)}%"></b></div>'
    else:
        droite = f'<span class="et-mot {niv}">{esc(etat_txt or "—")}</span>'
        jauge = ""
    cls = "el vc" + (" gov" if gov else "") + ("" if calcul else " fixe")
    return (f'<div class="{cls}"{ouvre}><div class="l1"><b>{esc(nom)}</b>{p}{droite}</div>'
            f'<div class="l2"><span class="cmp">{comparaison}</span>{jauge}</div></div>')


def verdict_valeur(x: Optional[float], unite: str, titre: str, comparaison: str = "", niv: str = "c", nd: int = 1,
                   symbole: str = "", calcul: str = "", classe: str = "") -> str:
    """Le VERDICT d'une valeur qui n'est PAS un pourcentage (un taux d'armature en kg/m³, une section en mm²…) : le
    même dessin que le verdict d'un taux (``chip(…, xl=True)`` puis la vérification et sa comparaison), la valeur
    (``fr(x, nd)``) en grand et l'unité en petit dans la pastille. ``titre`` est du TEXTE (échappé) ; ``comparaison``,
    ``unite`` et ``symbole`` sont du HTML du module (« ≥ A<sub>s,req</sub> = 1 206 mm² ») ; ``calcul`` : la fenêtre
    de calcul que le verdict ouvre ; ``classe`` : une classe de plus pour la feuille du module.

    La pastille ne se comprime jamais (``flex:none``) : sous une comparaison longue, la charte la réduisait à 114 px
    pour « 1 380,5 kg/m³ » qui en demande 164 (mesuré dans Chromium, colonne du résultat de 316 px)."""
    s = f"<small>{symbole}</small>" if symbole else ""
    u = f"<small>{_insecable(unite)}</small>" if unite else ""
    ouvre = f' data-calcul="{esc(calcul)}" title="Voir le calcul"' if calcul else ""
    cls = f"verdict {esc(classe)}" if classe else "verdict"
    return (f'<div class="{cls}"{ouvre}><span class="chip xl {niv}" style="flex:none">{s}{fr(x, nd)}{u}</span>'
            f'<div class="gov"><b>{esc(titre)}</b>{comparaison}</div></div>')


def lignes_fx(lignes: List[Dict]) -> str:
    """Les lignes « symbole = formule = valeurs = résultat » d'un groupe (les taux en pastille de couleur)."""
    out = ""
    for li in lignes:
        parts = li["parts"]
        h = f'<span class="s">{symb_math(parts[0])}</span>' + "".join(f'<span class="eq">=</span><span>{symb_math(p)}</span>' for p in parts[1:])
        res = f'<span class="chip {li["niveau"]}">{esc(li["res"])}</span>' if li.get("niveau") else f'<span class="r">{esc(li["res"])}</span>'
        out += f'<div class="fx">{h}<span class="eq">=</span>{res}</div>'
    return out


def txt_arrondi(pas_cm) -> str:
    """Le pas d'arrondi de l'axe de calcul en mots (beton/position.py) : « 0,5 cm », « aucun »."""
    return dict(ARRONDIS_TXT).get(pas_cm, f"{nb(pas_cm)} cm")


def rappel_jeu(jeu_cm: float, pas_cm: float, sujet: str = "Barres dessinées") -> str:
    """La ligne DISCRÈTE sous un dessin d'armatures (décision du propriétaire du 13/11/2026) : les barres sont dessinées à
    leur position réelle ; le jeu de pose et l'arrondi n'entrent que dans le calcul (beton/position.py). « » sans l'un
    ni l'autre. Les longueurs en mm, comme les cotes des dessins. ``sujet`` : le nom ET son participe accordé
    (« Barres dessinées », « Lits dessinés »)."""
    jeu, pas = float(jeu_cm or 0.0), float(pas_cm or 0.0)
    morceaux = [x for x in ((f"un jeu de pose de {nb(jeu * 10.0, 1)} mm" if jeu else ""),
                            (f"l'arrondi au {nb(pas * 10.0, 1)} mm" if pas else "")) if x]
    return f"{sujet} à leur position réelle · le calcul ajoute {' et '.join(morceaux)}" if morceaux else ""


def fx_axe(f: Dict) -> Dict:
    """La ligne du détail de l'AXE de calcul d'une couche (dalle, prédalle) ou d'un lit (poutre) — la règle de position du
    13/11/2026 (beton/position.py), la même ligne pour les trois modules : « e₁ = c + Ø/2 + jeu = 3 + 0,6 + 1 = 4,6 →
    5,0 cm » (la position réelle terme à terme, le jeu de pose, puis l'arrondi quand il change le nombre). ``f`` : une
    formule « e » du moteur, ``{"couche": rang, "detail": P.detail(...)}``."""
    det = f["detail"]
    jeu = det["jeu"] > 0

    def cm(x: float) -> str:
        return nb(round(x, 2), 2)

    def cm1(x: float) -> str:
        return fr(x, 1) if abs(round(x, 1) - x) < 1e-9 else fr(x, 2)
    parts = [f"e_{f['couche'] + 1}", " + ".join(t["symbole"] for t in det["termes"]) + (" + jeu" if jeu else ""),
             " + ".join(cm(t["valeur"]) for t in det["termes"]) + (f" + {cm(det['jeu'])}" if jeu else "")]
    res = (f"{cm1(det['brut'])} → " if abs(det["axe"] - det["brut"]) > 1e-9 else "") + f"{cm1(det['axe'])} cm"
    return {"parts": parts, "res": res, "niveau": ""}


COLONNES = ("Vérification", "Formule", "Valeur", "Limite", "Taux")


def _case_taux(li: Dict) -> str:
    """La dernière case d'une ligne du tableau : la pastille du taux ; une ligne qui n'a pas de taux mais un ÉTAT (une
    disposition constructive : « OK », « À vérifier ») le dit en mots, dans sa couleur (``etat_txt``)."""
    if li.get("etat_txt"):
        return f'<span class="chip {li["niveau"]}">{esc(li["etat_txt"])}</span>'
    return chip(li.get("taux"), li["niveau"])


def fenetre_calcul(ident: str, titre: str, sous_titre: str, hyps: Sequence[str], lignes: List[Dict], detail: str, pied: str = "",
                   colonnes: Sequence[str] = COLONNES) -> str:
    """La grande fenêtre de calcul, LA MÊME pour tous les modules : les hypothèses, un tableau (vérification, formule,
    valeur, limite, taux), puis le détail des formules sur deux colonnes. ``colonnes`` : les cinq titres du tableau,
    pour un module qui ne vérifie pas (un quantitatif : « Armature, Formule, Masse, Apport au taux, Part ») ; du HTML
    du module. Une ligne sans taux (``taux`` None) porte « — »."""
    if len(colonnes) != 5:
        raise ValueError(f"fenetre_calcul : cinq titres de colonnes attendus, pas {len(colonnes)}")
    c = list(colonnes)
    hyp = "".join(f"<span>{h}</span>" for h in hyps)
    rows = "".join(
        f'<tr class="{"gov" if li.get("gov") else ""}"><td class="nm">{li["nom"]}{("<span>" + li["sous"] + "</span>") if li.get("sous") else ""}</td>'
        f'<td class="fo">{li["formule"]}</td><td class="n v">{li["valeur"]}</td><td class="n">{li["limite"]}</td>'
        f'<td class="n">{_case_taux(li)}</td></tr>' for li in lignes)
    table = (f'<table class="vtab"><thead><tr><th>{c[0]}</th><th class="fo">{c[1]}</th><th class="n">{c[2]}</th><th class="n">{c[3]}</th>'
             f'<th class="n">{c[4]}</th></tr></thead><tbody>{rows}</tbody></table>')
    return (f'<div class="flyout calcul" id="ck-{esc(ident)}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="ck-hyp">{hyp}</div>'
            f'<div class="ck-body">{table}<div class="ck-sub">Le détail du calcul</div><div class="ck-cols">{detail}</div>{pied}</div></div>')


def selecteur(ident: str, titre: str, sous_titre: str, filtres: str, rows: List[Dict], debut: int, unite: str,
              etoile: str = "le plus léger qui tient", pastille: str = "taux") -> str:
    """Le sélecteur, LE MÊME pour les profilés et les armatures : trié par ordre de grandeur, le retenu (✓) et l'optimal
    (★, le plus léger qui tient) marqués ; un clic choisit (le calcul est relancé) ; sous la liste, les propriétés de la
    ligne survolée. ``rows`` : nom, col (la colonne d'ordre de grandeur), taux (arrondi ; None : pas de pastille),
    niveau, cur, opt, prop — et, facultatifs (sans eux, la sortie est la même à l'octet près) : ``etat`` (texte), ce
    qu'une ligne SANS taux dit à sa place, dans une pastille de sa couleur — courte, la pastille fait 52 px (« exclue »
    pour une cornière que la charge ne peut pas atteindre) ; ``inactif`` (vrai), une ligne que le moteur ne peut pas retenir — elle porte
    ``aria-disabled`` (grisée par la feuille de l'établi, le parcours ne la choisit pas ; un clic la demande quand même,
    et le refus du moteur se lit). Le pied dit « ✓ retenu · ★ <etoile> · <unite> · <pastille> » : ``etoile=""`` retire
    la légende ★ et ``pastille=""`` le « · taux » (un module de consultation, sans optimal ni vérification)."""
    def pastille_(r):
        if r.get("taux") is not None:
            return chip(r["taux"], r["niveau"])
        return f'<span class="chip {r["niveau"]}">{esc(r["etat"])}</span>' if r.get("etat") else ""
    # l'attribut d'une ligne inactive est préparé HORS de la f-string : un antislash dans son expression est une
    # erreur de syntaxe avant Python 3.12, et le harnais (contrôle des briques) charge ce module avec le python3 du
    # système (3.11) — trouvé par le harnais complet du 10/11/2026
    def inactif_(r):
        return ' aria-disabled="true"' if r.get("inactif") else ""
    lignes = "".join(
        f'<button class="pk-row{" cur" if r["cur"] else ""}{" opt" if r["opt"] else ""}"{inactif_(r)} '
        f'data-choix="{esc(r["nom"])}" data-prop="{esc(r.get("prop", ""))}">'
        f'<span class="mk">{icon("check") if r["cur"] else (icon("star") if r["opt"] else "")}</span>'
        f'<span class="nm">{esc(r["nom"])}</span><span class="kg">{r["col"]}</span>'
        f'{pastille_(r)}</button>' for r in rows)
    prop0 = next((r.get("prop", "") for r in rows if r["cur"]), "")
    vide = "" if rows else '<div class="pk-vide">Aucune section dans cette sélection.</div>'
    leg_etoile = f'<span>{icon("star")}{etoile}</span>' if etoile else ""
    leg_unite = f"{unite} · {pastille}" if pastille else unite
    return (f'<div class="flyout picker" id="pk-{esc(ident)}" data-pk="{esc(ident)}" data-debut="{debut}"><div class="fl-h"><div><b>{titre}</b><span>{sous_titre}</span></div>'
            f'<button class="pcoll" data-fermer="1" title="Fermer">{icon("x")}</button></div><div class="filters">{filtres}</div>'
            f'<div class="pk-list">{lignes}{vide}</div><div class="pk-prop">{prop0}</div>'
            f'<div class="fl-foot"><span>{icon("check")}retenu</span>{leg_etoile}<span>{leg_unite}</span></div></div>')


# ------------------------------------------------------------------ saisie
def _attr_chemin(chemin: Sequence) -> str:
    return esc("/".join(str(c) for c in chemin))


def ligne(label: str, champ: str, sym: str = "", aide: str = "", full: bool = False) -> str:
    """Une ligne de saisie : le symbole, le libellé (l'aide au survol de l'icône), le champ."""
    s = f'<span class="sym">{sym}</span>' if sym else ""
    a = f'<span class="aide" title="{esc(aide)}">{icon("info")}</span>' if aide else ""
    return f'<div class="row{" full" if full else ""}"><label>{s}<span>{label}</span>{a}</label>{champ}</div>'


def ligne2(*paires: Tuple[str, str, str]) -> str:
    """DEUX champs (ou plus) sur une seule ligne : les données condensées (demande de l'ingénieur, 11/2026 — « béton et
    acier sur la même ligne, la largeur et la hauteur sur la même ligne »). Chaque paire (libellé HTML court, champ, aide
    en texte : au survol de son icône ⓘ, jamais en paragraphe)."""
    out = ""
    for label, champ, aide in paires:
        a = f'<span class="aide" title="{esc(aide)}">{icon("info")}</span>' if aide else ""
        out += f'<label><span>{label}</span>{a}</label>{champ}'
    return f'<div class="row2 n{len(paires)}">{out}</div>'


def num(chemin: Sequence, valeur, unite: str = "", dec: int = 3, entier: bool = False, desactive: bool = False, mini=None, maxi=None,
        auto: str = "") -> str:
    """Un champ numérique lié à un CHEMIN de l'état (« geometrie/H ») : l'écran renvoie le texte tapé, le moteur le lit.

    ``auto`` (« vide = automatique ») : le texte indicatif du champ vide (« auto · 3,5 ») ; le champ accepte alors
    d'être VIDÉ. Il part comme un texte (l'écran refuse un nombre vide) et le module le lit par ``lire_auto`` : None
    pour un champ vidé (il prend l'automatique), le nombre sinon, un message pour tout autre texte. ``valeur`` None :
    le champ s'affiche vide, le texte indicatif en clair."""
    if valeur is None or not _fini(float(valeur)):
        v = ""
    elif entier:
        v = _fixe(float(valeur), 0)
    else:
        v = nb(valeur, dec)
    if v and not entier and abs(float(valeur)) >= 1000:
        ent, _, dec_ = v.partition(",")
        v = f"{int(ent):,}".replace(",", FINE) + ("," + dec_ if dec_ else "")
    lim = (f' data-min="{mini}"' if mini is not None else "") + (f' data-max="{maxi}"' if maxi is not None else "")
    u = f'<span class="u">{unite}</span>' if unite else ""
    lk = icon("lock") if desactive else ""
    sorte = "texte" if auto else ("entier" if entier else "nombre")
    vide = f' data-vide="auto" placeholder="{esc(auto)}"' if auto else ""
    return (f'<div class="field{" locked" if desactive else ""}">{lk}<input data-chemin="{_attr_chemin(chemin)}" data-sorte="{sorte}"'
            f' inputmode="decimal" value="{esc(v)}"{lim}{vide}{" disabled" if desactive else ""}>{u}</div>')


def texte(chemin: Sequence, valeur: str, indice: str = "", desactive: bool = False) -> str:
    """Un texte d'une ligne (un nom, une liste courte) ; ``desactive`` : verrouillé, comme ``num``."""
    lk = icon("lock") if desactive else ""
    return (f'<div class="field text{" locked" if desactive else ""}">{lk}<input data-chemin="{_attr_chemin(chemin)}" data-sorte="texte"'
            f' value="{esc(valeur or "")}" placeholder="{esc(indice)}"{" disabled" if desactive else ""}></div>')


def zone_texte(chemin: Sequence, valeur: str, lignes: int = 5, indice: str = "", desactive: bool = False) -> str:
    """Un texte sur plusieurs lignes (un tableau collé depuis Excel, une liste de couches…) : il part au moteur en
    quittant la zone, comme un champ ; ``desactive`` : verrouillé."""
    return (f'<div class="field area{" locked" if desactive else ""}"><textarea data-chemin="{_attr_chemin(chemin)}" data-sorte="texte" rows="{int(lignes)}"'
            f' placeholder="{esc(indice)}" spellcheck="false"{" disabled" if desactive else ""}>{esc(valeur or "")}</textarea></div>')


def fichier(chemin: Sequence, libelle: str, accept: str = ".csv,.txt") -> str:
    """Un bouton qui lit un fichier du PC : le moteur reçoit {"nom": …, "texte": …} au chemin donné (2 Mo au plus)."""
    return (f'<label class="btn sm fichier">{icon("upload")}{esc(libelle)}<input type="file" data-chemin="{_attr_chemin(chemin)}"'
            f' accept="{esc(accept)}" hidden></label>')


def choix(chemin: Sequence, valeur, options: Iterable[Tuple], desactive: bool = False) -> str:
    """Un menu déroulant ; ``options`` : (valeur, libellé)."""
    opts = "".join(f'<option value="{esc(k)}"{" selected" if k == valeur else ""}>{esc(t)}</option>' for k, t in options)
    return (f'<div class="field select"><select data-chemin="{_attr_chemin(chemin)}" data-sorte="choix"{" disabled" if desactive else ""}>{opts}</select>'
            f'{icon("chevron-down")}</div>')


def choix_groupes(chemin: Sequence, valeur, groupes: Iterable[Tuple], desactive: bool = False) -> str:
    """Un menu déroulant en GROUPES ; ``groupes`` : (titre, [valeurs]) — un titre ``None`` : des options hors groupe ;
    chaque option a sa valeur pour libellé (le même attribut ``data-chemin`` que ``choix``)."""
    def _opt(k):
        return f'<option value="{esc(k)}"{" selected" if k == valeur else ""}>{esc(k)}</option>'
    opts = "".join("".join(_opt(k) for k in noms) if t is None else
                   f'<optgroup label="{esc(t)}">{"".join(_opt(k) for k in noms)}</optgroup>' for t, noms in groupes if noms)
    return (f'<div class="field select"><select data-chemin="{_attr_chemin(chemin)}" data-sorte="choix"{" disabled" if desactive else ""}>{opts}</select>'
            f'{icon("chevron-down")}</div>')


def segments(chemin: Sequence, valeur, options: Iterable[Tuple], desactive: bool = False) -> str:
    """Un choix court en segments (deux à quatre options) ; ``desactive`` : verrouillé (aucun segment ne répond).
    Une option est (valeur, libellé) ou (valeur, libellé, explication au survol) — le troisième terme, facultatif
    (« Mixte » des dalles, 13/11/2026), ajoute un ``title`` ; sans lui, la sortie est la même, à l'octet près."""
    off = " disabled" if desactive else ""

    def survol(o):
        return f' title="{esc(o[2])}"' if len(o) > 2 and o[2] else ""
    bt = "".join(f'<button type="button" class="{"on" if o[0] == valeur else ""}" data-chemin="{_attr_chemin(chemin)}" data-sorte="choix" data-valeur="{esc(o[0])}"{survol(o)}{off}>{esc(o[1])}</button>'
                 for o in options)
    return f'<div class="seg full{" locked" if desactive else ""}">{bt}</div>'


def case(chemin: Sequence, valeur: bool, texte_: str, desactive: bool = False) -> str:
    """Une case à cocher ; ``desactive`` : verrouillée."""
    return (f'<label class="case"><input type="checkbox" data-chemin="{_attr_chemin(chemin)}" data-sorte="case"{" checked" if valeur else ""}'
            f'{" disabled" if desactive else ""}><span>{texte_}</span></label>')


def section(n: int, titre: str, resume: str, corps: str, ouvert: bool = True, cle: str = "", *, secondaire: bool = False) -> str:
    """Une section repliable du cadre Données (le numéro, le titre, un résumé quand elle est repliée). ``secondaire`` : une
    étape d'OPTIONS (méthode, réglages) que l'établi compact range sous « Options », repliée — le moteur le dit, l'écran
    le lit (data-secondaire). Sans lui, la sortie est la même, à l'octet près."""
    sec2 = ' data-secondaire="1"' if secondaire else ""
    return (f'<div class="sec{" open" if ouvert else ""}" data-sec="{esc(cle or n)}"{sec2}><div class="sechd" data-replier="1"><span class="n">{n}</span>'
            f'<span class="tt">{titre}</span><span class="sum">{resume}</span>{icon("chevron-down")}</div>'
            f'<div class="rows">{corps}</div></div>')


def aide(texte_: str) -> str:
    return f'<div class="hint">{texte_}</div>'


def message(texte_: str, grave: bool = False) -> str:
    return f'<div class="msg{" grave" if grave else ""}">{texte_}</div>'


# ------------------------------------------------------------------ télécharger un fichier propre au module
TYPES_FICHIER = ("application/json", "text/csv", "text/plain")
# les fichiers BINAIRES, rendus en base 64 (clé « base64 » à la place de « texte ») : la note de calcul PDF
TYPES_BINAIRES = ("application/pdf",)
BINAIRE_MAX = 15 * 1024 * 1024        # octets, décodés (15 Mo)
NOM_FICHIER_MAX = 120                  # caractères
TEXTE_FICHIER_MAX = 5 * 1024 * 1024    # octets, en UTF-8 (5 Mo)
_INTERDITS_NOM = re.compile(r'[\\/:*?"<>|]+')
# les caractères INVISIBLES d'un nom : contrôle, mise en forme (l'inversion du sens d'écriture, U+202E, fait lire un
# « .json » comme un « .exe »), séparateurs de ligne et de paragraphe
_INVISIBLES = ("Cc", "Cf", "Zl", "Zp")


def _visible(s: str) -> str:
    """``s`` sans ses caractères invisibles (chacun devient « - »)."""
    return "".join("-" if unicodedata.category(ch) in _INVISIBLES else ch for ch in s)


def bouton_telecharger(quoi: str, libelle: str, icone: str = "download") -> str:
    """Un bouton qui fait télécharger un fichier PROPRE AU MODULE (le JSON de l'écran Streamlit, un tableau CSV…) :
    l'écran envoie {"action": "telecharger", "quoi": <quoi>} et le routeur appelle ``telecharger(etat, quoi)`` du
    module, qui rend {"nom", "type", "texte"}. ``quoi`` et ``libelle`` sont échappés."""
    return f'<button type="button" class="btn sm" data-telecharger="{esc(quoi)}">{icon(icone)}{esc(libelle)}</button>'


def nom_fichier(base, extension: str) -> str:
    """Un nom de fichier SÛR tiré d'un nom saisi (« Poutre P1/2 : RDC » → « Poutre P1-2 - RDC.json ») : sans
    / \\ : * ? " < > | ni caractère de contrôle ou invisible (inversion du sens d'écriture…), sans « .. », 120
    caractères au plus extension comprise ; « note » si rien ne reste. Le nom rendu passe toujours le contrôle du
    routeur (``refus_fichier``)."""
    ext = ("." + str(extension).lstrip(".")) if extension else ""
    s = _INTERDITS_NOM.sub("-", _visible(str(base or "")))
    s = re.sub(r"\.{2,}", ".", " ".join(s.split())).strip(" .-") or "note"
    s = s[:max(1, NOM_FICHIER_MAX - len(ext))].rstrip(" .") or "note"
    return s + ext


def refus_fichier(f) -> Optional[str]:
    """Pourquoi un fichier rendu par ``telecharger`` d'un module est refusé (texte simple), ou None s'il est bon : un
    dictionnaire {"nom", "type", "texte"} ; un nom non vide, de 120 caractères au plus, sans « / », « \\ », « .. » ni
    caractère de contrôle ou invisible ; un type parmi TYPES_FICHIER ; un texte de 5 Mo au plus (UTF-8). Un fichier
    BINAIRE (TYPES_BINAIRES : la note PDF) porte « base64 » à la place de « texte » : du base 64 valide, 15 Mo au plus."""
    if not isinstance(f, dict):
        return "le module n'a pas rendu de fichier"
    nom, typ, texte_ = f.get("nom"), f.get("type"), f.get("texte")
    if not isinstance(nom, str) or not nom.strip(" ."):
        return "le nom du fichier est vide"
    if len(nom) > NOM_FICHIER_MAX:
        return f"le nom du fichier dépasse {NOM_FICHIER_MAX} caractères"
    if "/" in nom or "\\" in nom or ".." in nom:
        return "le nom du fichier ne peut contenir ni « / », ni « \\ », ni « .. »"
    if _visible(nom) != nom:
        return "le nom du fichier contient un caractère de contrôle ou invisible"
    if typ in TYPES_BINAIRES:
        b64 = f.get("base64")
        if not isinstance(b64, str) or not b64:
            return "le contenu du fichier n'est pas un binaire en base 64"
        try:
            n = len(base64.b64decode(b64, validate=True))
        except (ValueError, TypeError):
            return "le contenu du fichier n'est pas un binaire en base 64 valide"
        if n > BINAIRE_MAX:
            return f"le fichier dépasse 15 Mo ({fr(n / 1024 / 1024, 1)} Mo)"
        return None
    if typ not in TYPES_FICHIER:
        return f"type de fichier non permis : « {str(typ)[:40]} » (permis : {', '.join(TYPES_FICHIER + TYPES_BINAIRES)})"
    if not isinstance(texte_, str):
        return "le contenu du fichier n'est pas un texte"
    try:
        taille = len(texte_.encode("utf-8"))
    except UnicodeEncodeError:
        return "le contenu du fichier n'est pas un texte valide"
    if taille > TEXTE_FICHIER_MAX:
        return f"le fichier dépasse 5 Mo ({fr(taille / 1024 / 1024, 1)} Mo)"
    return None


# ------------------------------------------------------------------ importer la météo du lieu (Open-Meteo), à la demande
def bouton_meteo(chemin: Sequence, libelle: str, plan, petit: bool = False) -> str:
    """Un bouton qui IMPORTE la météo d'un lieu (Open-Meteo), à la demande de l'utilisateur — une importation de
    DONNÉES, jamais du code ; le calcul reste hors ligne.

    ``plan`` : le PLAN calculé par le moteur (``beton.maturite.meteo.plan``) — le lieu à géocoder (ou sa position), les
    plages de dates de l'archive et des prévisions, les variables. Il part sur le bouton (``data-meteo``, JSON échappé) ;
    au clic, l'écran (``js/meteo.js``) construit lui-même les adresses — vers les trois d'Open-Meteo, et elles seules —,
    fait les appels, puis envoie au module ``{"action": "modifier", "chemin": <chemin>, "valeur": {"lieu", "plan",
    "geocodage", "reponses"}}`` : les réponses BRUTES, que le module lit (``meteo.py``) et GARDE dans la note. Pendant
    la récupération, le bouton dit « Récupération de la météo… » ; un refus (pas de réseau, lieu introuvable, service en
    panne) arrive en message, jamais une exception.

    ``plan`` = ``{"erreur": "<message>"}`` : la récupération est impossible (pas de lieu, dates hors de portée…) — le
    bouton est grisé et le message écrit dessous. ``libelle`` et le message sont des TEXTES (échappés) ; ``petit`` : un
    bouton secondaire (le détail horaire)."""
    erreur = plan.get("erreur") if isinstance(plan, dict) else "le plan de la récupération manque"
    donnees = json.dumps(plan if isinstance(plan, dict) else {"erreur": erreur}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    b = (f'<button type="button" class="btn meteo{" sm" if petit else ""}" data-meteo="{esc(donnees)}" data-meteo-chemin="{_attr_chemin(chemin)}"'
         f'{" disabled" if erreur else ""}>{icon("cloud-sun")}<span>{esc(libelle)}</span></button>')
    return f'<div class="meteo-bt">{b}</div>' + (message(esc(erreur), True) if erreur else "")
