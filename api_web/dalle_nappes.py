# -*- coding: utf-8 -*-
"""Les armatures par NAPPE du module fusionné « Dalles béton armé » (docs/site/DALLES_FUSION.md § 3) : la nappe
inférieure et la nappe supérieure, chacune une VUE d'écran sur les deux faces de l'état du moteur (``inf_x`` et
``inf_y``, ``sup_x`` et ``sup_y``), qui ne change pas d'un octet.

Un treillis soudé est une nappe bidirectionnelle. Dans la nomenclature Øl/Øt/el/et (beton/treillis.py), les fils
PORTEURS Øl/el sont ceux de la direction calculée de la face ; les fils de répartition Øt/et ne sont pas comptés. Un
treillis unique « a/b/ea/eb, porteurs selon X » s'écrit donc, dans l'état inchangé : face X base treillis ``a/b/ea/eb``,
face Y base treillis ``b/a/eb/ea`` — la TRANSPOSÉE. Le moteur compte a/ea en X et b/eb en Y : exactement les deux
armatures du treillis, selon son orientation.

Le MODE d'une nappe n'est pas stocké : il se DÉDUIT (``mode``) des deux bases —
- deux bases treillis dont l'une est la transposée de l'autre → ``treillis`` (``treillis_renforts`` dès qu'une des deux
  faces porte un renfort) ;
- deux bases qui ne sont pas des treillis (barres Ø/e ou n barres) → ``barres`` ;
- sinon (deux treillis différents, ou un treillis d'un côté et des barres de l'autre : une ancienne note) →
  ``incoherent`` : l'écran le dit, et propose d'en faire un treillis.

L'ORIENTATION des porteurs se déduit aussi (``_porteurs``) : la direction dont les fils porteurs ont la plus forte
section au mètre ; à égalité (les dix treillis standards sont symétriques : même armature en X et en Y), la direction
principale. La désignation « vue depuis les porteurs » est la base de cette face. Une note peut donc avoir été écrite
« 8/10/100/150 porteurs X » et se relire « 10/8/150/100 porteurs Y » : c'est la même armature, posée de la même façon,
et la ligne « X : Ø8 e 100 · Y : Ø10 e 150 » le dit sans ambiguïté.

Ce fichier ne calcule rien : il lit l'état, écrit les deux bases d'une nappe et met en forme. Le détail par direction —
l'axe, le centre de gravité, les renforts — reste celui de api_web/plaque_compact.couche et .nappe, aux mêmes chemins et
sous les mêmes clés ``data-cible`` (« s0.inf_y-0 ») : le dessin les sélectionne.

    mode(s, cote, principale=None) -> dict
        ``s`` : une section de l'état (ses quatre faces), ``cote`` : « inf » | « sup », ``principale`` : « x » | « y »
        (départage un treillis symétrique ; sans elle, la direction principale par défaut de l'écran). Rend
        {"mode": "treillis" | "barres" | "treillis_renforts" | "incoherent",
         "porteurs": "x" | "y" | None,              la direction des fils porteurs (treillis seulement)
         "treillis": "Øl/Øt/el/et" | None,          la désignation vue depuis les porteurs (treillis seulement)
         "renforts": {"x": n, "y": n},              les renforts de chaque direction
         "coherent": bool}                          faux en mode « incoherent »
    ecrire(d, chemin, valeur) -> None
        ``chemin`` = ["sections", i, "nappes", "inf" | "sup", "mode" | "treillis" | "porteurs"] ; une écriture
        refusée lève ValueError (message en français, montré tel quel par api_web/dalle.modifier), la dalle intacte.
    html(EN, i, s, rs, verifs, principale, aide_axe, aide_renfort, verrou=False) -> str
        le HTML des DEUX nappes (inférieure, puis supérieure) de la section ``i`` : ``s`` l'état de la section, ``rs``
        son calcul (``MO.verifier(d)["sections"][i]`` ; le e_auto de chaque couche y est déjà celui du SITE :
        api_web/dalle._vue_section), ``verifs`` {clé de face: vérification}, ``principale`` « x » | « y ».
        ``aide_axe`` : le texte d'aide du champ « e axe » (str), OU une fonction ``(f, j, c, geo_c) -> dict`` des
        arguments nommés de plaque_compact.couche propres à la couche (le type prédalle y pose « details » — la pose et
        son pictogramme —, « aide_axe » et « dit_auto ») ; ``aide_renfort`` : le texte d'aide de « + Renfort » (str),
        OU une fonction ``(f) -> str`` (la prédalle dit « posées sur la prédalle » aux faces inférieures).
        ``verrou`` : tout se lit, aucun bouton ne modifie.
"""
from typing import Callable, Dict, List, Optional, Union

from beton import treillis as TR
from beton.dalle import entree as EN

from . import commun as C
from . import plaque_compact as PC

COTES = ("inf", "sup")
DIRECTIONS = ("x", "y")
# les trois segments de l'écran : (valeur du chemin « mode », libellé[, explication au survol]) — le troisième s'appelle
# « Mixte » (le propriétaire, 13/11/2026 : « Treillis + renforts » passait sur deux lignes ; « Mixte » se comprend : le
# treillis plus des barres de renfort, ce que le survol dit en entier)
MIXTE = "Mixte"
MODES = (("treillis", "Treillis soudé"), ("barres", "Barres"),
         ("treillis_renforts", MIXTE, "Treillis soudé et barres de renfort"))
CHAMPS = ("mode", "treillis", "porteurs")
# (« Armature », plus « Nappe » : le propriétaire, 13/11/2026 — le mot ne s'affiche plus nulle part)
TITRES = {"inf": "Armature inférieure", "sup": "Armature supérieure"}
AVERTISSEMENT = "Deux armatures de base différentes en X et en Y — ce n'est pas un treillis unique"


# ------------------------------------------------------------------ la lecture de l'état
def transposee(des: str) -> str:
    """« a/b/ea/eb » → « b/a/eb/ea » : la même nappe de treillis, lue depuis l'autre direction."""
    a, b, ea, eb = TR.fils(des)
    return TR.designation((b, a, eb, ea))


def _autre(dk: str) -> str:
    return "y" if dk == "x" else "x"


def _bases(s: Dict, cote: str):
    """(base X, base Y) d'une nappe : la couche 0 de chacune de ses deux faces."""
    return s["faces"][f"{cote}_x"]["couches"][0], s["faces"][f"{cote}_y"]["couches"][0]


def _treillis_unique(bx: Dict, by: Dict) -> bool:
    """Les deux bases sont des treillis, et l'une est la transposée de l'autre (un treillis symétrique est sa propre
    transposée)."""
    return bx["type"] == "treillis" and by["type"] == "treillis" and transposee(bx["treillis"]) == by["treillis"]


def _porteurs(bx: Dict, by: Dict, principale: str) -> str:
    """La direction des fils porteurs d'un treillis unique : celle dont la base (Øl/el) porte la plus forte section au
    mètre ; à égalité — treillis symétrique —, la direction principale."""
    ax, ay = TR.as_treillis_mm2_m(bx["treillis"]), TR.as_treillis_mm2_m(by["treillis"])
    if ax == ay:
        return principale
    return "x" if ax > ay else "y"


def _principale(principale: Optional[str]) -> str:
    return (principale or EN.DEFAUTS_DALLE["dir_principale"]).lower()


def mode(s: Dict, cote: str, principale: Optional[str] = None) -> Dict:
    """Le mode d'une nappe (« inf » ou « sup ») de la section ``s``, DÉDUIT de ses deux bases : voir l'en-tête."""
    if cote not in COTES:
        raise ValueError(f"Armature : « inf » ou « sup » attendu, pas « {cote} »")
    p = _principale(principale)
    bx, by = _bases(s, cote)
    renforts = {dk: len(s["faces"][f"{cote}_{dk}"]["couches"]) - 1 for dk in DIRECTIONS}
    out = {"mode": "incoherent", "porteurs": None, "treillis": None, "renforts": renforts, "coherent": True}
    if _treillis_unique(bx, by):
        pk = _porteurs(bx, by, p)
        out["mode"] = "treillis_renforts" if renforts["x"] or renforts["y"] else "treillis"
        out["porteurs"] = pk
        out["treillis"] = (bx if pk == "x" else by)["treillis"]
    elif bx["type"] != "treillis" and by["type"] != "treillis":
        out["mode"] = "barres"
    else:
        out["coherent"] = False
    return out


def recues(s: Dict, cote: str) -> Dict[str, tuple]:
    """Ce que chaque direction d'une nappe de treillis reçoit : {"x": (Ø, e), "y": (Ø, e)} en mm, lus dans la base de
    chaque face (les fils porteurs de la face, ceux que le moteur compte)."""
    bx, by = _bases(s, cote)
    tx, ty = TR.fils(bx["treillis"]), TR.fils(by["treillis"])
    return {"x": (tx[0], tx[2]), "y": (ty[0], ty[2])}


# ------------------------------------------------------------------ l'écriture : les chemins « nappes »
def _poser_treillis(c: Dict, des: str) -> None:
    """La couche devient un treillis de désignation ``des`` ; ses autres réglages (Ø, espacement, nombre, axe) restent,
    comme dans l'écran Streamlit où une couche garde ses quatre réglages quel que soit son type."""
    c["type"] = "treillis"
    c["treillis"] = des


def _lire_designation(valeur) -> str:
    t = TR.parse_designation(valeur) if isinstance(valeur, str) else None
    if t is None:
        raise ValueError(f"Treillis : désignation Øl/Øt/el/et attendue (« 10/10/100/100 »), pas « {valeur} »")
    return TR.designation(t)


def _en_treillis(s: Dict, cote: str, principale: str) -> None:
    """Les deux bases deviennent un treillis unique : si l'une l'est déjà on la garde (à deux treillis différents, celle
    de la direction principale) et sa transposée va sur l'autre ; sinon le treillis par défaut sur les deux. Les
    renforts, les axes et les centres de gravité ne bougent pas."""
    bx, by = _bases(s, cote)
    if _treillis_unique(bx, by):
        return
    bases = {"x": bx, "y": by}
    garde = next((dk for dk in (principale, _autre(principale)) if bases[dk]["type"] == "treillis"), None)
    if garde is None:
        _poser_treillis(bx, TR.TREILLIS_DEFAUT)
        _poser_treillis(by, TR.TREILLIS_DEFAUT)
        return
    _poser_treillis(bases[_autre(garde)], transposee(bases[garde]["treillis"]))


def _en_barres(s: Dict, cote: str) -> None:
    """Une base treillis devient des barres Ø/e (son Ø et son espacement, gardés, reprennent) ; une base qui n'est déjà
    pas un treillis (barres, n barres) reste ce qu'elle est."""
    for c in _bases(s, cote):
        if c["type"] == "treillis":
            c["type"] = "barres"


def _ecrire_mode(s: Dict, cote: str, valeur, principale: str) -> None:
    if valeur not in {o[0] for o in MODES}:
        raise ValueError(f"Type d'armature inconnu : « {valeur} » (treillis, barres ou treillis_renforts)")
    m = mode(s, cote, principale)
    if valeur == "barres":
        _en_barres(s, cote)
        return
    if valeur == "treillis" and m["mode"] == "treillis_renforts":
        n = m["renforts"]["x"] + m["renforts"]["y"]
        raise ValueError(f"Cette armature porte {n} renfort{'s' if n > 1 else ''} : « Treillis soudé » seul les retirerait. Les "
                         f"retirer un à un (« Retirer ce renfort », dans le détail de chaque renfort), ou garder « {MIXTE} ».")
    _en_treillis(s, cote, principale)
    if valeur == "treillis_renforts":
        m = mode(s, cote, principale)
        if not (m["renforts"]["x"] or m["renforts"]["y"]):
            # le premier renfort, sur la face des porteurs : le même geste que « + Renfort » de cette face
            # (sections/i/faces/<face>/couches/+ dans api_web/dalle._ecrire)
            couches = s["faces"][f"{cote}_{m['porteurs']}"]["couches"]
            if len(couches) >= EN.MAX_COUCHES:
                raise ValueError(f"Au plus {EN.MAX_COUCHES} couches par face (la base et trois renforts)")
            couches.append(EN.nouvelle_couche(base=False))


def _ecrire_treillis(s: Dict, cote: str, valeur, principale: str) -> None:
    """La désignation va sur la face des porteurs (ceux de la nappe si c'en est déjà un treillis unique, sinon la
    direction principale), sa transposée sur l'autre."""
    des = _lire_designation(valeur)
    m = mode(s, cote, principale)
    pk = m["porteurs"] or principale
    bases = dict(zip(DIRECTIONS, _bases(s, cote)))
    _poser_treillis(bases[pk], des)
    _poser_treillis(bases[_autre(pk)], transposee(des))


def _ecrire_porteurs(s: Dict, cote: str, valeur, principale: str) -> None:
    """« x » ou « y » : les deux désignations s'échangent (la nappe garde ses deux armatures, tournée d'un quart de
    tour) ; un treillis symétrique ne change pas."""
    dk = str(valeur).strip().lower() if isinstance(valeur, str) else None
    if dk not in DIRECTIONS:
        raise ValueError(f"Orientation des porteurs : X ou Y, pas « {valeur} »")
    m = mode(s, cote, principale)
    if m["mode"] not in ("treillis", "treillis_renforts"):
        raise ValueError("Orientation des porteurs : cette armature n'est pas un treillis unique (choisir d'abord « Treillis soudé »)")
    if dk == m["porteurs"]:
        return
    bx, by = _bases(s, cote)
    bx["treillis"], by["treillis"] = by["treillis"], bx["treillis"]


def ecrire(d: Dict, chemin: List, valeur) -> None:
    """Écrit une saisie de NAPPE dans la dalle ``d`` (une copie, que l'appelant renormalise) : ``chemin`` =
    ["sections", i, "nappes", "inf" | "sup", "mode" | "treillis" | "porteurs"]. ``ValueError`` (message français, à
    montrer tel quel) pour toute valeur refusée ; la dalle n'est alors pas touchée. Les renforts, les axes et les centres
    de gravité restent par face, par les chemins existants de api_web/dalle._ecrire."""
    if len(chemin) != 5 or chemin[0] != "sections" or chemin[2] != "nappes":
        raise ValueError(f"Chemin d'armature attendu (sections/i/nappes/inf|sup/champ), pas « {'/'.join(map(str, chemin))} »")
    try:
        i = int(chemin[1])
        s = d["sections"][i]
    except (ValueError, TypeError, IndexError):
        raise ValueError(f"Section introuvable : « {chemin[1]} »") from None
    cote, champ = chemin[3], chemin[4]
    if cote not in COTES:
        raise ValueError(f"Armature : « inf » ou « sup » attendu, pas « {cote} »")
    if champ not in CHAMPS:
        raise ValueError(f"Champ d'armature inconnu : « {champ} » (mode, treillis ou porteurs)")
    p = d["dir_principale"].lower()
    {"mode": _ecrire_mode, "treillis": _ecrire_treillis, "porteurs": _ecrire_porteurs}[champ](s, cote, valeur, p)


# ------------------------------------------------------------------ l'écran : les deux nappes de la section courante
def _lu(valeur: str) -> str:
    return f'<span class="et-lu">{C.esc(valeur)}</span>'


def _couche_lue(EN_, i: int, f: str, j: int, c: Dict, geo_c: Dict, details: str = "", aide_axe: str = "", dit_auto: bool = True,
                verrou: bool = False, role: str = "", deplacer: bool = False) -> str:
    """Une couche dont le type et la désignation se LISENT (la base d'une nappe de treillis : son treillis et son
    orientation se règlent par la nappe, pas par la direction) — même ligne, même repère, même clé ``data-cible`` et
    mêmes arguments nommés que plaque_compact.couche ; son axe reste réglable (vide = automatique, « ↺ »), son détail
    dit ses fils et sa section. ``verrou`` : l'axe aussi se lit, et aucun bouton ne modifie.

    ``role`` (« porteurs » ou « répartition » : la base d'une nappe de TREILLIS, 13/11/2026 — § 8.4) : la ligne dit ce que
    CETTE direction reçoit, « porteurs · Ø10 e 100 », au lieu de répéter le type et la désignation que la nappe porte
    déjà (le propriétaire : « éviter de réafficher plusieurs fois le type d'armature »)."""
    base = ["sections", i, "faces", f, "couches", j]
    tag = "B" if j == 0 else f"R{j}"
    nom = "Couche de base" if j == 0 else f"Renfort {j}"
    if c["type"] == "treillis":
        t = geo_c["treillis"]
        if role:
            valeur = f"Ø{t[0]} e {t[2]}"
            info = f"fils {'porteurs' if role == 'porteurs' else 'de répartition'} du treillis : Ø{t[0]} tous les {t[2]} mm"
        else:
            valeur = c["treillis"]
            info = (f"treillis soudé : porteurs Ø{t[0]} tous les {t[2]} mm, comptés dans cette direction ; répartition Ø{t[1]} "
                    f"tous les {t[3]} mm, non comptés ici")
    elif c["type"] == "barres":
        valeur = f"Ø{c['d']} e {c['esp']}"
        info = f"barres Ø{c['d']} tous les {c['esp']} mm"
    else:
        valeur = f"{c['n']} Ø{c['d']}"
        info = f"{c['n']} barres Ø{c['d']} réparties dans la bande"
    genre = role if role and c["type"] == "treillis" else dict(PC.TYPES)[c["type"]]
    tete = (f'<div class="et-l1"><button type="button" class="et-tag" data-deplier="1" title="{nom} : axe et détail">{tag}</button>'
            f'<div class="et-ty">{_lu(genre)}</div><div class="et-tr">{_lu(valeur)}</div>'
            f'<button type="button" class="et-dep" data-deplier="1" title="Axe et détail">{C.icon("chevron-down")}</button></div>')
    if verrou:
        axe = _lu(f"{PC._cm(geo_c['e_reel'])} cm · {'imposée' if c['dist'] is not None else 'auto'}")
    else:
        axe = PC.auto_ou_saisi(base + ["dist"], c["dist"], geo_c["reel_auto"], "Revenir à la position automatique",
                               base if deplacer else None)
        if c["dist"] is not None and dit_auto:
            info += f" · position saisie, l'automatique serait {PC._cm(geo_c['reel_auto'])} cm"
    det = (C.ligne("Position", axe, "", aide_axe or "Du parement à l'axe des barres, là où elles sont posées (le dessin). Vide : la position automatique ; une valeur saisie l'impose. Le calcul y ajoute le jeu de pose et arrondit (Options).")
           + details
           + f'<div class="hint et-info">{C.esc(info)} · A<sub>s</sub> = {C.fr(geo_c["As_pm"], 0)}{C.NBSP}mm²/m</div>')
    return f'<div class="et-piece" data-cible="s{i}.{f}-{j}">{tete}<div class="et-det">{det}</div></div>'


def _nappe_lue(i: int, f: str, face: Dict, geo: Dict, v: Dict, principale: str, couches: str) -> str:
    """Une direction d'une nappe VERROUILLÉE : le titre, la section nécessaire, ses couches (lues), la section posée et
    le centre de gravité — sans « + Renfort » ni sélecteur (plaque_compact.nappe les pose toujours)."""
    _, dk = f.split("_")
    princ = '<span class="et-princ" title="Direction principale">princ.</span>' if dk == principale else ""
    yg = C.ligne("y<sub>G</sub>", _lu(f"{PC._cm(geo['yG'])} cm · {'imposé' if face['yG'] is not None else 'calculé'}"), "",
                 "Centre de gravité des couches, mesuré depuis le parement.")
    return (f'<div class="et-grp"><div class="et-gh"><span class="t">{PC.titre_face(f)}</span>{princ}'
            f'<span class="et-req">A<sub>s</sub> néc. <b>{C.fr(v["valeur"], 0)}</b>{C.NBSP}mm²</span></div>'
            f'{couches}<div class="et-pied"><span class="et-pied-a">A<sub>s</sub> posé {C.fr(geo["As_pm"], 0)}{C.NBSP}mm²/m</span></div>{yg}</div>')


def _reglage_treillis(ch: List, m: Dict, s: Dict, cote: str, principale: str, verrou: bool) -> str:
    """Le réglage d'une nappe de treillis : UN sélecteur de treillis, l'orientation des porteurs (la principale marquée ;
    sans effet, donc grisée, pour un treillis symétrique), et la ligne de ce que chaque direction reçoit."""
    opts = TR.liste_choix()
    if m["treillis"] not in opts:
        opts = opts + [m["treillis"]]            # une ancienne note : la désignation reste utilisable
    sel = C.choix(ch + ["treillis"], m["treillis"], [(x, x) for x in opts], desactive=verrou)
    sym = transposee(m["treillis"]) == m["treillis"]
    # (les segments disent « X » et « Y » seulement : « Y · princ. » était coupé au petit portable — le harnais l'a vu ;
    # la direction principale est dite dans l'aide ⓘ, et marquée « princ. » sur le détail de sa direction, dessous)
    orient = C.segments(ch + ["porteurs"], m["porteurs"], [(dk, dk.upper()) for dk in DIRECTIONS], desactive=verrou or sym)
    aide_o = (("Treillis symétrique : même armature en X et en Y, l'orientation est sans effet." if sym
               else "La direction des fils porteurs (Øl tous les el) ; l'autre direction reçoit les fils de répartition (Øt tous les et).")
              + f" Direction principale : {principale.upper()}.")
    # (13/11/2026 : plus de ligne « X : Ø10 e 100 · Y : Ø10 e 100 » — chaque direction le dit sur la ligne de sa base,
    #  « porteurs · Ø10 e 100 », une fois ; plus de libellé « Treillis » devant le menu — « Treillis soudé », juste
    #  au-dessus, le dit déjà, et la désignation y était coupée, « 10/10/10… » : le menu prend la place, son explication
    #  au survol)
    aide_t = ("La désignation Øl/Øt/el/et pose les DEUX directions : les porteurs dans la direction choisie, la répartition "
              "dans l'autre (la transposée de la désignation).")
    return (f'<div class="row2 et-treillis"><div class="et-tr-sel" title="{C.esc(aide_t)}">{sel}</div>'
            f'<label><span>Porteurs</span><span class="aide" title="{C.esc(aide_o)}">{C.icon("info")}</span></label>{orient}</div>')


def _nappe_html(EN_, i: int, s: Dict, rs: Dict, verifs: Dict, cote: str, principale: str, aide_axe: Union[str, Callable],
                aide_renfort: Union[str, Callable], verrou: bool, apres: str = "") -> str:
    m = mode(s, cote, principale)
    ch = ["sections", i, "nappes", cote]
    tete = (f'<div class="et-nap-h"><span class="t">{TITRES[cote]}</span></div>'
            + C.segments(ch + ["mode"], m["mode"], MODES, desactive=verrou))
    en_treillis = m["mode"] in ("treillis", "treillis_renforts")
    corps = _reglage_treillis(ch, m, s, cote, principale, verrou) if en_treillis else ""
    if m["mode"] == "incoherent":
        bouton = ("" if verrou else f'<button type="button" class="et-nap-btn" data-chemin="{C._attr_chemin(ch + ["mode"])}" '
                                    f'data-valeur="treillis">En faire un treillis</button>')
        corps += f'<div class="et-avert et-nap-avert"><span>{C.esc(AVERTISSEMENT)}.</span>{bouton}</div>'
    faces = ""
    for dk in (principale, _autre(principale)):
        f = f"{cote}_{dk}"
        face, geo = s["faces"][f], rs["faces"][f]
        couches = ""
        for j, (c, g) in enumerate(zip(face["couches"], geo["couches"])):
            kw = aide_axe(f, j, c, g) if callable(aide_axe) else {"aide_axe": aide_axe}
            # (DAL5c) les flèches ▲▼ de l'axe : sur toute couche réglable, sauf les fils secondaires d'un treillis unique —
            # soudés sur les fils principaux, le treillis se déplace par ceux-ci (dalle_positions.deplacer le refuse aussi)
            kw["deplacer"] = not (j == 0 and en_treillis and dk != principale)
            if verrou or (j == 0 and en_treillis):
                role = ("porteurs" if m["porteurs"] == dk else "répartition") if (j == 0 and en_treillis) else ""
                couches += _couche_lue(EN_, i, f, j, c, g, verrou=verrou, role=role, **kw)
            else:
                couches += PC.couche(EN_, i, f, j, c, g, **kw)
        aide = aide_renfort(f) if callable(aide_renfort) else aide_renfort
        if verrou:
            faces += _nappe_lue(i, f, face, geo, verifs[f], principale, couches)
        else:
            faces += PC.nappe(EN_, i, f, face, geo, verifs[f], principale, couches, aide, verrou)
    return f'<div class="et-nap" data-nappe="{cote}">{tete}{corps}{faces}{apres}</div>'


def html(EN, i: int, s: Dict, rs: Dict, verifs: Dict, principale: str, aide_axe: Union[str, Callable],
         aide_renfort: Union[str, Callable], verrou: bool = False, apres_inf: str = "") -> str:
    """Les DEUX nappes (inférieure, puis supérieure) de la section ``i`` : pour chacune, son titre, les segments du mode
    (« Treillis soudé · Barres · Mixte », déduit), puis — treillis — le sélecteur unique et l'orientation
    des porteurs, — incohérente — l'avertissement et « En faire un treillis » ; puis le détail de chaque direction (la
    principale d'abord) : plaque_compact.nappe et .couche, la base d'un treillis en lecture — ce que SA direction reçoit,
    « porteurs · Ø10 e 100 » (son axe reste réglable). Les arguments : voir l'en-tête du module. ``apres_inf`` : ce que le
    module pose à la fin de la nappe inférieure (une prédalle : ses barres de couture, lot 5)."""
    p = _principale(principale)
    return "".join(_nappe_html(EN, i, s, rs, verifs, cote, p, aide_axe, aide_renfort, verrou, apres_inf if cote == "inf" else "")
                   for cote in COTES)
