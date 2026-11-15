# -*- coding: utf-8 -*-
"""Le CALCUL DES BANDES du « Générateur SAF — Fondation de grue » (lot 6, 15/11/2026 : « une note de calcul au droit des
bandes, aux appuis ») : la section de calcul de chaque bande (``saf.fondation_grue.sections``) mise en mots par le MODULE
DALLES — ses formules (``api_web/dalle._formule``), ses lignes de tableau, ses taux : l'écriture du module Dalles, une seule
pour les deux modules — et le diagramme de couverture de M(s) (``dessin_efforts_saf.couverture``).

L'écran en fait une fenêtre de calcul par direction (« bd-x », « bd-y », ``saf_armatures.ident_bandes`` : les lignes
Flexion, Hauteur et Effort tranchant de la carte ARMATURES l'ouvrent) ; la note, le chapitre « Calcul des bandes »
(``verifs_note``). Deux bandes pareilles (« BX1 = BX2 », ``saf_armatures.groupe_bandes``) sont écrites une fois : la plus
chargée (le premier à égalité).

Ce fichier ne calcule rien : il CHOISIT parmi les vérifications du moteur Dalles (la hauteur, les deux faces de la bande,
l'effort tranchant — la direction perpendiculaire est celle des autres bandes et des cartes) et les met en mots."""
from typing import Callable, Dict, List

from saf.fondation_grue import sections as SBM

from . import commun as C
from . import dessin_efforts_saf as DE
from . import saf_armatures as SA

N = C.NBSP
FACES_TXT = {"inf": "bas", "sup": "haut"}

PIED = ("La bande est calculée par le moteur du module Dalles comme une section b × h, avec ce que le principe y pose : le "
        "treillis de chaque face et le renfort qui gouverne (n barres dans la bande, à l'axe du treillis de sa direction). "
        "M⁺ et M⁻ sont pris dans la même section (ils ne sont pas au même endroit de la bande : côté sécurité). Le moteur "
        "Dalles y applique toute sa méthode, dont le minimum 0,25·A<sub>s,req</sub> de la face opposée et le maximum "
        "0,04·b·h ; h est lu au cm inférieur (d côté sécurité), A<sub>s,min</sub> au cm supérieur. "
        "h<sub>min</sub> = h<sub>u,min</sub> + y<sub>G</sub> ≤ h équivaut à d<sub>min</sub> = h<sub>u,min</sub> ≤ d = h − y<sub>G</sub> : "
        "la ligne Hauteur du principe.")


def _D():
    """Le module Dalles (ses formules mises en mots), chargé au premier besoin."""
    from . import dalle as D
    return D


# ------------------------------------------------------------------ ce qui est écrit
def groupes(fer: Dict, ef: Dict) -> List[Dict]:
    """Par direction (BX, puis BY) : {"sens", "ident" (la fenêtre), "nom" (« BX1 = BX2 », « BX1 · BX2 »), "sections":
    [(le nom écrit, la section)]} — deux bandes pareilles : une section, la plus chargée."""
    secs = {s["bande"]: s for s in fer.get("sections") or []}
    out = []
    for sens, _ in SA.SENS:
        g = SA.groupe_bandes(fer, ef, sens)
        ss = [secs[b["nom"]] for b in g["bandes"] if b["nom"] in secs]
        if not ss:
            continue
        if " = " in g["nom"] and not any(s.get("hors") for s in ss):
            # (la plus chargée, la première à 1e-6 près : deux bandes pareilles ne diffèrent que d'un bruit de calcul — qui
            #  n'est pas le même dans Pyodide et dans CPython, ``sections.EGAUX``)
            haut = max(s["taux_max"] for s in ss)
            paires = [(g["nom"], next(s for s in ss if s["taux_max"] >= haut * (1.0 - SBM.EGAUX)))]
        else:
            paires = [(s["bande"], s) for s in ss]
        out.append({"sens": sens, "ident": SA.ident_bandes(sens), "nom": g["nom"], "sections": paires})
    return out


def efforts_txt(s: Dict) -> str:
    """« M⁺ = 93,4 kN·m · M⁻ = 17,3 kN·m · V à d = 84,1 kN » : les efforts de la section, sur la largeur de la bande."""
    fi, fs = s["faces"]["inf"], s["faces"]["sup"]
    if s["V"] is None:
        v = f"V{N}à{N}d : ne se lit pas (non vérifié)"
    else:
        v = f"V{N}à{N}d{N}={N}{C.fr(s['V'], 1)}{N}kN" + (" (SCIA)" if s["scia_V"] else "")
    return f"M⁺{N}={N}{C.fr(fi['M'], 1)}{N}kN·m · M⁻{N}={N}{C.fr(fs['M'], 1)}{N}kN·m · {v}"


def armatures_txt(s: Dict) -> str:
    """« bas Ø12/150 + 4 Ø12 (R1) · haut Ø12/150 » : ce que la section porte, face par face."""
    out = []
    for face in ("inf", "sup"):
        f = s["faces"][face]
        lib = s["section"]["faces"][f["cle"]]["libelle"]
        out.append(f"{FACES_TXT[face]} {lib}" + (f" ({f['renfort']['rep']})" if f["renfort"] else ""))
    return " · ".join(out)


def origine_txt(s: Dict) -> str:
    """D'où vient chaque moment de la section."""
    out = []
    for face, sym in (("inf", "M⁺"), ("sup", "M⁻")):
        f = s["faces"][face]
        if f["renfort"]:
            n = len(f["renforts"])
            autres = f" (le plus chargé des {n} renforts de la bande)" if n > 1 else ""
            out.append(f"{sym} : celui du renfort {f['renfort']['rep']}{autres}, le plus fort de son tronçon")
        else:
            out.append(f"{sym} : le plus fort de la bande, sans renfort")
    return " ; ".join(out) + "."


def _vue(s: Dict) -> Dict:
    """La section telle que le module Dalles la lit : ses vérifications écrites (``s["verifications"]``)."""
    return dict(s["section"], verifications=s["verifications"])


def titre_verif(v: Dict) -> str:
    t = v["titre"]
    if v["cle"] == "hauteur" and v.get("face"):
        cote, dk = v["face"].split("_")
        t += f" · la famille {cote}. {dk.upper()} gouverne"
    return t


def lignes_verif(s: Dict, v: Dict) -> List[Dict]:
    """Les lignes « symbole = formule = valeurs = résultat » d'une vérification (le module Dalles), la ligne du cm supérieur
    de h quand elle compte, puis le taux."""
    D = _D()
    vue = _vue(s)
    out = [D._formule(f) for f in v["formules"]]
    p = s.get("plafond")
    if p and v["cle"] in p["As_min"]:
        lu = next(f["resultat"] for f in v["formules"] if f["cle"] == "besoin")
        out.append({"parts": ["A_s,nécessaire", f"max(A_s,nécessaire ; A_s,min à h = {p['h']} cm)",
                              f"max({D._n(lu, 'mm²')} ; {D._n(p['As_min'][v['cle']], 'mm²')})"],
                    "res": f"{D._n(v['valeur'], 'mm²')} mm²", "niveau": ""})
    out.append(D._taux_fx(v, vue))
    return out


def comparaison_txt(s: Dict, v: Dict) -> str:
    """La comparaison d'une vérification (HTML du module Dalles) : « 723,8 mm² ≤ A_s = 829,4 mm² »…"""
    return _D()._verdict(v, _vue(s))


def niveau(v: Dict) -> str:
    return _D()._niv(v)


# ------------------------------------------------------------------ l'écran : la fenêtre de calcul d'une direction
def _figure(bd: Dict, s: Dict, fer: Dict, w: float = 380.0) -> str:
    """Le diagramme de couverture de la bande ``bd`` (sa section ``s``)."""
    ech = fer.get("echelle") or {}
    nom_sens = "x" if bd["nom"].startswith("BX") else "y"
    barres = {r["rep"]: SA.barres(r).replace(N, " ") for r in fer["renforts"] if r["type"] == "bande" and r["bande"] == bd["nom"]}
    return DE.couverture(bd, s["couverture"], ech.get(f"inf_{nom_sens}", 1.0), ech.get(f"sup_{s['sens']}", 1.0), w, 160.0, barres)


def _bande(ef: Dict, nom: str) -> Dict:
    return next(b for b in ef["bandes"] if b["nom"] == nom)


def _legende(ef: Dict, w: float = 380.0) -> str:
    """La légende du diagramme de couverture (avec la zone grise des plots coulés, s'il y en a)."""
    return DE.legende_couverture(w, any(bool(b["dans_plot"].any()) for b in ef["bandes"]))


def fenetre(el: Dict, fer: Dict, ef: Dict) -> str:
    """Les fenêtres « bd-x » et « bd-y » : pour chaque section écrite, le tableau des vérifications du module Dalles, le
    diagramme de couverture de M(s), puis le détail de chaque vérification."""
    if fer.get("hors_domaine"):
        return ""
    D = _D()
    out = ""
    for gr in groupes(fer, ef):
        lignes, detail, hyps = [], "", []
        plusieurs = len(gr["sections"]) > 1
        for nom, s in gr["sections"]:
            if s.get("hors"):
                lignes.append({"nom": f"Bande {C.esc(nom)}", "formule": C.esc(s["hors"]), "valeur": "", "limite": "", "taux": None,
                               "niveau": "v", "etat_txt": "non calculée"})
                continue
            vue = _vue(s)
            gouv = D.gouvernante_section(vue)
            for v in s["verifications"]:
                li = D._ligne_tableau(v, vue, v is gouv)
                if v["cle"] == "tau":                     # (V à d lu par le module : au dixième, pas à la précision du calcul)
                    li = dict(li, sous=f"V{N}à{N}d{N}={N}{C.fr(s['V'], 1)}{N}kN · sans étriers")
                if plusieurs:
                    li = dict(li, nom=f"{C.esc(nom)} · {li['nom']}")
                lignes.append(li)
            detail += (f'<div class="ck-g sf-pleine"><div class="ck-t">Bande {C.esc(nom)} · b × h = {s["b"]} × {s["h"]} cm · '
                       f'{efforts_txt(s)}</div><div class="hint">{C.esc(armatures_txt(s))} · {C.esc(origine_txt(s))}</div>'
                       f'<div class="sf-couv">{_legende(ef)}{_figure(_bande(ef, s["bande"]), s, fer)}</div></div>')
            for v in s["verifications"]:
                detail += (f'<div class="ck-g"><div class="ck-t">{C.esc((nom + " · ") if plusieurs else "")}{C.esc(titre_verif(v))}</div>'
                           f'{C.lignes_fx(lignes_verif(s, v))}</div>')
            if not hyps:
                hyps = D._hypotheses(s["resultat"]["dalle"], s["resultat"]["materiaux"])
        ok = [s for _, s in gr["sections"] if not s.get("hors")]
        sous = (f"b × h = {ok[0]['b']} × {ok[0]['h']} cm · {efforts_txt(ok[0])} · moteur Dalles" if len(ok) == 1
                else f"{len(gr['sections'])} bandes · moteur Dalles")
        out += C.fenetre_calcul(gr["ident"], f"Bandes {C.esc(gr['nom'])} · section de calcul", sous, hyps, lignes, detail,
                                pied=C.aide(PIED))
    return out


# ------------------------------------------------------------------ la note : le chapitre « Calcul des bandes »
def verifs_note(el: Dict, fer: Dict, ef: Dict, largeur: float, figure: Callable[[str, str], str], riche: Callable[[str], str],
                tex: Callable[[str], str]) -> List[Dict]:
    """Les blocs du chapitre « Calcul des bandes » de la note : une section écrite par bloc — ses efforts, ses armatures, le
    diagramme de couverture, puis chaque vérification (ses formules, son verdict). ``figure(svg, nom)`` pose un dessin à la
    largeur de la colonne, ``riche(html)`` rend un texte riche de la note (ses indices composés), ``tex(s)`` une formule."""
    out = []
    intro = riche("Chaque bande est calculée par le moteur du module Dalles comme une section b × h — ses formules sont "
                  "celles de la note du module Dalles. " + PIED.split(". ", 1)[1])
    leg = figure(_legende(ef, largeur), "Couverture : légende")
    for gr in groupes(fer, ef):
        for nom, s in gr["sections"]:
            if s.get("hors"):
                out.append({"titre": f"Bande {nom}", "items": [("k", 0)],
                            "verdicts": [{"texte": f"Section non calculée : {s['hors']}.", "etat": "att"}]})
                continue
            items: List[tuple] = []
            if not out:
                items.append(("pr", intro))
            items += [("pr", riche(f"{efforts_txt(s)} · {armatures_txt(s)}.")), ("pr", riche(origine_txt(s))),
                      ("fig", leg, None),
                      ("fig", figure(_figure(_bande(ef, s["bande"]), s, fer, largeur), f"Couverture : {s['bande']}"),
                       f"M(s) [kN·m] ET CE QUE REPRENNENT LES ARMATURES · BANDE {s['bande']}")]
            verdicts = []
            for v in s["verifications"]:
                items.append(("s", titre_verif(v)))
                for li in lignes_verif(s, v):
                    items.append(("e", None, [tex(p) for p in li["parts"]], tex(li["res"])))
                items.append(("k", len(verdicts)))
                verdicts.append({"texte": riche(comparaison_txt(s, v)), "etat": "ok" if v["etat"] == "ok" else "ko",
                                 "cale": f"{C.pct(v['taux'])} %", "riche": True})
            out.append({"titre": f"Bande {nom} — section {s['b']} × {s['h']} cm", "items": items, "verdicts": verdicts})
    return out
