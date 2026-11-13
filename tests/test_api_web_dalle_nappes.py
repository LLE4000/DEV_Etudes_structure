# -*- coding: utf-8 -*-
"""Les armatures d'une dalle par NAPPE (api_web/dalle_nappes.py ; docs/site/DALLES_FUSION.md § 3), vérifiées en Python
ordinaire, sur des états construits avec beton.dalle.entree.nouvelle_dalle() / nouvelle_section() :

1. le MODE se déduit des deux bases, sans champ nouveau : treillis, treillis + renforts, barres, incohérent (deux
   treillis différents, ou treillis d'un côté et barres de l'autre) ; l'orientation des porteurs d'un treillis
   dissymétrique ; un côté inconnu est refusé ;
2. l'ÉCRITURE par nappe : un treillis dissymétrique 10/12/100/150 pose la désignation sur la face des porteurs et sa
   TRANSPOSÉE sur l'autre (X reçoit Ø10 tous les 100, Y Ø12 tous les 150 : les sections au mètre du moteur le prouvent),
   l'orientation s'échange, treillis → barres garde Ø et espacement, barres → treillis pose un treillis unique, une
   désignation ou un côté illisible est refusé, et rien d'autre ne bouge (renforts, axes, centres de gravité) ;
3. le HTML : les segments du mode, UN sélecteur de treillis par nappe, l'orientation des porteurs (la principale marquée,
   grisée pour un treillis symétrique), la base lue — ce que sa direction reçoit, « porteurs · Ø10 e 100 » — (son axe réglable) en
   mode treillis, l'écran d'aujourd'hui en mode barres, l'avertissement et « En faire un treillis » en mode incohérent,
   aucun doublon de data-cible, le verrou sans aucun bouton ;
4. trois MUTATIONS : chaque garantie ci-dessus rougit quand on casse ce qu'elle surveille.

    python3 tests/test_api_web_dalle_nappes.py
"""
import copy
import html
import math
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import dalle_nappes as NAP  # noqa: E402
from api_web import plaque_compact as PC  # noqa: E402
from beton import treillis as TR  # noqa: E402
from beton.dalle import entree as EN  # noqa: E402
from beton.dalle import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def refus(f, *a, **k):
    """Le message de la ValueError que ``f`` lève, ou None si elle n'en lève pas."""
    try:
        f(*a, **k)
    except ValueError as e:
        return str(e)
    return None


class Abime:
    """Abîme un attribut d'un module le temps d'un bloc (mutation)."""

    def __init__(self, mod, **attrs):
        self.mod, self.attrs, self.avant = mod, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.avant[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)

    def __exit__(self, *_):
        for k, v in self.avant.items():
            setattr(self.mod, k, v)


def base(s, f):
    return s["faces"][f]["couches"][0]


def poser(s, f, **c):
    """Règle la couche de base d'une face (type, treillis, d, esp…)."""
    base(s, f).update(c)


def des(s, cote):
    """(désignation X, désignation Y) des bases d'une nappe."""
    return base(s, f"{cote}_x")["treillis"], base(s, f"{cote}_y")["treillis"]


def html_de(d, i=0, principale=None, verrou=False, aide_axe="Aide axe", aide_renfort="Aide renfort"):
    """Le HTML des nappes de la section ``i`` d'une dalle, avec le calcul du moteur."""
    r = MO.verifier(EN.normaliser(d))
    rs = r["sections"][i]
    return NAP.html(EN, i, d["sections"][i], rs, {v["cle"]: v for v in rs["verifications"]},
                    principale or d["dir_principale"].lower(), aide_axe, aide_renfort, verrou)


def nappe_html(h, cote):
    """Le fragment d'UNE nappe (« inf » ou « sup ») dans le HTML des deux."""
    parts = re.split(r'(?=<div class="et-nap" data-nappe=")', h)
    return next(p for p in parts if f'data-nappe="{cote}"' in p)


SEG = r'<button type="button" class="(on)?" data-chemin="{}" data-sorte="choix" data-valeur="([^"]+)"( disabled)?>([^<]+)</button>'

# ------------------------------------------------------------------ 1. le mode se déduit des deux bases
print("=== 1 · Le mode, déduit")
s = EN.nouvelle_section("A")
m = NAP.mode(s, "inf")
chk("une section neuve : treillis 10/10/100/100 sur les deux faces → mode « treillis », porteurs = direction principale (Y par "
    "défaut), désignation 10/10/100/100, aucun renfort, cohérente",
    m == {"mode": "treillis", "porteurs": "y", "treillis": TR.TREILLIS_DEFAUT, "renforts": {"x": 0, "y": 0}, "coherent": True}, m)
chk("… la principale passée en X : porteurs X (treillis symétrique : les deux lectures sont possibles, la principale départage)",
    NAP.mode(s, "inf", "x")["porteurs"] == "x" and NAP.mode(s, "sup", "X")["porteurs"] == "x")
s["faces"]["inf_y"]["couches"].append(EN.nouvelle_couche(base=False))
m = NAP.mode(s, "inf")
chk("un renfort sur la face inférieure Y → « treillis_renforts », renforts {x: 0, y: 1} ; la nappe supérieure reste « treillis »",
    m["mode"] == "treillis_renforts" and m["renforts"] == {"x": 0, "y": 1} and NAP.mode(s, "sup")["mode"] == "treillis", m)
s = EN.nouvelle_section("A")
poser(s, "sup_x", type="barres")
poser(s, "sup_y", type="n_barres")
m = NAP.mode(s, "sup")
chk("deux bases qui ne sont pas des treillis (barres Ø/e en X, n barres en Y) → « barres », sans porteurs ni désignation",
    m["mode"] == "barres" and m["porteurs"] is None and m["treillis"] is None and m["coherent"], m)
poser(s, "inf_x", treillis="8/8/150/150")
m = NAP.mode(s, "inf")
chk("deux treillis DIFFÉRENTS (8/8/150/150 en X, 10/10/100/100 en Y : une ancienne note) → « incoherent », non cohérente",
    m["mode"] == "incoherent" and m["coherent"] is False and m["porteurs"] is None and m["treillis"] is None, m)
poser(s, "inf_x", treillis=TR.TREILLIS_DEFAUT, type="barres")
chk("un treillis d'un côté et des barres de l'autre → « incoherent »", NAP.mode(s, "inf")["mode"] == "incoherent")
s = EN.nouvelle_section("A")
poser(s, "inf_x", treillis="10/12/100/150")
poser(s, "inf_y", treillis="12/10/150/100")
m = NAP.mode(s, "inf")
chk("un treillis dissymétrique transposé (X 10/12/100/150, Y 12/10/150/100) → « treillis », porteurs X (Ø10 tous les 100 : la plus "
    "forte section au mètre), désignation vue depuis X",
    m["mode"] == "treillis" and m["porteurs"] == "x" and m["treillis"] == "10/12/100/150", m)
poser(s, "inf_x", treillis="12/10/150/100")
poser(s, "inf_y", treillis="10/12/100/150")
m = NAP.mode(s, "inf")
chk("… les deux faces échangées : porteurs Y, même désignation (la même armature, tournée d'un quart de tour)",
    m["porteurs"] == "y" and m["treillis"] == "10/12/100/150", m)
chk("un côté inconnu est refusé (« inf » ou « sup »)", "inf » ou « sup" in (refus(NAP.mode, s, "milieu") or ""))
chk("la transposée : « a/b/ea/eb » → « b/a/eb/ea » ; un treillis symétrique est sa propre transposée",
    NAP.transposee("10/12/100/150") == "12/10/150/100" and NAP.transposee("8/8/150/150") == "8/8/150/150")

# ------------------------------------------------------------------ 2. l'écriture par nappe
print("\n=== 2 · L'écriture")
CH = lambda cote, champ, i=0: ["sections", i, "nappes", cote, champ]  # noqa: E731


def controle_transposee(d):
    """Le contrôle de la transposée, réutilisé par la mutation : X 10/12/100/150 (π·5²·1000/100 mm²/m), Y 12/10/150/100
    (π·6²·1000/150 mm²/m), lues par le moteur."""
    s_ = d["sections"][0]
    bx, by = base(s_, "inf_x"), base(s_, "inf_y")
    return (des(s_, "inf") == ("10/12/100/150", "12/10/150/100")
            and math.isclose(MO.couche_as_par_m(bx, d["b"]), math.pi * 5 ** 2 * 1000 / 100, rel_tol=1e-12)
            and math.isclose(MO.couche_as_par_m(by, d["b"]), math.pi * 6 ** 2 * 1000 / 150, rel_tol=1e-12))


d = EN.nouvelle_dalle()                      # principale Y
NAP.ecrire(d, CH("inf", "treillis"), "10/12/100/150")
s = d["sections"][0]
chk("ecrire « treillis » 10/12/100/150 (principale Y, treillis symétrique avant) : la désignation va sur la face Y, sa "
    "transposée 12/10/150/100 sur la face X ; mode treillis, porteurs Y",
    des(s, "inf") == ("12/10/150/100", "10/12/100/150") and NAP.mode(s, "inf", "y") == dict(NAP.mode(s, "inf", "y"), mode="treillis", porteurs="y", treillis="10/12/100/150"),
    (des(s, "inf"), NAP.mode(s, "inf", "y")))
chk("… la dalle écrite est dans le domaine du moteur (normaliser la rend telle quelle)", EN.normaliser(d) == d)
NAP.ecrire(d, CH("inf", "porteurs"), "x")
chk("ecrire « porteurs » X : les deux désignations s'échangent — X 10/12/100/150, Y 12/10/150/100 ; le moteur compte en X "
    "π·5²·1000/100 = 785 mm²/m et en Y π·6²·1000/150 = 754 mm²/m", controle_transposee(d), des(s, "inf"))
m = NAP.mode(s, "inf", "y")
chk("… et la nappe se relit porteurs X, désignation 10/12/100/150", m["porteurs"] == "x" and m["treillis"] == "10/12/100/150", m)
avant = copy.deepcopy(d)
NAP.ecrire(d, CH("inf", "porteurs"), "X")
chk("« porteurs » X quand ils le sont déjà (majuscule admise) : rien ne bouge", d == avant)
NAP.ecrire(d, CH("inf", "porteurs"), "y")
chk("« porteurs » Y : l'échange inverse", des(s, "inf") == ("12/10/150/100", "10/12/100/150") and NAP.mode(s, "inf", "y")["porteurs"] == "y")
chk("« porteurs » sur la nappe supérieure (treillis symétrique) : accepté, rien ne change",
    refus(NAP.ecrire, d, CH("sup", "porteurs"), "x") is None and des(d["sections"][0], "sup") == (TR.TREILLIS_DEFAUT, TR.TREILLIS_DEFAUT))
chk("« porteurs » Z : refusé", "X ou Y" in (refus(NAP.ecrire, d, CH("inf", "porteurs"), "z") or ""))

# treillis → barres garde Ø et espacement
d2 = EN.nouvelle_dalle()
s2 = d2["sections"][0]
poser(s2, "inf_x", d=16, esp=200)
poser(s2, "inf_y", d=8, esp=125)
NAP.ecrire(d2, CH("inf", "mode"), "barres")
chk("mode « barres » depuis un treillis : les deux bases deviennent des barres Ø/e, chacune avec SON Ø et SON espacement (Ø16 e 200 "
    "en X, Ø8 e 125 en Y) ; la désignation du treillis reste rangée dans la couche (comme l'écran Streamlit)",
    [(c["type"], c["d"], c["esp"], c["treillis"]) for c in (base(s2, "inf_x"), base(s2, "inf_y"))]
    == [("barres", 16, 200, TR.TREILLIS_DEFAUT), ("barres", 8, 125, TR.TREILLIS_DEFAUT)] and NAP.mode(s2, "inf")["mode"] == "barres",
    [base(s2, f) for f in ("inf_x", "inf_y")])
chk("… la nappe supérieure n'a pas bougé", NAP.mode(s2, "sup")["mode"] == "treillis" and EN.normaliser(d2) == d2)
avant = copy.deepcopy(d2)
NAP.ecrire(d2, CH("inf", "mode"), "barres")
chk("mode « barres » quand la nappe l'est déjà : rien ne bouge", d2 == avant)
# barres → treillis pose un treillis unique
poser(s2, "inf_y", type="n_barres", n=5)
NAP.ecrire(d2, CH("inf", "mode"), "treillis")
chk("mode « treillis » depuis des barres (X Ø/e, Y n barres) : les deux bases deviennent le treillis par défaut 10/10/100/100 ; Ø, "
    "espacement et nombre restent rangés",
    des(s2, "inf") == (TR.TREILLIS_DEFAUT, TR.TREILLIS_DEFAUT) and NAP.mode(s2, "inf")["mode"] == "treillis"
    and (base(s2, "inf_x")["d"], base(s2, "inf_x")["esp"], base(s2, "inf_y")["n"]) == (16, 200, 5), [base(s2, f) for f in ("inf_x", "inf_y")])
d3 = EN.nouvelle_dalle()
s3 = d3["sections"][0]
poser(s3, "sup_x", type="treillis", treillis="8/8/150/150")
poser(s3, "sup_y", type="barres", d=12, esp=150)
NAP.ecrire(d3, CH("sup", "mode"), "treillis")
chk("mode « treillis » depuis « treillis 8/8/150/150 en X + barres en Y » (incohérent) : le treillis existant est gardé, sa transposée "
    "va sur Y → treillis unique 8/8/150/150",
    des(s3, "sup") == ("8/8/150/150", "8/8/150/150") and NAP.mode(s3, "sup")["mode"] == "treillis", des(s3, "sup"))
poser(s3, "inf_x", treillis="10/12/100/150")
poser(s3, "inf_y", treillis="8/8/150/150")
NAP.ecrire(d3, CH("inf", "mode"), "treillis")
chk("mode « treillis » depuis deux treillis différents (X 10/12/100/150, Y 8/8/150/150 ; principale Y) : celui de la direction "
    "principale est gardé, sa transposée va sur l'autre",
    des(s3, "inf") == ("8/8/150/150", "8/8/150/150"), des(s3, "inf"))
# treillis + renforts
d4 = EN.nouvelle_dalle()
s4 = d4["sections"][0]
NAP.ecrire(d4, CH("inf", "mode"), "treillis_renforts")
chk("mode « treillis_renforts » depuis un treillis sans renfort : un renfort (3 Ø12, le défaut de « + Renfort ») s'ajoute sur la face "
    "des porteurs (la principale Y) ; le mode devient « treillis_renforts »",
    [len(s4["faces"][f]["couches"]) for f in ("inf_x", "inf_y")] == [1, 2] and s4["faces"]["inf_y"]["couches"][1] == EN.COUCHE_RENFORT
    and NAP.mode(s4, "inf")["mode"] == "treillis_renforts" and EN.normaliser(d4) == d4, s4["faces"]["inf_y"])
avant = copy.deepcopy(d4)
NAP.ecrire(d4, CH("inf", "mode"), "treillis_renforts")
chk("mode « treillis_renforts » quand il y a déjà un renfort : rien ne s'ajoute", d4 == avant)
msg = refus(NAP.ecrire, d4, CH("inf", "mode"), "treillis")
chk("mode « treillis » (seul) sur une nappe qui porte un renfort : REFUSÉ avec un message — le mode se déduit, et « Treillis soudé » "
    "seul retirerait le renfort ; on le retire par « Retirer ce renfort »",
    msg is not None and "renfort" in msg and "Retirer ce renfort" in msg and d4 == avant, msg)
d5 = EN.nouvelle_dalle()
s5 = d5["sections"][0]
poser(s5, "inf_x", type="barres")
poser(s5, "inf_y", type="barres")
s5["faces"]["inf_x"]["couches"].append(EN.nouvelle_couche(base=False))
NAP.ecrire(d5, CH("inf", "mode"), "treillis")
chk("mode « treillis » depuis des barres QUI ONT un renfort : les bases deviennent un treillis, le renfort reste → le mode déduit est "
    "« treillis_renforts » (rien n'est retiré en silence)",
    NAP.mode(s5, "inf")["mode"] == "treillis_renforts" and len(s5["faces"]["inf_x"]["couches"]) == 2)
# les refus
chk("une désignation illisible est refusée : « 10/10/abc/100 »", "Øl/Øt/el/et" in (refus(NAP.ecrire, d, CH("inf", "treillis"), "10/10/abc/100") or ""))
chk("… « 0/10/100/100 » (un zéro), « » (vide), 12 (un nombre)",
    all("Øl/Øt/el/et" in (refus(NAP.ecrire, d, CH("inf", "treillis"), v) or "") for v in ("0/10/100/100", "", 12, "10/10/100")))
chk("une désignation écrite avec des espaces est lue et canonisée (« 8 / 8 / 150 / 150 » → 8/8/150/150)",
    refus(NAP.ecrire, copy.deepcopy(d), CH("sup", "treillis"), " 8 / 8 / 150 / 150 ") is None)
chk("un côté inconnu est refusé (« sections/0/nappes/milieu/mode »)", "inf » ou « sup" in (refus(NAP.ecrire, d, CH("milieu", "mode"), "barres") or ""))
chk("un champ inconnu est refusé", "mode, treillis ou porteurs" in (refus(NAP.ecrire, d, CH("inf", "axe"), "3") or ""))
chk("un mode inconnu est refusé", "Mode de nappe inconnu" in (refus(NAP.ecrire, d, CH("inf", "mode"), "grille") or ""))
chk("une section introuvable est refusée (ValueError, jamais IndexError)", "introuvable" in (refus(NAP.ecrire, d, CH("inf", "mode", 9), "barres") or ""))
chk("un chemin qui n'est pas celui d'une nappe est refusé", "Chemin de nappe" in (refus(NAP.ecrire, d, ["sections", 0, "faces", "inf_x", "yG"], "3") or ""))
d6 = EN.nouvelle_dalle()
NAP.ecrire(d6, CH("inf", "mode"), "barres")
chk("« porteurs » sur une nappe de barres : refusé (« pas un treillis unique »)", "treillis unique" in (refus(NAP.ecrire, d6, CH("inf", "porteurs"), "x") or ""))
# rien d'autre ne bouge
d7 = EN.nouvelle_dalle()
s7 = d7["sections"][0]
s7["faces"]["inf_x"]["couches"].append(dict(EN.nouvelle_couche(base=False), dist=8.0, d=16))
base(s7, "inf_y")["dist"] = 5.0
s7["faces"]["inf_x"]["yG"] = 4.2
s7["faces"]["sup_y"]["yG"] = 3.9
poser(s7, "inf_x", d=20, esp=250, n=7)
attendu = copy.deepcopy(d7)
NAP.ecrire(d7, CH("inf", "treillis"), "8/8/150/150")
NAP.ecrire(d7, CH("inf", "mode"), "barres")
NAP.ecrire(d7, CH("inf", "mode"), "treillis")
NAP.ecrire(d7, CH("inf", "treillis"), "10/12/100/150")
NAP.ecrire(d7, CH("inf", "porteurs"), "x")
for f_, t_ in (("inf_x", "10/12/100/150"), ("inf_y", "12/10/150/100")):
    attendu["sections"][0]["faces"][f_]["couches"][0]["treillis"] = t_
chk("après cinq écritures de nappe (treillis, barres, treillis, treillis dissymétrique, porteurs) : seules les désignations et le type "
    "des deux bases ont changé — le renfort (Ø16 à 8 cm), l'axe imposé 5 cm, les y_G imposés 4,2 et 3,9 cm, Ø 20 / e 250 / n 7 rangés, "
    "la nappe supérieure, tout le reste est intact",
    d7 == attendu, [(k, d7["sections"][0]["faces"][k]) for k in EN.FACES if d7["sections"][0]["faces"][k] != attendu["sections"][0]["faces"][k]])
chk("… et la dalle écrite est dans le domaine du moteur", EN.normaliser(d7) == d7)
chk("ecrire lit un indice de section en texte (« 0 », comme un chemin brut de l'écran)",
    refus(NAP.ecrire, copy.deepcopy(d7), ["sections", "0", "nappes", "sup", "mode"], "barres") is None)

# ------------------------------------------------------------------ 3. le HTML des deux nappes
print("\n=== 3 · Le HTML")
d8 = EN.nouvelle_dalle()
s8 = d8["sections"][0]
NAP.ecrire(d8, CH("inf", "treillis"), "10/12/100/150")
NAP.ecrire(d8, CH("inf", "porteurs"), "x")
s8["faces"]["inf_y"]["couches"].append(EN.nouvelle_couche(base=False))
h = html_de(d8)
inf, sup = nappe_html(h, "inf"), nappe_html(h, "sup")
chk("les deux nappes, l'inférieure puis la supérieure, titrées « Nappe inférieure » / « Nappe supérieure »",
    h.index('data-nappe="inf"') < h.index('data-nappe="sup"') and "Nappe inférieure" in inf and "Nappe supérieure" in sup
    and h.count('class="et-nap"') == 2)
segs = re.findall(SEG.format("sections/0/nappes/inf/mode"), inf)
chk("les segments du mode de la nappe inférieure : Treillis soudé · Barres · Treillis + renforts, le mode déduit marqué (ici « Treillis + "
    "renforts » : un renfort en Y)",
    [(v, t) for _, v, _, t in segs] == [("treillis", "Treillis soudé"), ("barres", "Barres"), ("treillis_renforts", "Treillis + renforts")]
    and [v for on, v, _, _ in segs if on] == ["treillis_renforts"], segs)
chk("… la supérieure (treillis symétrique, sans renfort) marque « Treillis soudé »",
    [v for on, v, _, _ in re.findall(SEG.format("sections/0/nappes/sup/mode"), sup) if on] == ["treillis"])
sels = re.findall(r'<select data-chemin="sections/0/nappes/inf/treillis" data-sorte="choix">(.*?)</select>', inf)
chk("UN sélecteur de treillis par nappe (chemin nappes/inf/treillis), qui propose les dix treillis de la base PLUS la désignation "
    "courante quand elle n'y est pas (10/12/100/150, retenue)",
    len(sels) == 1 and re.findall(r'<option value="([^"]+)"', sels[0]) == TR.liste_choix() + ["10/12/100/150"]
    and '<option value="10/12/100/150" selected>' in sels[0], sels[:1])
chk("… la supérieure : le sélecteur propose les dix treillis, 10/10/100/100 retenu, sans ajout",
    re.findall(r'<option value="([^"]+)"( selected)?', nappe_html(h, "sup").split("nappes/sup/treillis")[1].split("</select>")[0])
    == [(x, " selected" if x == TR.TREILLIS_DEFAUT else "") for x in TR.liste_choix()])
orient = re.findall(SEG.format("sections/0/nappes/inf/porteurs"), inf)
chk("l'orientation des porteurs : deux segments X / Y (« Y · princ. » était coupé au petit portable : la principale est dite dans l'aide), les porteurs X allumés, actifs (treillis "
    "dissymétrique)",
    [(v, t, bool(off)) for _, v, off, t in orient] == [("x", "X", False), ("y", "Y", False)] and [v for on, v, _, _ in orient if on] == ["x"], orient)
orient_s = re.findall(SEG.format("sections/0/nappes/sup/porteurs"), sup)
chk("… la supérieure (treillis symétrique) : les segments sont GRISÉS (l'orientation est sans effet), la principale allumée",
    all(off for _, _, off, _ in orient_s) and [v for on, v, _, _ in orient_s if on] == ["y"] and "sans effet" in sup, orient_s)
# (13/11/2026, § 8.4) plus de ligne « X : … · Y : … » sous le réglage : chaque direction dit CE QU'ELLE REÇOIT sur la ligne
# de sa base — « porteurs · Ø10 e 100 », « répartition · Ø12 e 150 » —, une fois, sans répéter « Treillis » ni la désignation
def _base_lue(h_, f):
    p_ = re.search(r'<div class="et-piece" data-cible="s0\.' + f + r'-0">.*?<div class="et-det">', h_, re.S)
    return re.findall(r'<span class="et-lu">([^<]+)</span>', p_.group(0)) if p_ else None
chk("ce que chaque direction reçoit, sur la ligne de SA base : inférieure X « porteurs · Ø10 e 100 », Y « répartition · Ø12 e 150 » ; "
    "supérieure (symétrique) : Y porteurs, X répartition, Ø10 e 100 ; plus de ligne « X : … · Y : … »",
    _base_lue(inf, "inf_x") == ["porteurs", "Ø10 e 100"] and _base_lue(inf, "inf_y") == ["répartition", "Ø12 e 150"]
    and _base_lue(sup, "sup_y") == ["porteurs", "Ø10 e 100"] and _base_lue(sup, "sup_x") == ["répartition", "Ø10 e 100"]
    and "et-nap-xy" not in h and "X : Ø10 e 100" not in h, (_base_lue(inf, "inf_x"), _base_lue(inf, "inf_y")))
chk("en mode treillis, la base de chaque direction se LIT (ni menu « type » ni menu « treillis » sur la couche 0) mais son axe reste réglable, "
    "et le renfort garde tous ses champs",
    'data-chemin="sections/0/faces/inf_x/couches/0/type"' not in inf and 'data-chemin="sections/0/faces/inf_x/couches/0/treillis"' not in inf
    and 'data-chemin="sections/0/faces/inf_y/couches/0/type"' not in inf
    and 'data-chemin="sections/0/faces/inf_x/couches/0/dist"' in inf and 'data-chemin="sections/0/faces/inf_y/couches/0/dist"' in inf
    and all(f'data-chemin="sections/0/faces/inf_y/couches/1/{k}"' in inf for k in ("type", "n", "d", "dist", "-")))
chk("… la base lue ne répète ni « Treillis » ni la désignation (la nappe les porte) ; son détail dit ses fils (« fils porteurs du "
    "treillis : Ø10 tous les 100 mm », « fils de répartition du treillis : Ø12 tous les 150 mm ») ; plus de « Streamlit »",
    '<span class="et-lu">Treillis</span>' not in inf and '<span class="et-lu">10/12/100/150</span>' not in inf
    and "fils porteurs du treillis : Ø10 tous les 100 mm" in inf and "fils de répartition du treillis : Ø12 tous les 150 mm" in inf
    and "Streamlit" not in h)
chk("le détail de chaque direction, la principale (Y) d'abord : « Inférieure Y » puis « Inférieure X », avec « + Renfort » ; y_G dès DEUX "
    "couches seulement (Y : base et renfort ; X, une couche : son axe suffit — 13/11/2026)",
    inf.index("Inférieure Y") < inf.index("Inférieure X") and all(f'data-chemin="sections/0/faces/{f}/couches/+"' in inf for f in ("inf_x", "inf_y"))
    and 'data-chemin="sections/0/faces/inf_y/yG"' in inf and 'data-chemin="sections/0/faces/inf_x/yG"' not in inf)
cibles = re.findall(r'data-cible="([^"]+)"', h)
chk("chaque couche de chaque face a sa pièce data-cible, une seule fois, sous la clé du dessin (s0.<face>-<j>)",
    len(cibles) == len(set(cibles)) and set(cibles) == {f"s0.{f}-{j}" for f in EN.FACES for j in range(len(s8["faces"][f]["couches"]))}, cibles)
chk("aucun avertissement, aucun bouton « En faire un treillis » sur des nappes cohérentes", "et-avert" not in h and "En faire un treillis" not in h)
# mode barres : l'écran d'aujourd'hui
d9 = EN.nouvelle_dalle()
NAP.ecrire(d9, CH("inf", "mode"), "barres")
r9 = MO.verifier(d9)
rs9 = r9["sections"][0]
v9 = {v["cle"]: v for v in rs9["verifications"]}
h9 = html_de(d9)
inf9 = nappe_html(h9, "inf")
aujourdhui = "".join(
    PC.nappe(EN, 0, f, d9["sections"][0]["faces"][f], rs9["faces"][f], v9[f], "y",
             "".join(PC.couche(EN, 0, f, j, c, rs9["faces"][f]["couches"][j], aide_axe="Aide axe") for j, c in enumerate(d9["sections"][0]["faces"][f]["couches"])),
             "Aide renfort")
    for f in ("inf_y", "inf_x"))
chk("en mode BARRES, le détail des deux directions est EXACTEMENT l'écran d'aujourd'hui (plaque_compact.nappe + couche, la principale d'abord)",
    aujourdhui in inf9 and inf9.endswith(aujourdhui + "</div>"), inf9[-300:])
chk("… « Barres » marqué, ni sélecteur de treillis, ni orientation, ni ligne X/Y",
    [v for on, v, _, _ in re.findall(SEG.format("sections/0/nappes/inf/mode"), inf9) if on] == ["barres"]
    and "nappes/inf/treillis" not in inf9 and "nappes/inf/porteurs" not in inf9 and "et-nap-xy" not in inf9)
# mode incohérent
d10 = EN.nouvelle_dalle()
poser(d10["sections"][0], "sup_x", treillis="8/8/150/150")
h10 = html_de(d10)
sup10 = nappe_html(h10, "sup")
av = re.search(r'<div class="et-avert et-nap-avert"><span>([^<]+)</span>(.*?)</div>', sup10)


def controle_incoherent(frag):
    """Le contrôle du mode incohérent, réutilisé par la mutation."""
    a = re.search(r'<div class="et-avert et-nap-avert"><span>([^<]+)</span>(.*?)</div>', frag)
    return (a is not None and html.unescape(a.group(1)) == NAP.AVERTISSEMENT + "."
            and 'data-chemin="sections/0/nappes/sup/mode" data-valeur="treillis">En faire un treillis</button>' in a.group(2))


chk("deux treillis différents (8/8/150/150 en X, 10/10/100/100 en Y) : l'avertissement « Deux armatures de base différentes en X et en Y — "
    "ce n'est pas un treillis unique » et le bouton « En faire un treillis » (chemin nappes/sup/mode, valeur treillis)",
    controle_incoherent(sup10), av and av.groups())
chk("… aucun segment du mode n'est marqué, et le détail par direction est celui d'aujourd'hui (les deux bases se règlent)",
    not [v for on, v, _, _ in re.findall(SEG.format("sections/0/nappes/sup/mode"), sup10) if on]
    and all(f'data-chemin="sections/0/faces/{f}/couches/0/{k}"' in sup10 for f in ("sup_x", "sup_y") for k in ("type", "treillis")))
chk("… la nappe inférieure, cohérente, n'a pas d'avertissement", "et-avert" not in nappe_html(h10, "inf"))
# le verrou
hv = html_de(d8, verrou=True)
actifs = re.findall(r'<(?:select|input|button)(?![^>]*\bdisabled\b)[^>]*data-chemin[^>]*>', hv)
chk("verrouillée : tout se lit — aucun champ ni bouton actif qui porte un chemin, ni « + Renfort », ni « Retirer », ni sélecteur ; les "
    "pièces gardent leur data-cible (le dessin les sélectionne encore)",
    not actifs and "et-add" not in hv and "et-ret" not in hv and "et-prop" not in hv and "En faire un treillis" not in hv
    and set(re.findall(r'data-cible="([^"]+)"', hv)) == set(cibles) and "Treillis + renforts" in hv, actifs[:3])
# aide_axe et aide_renfort : textes ou fonctions (le contrat posé par le lot FUS2, pour la prédalle)
hf = html_de(d8, aide_axe=lambda f, j, c, g: {"aide_axe": "AXE·" + f, "details": f"<i>POSE {f}-{j}</i>", "dit_auto": False},
             aide_renfort=lambda f: "RENFORT·" + f)
chk("aide_axe et aide_renfort peuvent être des FONCTIONS (la prédalle y pose la pose de chaque couche) : chaque couche porte son détail, "
    "chaque direction son aide",
    all(f"<i>POSE {f}-{j}</i>" in hf for f in EN.FACES for j in range(len(s8["faces"][f]["couches"])))
    and all(f'title="RENFORT·{f}"' in hf for f in EN.FACES) and all(f'title="AXE·{f}"' in hf for f in EN.FACES), hf[:200])
chk("les libellés et les textes sont échappés (C.esc) : un « < » dans une aide ne devient pas une balise",
    "<b>x</b>" not in html_de(d8, aide_axe="<b>x</b>") and "&lt;b&gt;x&lt;/b&gt;" in html_de(d8, aide_axe="<b>x</b>"))

# ------------------------------------------------------------------ 4. mutations
print("\n=== 4 · Mutations")
with Abime(NAP, transposee=lambda des_: des_):
    dm = EN.nouvelle_dalle()
    NAP.ecrire(dm, CH("inf", "treillis"), "10/12/100/150")
    NAP.ecrire(dm, CH("inf", "porteurs"), "x")
    chk("mutation « transposée = identité » : les deux faces reçoivent 10/12/100/150 — Y compterait Ø10 tous les 100 au lieu de Ø12 "
        "tous les 150 ; le contrôle de la transposée rougit",
        des(dm["sections"][0], "inf") == ("10/12/100/150", "10/12/100/150") and not controle_transposee(dm))
with Abime(NAP, _porteurs=lambda bx, by, p: p):
    dm = EN.nouvelle_dalle()
    NAP.ecrire(dm, CH("inf", "treillis"), "10/12/100/150")
    NAP.ecrire(dm, CH("inf", "porteurs"), "x")
    mm = NAP.mode(dm["sections"][0], "inf", "y")
    chk("mutation « porteurs = toujours la principale » : après l'échange, la nappe se relit encore porteurs Y, désignation 12/10/150/100 — "
        "l'écran mentirait sur l'orientation ; le contrôle de l'orientation rougit",
        mm["porteurs"] == "y" and mm["treillis"] == "12/10/150/100" and des(dm["sections"][0], "inf") == ("10/12/100/150", "12/10/150/100"))
with Abime(NAP, _treillis_unique=lambda bx, by: bx["type"] == "treillis" and by["type"] == "treillis"):
    hm = html_de(d10)
    chk("mutation « deux treillis = un treillis unique » : 8/8/150/150 en X et 10/10/100/100 en Y passent pour un treillis — plus "
        "d'avertissement, un sélecteur apparaît ; le contrôle du mode incohérent rougit",
        not controle_incoherent(nappe_html(hm, "sup")) and "nappes/sup/treillis" in hm and NAP.mode(d10["sections"][0], "sup")["mode"] == "treillis")

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)
