# -*- coding: utf-8 -*-
"""Le moteur des DONNÉES BÉTON (beton/donnees) : chaque grandeur qu'il ajoute est confrontée à sa source publiée, et
chaque donnée qu'il lit est celle du fichier du dépôt.

1. Les fonctions du tableau 3.1 de l'EN 1992-1-1 (beton/donnees/ec2.py), classe par classe de C12/15 à C90/105,
   arrondies comme le tableau, redonnent le tableau publié (transcrit ci-dessous, TABLEAU_3_1) ; le seul écart est
   celui du tableau imprimé lui-même (f_ctk,0,05 de C60/75), déclaré et figé.
2. Les classes du bureau (beton_classes.json) ont le f_ck et le f_ck,cube du tableau 3.1 ; le f_ctm du moteur des
   poutres (méthode du bureau) est celui de ec2.fctm pour chacune (deux écritures d'une même formule, liées ici).
3. consultation.classes() : chaque valeur est celle du fichier ou de la fonction nommée ; consultation.aciers() :
   f_yd = f_yk / γ_s (1,5 du bureau, 1,15 de l'EN 1992-1-1), E_s = 200 GPa, ε_yd = f_yd / E_s.
4. Le fichier des ancrages et recouvrements est VIDE (aucune valeur inventée) et se relit ; valider() refuse chaque
   forme fautive avec un message en français ; un tableau d'exemple (de forme, pas une donnée) passe.
5. Mutations : abîmer une formule, un coefficient ou le fichier fait rougir le contrôle qui le garde.

    python3 tests/test_donnees_beton_moteur.py
"""
import copy
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.donnees import ancrages as AN  # noqa: E402
from beton.donnees import consultation as K  # noqa: E402
from beton.donnees import ec2 as EC  # noqa: E402
from beton.poutre import entree as PE  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


# EN 1992-1-1:2004, tableau 3.1 — les lignes f_ck, f_ck,cube, f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95 (MPa) et E_cm (GPa),
# telles qu'imprimées, de C12/15 à C90/105 (transcription : la source à laquelle les fonctions sont confrontées)
TABLEAU_3_1 = """\
fck;12;16;20;25;30;35;40;45;50;55;60;70;80;90
fck_cube;15;20;25;30;37;45;50;55;60;67;75;85;95;105
fcm;20;24;28;33;38;43;48;53;58;63;68;78;88;98
fctm;1,6;1,9;2,2;2,6;2,9;3,2;3,5;3,8;4,1;4,2;4,4;4,6;4,8;5,0
fctk_005;1,1;1,3;1,5;1,8;2,0;2,2;2,5;2,7;2,9;3,0;3,1;3,2;3,4;3,5
fctk_095;2,0;2,5;2,9;3,3;3,8;4,2;4,6;4,9;5,3;5,5;5,7;6,0;6,3;6,6
Ecm;27;29;30;31;33;34;35;36;37;38;39;41;42;44"""
# l'écart du tableau IMPRIMÉ : f_ctk,0,05 de C60/75 vaut 0,7 × 4,355 = 3,05 MPa (3,0 arrondi), imprimé 3,1 (0,7 × 4,4,
# le f_ctm déjà arrondi). Hors des classes du bureau (C20/25 à C40/50) ; déclaré pour qu'aucun autre ne passe.
ECARTS_IMPRIMES = {("C60/75", "fctk_005")}


def tableau_publie() -> dict:
    lignes = {li.split(";")[0]: [float(x.replace(",", ".")) for x in li.split(";")[1:]] for li in TABLEAU_3_1.splitlines()}
    classes = [f"C{int(a)}/{int(b)}" for a, b in zip(lignes["fck"], lignes["fck_cube"])]
    return {c: {k: v[i] for k, v in lignes.items()} for i, c in enumerate(classes)}


PUBLIE = tableau_publie()
DECIMALES = {"fcm": 0, "fctm": 1, "fctk_005": 1, "fctk_095": 1, "Ecm": 0}


def arrondi(x: float, nd: int) -> float:
    """L'arrondi d'un tableau imprimé : au plus proche, la moitié vers le haut."""
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def ecarts_3_1(ec) -> set:
    """Les cases du tableau 3.1 que les fonctions de ``ec`` ne redonnent pas, arrondies comme le tableau."""
    out = set()
    for c, v in PUBLIE.items():
        for k, nd in DECIMALES.items():
            if arrondi(getattr(ec, k)(v["fck"]), nd) != v[k]:
                out.add((c, k))
    return out


# ---------------------------------------------------------------------------------------------------- 1. tableau 3.1
print("=== 1 · Les fonctions de l'EN 1992-1-1 face au tableau 3.1 publié")
chk(f"le tableau transcrit a 14 classes, de C12/15 à C90/105 ({len(PUBLIE)})", list(PUBLIE)[0] == "C12/15" and list(PUBLIE)[-1] == "C90/105" and len(PUBLIE) == 14)
e = ecarts_3_1(EC)
chk(f"f_cm, f_ctm, f_ctk,0,05, f_ctk,0,95, E_cm : les {len(PUBLIE) * len(DECIMALES)} cases sont redonnées, sauf l'écart imprimé déclaré",
    e == ECARTS_IMPRIMES, sorted(e))
chk("l'écart déclaré est bien un écart d'arrondi du tableau imprimé : 0,7 × f_ctm(C60/75) = 3,05 et 0,7 × 4,4 = 3,08",
    abs(EC.fctk_005(60) - 3.048) < 0.001 and arrondi(0.7 * PUBLIE["C60/75"]["fctm"], 1) == PUBLIE["C60/75"]["fctk_005"])
chk("f_ctm change de formule au-delà de C50/60 (2,12·ln(1 + f_cm/10))", abs(EC.fctm(55) - 2.12 * math.log(1 + 63 / 10)) < 1e-12
    and abs(EC.fctm(50) - 0.30 * 50 ** (2 / 3)) < 1e-12)
chk("les constantes de l'acier : E_s = 200 GPa (§3.2.7(4)), γ_s = 1,15 (tableau 2.1N)", EC.E_S == 200.0 and EC.GAMMA_S == 1.15
    and "3.2.7(4)" in EC.SOURCE_E_S and "2.1N" in EC.SOURCE_GAMMA_S)
chk("ε_yd = f_yd / E_s : 500 / 1,15 → 2,174 ‰", abs(EC.eps_yd(500 / 1.15) - 2.1739130434782608) < 1e-12)

# ---------------------------------------------------------------------------------------------------- 2. classes du bureau
print("\n=== 2 · Les classes du bureau (beton_classes.json)")
FICHIER = json.load(open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8"))
chk(f"les classes lues sont celles du fichier, dans son ordre ({', '.join(FICHIER)})", K.noms_classes() == list(FICHIER))
faux = [c for c, d in FICHIER.items() if c not in PUBLIE or (d["fck"], d["fck_cube"]) != (PUBLIE[c]["fck"], PUBLIE[c]["fck_cube"])]
chk("chaque classe du bureau a le f_ck et le f_ck,cube du tableau 3.1", not faux, faux)
faux = [x["nom"] for x in K.classes() if x["fctm"] != x["fctm_bureau"]]
chk("le f_ctm de la méthode du bureau (moteur des poutres, 0,30·f_ck^(2/3)) est celui de ec2.fctm, au bit près", not faux, faux)

# ---------------------------------------------------------------------------------------------------- 3. consultation
print("\n=== 3 · Les rubriques lisent leur source")
mauvais = []
for x in K.classes():
    d = FICHIER[x["nom"]]
    attendu = {"fck": d["fck"], "fck_cube": d["fck_cube"], "alpha_b": d["alpha_b"], "mu_a500": d["mu_a500"], "mu_a400": d["mu_a400"],
               "tau_lim_fichier": d["tau_lim"], "fcm": d["fck"] + 8, "fctm": EC.fctm(d["fck"]), "fctk_005": EC.fctk_005(d["fck"]),
               "fctk_095": EC.fctk_095(d["fck"]), "Ecm": EC.Ecm(d["fck"]),
               "tau_1": 0.016 * d["fck_cube"] / 1.05, "tau_2": 0.032 * d["fck_cube"] / 1.05, "tau_4": 0.064 * d["fck_cube"] / 1.05}
    mauvais += [(x["nom"], k, x[k], v) for k, v in attendu.items() if abs(x[k] - v) > 1e-12]
chk("classes() : f_ck, f_ck,cube, α_b, μ, « tau_lim » du fichier ; tableau 3.1 ; τ_adm = k·f_ck,cube/1,05 (k = 0,016 · 0,032 · 0,064)",
    not mauvais, mauvais[:4])
chk("chaque colonne annoncée existe dans chaque ligne", all(k in x for x in K.classes() for k, *_ in K.COLONNES_CLASSES))
c0 = K.classes()
c0[0]["fck"] = -1
chk("classes() rend une copie (la modifier ne touche à rien)", K.classes()[0]["fck"] == FICHIER[K.noms_classes()[0]]["fck"])
ac = {a["nom"]: a for a in K.aciers()}
chk("aciers() : les nuances des modules béton, B400 et B500", list(ac) == [f"B{n}" for n in PE.NUANCES] == ["B400", "B500"])
mauvais = [(n, k) for n, a in ac.items() for k, v in (("fyd_bureau", a["fyk"] / 1.5), ("fyd_ec2", a["fyk"] / 1.15), ("Es", 200.0),
                                                     ("eps_yd", a["fyk"] / 1.15 / 200000 * 1000), ("masse_volumique", 7860))
           if abs(a[k] - v) > 1e-9]
chk("aciers() : f_yd = f_yk/1,5 (bureau) et f_yk/1,15 (EN 1992-1-1), E_s = 200 GPa, ε_yd = f_yd/E_s, ρ = 7 860 kg/m³", not mauvais, mauvais)
chk("le γ_s du bureau est le défaut des modules béton (beton/poutre/entree : 1,5)", K.GAMMA_S_BUREAU == PE.DEFAUTS_POUTRE["gamma_s"] == 1.5)

# ---------------------------------------------------------------------------------------------------- 4. ancrages
print("\n=== 4 · Ancrages et recouvrements : le fichier à remplir")
brut = json.load(open(AN.CHEMIN, encoding="utf-8"))
d = AN.lire()
chk("le fichier est VIDE : aucun tableau, aucune valeur inventée", brut["tableaux"] == [] and d["tableaux"] == [])
chk("il dit ce qu'il attend : « Tableaux du bureau à venir », et où lire le format", d["etat"] == "Tableaux du bureau à venir"
    and "docs/beton/DONNEES_BETON.md" in d["aide"])
EXEMPLE = {"version": 1, "titre": "Exemple de forme", "etat": "—", "aide": "",
           "tableaux": [{"cle": "exemple_forme", "titre": "Exemple de forme (pas une donnée)", "grandeur": "l_x", "unite": "Ø",
                         "source": "test de forme", "conditions": ["condition A"],
                         "colonnes": {"titre": "Ø", "unite": "mm", "valeurs": [1, 2, 3]},
                         "lignes": [{"libelle": "ligne 1", "valeurs": [111, None, 333]}], "notes": ["note"]}]}
v = AN.valider(EXEMPLE)
chk("un tableau d'exemple (de forme) passe, une case vide (null) comprise", v["tableaux"][0]["lignes"][0]["valeurs"] == [111, None, 333]
    and AN.tableau(v, "exemple_forme")["titre"] == "Exemple de forme (pas une donnée)" and AN.tableau(v, "autre") is None)


def refus(abime, mot):
    try:
        AN.valider(abime)
    except ValueError as e:
        return mot in str(e), str(e)
    return False, "accepté"


def variante(**k):
    x = copy.deepcopy(EXEMPLE)
    t = x["tableaux"][0]
    for cle, val in k.items():
        if cle == "valeurs":
            t["lignes"][0]["valeurs"] = val
        elif val is None:
            t.pop(cle, None)
        else:
            t[cle] = val
    return x


for nom, abime, mot in (
        ("une ligne trop courte", variante(valeurs=[1, 2]), "3 valeur(s) attendue(s)"),
        ("une valeur qui n'est pas un nombre", variante(valeurs=[1, "40Ø", 3]), "un nombre"),
        ("une valeur infinie", variante(valeurs=[1, float("inf"), 3]), "un nombre"),
        ("une unité inconnue", variante(unite="pouces"), "n'est pas permise"),
        ("une source manquante", variante(source=None), "source"),
        ("une clé inconnue", variante(couleur="rouge"), "clé(s) inconnue(s)"),
        ("une clé de tableau illisible", variante(cle="a b"), "lettres, chiffres"),
        ("deux tableaux de même clé", {**EXEMPLE, "tableaux": EXEMPLE["tableaux"] * 2}, "même clé"),
        # relecture adverse : le sélecteur de l'écran désigne une ligne par son LIBELLÉ — deux lignes de même libellé, la
        # seconde ne se choisissait jamais (le sélecteur ouvrait la première)
        ("deux lignes de même libellé", {**EXEMPLE, "tableaux": [{**EXEMPLE["tableaux"][0],
                                                                  "lignes": EXEMPLE["tableaux"][0]["lignes"] * 2}]}, "même libellé"),
        ("une version inconnue", {**EXEMPLE, "version": 2}, "version"),
        ("« tableaux » qui n'est pas une liste", {**EXEMPLE, "tableaux": {}}, "doit être une liste")):
    ok, msg = refus(abime, mot)
    chk(f"valider() refuse {nom}, avec un message en français", ok, msg)

# ---------------------------------------------------------------------------------------------------- 5. mutations
print("\n=== 5 · Mutations : chaque contrôle rougit quand on abîme ce qu'il garde")


class Abime:
    """Le module ec2 avec UNE fonction remplacée."""

    def __init__(self, **fonctions):
        self.__dict__.update({k: getattr(EC, k) for k in DECIMALES}, **fonctions)


for nom, ab in (("f_ctm avec 0,31 au lieu de 0,30", Abime(fctm=lambda f: 0.31 * f ** (2 / 3) if f <= 50 else EC.fctm(f))),
                ("f_cm = f_ck + 7", Abime(fcm=lambda f: f + 7)),
                ("f_ctk,0,05 = 0,75·f_ctm", Abime(fctk_005=lambda f: 0.75 * EC.fctm(f))),
                ("E_cm avec l'exposant 0,31", Abime(Ecm=lambda f: 22 * ((f + 8) / 10) ** 0.31)),
                ("f_ctm sans sa seconde formule (au-delà de C50/60)", Abime(fctm=lambda f: 0.30 * f ** (2 / 3)))):
    chk(f"mutation « {nom} » : la confrontation au tableau 3.1 rougit", ecarts_3_1(ab) - ECARTS_IMPRIMES, "aucun écart")
chk("mutation « l'écart imprimé non déclaré » : la confrontation rougit", ecarts_3_1(EC) != set())
# le fichier des classes abîmé : classes() suit le fichier (la consultation n'a pas de copie à elle)
lu = PE._classes_lues()
garde = copy.deepcopy(lu)
try:
    lu["C30/37"]["alpha_b"] = 99.0
    K._classes.cache_clear()
    chk("mutation « α_b de C30/37 changé dans le fichier lu » : classes() le suit (aucune copie)",
        next(x for x in K.classes() if x["nom"] == "C30/37")["alpha_b"] == 99.0)
finally:
    lu.clear()
    lu.update(garde)
    K._classes.cache_clear()
chk("… et revient au fichier une fois rétabli", next(x for x in K.classes() if x["nom"] == "C30/37")["alpha_b"] == FICHIER["C30/37"]["alpha_b"])
garde_g = EC.GAMMA_S
try:
    EC.GAMMA_S = 1.2
    K._aciers.cache_clear()
    chk("mutation « γ_s = 1,2 » : f_yd de l'EN 1992-1-1 n'est plus f_yk/1,15 (le contrôle des aciers rougirait)",
        abs(next(a for a in K.aciers() if a["nuance"] == 500)["fyd_ec2"] - 500 / 1.15) > 1)
finally:
    EC.GAMMA_S = garde_g
    K._aciers.cache_clear()

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
