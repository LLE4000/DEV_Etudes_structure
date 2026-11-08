# -*- coding: utf-8 -*-
"""Le FERRAILLAGE DÉTAILLÉ et les DISPOSITIONS CONSTRUCTIVES du moteur poutre (beton/poutre/geometrie.py,
dispositions.py ; site, 11/2026), vérifiés à la main.

- sans aucune clé du détail, la poutre normalisée et son calcul sont ceux d'avant (la parité Streamlit est intacte) ;
- les positions imposées (« pos ») placent les lits ; d et y_G les suivent (Σ A·e / Σ A) ;
- les abscisses : uniforme entre les montants de l'étrier, « aligne » sur le lit 1, « entraxe » centré ;
- s_min = max(Ø ; d_g + 5 ; 20 mm) ; deux lits Ø16 au jeu du bureau (1 cm) : 14 mm libres → « à vérifier » ;
- des barres qui se chevauchent, ou qui sortent de l'étrier, sont « non conformes » ;
- les latérales automatiques suivent la règle de la note PDF ; imposées, elles la remplacent ;
- l'emprise d'un cadre partiel, d'une épingle ; un cadre intérieur sans barre haute dans ses angles est signalé ;
- A_st compte les branches réelles (cadre = 2, épingle = 1) ;
- les états nommés : « sollicitee » faux sans effort ; τ et étriers « non vérifiées » sans V ;
- le sélecteur marque « possible » et « serree » sans rien retirer (l'optimal du moteur est celui de Streamlit).

    python3 tests/test_poutre_dispositions.py
"""
import copy
import json
import math
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from beton.poutre import dispositions as DI  # noqa: E402
from beton.poutre import entree as EN  # noqa: E402
from beton.poutre import geometrie as G  # noqa: E402
from beton.poutre import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:240]}"))


def proche(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def poutre(**k):
    p = EN.nouvelle_poutre()
    p.update({x: v for x, v in k.items() if x != "s"})
    p["sections"][0].update(k.get("s", {}))
    return EN.normaliser(p)


# 1. sans le détail : l'état et le calcul d'avant
p0 = poutre(s={"M_inf": 80.0, "V": 40.0})
chk("une poutre sans clé du détail se normalise sans en ajouter aucune",
    set(p0) <= set(EN.DEFAUTS_POUTRE) | {"sections"} and all(set(l_) == {"n", "d"} for f in ("inf", "sup") for l_ in p0["sections"][0][f])
    and all(set(e) == {"type", "d"} for e in p0["sections"][0]["etriers"]) and "laterales" not in p0["sections"][0], p0)
p1 = EN.normaliser(dict(p0, dg=20.0, peau_d=10, peau_smax=30.0))
chk("des clés du détail à leur valeur par défaut disparaissent (la poutre d'avant, à l'octet)", json.dumps(p1, sort_keys=True) == json.dumps(p0, sort_keys=True))

# 2. les positions imposées
p = poutre(s={"M_inf": 120.0, "inf": [{"n": 3, "d": 20}, {"n": 2, "d": 16, "pos": 4.0}]})
d = MO.verifier(p)["sections"][0]["geometrie"]["inf"]["lits"]
e1 = 3.0 + G.arrondi_demi_cm(1.0 + 1.0)            # enrobage + arr½(Ø étrier 10/10 + Ø20/20) = 3 + 2 = 5 cm
chk("lit 1 à la position du module (5 cm), lit 2 à 4 cm d'entraxe vertical (9 cm)", proche(d[0]["distance"], e1) and proche(d[1]["distance"], e1 + 4.0),
    [x["distance"] for x in d])
A1, A2 = 3 * math.pi * 100, 2 * math.pi * 64
g = MO.verifier(p)["sections"][0]["grandeurs"]
chk("y_G = Σ A·e / Σ A et d = h − y_G suivent les positions", proche(g["yG_inf"], (A1 * 5 + A2 * 9) / (A1 + A2)) and proche(g["d_inf"], 40 - g["yG_inf"]),
    (g["yG_inf"], g["d_inf"]))
p = poutre(s={"inf": [{"n": 2, "d": 16, "pos": 6.5}]})
chk("une position imposée du lit 1 est la distance parement → axe", proche(MO.verifier(p)["sections"][0]["geometrie"]["inf"]["lits"][0]["distance"], 6.5))

# 3. les abscisses
p = poutre(b=30.0, s={"inf": [{"n": 4, "d": 20}, {"n": 2, "d": 16, "rep": "aligne"}, {"n": 3, "d": 12, "rep": "entraxe", "sh": 7.0}]})
xs = [l_["x"] for l_ in MO.verifier(p)["sections"][0]["geometrie"]["inf"]["lits"]]
x0 = 3.0 + 1.0 + 1.0                                  # enrobage + Ø étrier + Ø/2 (cm)
chk("uniforme : la première et la dernière barre au contact des montants, réparties", proche(xs[0][0], x0) and proche(xs[0][-1], 30 - x0)
    and proche(xs[0][1] - xs[0][0], (30 - 2 * x0) / 3), xs[0])
chk("« aligne » : deux barres sur les angles du lit 1", xs[1] == [xs[0][0], xs[0][-1]], xs[1])
chk("« entraxe » : trois barres centrées à 7 cm", proche(xs[2][1], 15.0) and proche(xs[2][2] - xs[2][1], 7.0) and proche(xs[2][1] - xs[2][0], 7.0), xs[2])

# 4. l'espacement minimal et les états
chk("s_min = max(Ø ; d_g + 5 ; 20) : Ø16, d_g 20 → 25 mm ; Ø32 → 32 mm ; d_g 10 → 20 mm",
    DI.s_min(16, 16, 20) == 25 and DI.s_min(32, 16, 20) == 32 and DI.s_min(12, 12, 10) == 20)
p = poutre(s={"inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]})
it = next(x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
chk("deux lits Ø16 au jeu du bureau : 3 cm d'entraxe, 14 mm libres < 25 mm → « à vérifier »",
    proche(it["libre"], 14.0) and it["mini"] == 25 and it["etat"] == DI.WARN, it)
p = poutre(jeu_entre_lits=2.1, s={"inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]})
it = next(x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
chk("… avec un jeu de 2,1 cm : 25 mm libres → conforme", proche(it["libre"], 25.0) and it["etat"] == DI.OK, it)
p = poutre(s={"inf": [{"n": 8, "d": 32}]})
dis = MO.verifier(p)["sections"][0]["dispositions"]
chk("8Ø32 dans 20 cm : les barres se chevauchent → « non conforme »", dis["etat"] == DI.NOK
    and any(x["cle"] == "lit" and x["etat"] == DI.NOK and x["libre"] < 0 for x in dis["items"]), [x for x in dis["items"] if x["etat"] != "ok"])
p = poutre(s={"inf": [{"n": 3, "d": 16, "rep": "entraxe", "sh": 9.0}]})
dis = MO.verifier(p)["sections"][0]["dispositions"]
chk("un entraxe qui sort les barres de l'étrier → « non conforme » (barres hors de l'étrier)",
    any(x["cle"] == "cadre" and x["etat"] == DI.NOK for x in dis["items"]), [x["texte"] for x in dis["items"] if x["etat"] != "ok"])
p = poutre(enrobage=1.5, s={"inf": [{"n": 2, "d": 32}], "etriers": [{"type": "etrier", "d": 6}]})
it = next(x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "enrobage")
chk("enrobage 1,5 cm + Ø6 = 21 mm < Ø32 → « à vérifier » (adhérence)", proche(it["libre"], 21.0) and it["etat"] == DI.WARN, it)

# 5. les latérales
p = poutre(h=90.0)
lat = MO.verifier(p)["sections"][0]["geometrie"]["laterales"]
e_inf = MO.verifier(p)["sections"][0]["geometrie"]["inf"]["lits"][0]["distance"]
ecart = 90 - 2 * e_inf
chk("automatiques : n = ⌈écart / 30⌉ − 1 par face, réparties entre les lits 1", lat["auto"] and lat["n"] == math.ceil(ecart / 30) - 1
    and proche(lat["ys"][0], e_inf + ecart / (lat["n"] + 1)), lat)
chk("une poutre de 40 cm n'en a pas (écart ≤ 30 cm)", MO.verifier(poutre())["sections"][0]["geometrie"]["laterales"]["n"] == 0)
p = poutre(s={"laterales": {"n": 2, "d": 12}})
lat = MO.verifier(p)["sections"][0]["geometrie"]["laterales"]
chk("imposées : 2 Ø12 par face, même sous 30 cm d'écart", not lat["auto"] and lat["n"] == 2 and lat["d"] == 12, lat)

# 6. les cadres
p = poutre(b=40.0, s={"inf": [{"n": 5, "d": 20}], "sup": [{"n": 2, "d": 12}],
                      "etriers": [{"type": "etrier", "d": 8}, {"type": "etrier", "d": 8, "de": 2, "a": 4}, {"type": "epingle", "d": 8}]})
r = MO.verifier(p)["sections"][0]
cad = r["geometrie"]["etriers"]["cadres"]
xs1 = r["geometrie"]["inf"]["lits"][0]["x"]
chk("cadre complet au nu de l'enrobage ; cadre partiel autour des barres 2 à 4 ; épingle sur la barre du milieu",
    cad[0]["complet"] and proche(cad[0]["x0"], 3.0) and not cad[1]["complet"] and proche(cad[1]["x0"], xs1[1] - 1.0 - 0.8)
    and proche(cad[1]["x1"], xs1[3] + 1.0 + 0.8) and cad[2]["de"] == 3 and cad[2]["brins"] == 1, cad)
chk("A_st compte les branches réelles : 2 + 2 + 1 brins", proche(r["grandeurs"]["Ast"], 5 * math.pi * 16) and r["geometrie"]["etriers"]["brins"] == 5)
chk("le cadre intérieur sous deux barres hautes : angles supérieurs sans barre → « à vérifier »",
    any(x["cle"] == "angles_cadre" and "C2" in x["texte"] and x["etat"] == DI.WARN for x in r["dispositions"]["items"]))
p = poutre(b=40.0, s={"inf": [{"n": 5, "d": 20}], "sup": [{"n": 5, "d": 12}], "etriers": [{"type": "etrier", "d": 8}, {"type": "etrier", "d": 8, "de": 2, "a": 4}]})
chk("… cinq barres hautes alignées sur le bas : chaque angle tient sa barre", not any(x["cle"] == "angles_cadre" for x in MO.verifier(p)["sections"][0]["dispositions"]["items"]))
p = poutre(s={"etriers": [{"type": "etrier", "d": 8, "de": 1, "a": 9}]})
chk("une barre de cadre au-delà du lit 1 est signalée", any(x["cle"] == "position_cadre" for x in MO.verifier(p)["sections"][0]["dispositions"]["items"]))

# 7. les états nommés
r = MO.verifier(poutre())
chk("sans aucun effort : « sollicitee » faux (section et poutre) ; τ et étriers non vérifiées, et pourquoi",
    r["sollicitee"] is False and r["sections"][0]["sollicitee"] is False
    and [x["cle"] for x in r["sections"][0]["non_verifiees"]] == ["tau", "pas"] and "V = 0" in r["sections"][0]["non_verifiees"][0]["raison"])
r = MO.verifier(poutre(s={"V": 30.0}))
chk("avec V : rien de non vérifié", r["sections"][0]["sollicitee"] and r["sections"][0]["non_verifiees"] == [])
r = MO.verifier(poutre(s={"M_inf": 40.0, "V": 20.0, "pas": 20.0, "inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]}))
chk("« statut » reste celui de la méthode ; « statut_complet » y ajoute les dispositions", r["statut"] == MO.OK and r["statut_complet"] == MO.WARN, (r["statut"], r["statut_complet"]))

# 8. le sélecteur : des marques, rien de retiré, l'optimal de Streamlit
p = poutre(s={"M_inf": 120.0})
c = MO.candidats_armatures(p, "A", "inf")
chk("le sélecteur garde ses 30 dispositions et marque possible / serrée", len(c) == 30 and all("possible" in x and "serree" in x for x in c)
    and any(not x["possible"] for x in c) and any(x["serree"] for x in c))
chk("l'optimal du moteur reste la plus petite aire « ok » (Streamlit)", [x["nom"] for x in c if x["optimal"]] == [next(x["nom"] for x in c if x["ok"])])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
