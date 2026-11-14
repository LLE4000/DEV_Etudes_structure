# -*- coding: utf-8 -*-
"""Le FERRAILLAGE DÉTAILLÉ et les DISPOSITIONS CONSTRUCTIVES du moteur poutre (beton/poutre/geometrie.py,
dispositions.py ; site, 11/2026), vérifiés à la main.

- sans aucune clé du détail, la poutre normalisée est celle d'avant (aucune clé ajoutée) ;
- les positions (beton/position.py, règle du 13/11/2026) : la position RÉELLE d'un lit (c + Ø étrier + Ø/2, le dessin ;
  une position imposée « pos » en est une) et son axe de CALCUL (+ jeu de pose 1 cm, arrondi au 0,5 cm) ; d et y_G
  suivent les axes de calcul (Σ A·e / Σ A), les dispositions et les latérales les positions réelles ;
- les abscisses : uniforme entre les montants de l'étrier, « aligne » sur le lit 1, « entraxe » centré ;
- s_min = max(Ø ; d_g + 5 ; 20 mm) ; deux lits Ø16 au jeu d'avant le 13/11/2026 (1 cm) : 10 mm libres → « non conforme »
  (décision du bureau, 11/2026 : un espacement sous le minimum de l'EC2 est une non-conformité, pas une observation) ; au
  jeu du bureau d'aujourd'hui (2,5 cm, le défaut) : 25 mm libres → conforme ;
- des barres qui se chevauchent, ou qui sortent de l'étrier, sont « non conformes » et « impossibles » (le sélecteur ne
  les propose pas) ; un espacement positif sous s_min est non conforme mais se pose (« serrée ») ;
- EC2 8.2(3) : des lits dont les barres ne sont pas superposées sont « à examiner » ;
- une observation « à examiner » se justifie (pour l'empreinte examinée) ; une non-conformité jamais ;
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
# lit 1 : réel 3 + 1,0 (étrier Ø10) + 1,0 (Ø20/2) = 5,0 cm, calcul 5,0 + 1 (jeu) = 6,0 → 6,0 ; lit 2 imposé à 4 cm
# d'entraxe : réel 9,0, calcul 10,0 → 10,0
chk("lit 1 : réel 3 + 1 + 1 = 5 cm, calcul 5 + 1 (jeu) = 6 cm ; lit 2 à 4 cm d'entraxe : réel 9 cm, calcul 10 cm",
    [x["reel"] for x in d] == [5.0, 9.0] and [x["distance"] for x in d] == [6.0, 10.0], [(x["reel"], x["distance"]) for x in d])
A1, A2 = 3 * math.pi * 100, 2 * math.pi * 64
g = MO.verifier(p)["sections"][0]["grandeurs"]
chk("y_G = Σ A·e / Σ A et d = h − y_G suivent les axes de CALCUL (6 et 10 cm)",
    proche(g["yG_inf"], (A1 * 6 + A2 * 10) / (A1 + A2)) and proche(g["d_inf"], 40 - g["yG_inf"]), (g["yG_inf"], g["d_inf"]))
p = poutre(s={"inf": [{"n": 2, "d": 16, "pos": 6.5}]})
lit = MO.verifier(p)["sections"][0]["geometrie"]["inf"]["lits"][0]
chk("une position imposée du lit 1 est la position RÉELLE (6,5 cm, le dessin) ; le calcul y ajoute le jeu : 6,5 + 1 = 7,5 cm",
    lit["reel"] == 6.5 and lit["distance"] == 7.5, lit)

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
p = poutre(jeu_entre_lits=1.0, s={"inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]})
it = next(x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
chk("deux lits Ø16 au jeu d'avant (1 cm) : 0,8 + 1 + 0,8 = 2,6 cm d'entraxe réel, 10 mm libres < 25 mm → « non conforme », "
    "mais posable", proche(it["libre"], 10.0) and it["mini"] == 25 and it["etat"] == DI.NOK and not it["impossible"], it)
p = poutre(s={"inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]})
it = next(x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
chk("… au jeu entre lits du bureau (2,5 cm, le défaut depuis le 13/11/2026) : 25 mm libres → conforme",
    EN.DEFAUTS_POUTRE["jeu_entre_lits"] == 2.5 and proche(it["libre"], 25.0) and it["etat"] == DI.OK, it)
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
chk("enrobage 1,5 cm + Ø6 = 21 mm < Ø32 → « non conforme » (adhérence, c_min,b = Ø)", proche(it["libre"], 21.0) and it["etat"] == DI.NOK, it)

# 5. les latérales
p = poutre(h=90.0)
lat = MO.verifier(p)["sections"][0]["geometrie"]["laterales"]
# les latérales se mesurent aux positions RÉELLES des lits 1 (3 + 1,0 + 0,8 = 4,8 cm) : h = 90, écart 90 − 9,6 = 80,4 cm,
# n = ⌈80,4 / 30⌉ − 1 = 2, la première à 4,8 + 80,4 / 3 = 31,6 cm
chk("automatiques : n = ⌈écart / 30⌉ − 1 par face, réparties entre les lits 1 (aux positions réelles : 2 barres, "
    "écart 80,4 cm, la première à 31,6 cm)", lat["auto"] and lat["n"] == 2 and proche(lat["ecart"], 80.4)
    and proche(lat["ys"][0], 31.6), lat)
lat40 = MO.verifier(poutre())["sections"][0]["geometrie"]["laterales"]
chk("une poutre de 40 cm en a UNE par face : écart réel 40 − 2 × 4,8 = 30,4 cm > 30 cm (avant le 13/11/2026, aux "
    "positions arrondies 5,0 cm, l'écart faisait 30 cm et il n'y en avait pas)", lat40["n"] == 1 and proche(lat40["ecart"], 30.4), lat40)
chk("une poutre de 39 cm n'en a pas : 39 − 9,6 = 29,4 cm ≤ 30 cm", MO.verifier(poutre(h=39.0))["sections"][0]["geometrie"]["laterales"]["n"] == 0)
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
r = MO.verifier(poutre(jeu_entre_lits=1.0, s={"M_inf": 40.0, "V": 20.0, "pas": 20.0, "inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]}))
chk("« statut » reste celui de la méthode ; « statut_complet » y ajoute les dispositions (10 mm libres : non conforme)",
    r["statut"] == MO.OK and r["statut_complet"] == MO.NOK, (r["statut"], r["statut_complet"]))

# 9. EC2 8.2(3) : les lits superposés
p = poutre(b=30.0, s={"inf": [{"n": 4, "d": 20}, {"n": 3, "d": 20}]})
dis = MO.verifier(p)["sections"][0]["dispositions"]
it = next((x for x in dis["items"] if x["cle"] == "superposes"), None)
chk("L1 4Ø20, L2 3Ø20 répartis : la barre du milieu de L2 n'est au-dessus d'aucune barre → « à examiner » (EC2 8.2(3))",
    it is not None and it["etat"] == DI.WARN and it["id"] == "superposes:inf:1" and "1 barre non superposée" in it["texte"], it)
p = poutre(b=30.0, s={"inf": [{"n": 4, "d": 20}, {"n": 3, "d": 20, "rep": "aligne"}]})
chk("… « superposée » (sur les barres de L1) : rien à examiner", not any(x["cle"] == "superposes" for x in MO.verifier(p)["sections"][0]["dispositions"]["items"]))
p = poutre(b=30.0, s={"inf": [{"n": 4, "d": 20}, {"n": 2, "d": 16}]})
chk("… deux barres réparties sous quatre : sur les angles, superposées (Ø différents, à ½ Ø près)",
    not any(x["cle"] == "superposes" for x in MO.verifier(p)["sections"][0]["dispositions"]["items"]))

p = poutre(b=30.0, s={"inf": [{"n": 3, "d": 20}], "sup": [{"n": 2, "d": 16}], "etriers": [{"type": "etrier", "d": 10}, {"type": "epingle", "d": 8}]})
it = next((x for x in MO.verifier(p)["sections"][0]["dispositions"]["items"] if x["cle"] == "epingle"), None)
chk("une épingle sur la barre du milieu sous deux barres hautes d'angle : rien à accrocher en haut → « à examiner »",
    it is not None and it["id"] == "epingle:1" and it["etat"] == DI.WARN, it)
p = poutre(b=30.0, s={"inf": [{"n": 3, "d": 20}], "sup": [{"n": 3, "d": 16}], "etriers": [{"type": "etrier", "d": 10}, {"type": "epingle", "d": 8}]})
chk("… trois barres hautes : l'épingle tient sa barre haute", not any(x["cle"] == "epingle" for x in MO.verifier(p)["sections"][0]["dispositions"]["items"]))

# 10. les identifiants, les empreintes et les justifications de l'ingénieur
p = poutre(b=40.0, s={"M_inf": 60.0, "V": 30.0, "pas": 20.0, "inf": [{"n": 5, "d": 20}], "sup": [{"n": 2, "d": 16}],
                      "etriers": [{"type": "etrier", "d": 8}, {"type": "etrier", "d": 8, "de": 2, "a": 4}]})
r = MO.verifier(p)["sections"][0]
chk("le cas d'essai passe la méthode du bureau (seules les dispositions parlent)", r["statut"] == MO.OK, [(v["cle"], v["etat"]) for v in r["verifications"]])
ids = [x["id"] for x in r["dispositions"]["items"]]
chk("chaque observation a un identifiant stable, unique dans la section", len(ids) == len(set(ids)) and "angles_cadre:1" in ids and "faces" in ids, ids)
ang = next(x for x in r["dispositions"]["items"] if x["id"] == "angles_cadre:1")
chk("sans justification : le cadre intérieur est « à examiner », la section aussi", ang["etat_eff"] == DI.WARN and r["statut_complet"] == MO.WARN
    and r["dispositions"]["compte"][DI.WARN] == 1, (ang, r["statut_complet"]))
pj = copy.deepcopy(p)
pj["sections"][0]["justifs"] = {"angles_cadre:1": {"texte": "  Deux barres de montage HA10 prévues au plan.  ", "empreinte": ang["empreinte"]}}
pj = EN.normaliser(pj)
chk("la justification se normalise (texte rogné)", pj["sections"][0]["justifs"]["angles_cadre:1"]["texte"] == "Deux barres de montage HA10 prévues au plan.")
rj = MO.verifier(pj)["sections"][0]
aj = next(x for x in rj["dispositions"]["items"] if x["id"] == "angles_cadre:1")
chk("justifiée pour l'empreinte examinée : « examinée » (just), jamais « ok » ; la section n'est plus « à vérifier »",
    aj["etat"] == DI.WARN and aj["etat_eff"] == DI.JUST and aj["justification"].startswith("Deux barres") and rj["statut_complet"] == MO.OK
    and rj["dispositions"]["compte"][DI.JUST] == 1, (aj, rj["statut_complet"]))
pk = copy.deepcopy(pj)
pk["sections"][0]["etriers"][1]["a"] = 3                     # le cadre change : ce qui a été examiné n'est plus là
rk = MO.verifier(pk)["sections"][0]
ak = next(x for x in rk["dispositions"]["items"] if x["id"] == "angles_cadre:1")
chk("la disposition change : la justification devient « à revoir », l'observation redevient « à examiner »",
    ak["etat_eff"] == DI.WARN and "justification" not in ak and ak["justification_perimee"].startswith("Deux barres") and rk["statut_complet"] == MO.WARN, ak)
pn = copy.deepcopy(poutre(jeu_entre_lits=1.0, s={"M_inf": 40.0, "V": 20.0, "pas": 20.0, "inf": [{"n": 2, "d": 16}, {"n": 2, "d": 16}]}))
itn = next(x for x in MO.verifier(pn)["sections"][0]["dispositions"]["items"] if x["cle"] == "entre_lits")
pn["sections"][0]["justifs"] = {itn["id"]: {"texte": "Validé", "empreinte": itn["empreinte"]}}
rn = MO.verifier(EN.normaliser(pn))["sections"][0]
itn2 = next(x for x in rn["dispositions"]["items"] if x["cle"] == "entre_lits")
chk("une NON-CONFORMITÉ ne se justifie pas : 10 mm pour 25 mm reste non conforme, justification ou pas",
    itn2["etat_eff"] == DI.NOK and "justification" not in itn2 and rn["statut_complet"] == MO.NOK, itn2)
for mauvais, quoi in (({"x": "texte"}, "valeur non dict"), ({"x": {"texte": "a", "auteur": "b"}}, "clé inconnue"),
                      ({"x": {"texte": "a" * 601}}, "texte trop long"), ({"": {"texte": "a"}}, "identifiant vide")):
    q = copy.deepcopy(poutre())
    q["sections"][0]["justifs"] = mauvais
    try:
        EN.normaliser(q)
        refuse = False
    except ValueError:
        refuse = True
    chk(f"justifications refusées : {quoi}", refuse)
q = copy.deepcopy(poutre())
q["sections"][0]["justifs"] = {}
chk("aucune justification : la clé disparaît (la section d'avant, à l'octet)", "justifs" not in EN.normaliser(q)["sections"][0])

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
