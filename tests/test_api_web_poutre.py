# -*- coding: utf-8 -*-
"""L'API du site pour les poutres béton armé (api_web/poutre.py), vérifiée en Python ordinaire.

- le résultat affiché est celui du moteur beton/poutre : taux, couleur et vérification gouvernante, section par section ;
- une vérification que le moteur déclare non conforme est rouge même sous 100 % (acier excessif) ;
- chaque champ du cadre Données pointe un endroit réel de la poutre, et le réécrire avec sa valeur ne change rien ;
- les boutons « Ajouter » et « Retirer » (sections, lits, lignes d'étriers) passent par le même chemin, avec leurs bornes ;
- une saisie hors domaine est refusée avec un message, et la poutre reste intacte ;
- le sélecteur d'armatures est trié par section d'acier, marque la disposition retenue et l'optimale du moteur ;
- les poutres de la note s'ajoutent, se copient, se suppriment (jamais la dernière) ; le résumé est la plus chargée ;
- un nom saisi est échappé partout où il s'affiche.

    python3 tests/test_api_web_poutre.py
"""
import copy
import itertools
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import poutre as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.poutre import entree as EN  # noqa: E402
from beton.poutre import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:240]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(module="poutre", action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def niveau_de(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


LIGNE = re.compile(r'<div class="el( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)">([^<]+)</span>', re.S)

# 1. une poutre neuve : le résultat est celui du moteur
r = req(module="poutre", action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
p0 = EN.nouvelle_poutre()
p0["nom"] = "V1"
chk("la poutre neuve est celle du module (20 × 40, C30/37, B500)", r["etat"]["poutres"] == [EN.normaliser(p0)])
ref = MO.verifier(p0)
chk("le taux affiché est celui du moteur, arrondi dans sa bande", r["resume"]["taux"] == C.pct(ref["taux_max"]), (r["resume"], ref["taux_max"]))

# 2. une poutre chargée : M = 122 kN·m, V = 60 kN, puis une seconde section sur appui
r = saisir(r["etat"], "sections/0/M_inf", "122")
r = saisir(r["etat"], "sections/0/V", "60")
r = req(module="poutre", action="modifier", etat=r["etat"], chemin=["sections", "+"], valeur=1)
r = saisir(r["etat"], "sections/1/M_sup", "45")
r = saisir(r["etat"], "sections/1/V", "35")
etat = r["etat"]
p = etat["poutres"][0]
chk("« Ajouter une section » crée la section B", [s["nom"] for s in p["sections"]] == ["A", "B"], [s["nom"] for s in p["sections"]])
ref = MO.verifier(p)
attendu = [(v, s) for s in ref["sections"] for v in s["verifications"]]
lignes = LIGNE.findall(r["resultat"])
chk(f"une ligne de résultat par vérification du moteur ({len(attendu)})", len(lignes) == len(attendu), (len(lignes), len(attendu)))
chk("chaque ligne porte le nom court, le taux et la couleur de sa vérification",
    all(nom == C.esc(P.COURTS[v["cle"]]) and n == niveau_de(v) and t == f"{C.pc(C.pct(v['taux']))}{C.NBSP}%"
        for (_, nom, n, t), (v, _s) in zip(lignes, attendu)), lignes[:3])
g = ref["gouvernant"]
gov = [i for i, (x, *_) in enumerate(lignes) if x]
chk("une seule ligne gouvernante, celle du moteur",
    len(gov) == 1 and attendu[gov[0]][1]["nom"] == g["section"] and attendu[gov[0]][0]["cle"] == g["cle"], (gov, g))
niv = "r" if ref["statut"] == MO.NOK else C.niveau(ref["taux_max"])
chk("le résumé : taux, couleur et gouvernant du moteur",
    r["resume"]["taux"] == C.pct(ref["taux_max"]) and r["resume"]["niveau"] == niv and r["resume"]["gouvernant"].endswith(f"section {g['section']} · {g['nom']}"),
    (r["resume"], ref["taux_max"], g))
grp = re.findall(r'<div class="elgrp">Section ([^<]+)<span class="pct (\w)">([^<]+)</span>', r["resultat"])
chk("un en-tête par section, avec le taux de la section",
    grp == [(s["nom"], "r" if s["statut"] == MO.NOK else C.niveau(s["taux_max"]), f"{C.pc(C.pct(s['taux_max']))}{C.NBSP}%") for s in ref["sections"]], grp)
chk("une fenêtre de calcul par section, et chaque ligne ouvre celle de sa section",
    all(f'id="ck-s{i}"' in r["volets"] for i in range(len(p["sections"])))
    and re.findall(r'data-calcul="(s\d)"', r["resultat"]) == [f"s{i}" for i, s in enumerate(ref["sections"]) for _ in s["verifications"]])
chk("une vue par section, et « Toutes » quand il y en a plusieurs", [v["cle"] for v in r["vues"]] == ["s0", "s1", "toutes"], [v["cle"] for v in r["vues"]])
chk("le choix d'armatures s'ouvre sur les deux faces de chaque section",
    re.findall(r'data-pk="([^"]+)"', r["resultat"]) == ["inf-0", "sup-0", "inf-1", "sup-1"], re.findall(r'data-pk="([^"]+)"', r["resultat"]))
# la fenêtre de calcul : chaque formule du moteur a sa ligne, et son résultat est celui du moteur
fx = sum(len(v["formules"]) for v in ref["sections"][0]["verifications"])
ck0 = r["volets"].split('id="ck-s1"')[0]
chk(f"la fenêtre de la section A détaille les {fx} formules du moteur", ck0.count('class="fx') >= fx, ck0.count('class="fx'))
v_inf = next(v for v in ref["sections"][0]["verifications"] if v["cle"] == "inf")
f_req = next(f for f in v_inf["formules"] if f["cle"] == "As_req")
chk("A_s,req s'écrit avec les valeurs du moteur et son résultat", f"{C.fr(f_req['resultat'], 0)} mm²" in ck0 and "122 × 10⁶" in ck0, f_req["resultat"])

# 3. rouge même sous 100 % : l'acier excessif (A_s > A_s,max) est non conforme quel que soit son taux
e = copy.deepcopy(etat)
for j in range(EN.MAX_LITS):
    if j:
        req_ = req(module="poutre", action="modifier", etat=e, chemin=["sections", "0", "inf", "+"], valeur=1)
        e = req_["etat"]
    e = saisir(e, f"sections/0/inf/{j}/n", "6")["etat"]
    out = saisir(e, f"sections/0/inf/{j}/d", "32")
    e = out["etat"]
v = next(v for v in MO.verifier(e["poutres"][0])["sections"][0]["verifications"] if v["cle"] == "inf")
ligne_inf = [x for x in LIGNE.findall(out["resultat"]) if x[1] == "Armatures inférieures"][0]
chk("acier excessif : le moteur le déclare non conforme sous 100 %", v["etat"] == MO.NOK and v["taux"] < 1, (v["etat"], v["taux"]))
chk("… et la ligne est rouge, pas verte", ligne_inf[2] == "r", ligne_inf)

# 4. chaque champ pointe la poutre ; le réécrire avec sa propre valeur ne change rien
champs = re.findall(r'<input data-chemin="([^"]+)" data-sorte="(\w+)"[^>]*?value="([^"]*)"', r["donnees"])
champs += [(c_, "choix", v_) for c_, v_ in re.findall(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>.*?<option value="([^"]*)" selected', r["donnees"], re.S)]
champs += re.findall(r'<button type="button" class="on" data-chemin="([^"]+)" data-sorte="(choix)" data-valeur="([^"]*)"', r["donnees"])
# poutre (nom, béton, acier, b, h, enrobage) + par section (M_inf, M_sup, V ; n et Ø par lit ; type et Ø par ligne
# d'étriers ; pas) + méthode (γs et les deux jeux)
n_attendu = 6 + sum(3 + 2 * (len(s_["inf"]) + len(s_["sup"]) + len(s_["etriers"])) + 1 for s_ in p["sections"]) + 3
chk(f"le cadre Données porte tous les champs de la poutre ({len(champs)})", len(champs) == n_attendu, (len(champs), n_attendu))
vus = {c_ for c_, _, _ in champs}
chk("les champs couvrent la poutre, les efforts, les lits, les étriers et la méthode",
    {"nom", "beton", "fyk", "b", "h", "enrobage", "gamma_s", "jeu_premier_lit", "jeu_entre_lits", "sections/0/M_inf", "sections/1/M_sup",
     "sections/0/V", "sections/0/inf/0/n", "sections/0/inf/0/d", "sections/1/sup/0/d", "sections/0/etriers/0/type", "sections/0/etriers/0/d",
     "sections/1/pas"} <= vus, sorted(vus))
mauvais = []
for chemin, sorte, valeur in champs:
    e2 = copy.deepcopy(etat)
    msg = P.modifier(e2, chemin.split("/"), valeur.replace("&amp;", "&"))
    if msg or e2 != etat:
        mauvais.append((chemin, valeur, msg))
chk("réécrire chaque champ avec sa valeur affichée ne change rien", not mauvais, mauvais[:3])

# 5. les saisies : nombres, entiers, refus
e = copy.deepcopy(etat)
P.modifier(e, ["h"], "62,5")
chk("« 62,5 » se lit 62,5 cm", e["poutres"][0]["h"] == 62.5)
P.modifier(e, ["sections", "0", "inf", "0", "n"], "3")
P.modifier(e, ["fyk"], "400")
chk("le nombre de barres et la nuance restent des entiers",
    e["poutres"][0]["sections"][0]["inf"][0]["n"] == 3 and isinstance(e["poutres"][0]["sections"][0]["inf"][0]["n"], int)
    and e["poutres"][0]["fyk"] == 400 and isinstance(e["poutres"][0]["fyk"], int))
for chemin, valeur, quoi in ((["h"], "douze", "un nombre illisible"), (["h"], "0", "une hauteur nulle"), (["b"], "-20", "une largeur négative"),
                             (["sections", "0", "V"], "-5", "un effort tranchant négatif"), (["beton"], "C99/999", "une classe de béton inconnue"),
                             (["fyk"], "235", "une nuance inconnue"), (["sections", "0", "inf", "0", "d"], "13", "un diamètre hors liste"),
                             (["sections", "0", "pas"], "0", "un pas nul")):
    avant = copy.deepcopy(e)
    msg = P.modifier(e, chemin, valeur)
    if not (msg and e == avant):
        chk(f"{quoi} est refusé, la poutre reste intacte", False, (chemin, valeur, msg))
        break
else:
    chk("les saisies hors domaine (illisible, nulle, négative, classe, nuance, Ø, pas) sont refusées, la poutre intacte", True)
out = req(module="poutre", action="modifier", etat=etat, chemin=["h"], valeur="0")
chk("refusée par l'écran : un message, et l'état renvoyé est l'ancien", out["ok"] and out["message"] and out["etat"] == etat, out.get("message"))

# 6. ajouter, retirer : sections, lits, lignes d'étriers
e = copy.deepcopy(etat)
for _ in range(EN.MAX_LITS - 1):
    P.modifier(e, ["sections", "0", "sup", "+"], 1)
chk(f"« Ajouter un lit » jusqu'à {EN.MAX_LITS} lits", len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS)
msg = P.modifier(e, ["sections", "0", "sup", "+"], 1)
chk(f"un {EN.MAX_LITS + 1}ᵉ lit est refusé avec un message lisible (pas celui, technique, du moteur)",
    msg == f"Au plus {EN.MAX_LITS} lits par face" and len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS, msg)
h = P.formulaire(e)
chk("au maximum, le bouton « Ajouter un lit » de cette face disparaît", 'data-chemin="sections/0/sup/+"' not in h and 'data-chemin="sections/0/inf/+"' in h)
P.modifier(e, ["sections", "0", "sup", "1", "-"], 1)
chk("« Retirer ce lit » retire le bon lit", len(e["poutres"][0]["sections"][0]["sup"]) == EN.MAX_LITS - 1)
P.modifier(e, ["sections", "0", "etriers", "+"], 1)
P.modifier(e, ["sections", "0", "etriers", "1", "type"], "epingle")
chk("une ligne d'étriers s'ajoute, et devient une épingle", [x["type"] for x in e["poutres"][0]["sections"][0]["etriers"]] == ["etrier", "epingle"])
P.modifier(e, ["sections", "0", "-"], 1)
chk("retirer la section A garde la section B", [s["nom"] for s in e["poutres"][0]["sections"]] == ["B"])
msg = P.modifier(e, ["sections", "0", "-"], 1)
chk("la dernière section ne se retire pas", msg and len(e["poutres"][0]["sections"]) == 1, msg)
msg = P.modifier(e, ["sections", "0", "inf", "0", "-"], 1)
chk("le dernier lit d'une face ne se retire pas", msg and len(e["poutres"][0]["sections"][0]["inf"]) == 1, msg)
h = P.formulaire(e)
chk("une section seule n'a pas de bouton « Retirer »", 'data-chemin="sections/0/-"' not in h)
P.modifier(e, ["sections", "+"], 1)
chk("la section ajoutée prend la première lettre libre (A)", [s["nom"] for s in e["poutres"][0]["sections"]] == ["B", "A"])

# 7. le sélecteur d'armatures
s = req(module="poutre", action="selecteur", etat=etat, role="inf-0")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)".*?<span class="kg">([^<]+)</span>', s["html"], re.S)
aires = [float(re.sub(r"[\s  ]", "", a).replace(",", ".")) for *_, a in rows]
cands = MO.candidats_armatures(etat["poutres"][0], "A", "inf")
chk(f"le sélecteur liste les dispositions du moteur ({len(rows)})", [n for _, _, n, _ in rows] == [c_["nom"] for c_ in cands] and len(rows) > 20, len(rows))
chk("triées par section d'acier croissante", aires == sorted(aires), aires[:6])
chk("une seule disposition retenue : 2Ø16, celle de la section", [n for cur, _, n, _ in rows if cur] == ["2Ø16"], [n for cur, _, n, _ in rows if cur])
best = next((c_["nom"] for c_ in cands if c_["ok"]), None)
chk("l'étoile est l'optimale du moteur (la plus petite section qui tient)", [n for _, opt, n, _ in rows if opt] == ([best] if best else []),
    ([n for _, opt, n, _ in rows if opt], best))
r2 = req(module="poutre", action="choisir", etat=etat, role="inf-0", nom=best)
v2 = next(v for v in MO.verifier(r2["etat"]["poutres"][0])["sections"][0]["verifications"] if v["cle"] == "inf")
chk(f"choisir {best} le pose en un lit et recalcule : la face inférieure passe",
    r2["ok"] and r2["etat"]["poutres"][0]["sections"][0]["inf"] == [{"n": int(best.split("Ø")[0]), "d": int(best.split("Ø")[1])}] and v2["etat"] == MO.OK,
    (r2.get("erreur"), v2["etat"]))
s = req(module="poutre", action="selecteur", etat=etat, role="sup-1")
chk("le sélecteur de la face supérieure de la section B se titre ainsi", "Armatures supérieures · section B" in s["html"])

# 8. les poutres de la note
e = copy.deepcopy(etat)
P.gerer_elements(e, "ajouter")
P.gerer_elements(e, "dupliquer")
chk("ajouter puis copier : trois poutres, la dernière ouverte, nommée V2 (copie)",
    len(e["poutres"]) == 3 and e["actif"] == 2 and e["poutres"][2]["nom"] == "V2 (copie)", [x["nom"] for x in e["poutres"]])
e["actif"] = 1
rn = req(module="poutre", action="calculer", etat=e)
t_max = max(MO.verifier(x)["taux_max"] for x in e["poutres"])
chk("le résumé de la note est sa poutre la plus chargée (ici V1, fermée)",
    rn["resume"]["taux"] == C.pct(t_max) and rn["resume"]["gouvernant"].startswith("V1 ·"), (rn["resume"], t_max))
chk("la liste montre chaque poutre avec son taux", len(re.findall(r'class="ni[ "]', rn["elements"])) == 3 and 'class="ni on" data-activer="1"' in rn["elements"])
P.gerer_elements(e, "supprimer", 0)
chk("supprimer la première garde la poutre ouverte", len(e["poutres"]) == 2 and e["actif"] == 0 and e["poutres"][0]["nom"] == "V2")
P.gerer_elements(e, "supprimer", 0)
P.gerer_elements(e, "supprimer", 0)
chk("la dernière poutre ne se supprime pas", len(e["poutres"]) == 1)

# 9. toutes les classes de béton, les deux aciers, des efforts du nul au démesuré se calculent
fautes = []
for beton, fyk, (m_inf, m_sup, v_) in itertools.product(EN.classes_beton(), EN.NUANCES, ((0, 0, 0), (60, 0, 20), (0, 90, 70), (400, 300, 400))):
    e = P.nouvel_etat()
    for chemin, x in ((["beton"], beton), (["fyk"], fyk), (["sections", "0", "M_inf"], m_inf), (["sections", "0", "M_sup"], m_sup), (["sections", "0", "V"], v_)):
        msg = P.modifier(e, chemin, x)
        if msg:
            fautes.append((beton, fyk, chemin, msg))
    out = req(module="poutre", action="calculer", etat=e)
    if not out["ok"]:
        fautes.append((beton, fyk, m_inf, m_sup, v_, out.get("erreur")))
n_cas = len(EN.classes_beton()) * len(EN.NUANCES) * 4
chk(f"les {n_cas} combinaisons béton × acier × efforts se calculent", not fautes, fautes[:3])

# 10. l'échappement. Le contrat : les fragments HTML (liste, données, résultat, fenêtres, dessins, phrase) sont échappés
# ici ; les champs TEXTE (titres des vues, résumé) sont des données, que la page échappe en les posant
# (verifier_site.js : « .stab piégé »).
def fragments_html(out):
    return {"liste des poutres": out["elements"], "cadre Données": out["donnees"], "résultat": out["resultat"], "fenêtres de calcul": out["volets"],
            "dessins": "".join(v["svg"] for v in out["vues"]), "phrase": out["phrase"], "statut": out["statut_html"]}


e = copy.deepcopy(etat)
P.modifier(e, ["nom"], '<script>alert("x")</script>')
out = req(module="poutre", action="calculer", etat=e)
fuites = [k for k, h in fragments_html(out).items() if "<script>" in h]
chk("un nom de poutre saisi est échappé dans chaque fragment HTML", not fuites and "&lt;script&gt;" in out["elements"] and "&lt;script&gt;" in out["volets"], fuites)
# un nom de section ne se saisit pas à l'écran, mais un fichier .stab peut en porter un
e = copy.deepcopy(etat)
e["poutres"][0]["sections"][0]["nom"] = '<img src=x onerror=alert(1)>'
out = req(module="poutre", action="calculer", etat=e)
fuites = [k for k, h in fragments_html(out).items() if "<img" in h]
chk("un nom de section venu d'un fichier est échappé dans chaque fragment HTML", out["ok"] and not fuites, fuites or out.get("erreur"))

# 11. deux sections du même nom (possible dans un fichier .stab, pas à l'écran) : le site désigne une section par son
# nom (sélecteur, ligne gouvernante) — la seconde prend la première lettre libre, et son sélecteur est bien le sien
e = copy.deepcopy(etat)
e["poutres"][0]["sections"][1]["nom"] = "A"
e["poutres"][0]["sections"][1]["M_inf"] = 200.0
out = req(module="poutre", action="calculer", etat=e)
noms = [s_["nom"] for s_ in out["etat"]["poutres"][0]["sections"]] if out["ok"] else out.get("erreur")
chk("deux sections du même nom venues d'un fichier : la seconde prend la première lettre libre (B)", noms == ["A", "B"], noms)
s = req(module="poutre", action="selecteur", etat=out["etat"], role="inf-1")
taux_affiches = re.findall(r'class="chip \w">([^<]+)<', s["html"])
# l'attendu : la seconde section SEULE dans sa poutre (aucun nom à confondre)
p2 = dict(out["etat"]["poutres"][0])
p2["sections"] = [p2["sections"][1]]
attendus = [f"{C.pc(C.pct(min(c_['taux'], 9.99)))}{C.NBSP}%" for c_ in MO.candidats_armatures(p2, p2["sections"][0]["nom"], "inf")]
chk("… et le sélecteur de la seconde calcule SA section (M = 200 kN·m), pas la première", taux_affiches == attendus, (taux_affiches[:3], attendus[:3]))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
