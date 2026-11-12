# -*- coding: utf-8 -*-
"""Le module « Flambement » du site (api_web/flambement.py) : ce que l'écran affiche est ce que le moteur
(acier/flambement) calcule ; les saisies, leurs refus et leurs effets de bord ; le sélecteur de profilés ; les éléments
de la note ; les dessins ; l'échappement. Le calcul lui-même est prouvé par tests/test_flambement_moteur.py.

    python3.12 tests/test_api_web_flambement.py
"""
import copy
import json
import math
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from acier.flambement import entree as EN  # noqa: E402
from acier.flambement import moteur as MO  # noqa: E402
from acier.flambement import section as SE  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import dessin_flambement as DF  # noqa: E402
from api_web import flambement as F  # noqa: E402
from api_web import routeur as R  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="flambement", **k))))


def modif(etat, chemin, valeur):
    return req(action="modifier", etat=copy.deepcopy(etat), chemin=chemin, valeur=valeur)


def texte(h):
    """Le texte d'un fragment HTML (balises ôtées, entités rendues, indices recollés)."""
    import html
    return html.unescape(re.sub(r"<[^>]+>", "", h)).replace("\u202f", " ").replace("\xa0", " ")


# ------------------------------------------------------------------ 1. une note neuve : l'écran = le moteur
r0 = req(action="nouveau")
chk("une note neuve se calcule", r0.get("ok"), r0.get("erreur"))
e0 = r0["etat"]
chk("la note neuve : une barre P1, HEA 200, S235, N_Ed = 500 kN, L = 4 m, bi-articulée", e0 == F.nouvel_etat() and e0["barres"][0]["profil"] == "HEA 200")
m0 = MO.verifier(e0["barres"][0])
res = r0["resultat"]
chk("l'avertissement « nouveau calcul, à valider par le bureau » est EN TÊTE du résultat",
    res.index("Nouveau calcul") < res.index('class="verdict"') and "à valider par le bureau" in res)
chk("… et dans la fenêtre de calcul", "à valider par le bureau" in r0["volets"])
chk("le verdict porte le taux du moteur, arrondi sans quitter sa bande (C.pct)", f'{C.pc(C.pct(m0["taux_max"]))}' in texte(res.split('class="et-blocs"')[0])
    and C.pct(m0["taux_max"]) == 63)
chk("le verdict nomme la vérification gouvernante et sa comparaison N_Ed ≤ N_b,z,Rd = 794,0 kN",
    "Flambement autour de z-z" in res and "794,0" in texte(res.split('class="et-blocs"')[0]))
for v in m0["verifications"]:
    chk(f"ligne « {v['nom']} » : taux {C.pct(v['taux'])} % du moteur, sa couleur", f'<span class="pct {C.niveau(v["taux"])}">{C.pc(C.pct(v["taux"]))}' in res)
chk("trois lignes de résultat (deux lignes chacune : la comparaison, la jauge), chacune ouvre la fenêtre de calcul",
    res.count('class="el vc') == 3 and len(re.findall(r'class="el vc[^"]*" data-calcul="s0"', res)) == 3 and 'id="ck-s0"' in r0["volets"])
chk("le profilé est sur la ligne « Section » et ouvre le sélecteur (★ : un plus léger tient)", 'data-pk="profil"' in res and "Plus léger qui tient : HEA 180" in res)
chk("le pied : la masse de la barre (42,3 kg/m × 4,00 m = 169 kg) et la légende", "169" in texte(res.split("pfoot")[1]) and 'class="legend"' in res)
chk("une barre seule : pas de statut de note (le verdict le dit déjà)", r0["statut_html"] == "")
chk("le résumé : 63 %, vert pâle, la barre et la vérification gouvernante",
    r0["resume"] == {"taux": 63, "niveau": "p", "statut": "Conforme", "gouvernant": "P1 · HEA 200 · flambement z-z"}, r0["resume"])

# ------------------------------------------------------------------ 2. la fenêtre de calcul : chaque formule substituée
vol = texte(r0["volets"])
for frag in ("Acier S235", "fy = 235 N/mm²", "E = 210 000 N/mm²", "γM0 = 1,0 · γM1 = 1,0 (ANB)",
             "Tableau 3.1 · S235 · t = 10 mm ≤ 40 mm", "√(235 / 235)", "c / t (semelle)", "78,8 / 10", "classe 1",
             "π² × 210 000 × 3 692,4·10⁴ / 4 000²", "4 783,0 kN", "π² × 210 000 × 1 335,5·10⁴ / 4 000²", "1 730,0 kN",
             "√(5 383 × 235 / (1 730,0·10³))", "0,855", "4 000 / (49,8 × 93,9)", "laminé en I, h/b ≤ 1,2, tf ≤ 100 mm",
             "½·[1 + 0,49·(0,855 − 0,2) + 0,855²]", "1,026", "1 / (1,026 + √(1,026² − 0,855²))", "0,628",
             "0,628 × 5 383 × 235 / 1,0", "794,0 kN", "500 / 794,0", "5 383 × 235 / 1,0", "1 265,1 kN", "39,5 %", "63 %"):
    chk(f"fenêtre de calcul : « {frag} »", frag in vol, frag)
chk("fenêtre : λ̄ surligné et les indices des lettres grecques (χ_z, φ_z, λ̄_z) en <sub>", 'class="fl-barre"' in r0["volets"]
    and "χ<sub>z</sub>" in r0["volets"] and "φ<sub>z</sub>" in r0["volets"] and "λ_" not in r0["volets"] and "χ_" not in r0["volets"])
chk("fenêtre : le tableau des trois vérifications, la gouvernante marquée", r0["volets"].count("<tr class=") == 3 and r0["volets"].count('<tr class="gov">') == 1)

# ------------------------------------------------------------------ 3. le cadre Données
d = r0["donnees"]
for n, t in ((1, "Barre"), (2, "Effort et longueur"), (3, "Flambement autour de y-y"), (4, "Flambement autour de z-z"), (5, "Hypothèses")):
    chk(f"section {n} « {t} » numérotée et repliable", f'<span class="n">{n}</span><span class="tt">{t}</span>' in d)
for chemin in ("nom", "famille", "profil", "nuance", "N", "L", "axes/y/appuis", "axes/z/appuis", "axes/y/maintiens", "axes/z/maintiens", "gamma_M0", "gamma_M1"):
    chk(f"champ « {chemin} » présent", f'data-chemin="{chemin}"' in d)
chk("le menu des profilés ne liste que la famille du profilé (HE A), avec la masse", d.count('<option value="HEA ') == 24 and "HEA 200 · 42,3 kg/m" in d
    and '<option value="IPE 200"' not in d)
chk("le menu des appuis donne k (Euler) dans chaque libellé", "Encastrée – encastrée · k = 0,5" in d and "Encastrée – libre · k = 2" in d)
chk("f_y et la classe rappelés sous la nuance", "Tableau 3.1, t = 10 mm ≤ 40 mm" in texte(d) and "classe 1 en compression" in texte(d))

# ------------------------------------------------------------------ 4. saisies, effets de bord, refus
r = modif(e0, ["N"], "1 250,5")
chk("N_Ed « 1 250,5 » (espace, virgule) est lu 1 250,5 kN", r["etat"]["barres"][0]["N"] == 1250.5 and not r["message"])
chk("… et le résultat suit le moteur", r["resume"]["taux"] == C.pct(MO.verifier(r["etat"]["barres"][0])["taux_max"]))
refus = [(["N"], "-5", "COMPRESSION"), (["L"], "0", "positive"), (["L"], "abc", "illisible"), (["axes", "y", "k"], "0", "positif"),
         (["axes", "z", "Lcr"], "-1", "positive"), (["axes", "z", "maintiens"], "10", "maintiens"), (["axes", "z", "maintiens"], "1,5", "entier"),
         (["gamma_M1"], "0,9", "ANB"), (["gamma_M0"], "2", "ANB"), (["nuance"], "S690", "inconnue"), (["famille"], "XYZ", "inconnue"),
         (["profil"], "HEA 999", "inconnu"), (["axes", "y", "appuis"], "ZZ", "inconnus"), (["axes", "x", "k"], "1", "inconnu")]
for chemin, v, mot in refus:
    r = modif(e0, chemin, v)
    chk(f"refus avec un message : {'/'.join(chemin)} = {v!r} (« … {mot} … »), la note reste intacte",
        r.get("ok") and r.get("message") and mot.lower() in r["message"].lower() and r["etat"] == e0, r.get("message"))
r = modif(e0, ["famille"], "IPE")
chk("changer de famille retient le plus léger qui tient dans la nouvelle famille (IPE)", r["etat"]["barres"][0]["profil"]
    == MO.plus_leger(MO.candidats(e0["barres"][0], ["IPE"]))["nom"], r["etat"]["barres"][0]["profil"])
r = modif(e0, ["famille"], "HEA")
chk("réécrire la famille du profilé retenu ne change rien", r["etat"] == e0)
e1 = modif(e0, ["L"], "6")["etat"]
r = modif(e1, ["axes", "y", "appuis"], "k")
chk("passer à « coefficient saisi » part du coefficient en cours (k = 1)", r["etat"]["barres"][0]["axes"]["y"]["k"] == 1.0
    and 'data-chemin="axes/y/k"' in r["donnees"])
e2 = modif(e1, ["axes", "z", "maintiens"], "2")["etat"]
r = modif(e2, ["axes", "z", "appuis"], "L")
chk("passer à « longueur saisie » part de la L_cr en cours (6 / 3 = 2,00 m)", r["etat"]["barres"][0]["axes"]["z"]["Lcr"] == 2.0
    and 'data-chemin="axes/z/Lcr"' in r["donnees"])
r = modif(e1, ["axes", "y", "appuis"], "EA")
chk("encastrée–articulée : L_cr,y = 0,7 × 6,00 = 4,20 m, dans le résultat et la fenêtre", "Lcr 4,20 m" in texte(r["resultat"])
    and "0,7 × 6,00" in texte(r["volets"]))
r = modif(e1, ["axes", "z", "maintiens"], "1")
chk("un maintien autour de z : L_cr,z = 6,00 / (1 + 1) = 3,00 m", "6,00 / (1 + 1)" in texte(r["volets"]) and "Lcr 3,00 m" in texte(r["resultat"]))
r = modif(e0, ["nuance"], "S460")
chk("S460 : f_y = 460 et les courbes du S460", r["etat"]["barres"][0]["nuance"] == "S460" and "460 N/mm²" in texte(r["volets"]))

# ------------------------------------------------------------------ 5. les cas particuliers
e = modif(modif(e0, ["famille"], "IPE")["etat"], ["profil"], "IPE 400")["etat"]
e = modif(e, ["nuance"], "S355")["etat"]
r = req(action="calculer", etat=e)
chk("IPE 400 S355 : classe 4, A_eff dans la fenêtre (λ̄_p, ρ de l'âme) et l'avertissement", "classe 4" in r["resultat"] and "ρ<sub>âme</sub>" in r["volets"]
    and "A<sub>eff</sub>" in r["volets"] and "aire efficace" in r["resultat"])
e = modif(e0, ["famille"], "L")["etat"]
r = req(action="calculer", etat=e)
chk("cornière : axes u-u et v-v dans les sections, le résultat, la fenêtre et les dessins", "Flambement autour de u-u" in r["donnees"]
    and "Flambement v-v" in r["resultat"] and "N<sub>b,v,Rd</sub>" in r["volets"] and "u · courbe b" in r["vues"][1]["svg"])
chk("cornière : torsion, flexion-torsion et attaches non prises en compte, DIT dans le résultat", "annexe BB.1.2" in r["resultat"])
e = modif(e0, ["famille"], "HD")["etat"]
e = modif(e, ["profil"], "HD 400x1086.0")["etat"]
r = req(action="calculer", etat=e)
chk("HD 400×1086 (t_f > 80 mm) : « Calcul impossible » avec le motif, sans exception", r.get("ok") and "Calcul impossible" in r["resultat"]
    and "80 mm" in r["resultat"] and r["resume"]["niveau"] == "x" and r["volets"] == "")
chk("… la section se dessine quand même, et le sélecteur reste ouvert pour en changer", r["vues"] and r["vues"][0]["cle"] == "section"
    and 'data-pk="profil"' in r["resultat"])
e = modif(e0, ["famille"], "UPN")["etat"]
r = req(action="calculer", etat=e)
chk("U : la flexion-torsion non vérifiée est DITE", "flexion-torsion" in r["resultat"])
e = modif(e0, ["N"], "0")["etat"]
r = req(action="calculer", etat=e)
chk("N_Ed = 0 : taux nul, conforme", r["resume"]["taux"] == 0 and r["resume"]["niveau"] == "p")
e = modif(e0, ["N"], "5000")["etat"]
r = req(action="calculer", etat=e)
chk("N_Ed = 5 000 kN : non conforme, rouge, « > » dans le verdict", r["resume"]["niveau"] == "r" and '<section class="et-bloc nok"' in r["resultat"] and "Non conforme" in r["resultat"]
    and "&gt; N<sub>b,z,Rd</sub>" in r["resultat"])
e = modif(e0, ["L"], "0,5")["etat"]
r = req(action="calculer", etat=e)
chk("barre courte (λ̄ ≤ 0,2) : χ = 1, le résultat le signale (§6.3.1.2(4))", "§6.3.1.2(4)" in r["resultat"])

# ------------------------------------------------------------------ 6. le sélecteur
s = req(action="selecteur", etat=e0, role="profil")["html"]
noms = re.findall(r'data-choix="([^"]+)"', s)
masses = [float(x.replace(" ", "").replace(",", ".")) for x in re.findall(r'<span class="kg">([^<]+)</span>', s)]
chk("le sélecteur : IPE, HE A, HE B par défaut (les familles du profilé retenu), triés par masse", masses == sorted(masses) and len(noms) == 71
    and all(n.split()[0] in ("IPE", "HEA", "HEB") for n in noms))
best = MO.plus_leger(MO.candidats(e0["barres"][0], ["IPE", "HEA", "HEB"]))
chk("★ = le plus léger qui tient selon le moteur (HEA 180) ; ✓ = le retenu (HEA 200)", f'class="pk-row opt" data-choix="{best["nom"]}"' in s
    and 'class="pk-row cur" data-choix="HEA 200"' in s and best["nom"] == "HEA 180")
c_tous = {c["nom"]: c for c in MO.candidats(e0["barres"][0], ["IPE", "HEA", "HEB"])}
ecarts = [n for n, t in re.findall(r'data-choix="([^"]+)"[^>]*>.*?<span class="chip [a-z]">([0-9,]+)', s) if C.pc(C.pct(min(c_tous[n]["taux"], 9.99))) != t]
chk("chaque ligne porte le taux du moteur pour ce profilé", not ecarts, ecarts[:3])
s = req(action="selecteur", etat=e0, role="profil", familles=["CHS"], froid=False)["html"]
chk("puce CHS sans les formés à froid : tubes ronds finis à chaud seulement", re.findall(r'data-choix="([^"]+)"', s)
    and all(n.startswith("CHS") and "froid" not in n for n in re.findall(r'data-choix="([^"]+)"', s)))
s = req(action="selecteur", etat=e0, role="profil", familles=["CHS"], froid=True)["html"]
chk("… avec les formés à froid", any("froid" in n for n in re.findall(r'data-choix="([^"]+)"', s)))
s = req(action="selecteur", etat=modif(e0, ["nuance"], "S460")["etat"], role="profil", familles=["CHS"], froid=False)["html"]
chk("S460 : les tubes ronds de classe 4 sont écartés, et le compte est DIT", "hors du champ écarté" in s)
s = req(action="selecteur", etat=e0, role="profil", familles=["AUTRES"])["html"]
chk("puce « Autres séries » : IPE A, IPE O, IPN, HE AA, HL, HP", {n.split()[0] for n in re.findall(r'data-choix="([^"]+)"', s)} <= {"IPE", "IPN", "HEAA", "HL", "HP"})
r = req(action="choisir", etat=e0, role="profil", nom="HEB 160")
chk("choisir HEB 160 dans le sélecteur le retient et recalcule", r["etat"]["barres"][0]["profil"] == "HEB 160"
    and r["resume"]["taux"] == C.pct(MO.verifier(r["etat"]["barres"][0])["taux_max"]))
r = req(action="choisir", etat=e0, role="profil", nom="HEA 180")
chk("le plus léger retenu : plus d'étoile sur la ligne « Section »", "Plus léger qui tient" not in r["resultat"])

# ------------------------------------------------------------------ 7. plusieurs barres par note
a = req(action="ajouter", etat=e0)
chk("ajouter : une barre P2 neuve, ouverte", len(a["etat"]["barres"]) == 2 and a["etat"]["barres"][1]["nom"] == "P2" and a["etat"]["actif"] == 1)
b = modif(a["etat"], ["N"], "1500")
chk("la saisie va à la barre ouverte seulement", b["etat"]["barres"][1]["N"] == 1500 and b["etat"]["barres"][0]["N"] == 500)
chk("le résumé de la note : sa barre la plus chargée (P2)", b["resume"]["gouvernant"].startswith("P2") and b["resume"]["niveau"] == "r")
chk("deux barres : la vue « Toutes les barres » (tableau, la barre ouverte surlignée)", b["vues"][-1]["cle"] == "note" and '<tr class="on">' in b["vues"][-1]["svg"]
    and b["vues"][-1]["svg"].count("<tr class=") == 2)
dup = req(action="dupliquer", etat=b["etat"])
chk("copier : « P2 (copie) », mêmes données", dup["etat"]["barres"][2]["nom"] == "P2 (copie)" and dup["etat"]["barres"][2]["N"] == 1500)
act = req(action="activer", etat=dup["etat"], index=0)
chk("ouvrir P1 : le résultat est celui de P1", act["etat"]["actif"] == 0 and '<div class="et-rp"><b>P1</b><span>HEA 200' in act["resultat"])
sup = req(action="supprimer", etat=act["etat"], index=0)
chk("supprimer P1 : il en reste deux, l'ouverte reste valide", len(sup["etat"]["barres"]) == 2 and sup["etat"]["actif"] == 0)
seul = req(action="supprimer", etat=e0, index=0)
chk("la dernière barre ne se supprime pas", len(seul["etat"]["barres"]) == 1)
r = req(action="ajouter", etat=req(action="ajouter", etat=e0)["etat"])
chk("noms suivants : P2 puis P3", [x["nom"] for x in r["etat"]["barres"]] == ["P1", "P2", "P3"])

# ------------------------------------------------------------------ 8. les dessins
v = {x["cle"]: x for x in r0["vues"]}
chk("trois vues : la barre, la section, les courbes χ (une quatrième « Toutes » à partir de deux barres)", list(v) == ["barre", "section", "courbes"])
sv = v["barre"]["svg"]
chk("la barre : deux plans, N_Ed en tête, L = 4 000 coté, L_cr de chaque axe, η de chaque axe dans sa bande",
    "Flambement autour de y-y" in sv and "Flambement autour de z-z" in sv and sv.count("500 kN") == 2 and "L = 4\u202f000" in sv
    and sv.count("L<tspan") >= 2 and "η = 45 %" in sv and "η = 63 %" in sv)
chk("la barre est à l'échelle : sa largeur dans chaque plan est celle du profilé (h = 190, b = 200)",
    'width="190"' in sv and 'width="200"' in sv)
pts = DF.mode_norme("EA", 0, 400)
chk("mode encastré–articulé : w(0) = 0, w'(0) = 0, w(1) = 0, max = 1", abs(pts[0][1]) < 1e-12 and abs(pts[1][1] - pts[0][1]) < 1e-3
    and abs(pts[-1][1]) < 1e-9 and abs(max(w for _, w in pts) - 1) < 1e-12)
x0, x1 = DF.troncon_lcr("EA", 0)
chk("mode encastré–articulé : la part entre points d'inflexion vaut 0,699·L (tan βL = βL)", abs((x1 - x0) - 0.6992) < 1e-3)
chk("mode bi-articulé à n maintiens : n + 1 demi-ondes", all(abs(DF.mode("AA", 2, k / 3)) < 1e-12 for k in range(4)))
chk("mode encastré–encastré et console : 1 − cos 2πξ, 1 − cos πξ/2", abs(DF.mode("EE", 0, 0.5) - 2) < 1e-12 and abs(DF.mode("EL", 0, 1.0) - 1) < 1e-12)
sv = v["section"]["svg"]
chk("la section : contour, hachures, axes y et z avec leur courbe, cotes h et b, fiche du calcul", "y · courbe b" in sv and "z · courbe c" in sv
    and "h = 190" in sv and "b = 200" in sv and "LE CALCUL" in sv and sv.count("<line") > 40)
sv = v["courbes"]["svg"]
chk("les courbes χ : cinq courbes, Euler, le point de chaque axe", sv.count("α = ") == 5 and "Euler" in sv and sv.count('r="7"') == 2
    and "0,878" in sv and "0,628" in sv)
chk("les courbes χ ne se disent pas « à l'échelle » (« echelle »: false, lu par site/web/js/etabli.js) ; la barre et la section si",
    v["courbes"].get("echelle") is False and v["barre"].get("echelle", True) and v["section"].get("echelle", True)
    and "v.echelle !== false" in open(os.path.join(RACINE, "site", "web", "js", "etabli.js"), encoding="utf-8").read())
e = modif(e0, ["famille"], "L")["etat"]
r = req(action="calculer", etat=e)
chk("les noms du catalogue (« 100x100x10 ») s'écrivent « 100×100×10 » à l'écran, la clé reste celle du catalogue",
    "×" in r["vues"][1]["titre"] and "x" not in r["vues"][1]["titre"].split()[-1] and r["etat"]["barres"][0]["profil"].count("x") == 2)

# ------------------------------------------------------------------ 9. échappement, sécurité, fichier
PIEGE = '<img src=x onerror=alert(1)>'
r = modif(e0, ["nom"], PIEGE)
frag = r["resultat"] + r["volets"] + r["donnees"] + r["elements"] + "".join(x["svg"] for x in r["vues"])
chk("un nom piégé est échappé partout (résultat, fenêtre, données, éléments, dessins)", "<img" not in frag and "&lt;img" in frag)
tout0 = r0["resultat"] + r0["volets"] + r0["donnees"] + r0["elements"] + "".join(x["svg"] for x in r0["vues"])
chk("aucun fragment ne porte de script ni de gestionnaire « on…= »", not re.search(r"<script|\son[a-z]+\s*=", tout0, re.I))
stab = json.loads(json.dumps(dup["etat"]))
chk("la note survit à un aller-retour .stab", F.normaliser(stab) == dup["etat"])
for abime in ({"barres": [{"profil": "HEA 999"}]}, {"barres": [{"N": "x"}]}, {"barres": "abîmé"}, [1, 2], {"barres": [], "actif": 5}):
    out = json.loads(R.traiter(json.dumps({"module": "flambement", "action": "calculer", "etat": abime})))
    chk(f"note abîmée {str(abime)[:40]} : réponse lisible", isinstance(out, dict) and (out.get("ok") or out.get("erreur")))
chk("une barre de profilé inconnu est refusée avec son nom", "HEA 999" in json.loads(R.traiter(json.dumps(
    {"module": "flambement", "action": "calculer", "etat": {"barres": [{"profil": "HEA 999"}]}})))["erreur"])
r = req(action="calculer", etat={"version": 1, "barres": [EN.nouvelle_barre()], "actif": "x"})
chk("un « actif » illisible ramène à la première barre", r.get("ok") and r["etat"]["actif"] == 0)

# ------------------------------------------------------------------ 10. les chiffres AFFICHÉS = un calcul à la main indépendant
# Le module n'a pas d'écran Streamlit de référence (« Page en construction »). Les chiffres que le SITE affiche sont donc
# comparés à un calcul refait ici de bout en bout, SANS le moteur (catalogue brut, Tableaux 3.1, 5.2, 6.1, 6.2, EN 1993-1-5
# §4.4), et χ, λ̄ à ceux d'ec3_buckling_chi, la seule fonction de flambement de l'application Streamlit
# (modules/choix_profile.py, chargée avec un faux « streamlit »), avec la même aire, la même courbe et le même f_y.
import importlib.util  # noqa: E402
import types  # noqa: E402

from acier.profiles import catalogue as CAT  # noqa: E402

P_CAT = CAT.par_nom()
FY_MAIN = {"S235": (235, 215), "S275": (275, 255), "S355": (355, 335), "S460": (460, 430)}
ALPHA_MAIN = {"a0": .13, "a": .21, "b": .34, "c": .49, "d": .76}
LAMINES_I = {"IPE", "IPE A", "IPE O", "IPN", "HEA", "HEB", "HEM", "HEAA", "HL", "HD", "HP"}


def a_la_main(nom, nu, N, L, ky, kz):
    """Le calcul d'une barre bi-axe, écrit sans le moteur."""
    p = P_CAT[nom]
    f = p["famille"]
    t = max(p.get("tf") or 0, p.get("tw") or 0, p.get("t") or 0)
    fy = FY_MAIN[nu][0] if t <= 40 else FY_MAIN[nu][1]
    eps = math.sqrt(235 / fy)
    A = p["A"] * 100

    def classe_de(ct, lims):
        return next((i for i, lim in enumerate(lims, 1) if ct <= lim * eps), 4)

    def rho(lp, interne):
        if lp <= (0.673 if interne else 0.748):
            return 1.0
        return min(1.0, (lp - (0.22 if interne else 0.188)) / lp ** 2)
    if f in LAMINES_I or f in ("UPN", "UPE"):
        tw, tf = p["tw"], p["tf"]
        r_ = p.get("r") or p.get("r1")                              # congé de raccordement (r1 pour les U)
        cf = (p["b"] - tw) / 2 - r_ if f in LAMINES_I else p["b"] - tw - r_
        cw = p["h"] - 2 * tf - 2 * r_
        classe = max(classe_de(cf / tf, (9, 10, 14)), classe_de(cw / tw, (33, 38, 42)))
        Aeff = A
        if classe == 4:
            n_sem = 4 if f in LAMINES_I else 2
            Aeff = (A - n_sem * (1 - rho(cf / tf / (28.4 * eps * math.sqrt(.43)), False)) * cf * tf
                    - (1 - rho(cw / tw / (28.4 * eps * 2), True)) * cw * tw)
    elif f in ("SHS", "RHS", "SHS-F", "RHS-F"):
        tt = p["t"]
        cs = (p["b"] - 3 * tt, p["h"] - 3 * tt)                   # EN 1993-1-5 §4.4(2) : b − 3t
        classe = max(classe_de(c / tt, (33, 38, 42)) for c in cs)
        Aeff = A - (sum(2 * (1 - rho(c / tt / (28.4 * eps * 2), True)) * c * tt for c in cs) if classe == 4 else 0)
    elif f in ("CHS", "CHS-F"):
        dt = p["d"] / p["t"]
        classe = next((i for i, lim in enumerate((50, 70, 90), 1) if dt <= lim * eps ** 2), 4)
        Aeff = A
    else:
        h, bb, tt = p["h"], p["b"], p["t"]
        classe = 3 if (h / tt <= 15 * eps and (bb + h) / (2 * tt) <= 11.5 * eps) else 4
        Aeff = A - ((1 - rho(h / tt / (28.4 * eps * math.sqrt(.43)), False)) * (h + bb) * tt if classe == 4 else 0)
    s460 = nu == "S460"
    if f in LAMINES_I:
        if p["h"] / p["b"] > 1.2:
            cy, cz = (("a0", "a0") if s460 else ("a", "b")) if p["tf"] <= 40 else (("a", "a") if s460 else ("b", "c"))
        else:
            cy, cz = (("a", "a") if s460 else ("b", "c")) if p["tf"] <= 100 else (("c", "c") if s460 else ("d", "d"))
    elif f in ("SHS", "RHS", "CHS"):
        cy = cz = "a0" if s460 else "a"
    elif f in ("SHS-F", "RHS-F", "CHS-F", "UPN", "UPE"):
        cy = cz = "c"
    else:
        cy = cz = "b"
    Iy, Iz = (p["Iu"], p["Iv"]) if f in ("L", "LI") else (p["Iy"], p["Iz"])
    out = {"classe": classe, "fy": fy, "Aeff": Aeff, "NcRd": Aeff * fy / 1000}
    for a, I, k, crb in (("y", Iy, ky, cy), ("z", Iz, kz, cz)):
        Lcr = k * L * 1000
        Ncr = math.pi ** 2 * 210000 * I * 1e4 / Lcr ** 2
        lb = math.sqrt(Aeff * fy / Ncr)
        ph = .5 * (1 + ALPHA_MAIN[crb] * (lb - .2) + lb ** 2)
        chi = min(1, 1 / (ph + math.sqrt(ph ** 2 - lb ** 2)))
        out[a] = {"courbe": crb, "Lcr": Lcr / 1000, "I": I, "lb": lb, "chi": chi, "Nb": chi * Aeff * fy / 1000}
    out["taux"] = max(N / out["y"]["Nb"], N / out["z"]["Nb"], N / out["NcRd"])
    return out


def ecran_choix_profile():
    st = types.ModuleType("streamlit")
    st.cache_data = lambda f=None, **k: f if f else (lambda g: g)
    ancien = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st
    try:
        spec = importlib.util.spec_from_file_location("ecran_choix_profile_site", os.path.join(RACINE, "modules", "choix_profile.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        if ancien is None:
            del sys.modules["streamlit"]
        else:
            sys.modules["streamlit"] = ancien
    return m


ECRAN = ecran_choix_profile()
APPUIS_K = {1.0: "AA", 0.7: "EA", 0.5: "EE", 2.0: "EL"}
CAS_MAIN = [  # profilé, nuance, N_Ed (kN), L (m), k_y, k_z — courants, extrêmes, chaque famille, chaque classe, chaque courbe
    ("HEA 200", "S235", 500, 4.0, 1.0, 1.0), ("HEA 200", "S235", 500, 4.0, 1.0, 0.5), ("HEB 300", "S355", 2500, 6.0, 0.7, 1.0),
    ("IPE 400", "S355", 900, 7.0, 0.7, 0.5), ("IPE 300", "S275", 300, 5.0, 1.0, 1.0), ("HEM 1000", "S355", 5000, 12.0, 1.0, 0.5),
    ("HL 920x588", "S460", 8000, 10.0, 1.0, 1.0), ("IPE 600", "S460", 1500, 8.0, 1.0, 0.5), ("HEB 400", "S460", 3000, 9.0, 2.0, 1.0),
    ("UPN 200", "S235", 150, 3.0, 1.0, 1.0), ("UPE 300", "S355", 400, 4.0, 1.0, 0.5),
    ("L 100x100x10", "S235", 50, 2.0, 1.0, 1.0), ("L 100x65x7", "S355", 40, 1.5, 1.0, 1.0), ("L 180x180x13", "S235", 60, 4.0, 2.0, 0.5),
    ("CHS 168.3x4", "S355", 300, 5.0, 1.0, 1.0), ("CHS 168.3x4", "S460", 300, 5.0, 1.0, 1.0),
    ("SHS 160x160x5 (froid)", "S235", 500, 4.0, 0.7, 1.0), ("RHS 200x100x5", "S355", 400, 3.5, 1.0, 1.0),
    ("RHS 200x100x5 (froid)", "S460", 300, 3.5, 1.0, 0.5), ("SHS 350x350x10 (froid)", "S460", 3000, 6.0, 1.0, 1.0),
    ("HEA 200", "S235", 1, 0.3, 1.0, 1.0), ("IPE 80", "S235", 20, 12.0, 2.0, 2.0), ("HEB 300", "S355", 3294.3, 5.0, 1.0, 1.0),
]
couverts = {"classe": set(), "courbe": set(), "famille": set()}
for cas in CAS_MAIN:
    nom, nu, N, L, ky, kz = cas
    ref = a_la_main(*cas)
    e = copy.deepcopy(e0)
    b_ = e["barres"][0]
    b_.update(profil=nom, nuance=nu, N=float(N), L=float(L))
    b_["axes"]["y"]["appuis"], b_["axes"]["z"]["appuis"] = APPUIS_K[ky], APPUIS_K[kz]
    r = req(action="calculer", etat=e)
    vol, tab = texte(r["volets"]), texte(r["volets"].split("ck-sub")[0])
    attendu = [f"{C.fr(ref[a]['Nb'], 1)} kN" for a in ("y", "z")] + [f"{C.fr(ref['NcRd'], 1)} kN", f"courbe {ref['y']['courbe']}",
                                                                      f"courbe {ref['z']['courbe']}", f"fy = {ref['fy']} N/mm²"]
    attendu += [f"λ = {C.fr(ref[a]['lb'], 3)} · χ = {C.fr(ref[a]['chi'], 3)}" for a in ("y", "z")]
    attendu = [x.replace("\u202f", " ") for x in attendu]
    manque = [x for x in attendu if x not in tab and x not in vol]
    # ec3_buckling_chi de l'écran Streamlit, avec la même aire (A_eff), la même inertie, la même courbe, le même f_y
    ecran = [ECRAN.ec3_buckling_chi(ref["Aeff"] / 100, ref[a]["I"], ref[a]["Lcr"], fy=ref["fy"], alpha=ALPHA_MAIN[ref[a]["courbe"]]) for a in ("y", "z")]
    ecart_ecran = [a for a, x in zip("yz", ecran) if abs(x["chi"] - ref[a]["chi"]) > 1e-12 or abs(x["lambdabar"] - ref[a]["lb"]) > 1e-12]
    t_aff = r["resume"]["taux"]
    cl = ref["classe"] if not nom.startswith("L ") or ref["classe"] == 4 else "1 à 3"
    chk(f"chiffres affichés = calcul à la main = ec3_buckling_chi : {nom} {nu}, N {C.nb(N)} kN, L {C.nb(L)} m, k {C.nb(ky)}/{C.nb(kz)} "
        f"(classe {cl}, η {t_aff} %)", not manque and not ecart_ecran and t_aff == C.pct(ref["taux"]) and f"classe {cl}" in tab,
        (manque, ecart_ecran, t_aff, C.pct(ref["taux"])))
    couverts["classe"].add(ref["classe"])
    couverts["courbe"] |= {ref["y"]["courbe"], ref["z"]["courbe"]}
    couverts["famille"].add(SE.FABRICATION[P_CAT[nom]["famille"]])
chk(f"les {len(CAS_MAIN)} cas couvrent les classes 1 à 4, les courbes a0, a, b, c et les cinq fabrications",
    couverts["classe"] == {1, 2, 3, 4} and couverts["courbe"] >= {"a0", "a", "b", "c"} and len(couverts["famille"]) == 5, couverts)
r = req(action="calculer", etat=dict(e0, barres=[dict(e0["barres"][0], profil="HEB 300", nuance="S355", N=3294.3, L=5.0)]))
chk("pile à la limite (η = 99,99… %) : vert foncé, « Conforme, attention », jamais arrondi à un taux rouge", r["resume"]["niveau"] == "f"
    and r["resume"]["taux"] == 100 and '<section class="et-bloc warn"' in r["resultat"] and "attention" in r["resultat"], r["resume"])

# ------------------------------------------------------------------ 11. saisies absurdes : refusées proprement, jamais d'exception
for chemin, v, mot in ((["L"], "1e200", "comprise entre"), (["L"], "1e-300", "comprise entre"), (["L"], "150", "100 m"), (["N"], "1e300", "au plus"),
                       (["N"], "2 000 000", "au plus"), (["axes", "y", "k"], "1e200", "compris entre"), (["axes", "y", "Lcr"], "1e200", "comprise entre"),
                       (["axes", "y", "Lcr"], "1e-300", "comprise entre"), (["N"], "inf", "illisible"), (["N"], "nan", "illisible"),
                       (["N"], PIEGE, "illisible")):
    e = copy.deepcopy(e0)
    if chemin[-1] in ("k", "Lcr"):
        e = modif(e, ["axes", "y", "appuis"], chemin[-1] if chemin[-1] == "k" else "L")["etat"]
    r = modif(e, chemin, v)
    chk(f"saisie absurde refusée avec un message (du TEXTE, posé par textContent) : {'/'.join(chemin)} = {v!r}",
        r.get("ok") and mot in (r.get("message") or "") and r["etat"] == e, (r.get("message"), r.get("erreur")))
for abime in ({"barres": [{"N": 1e308}]}, {"barres": [{"L": 1e200}]}, {"barres": [{"axes": {"y": {"appuis": "L", "Lcr": 1e300}}}]}):
    out = json.loads(R.traiter(json.dumps({"module": "flambement", "action": "calculer", "etat": abime})))
    chk(f"note abîmée {str(abime)[:50]} : refusée avec le motif, pas d'« OverflowError »", not out.get("ok") and "Overflow" not in out.get("erreur", "")
        and "ValueError" in out.get("erreur", ""), out.get("erreur"))
r = req(action="calculer", etat={"version": 1, "barres": [dict(EN.nouvelle_barre(), nom=None)], "actif": 0})
chk("un nom absent (null) se relit vide, pas « None »", r["etat"]["barres"][0]["nom"] == "")
for L, prof in ((0.01, "HEB 300"), (100.0, "IPE 80"), (0.3, "HEM 1000")):
    e = copy.deepcopy(e0)
    e["barres"][0].update(profil=prof, L=L)
    r = req(action="calculer", etat=e)
    sv = r["vues"][0]["svg"]
    vb = [float(x) for x in re.search(r'viewBox="([^"]+)"', sv).group(1).split()]
    chk(f"barre de {C.nb(L)} m en {prof} : le dessin se fait, ses textes restent lisibles (≥ 1/80 de la largeur du dessin)",
        r.get("ok") and min(float(x) for x in re.findall(r'<text[^>]*font-size="([0-9.]+)"', sv)) >= vb[2] / 80, vb)

# ------------------------------------------------------------------ 12. dessins : la cornière dans ses plans principaux, la phrase
e = modif(e0, ["famille"], "L")["etat"]
e["barres"][0]["profil"] = "L 100x100x10"
r = req(action="calculer", etat=e)
larg = [float(x) for x in re.findall(r'<rect x="[^"]+" y="0" width="([0-9.]+)" height="4000" fill="' + DF.ACIER, r["vues"][0]["svg"])]
chk("cornière L 100×100×10 : autour de u-u la barre a sa largeur selon v (≈ √2·h = 141 mm), autour de v-v selon u (≈ 71 mm + t)",
    len(larg) == 2 and 135 < larg[0] < 142 and 70 < larg[1] < 80, larg)
TETE = re.search(r'<div class="et-rp">.*?</div>', r0["resultat"]).group(0)
chk("la tête du résultat est courte (le nom, le profilé, la nuance, N_Ed, L) ; plus de phrase sous le titre de la page",
    "L<sub>cr" not in TETE and "500 kN" in texte(TETE) and "4,00 m" in texte(TETE) and "HEA 200 · S235" in texte(TETE) and r0["phrase"] == "")
e = copy.deepcopy(e0)
e["barres"][0].update(profil="SHS 350x350x10 (froid)", nuance="S460", N=3000.0, L=6.0)
r = req(action="calculer", etat=e)
chk("tube formé à froid : la fenêtre écrit la paroi c = b − 3t, substituée, et son λ̄_p indicé b / h", "(b − 3·t) / t" in texte(r["volets"])
    and "(350 − 3 × 10) / 10" in texte(r["volets"]) and "<sub>p,b</sub>" in r["volets"] and "paroib" not in r["volets"])
r = modif(e0, ["axes", "y", "appuis"], "EA")
chk("sous chaque menu d'appuis, k et L_cr rappelés (le libellé du menu est coupé sur une tablette en portrait)",
    "k = 0,7 · L<sub>cr</sub> = 2,80 m" in r["donnees"] and "k = 1,0 · L<sub>cr</sub> = 4,00 m" in r["donnees"])
css = open(os.path.join(RACINE, "site", "web", "css", "modules", "flambement.css"), encoding="utf-8").read()
chk("les quatre nuances en segments peuvent rétrécir (min-width: 0) : « S460 » ne sort plus du cadre Données étroit",
    ".rows .seg.full button { min-width: 0;" in css)

# ------------------------------------------------------------------ 13. l'établi compact (la mise en forme des poutres, 11/11/2026)
d = r0["donnees"]
chk("la colonne de gauche est celle de l'établi compact : la ligne de la barre ouverte, puis ses rubriques",
    'class="et-arbre et-arbre-c" data-barre="0"' in d and d.count('data-chemin="nom"') == 1)
RUBRIQUES = re.findall(r'<div class="sec( open)?" data-sec="([^"]+)"( data-secondaire="1")?>', d)
chk("cinq rubriques — la barre (seule ouverte), l'effort, chaque axe, les hypothèses (secondaire, rangée sous « Options »)",
    RUBRIQUES == [(" open", "br", ""), ("", "ef", ""), ("", "axy", ""), ("", "axz", ""), ("", "me", ' data-secondaire="1"')], RUBRIQUES)
chk("le nom de la barre ne se saisit plus dans une rubrique : il est sur sa ligne, en tête de la colonne",
    'data-chemin="nom"' not in d.split('<div class="sec')[1] and ">Nom<" not in d)
chk("chaque rubrique repliée dit l'essentiel : le profilé et la nuance, N_Ed et L, les appuis et L_cr de chaque axe, γ_M0 et γ_M1",
    all(x in texte(d) for x in ("HEA 200 · S235", "NEd 500 kN · L 4,00 m", "Articulée – articulée · Lcr 4,00 m", "γM0 1,0 · γM1 1,0 · ANB")), texte(d)[:400])
chk("la tête du résultat : le nom de la barre, puis le profilé, la nuance, N_Ed et L", '<div class="et-rp"><b>P1</b><span>HEA 200 · S235 · N<sub>Ed</sub> = 500 kN · L = 4,00 m</span></div>' in res, res[:300])
BLOC = re.search(r'<section class="et-bloc (\w+)" data-theme="0">.*?</section>', res, re.S)
chk("un bloc « Vérifications » : son état et son taux (ceux du verdict), le rappel du matériau, la fenêtre « tout le calcul »",
    BLOC and BLOC.group(1) == "ok" and "Vérifications" in BLOC.group(0) and "Conforme · 63 %" in texte(BLOC.group(0))
    and "fy = 235 N/mm² · classe 1 en compression · γM1 = 1,0" in texte(BLOC.group(0)) and 'class="et-tout" data-calcul="s0"' in BLOC.group(0))
CMP = [texte(x) for x in re.findall(r'<span class="cmp">(.*?)</span>', res)]
chk("chaque ligne dit ce qu'elle compare : N_Ed face à la résistance de l'axe et sa courbe, puis à la résistance de la section",
    CMP == [f"NEd 500 ≤ Nb,{a},Rd {texte(C.fr(m0['axes'][a]['NbRd'], 1))} kN · courbe {m0['axes'][a]['courbe']}" for a in ("y", "z")]
    + [f"NEd 500 ≤ Nc,Rd {texte(C.fr(m0['section']['NcRd'], 1))} kN · classe 1"] and "794,0 kN · courbe c" in CMP[1] and "1 265,1" in CMP[2], CMP)
chk("la longueur de flambement de chaque axe est sur sa ligne ; la ligne gouvernante est marquée",
    res.count('<span class="prof"><span>L<sub>cr</sub> 4,00') == 2 and res.count('class="el vc gov"') == 1
    and re.search(r'class="el vc gov"[^>]*><div class="l1"><b>Flambement z-z</b>', res))
chk("le verdict ouvre la même fenêtre ; il n'y a qu'UNE fenêtre de calcul", '<div class="verdict" data-calcul="s0"' in res and r0["volets"].count('id="ck-') == 1)
chk("chaque vue porte son en-tête technique : le titre, le profilé et la nuance, L, la longueur de flambement et la courbe de chaque axe ; "
    "à droite l'effort N_Ed (une fois, dans la bande)",
    all('class="et-ent"' in x.get("entete", "") for x in r0["vues"]) and "<b>BARRE</b><span>· HEA 200 · S235</span>" in r0["vues"][0]["entete"]
    and "COURBES DE FLAMBEMENT" in r0["vues"][2]["entete"]
    and all("L = 4,00 m" in texte(x["entete"]) and "y-y : Lcr 4,00 m, courbe b · z-z : Lcr 4,00 m, courbe c" in texte(x["entete"])
            and x["entete"].count('<div class="et-eff"><span><i>N</i><sub>Ed</sub><b>500,0</b><small>kN</small></span></div>') == 1 for x in r0["vues"]),
    r0["vues"][0].get("entete"))
chk("aucun en-tête n'écrit un indice brut (« N_Ed », « L_cr », « t_f ») : ses lignes sont du texte", not any("_" in texte(x["entete"]) for x in r0["vues"]),
    [texte(x["entete"]) for x in r0["vues"]])
chk("plus de cadre « Dans cette note » ni de phrase : les barres sont dans la liste de la ligne de l'élément", r0["elements"] == "" and r0["phrase"] == "")
n2 = req(action="ajouter", etat=e0)
chk("deux barres : la liste les nomme avec leur profilé et leur longueur, et la note dit son statut", "· HEA 200 · 4,00" in texte(n2["donnees"])
    and texte(n2["donnees"]).count("HEA 200 · 4,00") >= 2 and "Conforme" in n2["statut_html"], n2["statut_html"])
n3 = req(action="dupliquer", etat=req(action="ajouter", etat=n2["etat"])["etat"], index=0)
chk("copier P1 (par son rang) : la copie s'insère juste après elle, et s'ouvre", [x["nom"] for x in n3["etat"]["barres"]] == ["P1", "P1 (copie)", "P2", "P3"]
    and n3["etat"]["actif"] == 1, [x["nom"] for x in n3["etat"]["barres"]])
n4 = req(action="supprimer", etat=n3["etat"], index=0)
chk("supprimer une barre AVANT l'ouverte : l'ouverte reste la même (son rang recule)", [x["nom"] for x in n4["etat"]["barres"]] == ["P1 (copie)", "P2", "P3"]
    and n4["etat"]["actif"] == 0)
n5 = req(action="supprimer", etat=req(action="activer", etat=n3["etat"], index=3)["etat"], index=3)
chk("supprimer la dernière, ouverte : la précédente s'ouvre", n5["etat"]["actif"] == 2 and len(n5["etat"]["barres"]) == 3)
rx = req(action="calculer", etat=dict(e0, barres=[dict(e0["barres"][0], N=5000.0)]))
CMPX = [texte(x) for x in re.findall(r'<span class="cmp">(.*?)</span>', rx["resultat"])]
chk("une vérification dépassée s'écrit « > », jamais « ≤ »", all(" > " in x and "≤" not in x for x in CMPX) and len(CMPX) == 3, CMPX)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
