# -*- coding: utf-8 -*-
"""L'API du site pour les dalles béton armé (api_web/dalle.py), vérifiée en Python ordinaire.

- le résultat affiché est celui du moteur beton/dalle : une ligne par vérification, son taux, sa couleur, la
  vérification gouvernante, section par section, et le résumé de la note (la dalle la plus chargée) ;
- une vérification que le moteur déclare non conforme est rouge même sous 100 % (acier excessif, d utile ≤ 0) ;
- la fenêtre de calcul : une ligne par vérification, et chaque formule « symbole = expression = valeurs = résultat »
  porte le résultat du moteur ;
- chaque champ pointe un endroit réel de la dalle ; une saisie hors du domaine de l'écran Streamlit est REFUSÉE avec un
  message (l'écran la ramenait en silence à une autre valeur) et la dalle reste intacte ;
- les boutons ajouter / copier / retirer (sections, renforts), l'axe et le centre de gravité automatiques ou imposés,
  γs et le jeu communs à la note, le verrou ;
- le sélecteur de la nappe de base trié par section au mètre, la retenue et la plus petite qui tient (celles du
  moteur) ; le choix garde les renforts ; l'étoile des lignes du résultat ;
- les dalles de la note s'ajoutent, se copient, se suppriment (jamais la dernière) ;
- la reprise d'un fichier de l'écran Streamlit (dalle_ba.json) ;
- les dessins à l'échelle (barres, cotes, vue partielle) ; un nom saisi est échappé partout.

    python3 tests/test_api_web_dalle.py
"""
import copy
import json
import os
import random
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dalle as P  # noqa: E402
from api_web import dessin_dalle as DD  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton import treillis as TR  # noqa: E402
from beton.dalle import entree as EN  # noqa: E402
from beton.dalle import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(module="dalle", action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


def niveau_de(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def dalle_de(etat, i=None):
    d = etat["dalles"][etat["actif"] if i is None else i]
    return {k: v for k, v in d.items() if k != "verrou"}


LIGNE = re.compile(r'<div class="el( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>(.*?)</div><div class="l2">.*?<span class="pct (\w)">([^<]+)</span>', re.S)

# ------------------------------------------------------------------ 1. une note neuve
r = req(module="dalle", action="nouveau")
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
attendu = EN.nouvelle_dalle()
chk("la dalle neuve est celle de l'écran Streamlit (bande 100 × 20, C30/37, B500, principale Y, treillis 10/10/100/100)",
    dalle_de(r["etat"]) == attendu and r["etat"]["dalles"][0]["verrou"] is False, dalle_de(r["etat"]))
ref = MO.verifier(attendu)
chk("le taux de la note est celui du moteur, arrondi dans sa bande", r["resume"]["taux"] == C.pct(ref["taux_max"]), (r["resume"], ref["taux_max"]))
chk("deux vues par section : la coupe et le plan des nappes", [v["cle"] for v in r["vues"]] == ["c0", "p0"], [v["cle"] for v in r["vues"]])

# ------------------------------------------------------------------ 2. le cas de tests/test_dalle_bidir.py, saisi comme à l'écran
for ch, x in (("sections/0/Mx_inf", "28"), ("sections/0/Mx_sup", "12"), ("sections/0/My_inf", "35"), ("sections/0/My_sup", "9"),
              ("sections/0/V", "60")):
    r = saisir(r["etat"], ch, x)
r = saisir(r["etat"], "sections/0/faces/inf_y/couches/+", "1")
r = saisir(r["etat"], "sections/0/faces/inf_y/couches/1/dist", "8")
r = saisir(r["etat"], "sections/+", "1")
r = saisir(r["etat"], "sections/1/Mx_sup", "18,5")
r = saisir(r["etat"], "sections/1/My_sup", "24")
r = saisir(r["etat"], "sections/1/V", "75")
etat = r["etat"]
d = dalle_de(etat)
chk("« Ajouter une section » crée la section B ; « Ajouter un renfort » pose 3 Ø12 (le défaut de l'écran)",
    [s["nom"] for s in d["sections"]] == ["A", "B"] and d["sections"][0]["faces"]["inf_y"]["couches"][1] == dict(EN.COUCHE_RENFORT, dist=8.0),
    d["sections"][0]["faces"]["inf_y"])
ref = MO.verifier(d)
g = ref["sections"][0]["grandeurs"]
chk("les chiffres de test_dalle_bidir : y_G = 4,86 cm et As,req inf. Y = 770 mm² (renfort posé à 8 cm)",
    round(g["yG_inf_y"], 2) == 4.86 and round(g["As_req_inf_y"]) == 770, (g["yG_inf_y"], g["As_req_inf_y"]))
lignes = LIGNE.findall(r["resultat"])
attendues = [(v, s) for s in ref["sections"] for v in s["verifications"]]
chk(f"une ligne de résultat par vérification du moteur ({len(attendues)})", len(lignes) == len(attendues), (len(lignes), len(attendues)))
chk("chaque ligne porte le nom, le taux et la couleur de sa vérification",
    all(nom == C.esc(P.nom_ligne(v)) and n == niveau_de(v) and t == f"{C.pc(C.pct(v['taux']))}{C.NBSP}%"
        for (_, nom, _p, n, t), (v, _s) in zip(lignes, attendues)), lignes[:3])
chk("l'ordre de l'écran : hauteur, inf. principale, inf. secondaire, sup. principale, sup. secondaire, tranchant",
    [x[1] for x in lignes[:6]] == ["Hauteur", "Inférieure Y", "Inférieure X", "Supérieure Y", "Supérieure X", "Effort tranchant"],
    [x[1] for x in lignes[:6]])
gv = ref["gouvernant"]
i_gov = [k for k, x in enumerate(lignes) if x[0]]
chk("la ligne gouvernante est celle du moteur, et elle seule", len(i_gov) == 1 and attendues[i_gov[0]][0]["cle"] == gv["cle"]
    and attendues[i_gov[0]][1]["nom"] == gv["nom"], (i_gov, gv))
v_g = next(v for v in ref["sections"][gv["section"]]["verifications"] if v["cle"] == gv["cle"])
verdict = re.search(r'<div class="verdict">(.*?)</div></div>', r["resultat"], re.S).group(1)
val_g, lim_g = P._comparaison(v_g, ref["sections"][gv["section"]])
chk("le verdict : le taux maximal, la vérification gouvernante et sa comparaison « valeur ≤ limite » (ici τ ≤ τ_adm,I)",
    f"{C.pc(C.pct(ref['taux_max']))}{C.NBSP}%" in verdict and f"{val_g} ≤ {lim_g}" in verdict
    and gv["cle"] == "tau" and "τ{0}={0}0,50{0}N/mm² ≤ τ<sub>adm,I</sub>{0}={0}0,56{0}N/mm²".format(C.NBSP) in verdict, verdict)
chk("chaque nappe ouvre le sélecteur de sa couche de base ; hauteur et tranchant n'en ont pas",
    sorted(set(re.findall(r'data-pk="([^"]+)"', r["resultat"]))) == sorted(f"{f}-{i}" for i in range(2) for f in EN.FACES))
chk("chaque ligne ouvre la fenêtre de calcul de sa section", set(re.findall(r'data-calcul="([^"]+)"', r["resultat"])) == {"s0", "s1"}
    and r["volets"].count('class="flyout calcul"') == 2)
chk("le renfort posé à 8 cm se lit sur la ligne : « 10/10/100/100 + 3 Ø12 » (la base, puis le renfort)",
    "<span>10/10/100/100 + 3 Ø12</span>" in lignes[1][2], lignes[1][2])
chk("deux sections : quatre vues (coupe et plan de chacune)", [v["cle"] for v in r["vues"]] == ["c0", "p0", "c1", "p1"])

# la fenêtre de calcul : une ligne par vérification, les formules du moteur substituées
ck = re.search(r'id="ck-s0".*?(?=<div class="flyout calcul"|$)', r["volets"], re.S).group(0)
chk("fenêtre de calcul : une ligne du tableau par vérification de la section", ck.count("<tr class=") == len(ref["sections"][0]["verifications"]))
fx = re.findall(r'<div class="fx">(.*?)</div>', ck, re.S)
n_formules = sum(len(v["formules"]) + 1 for v in ref["sections"][0]["verifications"])
chk(f"fenêtre de calcul : chaque formule du moteur, plus le taux de chaque vérification ({n_formules} lignes)", len(fx) == n_formules, len(fx))
textes = [re.sub(r"<[^>]+>", "", x) for x in fx]
chk("A_s,req = M·10⁶ / (f_yd·0,9·d) = 35 × 10⁶ / (333,3 × 0,9 × 151,4) = 770 mm² (valeurs substituées, comme à la main)",
    "As,req=M·10⁶ / (fyd·0,9·d)=35 × 10⁶ / (333,3 × 0,9 × 151,4)=770 mm²" in textes,
    [t for t in textes if t.startswith("As,req")][:2])
chk("y_G = Σ A_s·e / Σ A_s = (785 × 3,5 + 339 × 8) / (785 + 339) = 4,9 cm",
    any(t.startswith("yG=Σ As·e / Σ As=(785 × 3,5 + 339 × 8) / (785 + 339)") and t.endswith("4,9 cm") for t in textes),
    [t for t in textes if t.startswith("yG")])
res_moteur = [f["resultat"] for v in ref["sections"][0]["verifications"] for f in v["formules"]]
res_site = [re.search(r'<span class="r">([^<]+)</span>', x).group(1) for x in fx if '<span class="r">' in x]
chk("chaque résultat de formule affiché est celui du moteur, à la précision d'usage",
    len(res_site) == len(res_moteur) and all(s.split(" ")[0] == P._n(m, f["unite"]) for s, m, f in
                                             zip(res_site, res_moteur, [f for v in ref["sections"][0]["verifications"] for f in v["formules"]])),
    list(zip(res_site, res_moteur))[:4])

# ------------------------------------------------------------------ 3. rouge sous 100 % quand le moteur dit non conforme
e2 = req(module="dalle", action="nouveau")["etat"]      # le cas ASMAX de la parité : bande de 20 cm, h = 8 cm, Ø25/100
for ch, x in (("b", "20"), ("h", "8"), ("sections/0/Mx_inf", "1"), ("sections/0/My_inf", "1"), ("sections/0/V", "2")):
    e2 = saisir(e2, ch, x)["etat"]
for f, dd, esp in (("inf_x", "25", "100"), ("inf_y", "25", "100"), ("sup_x", "6", "250"), ("sup_y", "6", "250")):
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/type", "barres")["etat"]
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/d", dd)["etat"]
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/esp", esp)["etat"]
r2 = req(module="dalle", action="calculer", etat=e2)
ref2 = MO.verifier(dalle_de(e2))
excessif = [v for s in ref2["sections"] for v in s["verifications"] if "acier_excessif" in v["motifs"] and v["taux"] <= 1]
lignes2 = LIGNE.findall(r2["resultat"])
att2 = [v for s in ref2["sections"] for v in s["verifications"]]
chk("acier excessif (A_s > A_s,max) sous 100 % : la ligne est rouge, comme le moteur la déclare",
    bool(excessif) and all(n == "r" for (_, _, _, n, _), v in zip(lignes2, att2) if v in excessif), [x[3:] for x in lignes2[:4]])

# ------------------------------------------------------------------ 4. les saisies refusées (la dalle reste intacte)
REFUS = [("sections/0/Mx_inf", "-20", "positifs"), ("sections/0/V", "-1", "positifs"), ("b", "15", "entre 20 et 500"),
         ("b", "100,5", "entier"), ("h", "101", "entre 5 et 100"), ("enrobage", "25", "entre 0 et 20"), ("gamma_s", "0,9", "entre 1 et 2"),
         ("jeu_premier_lit", "-0,5", "au moins 0"), ("sections/0/faces/inf_x/couches/0/type", "grille", "inconnu"),
         ("sections/0/faces/inf_x/couches/0/d", "14", "Ø14"), ("sections/0/faces/sup_x/couches/0/treillis", "10/10/abc/100", "Øl/Øt/el/et"),
         ("sections/0/faces/inf_y/couches/1/esp", "20", "entre 25 et 500"), ("sections/0/faces/inf_y/couches/1/n", "51", "entre 1 et 50"),
         ("sections/0/faces/inf_y/couches/1/n", "2,5", "entier"), ("sections/0/faces/inf_y/couches/1/dist", "0", "plus de 0"),
         ("sections/0/faces/inf_y/yG", "-3", "plus de 0"), ("sections/0/faces/inf_y/couches/0/-", "1", "base ne se retire pas"),
         ("sections/0/Mx_inf", "abc", "illisible"), ("fyk", "450", "B400 ou B500"), ("beton", "C16/20", "inconnue"),
         ("dir_principale", "Z", "X ou Y"), ("sections/0/nom", "  ", "vide"), ("sections/9/V", "1", "Saisie refusée")]
for ch, x, mot in REFUS:
    out = req(module="dalle", action="modifier", etat=copy.deepcopy(etat), chemin=ch.split("/"), valeur=x)
    chk(f"refus : {ch} = « {x} » → message ({mot}), dalle intacte",
        out["ok"] and out["message"] and mot in out["message"] and out["etat"] == etat, out.get("message") or out.get("erreur"))
e1 = P.normaliser(copy.deepcopy(etat))
e1["dalles"][0]["sections"] = e1["dalles"][0]["sections"][:1]
out = req(module="dalle", action="modifier", etat=e1, chemin=["sections", "0", "-"], valeur="1")
chk("la dernière section ne se retire pas", out["message"] == "Il faut au moins une section" and out["etat"] == e1, out.get("message"))
e4 = copy.deepcopy(etat)
for _ in range(2):
    e4 = saisir(e4, "sections/0/faces/inf_y/couches/+", "1")["etat"]
out = req(module="dalle", action="modifier", etat=e4, chemin="sections/0/faces/inf_y/couches/+".split("/"), valeur="1")
chk("au plus quatre couches par face (la base et trois renforts) ; à quatre, plus de bouton « Ajouter un renfort »",
    "Au plus 4 couches" in (out["message"] or "") and 'data-chemin="sections/0/faces/inf_y/couches/+"' not in saisir(e4, "nom", "Dalle 1")["donnees"])

# ------------------------------------------------------------------ 5. les effets de bord
r5 = saisir(etat, "sections/0/copie", "1")
d5 = dalle_de(r5["etat"])
c5 = copy.deepcopy(d["sections"][0])
c5["nom"] = "C"
chk("« Copier la section » : une section C avec les sollicitations et les armatures de A", d5["sections"][2] == c5, d5["sections"][2]["nom"])
r5 = saisir(etat, "sections/0/faces/inf_y/couches/1/-", "1")
chk("« Retirer ce renfort » le retire", len(dalle_de(r5["etat"])["sections"][0]["faces"]["inf_y"]["couches"]) == 1)
r5 = saisir(etat, "sections/0/faces/inf_y/couches/1/dist", "")
chk("un axe vidé revient à l'automatique (null)", dalle_de(r5["etat"])["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] is None)
auto = MO.distance_auto(EN.COUCHE_BASE, 3.0, 0.0)
r5 = saisir(etat, "sections/0/faces/inf_x/couches/0/dist", C.nb(auto))
chk(f"l'axe automatique affiché ({C.nb(auto)} cm), réécrit, reste automatique (comme le champ de l'écran)",
    dalle_de(r5["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None)
r5 = saisir(etat, "sections/0/faces/inf_x/couches/0/dist", "4,5")
chk("un autre axe est imposé, et le bouton « saisi ↺ » (retour à l'automatique) apparaît", dalle_de(r5["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] == 4.5
    and 'class="dl-raz" data-chemin="sections/0/faces/inf_x/couches/0/dist" data-valeur=""' in r5["donnees"])
r5 = saisir(etat, "sections/0/faces/inf_y/yG", "4,2")
chk("un centre de gravité saisi est imposé (y_G = 4,2 cm dans le calcul)", MO.verifier(dalle_de(r5["etat"]))["sections"][0]["grandeurs"]["yG_inf_y"] == 4.2)
r5 = saisir(r5["etat"], "sections/0/faces/inf_y/yG", "")
chk("… et vidé, il redevient calculé", dalle_de(r5["etat"])["sections"][0]["faces"]["inf_y"]["yG"] is None)
r5 = saisir(etat, "sections/0/faces/inf_y/yG", C.nb(g["yG_inf_y"]))
chk("le centre de gravité calculé affiché, réécrit, reste calculé", dalle_de(r5["etat"])["sections"][0]["faces"]["inf_y"]["yG"] is None)
r5 = saisir(etat, "sections/0/faces/sup_x/couches/0/type", "n_barres")
r5 = saisir(r5["etat"], "sections/0/faces/sup_x/couches/0/n", "7")
c = dalle_de(r5["etat"])["sections"][0]["faces"]["sup_x"]["couches"][0]
chk("le type « n barres » montre le nombre de barres et le Ø ; la couche garde ses autres réglages (comme l'écran)",
    c["type"] == "n_barres" and c["n"] == 7 and c["treillis"] == "10/10/100/100"
    and 'data-chemin="sections/0/faces/sup_x/couches/0/n"' in r5["donnees"], c)

# γs et le jeu : communs à la note
e6 = req(module="dalle", action="ajouter", etat=etat)["etat"]
e6 = saisir(e6, "gamma_s", "1,15")["etat"]
e6 = saisir(e6, "jeu_premier_lit", "0,5")["etat"]
chk("γs et le jeu premier lit s'appliquent à toutes les dalles de la note (paramètres communs de l'écran)",
    all(x["gamma_s"] == 1.15 and x["jeu_premier_lit"] == 0.5 for x in e6["dalles"]))
e7 = req(module="dalle", action="ajouter", etat=e6)["etat"]
chk("une dalle ajoutée prend γs et le jeu de la note, et le premier nom libre",
    e7["dalles"][-1]["gamma_s"] == 1.15 and e7["dalles"][-1]["jeu_premier_lit"] == 0.5 and e7["dalles"][-1]["nom"] == "Dalle 3",
    [x["nom"] for x in e7["dalles"]])

# le verrou
r8 = saisir(etat, "verrou", True)
out = req(module="dalle", action="modifier", etat=r8["etat"], chemin=["sections", "0", "Mx_inf"], valeur="99")
chk("dalle verrouillée : une saisie est refusée, la dalle reste intacte", "verrouillée" in (out["message"] or "") and out["etat"] == r8["etat"])
chk("dalle verrouillée : le cadre Données est en lecture seule (aucun champ, la case pour déverrouiller) et aucun sélecteur",
    'data-sorte="nombre"' not in r8["donnees"] and 'data-chemin="verrou"' in r8["donnees"] and "data-pk=" not in r8["resultat"]
    and C.icon("lock") in r8["elements"])
chk("… et déverrouillée, elle se modifie de nouveau", saisir(r8["etat"], "verrou", False)["etat"] == etat)

# ------------------------------------------------------------------ 6. le sélecteur de la nappe de base
s = req(module="dalle", action="selecteur", etat=etat, role="inf_y-0")
cands = MO.candidats_base(d, 0, "inf_y")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)"', s["html"])
chk(f"sélecteur : les {len(cands)} nappes du moteur, triées par section au mètre", [x[2] for x in rows] == [c["nom"] for c in cands]
    and [c["As_pm"] for c in cands] == sorted(c["As_pm"] for c in cands))
chk("sélecteur : la retenue (✓) et la plus petite qui tient (★) sont celles du moteur",
    [x[2] for x in rows if x[0]] == [c["nom"] for c in cands if c["retenue"]] and [x[2] for x in rows if x[1]] == [c["nom"] for c in cands if c["optimal"]],
    ([x for x in rows if x[0] or x[1]], [c["nom"] for c in cands if c["retenue"] or c["optimal"]]))
s_t = req(module="dalle", action="selecteur", etat=etat, role="inf_y-0", familles=["treillis"])
chk("sélecteur : la puce « Treillis » seule ne garde que les dix treillis de la base",
    re.findall(r'data-choix="([^"]+)"', s_t["html"]) == [f"Treillis {x}" for x in TR.liste_choix()])
opt = next(c for c in cands if c["optimal"])
r9 = req(module="dalle", action="choisir", etat=etat, role="inf_y-0", nom=opt["nom"])
f9 = dalle_de(r9["etat"])["sections"][0]["faces"]["inf_y"]
chk(f"choisir « {opt['nom']} » remplace la couche de base ; le renfort posé à 8 cm reste",
    MO.candidat_nom(f9["couches"][0]) == opt["nom"] and f9["couches"][1] == d["sections"][0]["faces"]["inf_y"]["couches"][1], f9)
chk("… et le taux de la face devient celui que le sélecteur annonçait",
    MO.verifier(dalle_de(r9["etat"]))["sections"][0]["grandeurs"]["besoin_inf_y"] / MO.verifier(dalle_de(r9["etat"]))["sections"][0]["grandeurs"]["As_inf_y"]
    == opt["taux"])
leger = re.findall(r'<span class="leger" title="([^"]+)">', r["resultat"])
chk("l'étoile d'une ligne dit la plus petite nappe de base qui tient, quand ce n'est pas la retenue",
    any(opt["nom"] in x for x in leger), leger[:2])
r10 = req(module="dalle", action="choisir", etat=etat, role="sup_x-1", nom="Ø12/150")
chk("choisir « Ø12/150 » pose des barres Ø12 tous les 150 mm", {k: dalle_de(r10["etat"])["sections"][1]["faces"]["sup_x"]["couches"][0][k]
                                                               for k in ("type", "d", "esp")} == {"type": "barres", "d": 12, "esp": 150})

# ------------------------------------------------------------------ 7. les dalles de la note
a = req(module="dalle", action="ajouter", etat=etat)
chk("« Ajouter une dalle » : Dalle 2 ouverte, neuve", len(a["etat"]["dalles"]) == 2 and a["etat"]["actif"] == 1
    and dalle_de(a["etat"]) == dict(EN.nouvelle_dalle(), nom="Dalle 2"))
cp = req(module="dalle", action="dupliquer", etat=etat)
chk("« Copier » : la dalle ouverte, nommée « (copie) »", dalle_de(cp["etat"]) == dict(d, nom="Dalle 1 (copie)"))
chk("le résumé de la note est sa dalle la plus chargée", cp["resume"]["taux"] == C.pct(ref["taux_max"]))
sp = req(module="dalle", action="supprimer", etat=cp["etat"], index=0)
chk("supprimer la première dalle garde la copie ouverte", len(sp["etat"]["dalles"]) == 1 and sp["etat"]["dalles"][0]["nom"] == "Dalle 1 (copie)")
chk("la dernière dalle ne se supprime pas", len(req(module="dalle", action="supprimer", etat=sp["etat"], index=0)["etat"]["dalles"]) == 1)
lourd = saisir(a["etat"], "sections/0/My_inf", "80")
chk("une seconde dalle plus chargée devient le résumé de la note, nommée dans le gouvernant",
    lourd["resume"]["taux"] == C.pct(MO.verifier(dalle_de(lourd["etat"]))["taux_max"]) and lourd["resume"]["gouvernant"].startswith("Dalle 2"),
    lourd["resume"])

# ------------------------------------------------------------------ 8. le fichier de l'écran Streamlit
V1 = {"dalles": [{"id": 1, "nom": "Plancher", "sections": [{"id": 1, "nom": "A"}]}],
      "values": {"dal1_sec1_M_inf": 28.0, "dal1_sec1_M_sup_raw": "-12,0", "dal1_sec1_V": 55.0, "dal1_sec1_ncouches_inf": 2,
                 "dal1_sec1_arm_type_inf_c2": "Barres", "dal1_sec1_ø_barres_inf_c2": 12, "gamma_s": 1.5}}
f = req(module="dalle", action="modifier", etat=etat, chemin=["fichier_streamlit"], valeur={"nom": "dalle_ba.json", "texte": json.dumps(V1)})
d_f = dalle_de(f["etat"])
chk("un fichier v1 de l'écran Streamlit remplace les dalles de la note (migration : M_inf → Mx,inf, couches → direction X)",
    len(f["etat"]["dalles"]) == 1 and d_f["nom"] == "Plancher" and d_f["sections"][0]["Mx_inf"] == 28.0
    and len(d_f["sections"][0]["faces"]["inf_x"]["couches"]) == 2, d_f["sections"][0]["Mx_inf"])
chk("… et ce que l'écran y remplaçait en silence est dit (M_sup « -12,0 » lu 0)", "lu 0" in (f["message"] or ""), f["message"])
f2 = req(module="dalle", action="modifier", etat=etat, chemin=["fichier_streamlit"], valeur={"nom": "x.json", "texte": "{pas du json"})
chk("un fichier illisible est refusé avec un message, la note reste", "JSON" in (f2["message"] or "") and f2["etat"] == etat, f2.get("message"))
f3 = req(module="dalle", action="modifier", etat=etat, chemin=["fichier_streamlit"],
         valeur={"nom": "x.json", "texte": json.dumps({"values": {"dal1_b": 800}})})
chk("un fichier hors des bornes de l'écran (b = 800 cm) est refusé avec un message", "refusé" in (f3["message"] or "") and f3["etat"] == etat)

# ------------------------------------------------------------------ 9. l'échappement
PIEGE = '<img src=x onerror=alert(1)>'
e9 = saisir(etat, "nom", PIEGE)["etat"]
r9 = saisir(e9, "sections/1/nom", PIEGE)
frag = {k: r9[k] for k in ("resultat", "volets", "donnees", "elements", "phrase")}
frag["dessins"] = "".join(v["svg"] for v in r9["vues"])
chk("un nom de dalle ou de section saisi est échappé partout", not [k for k, h in frag.items() if "<img" in h],
    [k for k, h in frag.items() if "<img" in h])
chk("… et le sélecteur aussi", "<img" not in req(module="dalle", action="selecteur", etat=r9["etat"], role="inf_x-1")["html"])

# ------------------------------------------------------------------ 10. les dessins, à l'échelle
s0 = ref["sections"][0]
svg = DD.svg_coupe("t", s0, ref["dalle"])
n_points = sum(len([x for x in DD.positions(c, 1000.0, j) if c["d"] / 2 <= x <= 1000 - c["d"] / 2])
               for dk in ("x", "y") for cote in ("inf", "sup") for j, c in enumerate(s0["faces"][f"{cote}_{dk}"]["couches"]))
chk(f"coupe : chaque barre coupée est un cercle de son Ø ({n_points} dans les deux coupes)", svg.count("<circle cx") - 12 == n_points
    or svg.count('stroke="#FFFFFF"') == n_points, (svg.count("<circle"), n_points))
chk("coupe : le renfort 3 Ø12 posé à 8 cm est un point de rayon 6 mm à 80 mm du dessous (y = 2,4u + 200 − 80)",
    re.search(r'<circle cx="[\d.]+" cy="([\d.]+)" r="6" fill="#1F5FA8"', svg) is not None)
chk("coupe : les cotes h = 200, b = 1 000 et d = 151,4 (y_G = 4,86 cm)", all(t in svg for t in ("h = 200", "b = 1 000", "d = 151,4")))
chk("« n barres » : n barres également réparties dans la bande (une demi-maille aux bords)",
    DD.positions({"type": "n_barres", "n": 4, "d": 12, "esp": 150}, 1000.0, 1) == [125.0, 375.0, 625.0, 875.0])
chk("barres et treillis : b / e barres, centrées ; un renfort de même pas s'intercale d'une demi-maille",
    DD.positions({"type": "barres", "d": 12, "esp": 150}, 1000.0, 0) == [50.0 + 150 * k for k in range(7)]
    and DD.positions({"type": "barres", "d": 12, "esp": 150}, 1000.0, 1) == [125.0 + 150 * k for k in range(6)])
large = MO.verifier(dict(EN.nouvelle_dalle(), b=500))
chk("une bande de 5 m se dessine en partie (1,20 m) et la cote le dit",
    "1 200 vus sur b = 5 000" in DD.svg_coupe("t", large["sections"][0], large["dalle"])
    and "1 200 vus sur b = 5 000" in DD.svg_plan("t", large["sections"][0], large["dalle"]))
plan = DD.svg_plan("t", s0, ref["dalle"])
chk("plan : les deux nappes, leurs espacements cotés et le repère (principale Y)",
    all(t in plan for t in ("Nappe inférieure", "Nappe supérieure", ">100<", "direction principale : Y")), plan[:200])

# ------------------------------------------------------------------ 11. le site affiche le moteur, sur des dalles tirées au sort
rng = random.Random(20261107)
fautes = []
for k in range(25):
    e = req(module="dalle", action="nouveau")["etat"]
    for ch, x in (("b", rng.choice(("50", "100", "120", "150"))), ("h", rng.choice(("16", "20", "25"))),
                  ("beton", rng.choice(sorted(MO.classes_beton()))), ("fyk", rng.choice(("400", "500"))),
                  ("dir_principale", rng.choice("XY")), ("enrobage", rng.choice(("0", "2,5", "3")))):
        e = saisir(e, ch, x)["etat"]
    for m in EN.MOMENTS + ("V",):
        e = saisir(e, f"sections/0/{m}", C.nb(round(rng.uniform(0, 60), 2)))["etat"]
    f = rng.choice(EN.FACES)
    e = saisir(e, f"sections/0/faces/{f}/couches/0/type", rng.choice(("treillis", "barres", "n_barres")))["etat"]
    rr = req(module="dalle", action="calculer", etat=e)
    rf = MO.verifier(dalle_de(e))
    li = LIGNE.findall(rr["resultat"])
    att = [v for s_ in rf["sections"] for v in s_["verifications"]]
    if len(li) != len(att) or any(t != f"{C.pc(C.pct(v['taux']))}{C.NBSP}%" or n != niveau_de(v) for (_, _, _, n, t), v in zip(li, att)) \
            or rr["resume"]["taux"] != C.pct(rf["taux_max"]):
        fautes.append(k)
chk("25 dalles tirées au sort : chaque ligne affichée porte le taux et la couleur du moteur", not fautes, fautes)

# ------------------------------------------------------------------ 12. ce que la relecture indépendante a trouvé
# a) une dalle NON CONFORME sous 100 % (acier excessif) : le verdict ne nomme pas une vérification conforme
ex = req(module="dalle", action="nouveau")["etat"]
for ch, x in (("sections/0/Mx_inf", "10"), ("sections/0/faces/inf_x/couches/0/type", "barres"),
              ("sections/0/faces/inf_x/couches/0/d", "40"), ("sections/0/faces/inf_x/couches/0/esp", "25")):
    ex = saisir(ex, ch, x)["etat"]
rx = req(module="dalle", action="calculer", etat=ex)
mx = MO.verifier(dalle_de(ex))
vx = {v["cle"]: v for v in mx["sections"][0]["verifications"]}
verdict = re.search(r'<div class="verdict">(.*?)</div></div>', rx["resultat"], re.S).group(1)
chk("acier excessif sous 100 % (Ø40/25 : A_s > A_s,max) : la dalle est non conforme, et le moteur nomme la hauteur (la plus chargée)",
    mx["statut"] == MO.NOK and mx["gouvernant"]["cle"] == "hauteur" and vx["inf_x"]["etat"] == MO.NOK and vx["inf_x"]["taux"] < vx["hauteur"]["taux"] <= 1)
chk("… le verdict nomme la vérification NON CONFORME, en rouge, avec la comparaison qui la rend non conforme (A_s > A_s,max)",
    "inférieure X" in verdict and 'class="chip xl r"' in verdict and f"{C.pc(C.pct(vx['inf_x']['taux']))}{C.NBSP}%" in verdict
    and "A<sub>s,max</sub>{0}={0}8{1}000{0}mm²".format(C.NBSP, C.fr(1000, 0)[1]) in verdict and "acier excessif" in verdict, verdict)
chk("… le résumé de la note la nomme aussi, en rouge", rx["resume"]["niveau"] == "r" and rx["resume"]["gouvernant"].endswith("inférieure X")
    and rx["resume"]["taux"] == C.pct(vx["inf_x"]["taux"]), rx["resume"])
lx = LIGNE.findall(rx["resultat"])
chk("… la ligne dit pourquoi elle est rouge (« acier excessif ») et c'est la ligne gouvernante",
    any(n == "Inférieure X" and "acier excessif" in p and gv_ for gv_, n, p, _, _ in lx), [(x[0], x[1]) for x in lx])
ck = re.search(r'id="ck-s0".*?(?=<div class="flyout calcul"|$)', rx["volets"], re.S).group(0)
chk("… et une seule ligne gouvernante dans le tableau de la fenêtre de calcul (la non conforme)",
    ck.count('<tr class="gov">') == 1 and re.search(r'<tr class="gov"><td class="nm">Inférieure X', ck) is not None)
neuf = req(module="dalle", action="nouveau")
chk("une note neuve (quatre nappes à 38 %) : une seule ligne gouvernante dans le tableau, la première à égalité",
    neuf["volets"].count('<tr class="gov">') == 1 and re.search(r'<tr class="gov"><td class="nm">Inférieure Y', neuf["volets"]) is not None)

# b) l'absurde est refusé avec un message (l'écran le calculait en chiffres démesurés), la dalle reste intacte
for ch, x, mot in (("sections/0/Mx_inf", "200000", "Au plus 100"), ("sections/0/V", "1e9", "Au plus 100"),
                   ("jeu_premier_lit", "150", "au plus 100 cm"), ("sections/0/faces/inf_x/couches/0/dist", "150", "au plus 100 cm"),
                   ("sections/0/faces/inf_x/yG", "1e6", "au plus 100 cm"), ("sections/0/Mx_inf", "1e400", "illisible"),
                   ("verrou", "peut-être", "coché ou décoché"), ("sections/7/Mx_inf", "5", "champ introuvable « sections/7/Mx_inf »"),
                   ("sections/0/faces/inf_q/yG", "3", "champ introuvable")):
    out = req(module="dalle", action="modifier", etat=copy.deepcopy(etat), chemin=ch.split("/"), valeur=x)
    chk(f"refus : {ch} = « {x} » → message ({mot}), dalle intacte",
        out["ok"] and out["message"] and mot in out["message"] and out["etat"] == etat, out.get("message") or out.get("erreur"))
chk("le verrou lit « false » comme décoché (et non comme une chaîne non vide, donc vraie)",
    saisir(etat, "verrou", "false")["etat"]["dalles"][0]["verrou"] is False and saisir(etat, "verrou", True)["etat"]["dalles"][0]["verrou"] is True)
grand = copy.deepcopy(etat)
grand["dalles"][0]["sections"][0]["Mx_inf"] = 1e308
for nom, e_ in (("un moment de 10³⁰⁸ kN·m (le calcul déborderait)", grand), ("« dalles » qui n'est pas une liste", {"dalles": "x"}),
                ("une dalle qui n'est pas un objet", {"dalles": [5]})):
    out = req(module="dalle", action="calculer", etat=e_)
    chk(f".stab abîmé — {nom} : une erreur claire, jamais une exception de Python",
        not out["ok"] and out["erreur"].startswith("ValueError") and "Overflow" not in out["erreur"], out.get("erreur", "")[:120])

vide = req(module="dalle", action="calculer", etat={"dalles": [{"sections": [{"nom": ""}, {"nom": " "}]}]})
chk(".stab dont les sections n'ont pas de nom (impossible à l'écran) : elles prennent les premières lettres libres",
    vide["ok"] and [x["nom"] for x in vide["etat"]["dalles"][0]["sections"]] == ["A", "B"], vide.get("erreur"))

# c) les dessins : la hauteur utile cotée est celle de la face TENDUE ; la légende dit l'axe (et non « e », l'espacement)
ap = req(module="dalle", action="nouveau")["etat"]
for ch, x in (("sections/0/My_sup", "30"), ("sections/0/My_inf", "5"), ("sections/0/Mx_inf", "12"),
              ("sections/0/faces/sup_y/couches/0/type", "barres"), ("sections/0/faces/sup_y/couches/0/d", "12")):
    ap = saisir(ap, ch, x)["etat"]
ra = MO.verifier(dalle_de(ap))
ca = DD.svg_coupe("t", ra["sections"][0], ra["dalle"])
coupes = ca.split("Coupe ⊥")
chk("coupe ⊥ Y (M_y,sup > M_y,inf) : d = 160 de la nappe supérieure Y (axe Ø12 à 40 mm), et non 165 de l'inférieure",
    "d = 160" in coupes[1] and "d = 165" not in coupes[1], re.findall(r"d = \d+", ca))
chk("coupe ⊥ X (M_x,inf seul) : d = 165 de la nappe inférieure X", "d = 165" in coupes[2], re.findall(r"d = \d+", coupes[2]))
chk("la légende des couches dit « axe à 35 » (l'espacement s'écrit Ø/e : « Ø12/150 »)", "axe à 35" in ca and "e = 35" not in ca)
fo = saisir(ap, "nom", "Dalle 1")["donnees"]
chk("une couche de treillis dit ses fils porteurs « Ø10/100 », son axe est nommé et « auto »",
    ">Ø10/100<" in fo and ">auto<" in fo and fo.count('class="dl-axe"') == 4, fo[:120])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)
