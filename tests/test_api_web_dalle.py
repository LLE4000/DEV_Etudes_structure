# -*- coding: utf-8 -*-
"""L'API du site pour les dalles béton armé (api_web/dalle.py), vérifiée en Python ordinaire.

- l'écran est l'établi COMPACT de la poutre (api_web/etabli_compact.py, api_web/plaque_compact.py) : la colonne de
  gauche compacte (la LIGNE de la dalle ouverte — son nom en champ, la liste des dalles, copier, supprimer — puis ses
  quatre rubriques), les couches des données et des
  dessins sous la même clé, la section courante, un bloc de résultats par section dont l'état ne cache jamais une
  non-conformité ;
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

Depuis la fusion des modules Dalle et Prédalle (docs/site/DALLES_FUSION.md, 11/2026), cette suite exerce le TYPE « Dalle
CSP » (coulée sur place) du module fusionné : ce qu'elle prouve, c'est que le site affiche le moteur beton/dalle chiffre
pour chiffre — le moteur tel qu'il reçoit la dalle (``moteur_de`` : la copie de api_web/dalle_positions.pour_moteur, ses
nappes posées l'une sur l'autre, toujours depuis la décision de l'ingénieur du 09/11/2026 ; l'option « nappes empilées »,
que cette suite décochait pour lire le moteur sur la dalle brute, n'existe plus). Le type prédalle, la bascule, les
niveaux et les trois formes d'état : tests/test_api_web_dalles_fusion.py, tests/test_api_web_dalle_positions.py.

    python3 tests/test_api_web_dalle.py
"""
import copy
import html
import json
import os
import random
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dalle as P  # noqa: E402
from api_web import dalle_positions as POS  # noqa: E402
from api_web import dessin_dalle as DD  # noqa: E402
from api_web import plaque_compact as PC  # noqa: E402
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


# (le nom d'une dalle de la liste peut porter la classe de son état, « nom et-n-ok » : 13/11/2026, § 8.2)
NOMS = re.compile(r'(?:<span class="nom(?: et-n-\w+)?">([^<]+)</span>|<input data-chemin="nom" data-sorte="texte" value="([^"]*)"[^>]*>)'
                  r'<span class="dim">([^<]+)</span>')


def noms_arbre(html):
    """(nom, dimensions) de chaque élément nommé dans ``html`` : une ligne d'arborescence (le nom d'un élément fermé
    s'écrit, celui de l'ouvert est un champ) ou une entrée de la liste des éléments de la colonne compacte (09/11/2026)."""
    return [(a or b, d) for a, b, d in NOMS.findall(html)]


def dalle_de(etat, i=None):
    """La dalle de l'ÉTAT, sans les clés du site (verrou, type — api_web/dalle.CLES_SITE) : ce que la note garde."""
    d = etat["dalles"][etat["actif"] if i is None else i]
    return {k: v for k, v in d.items() if k not in P.CLES_SITE}


def moteur_de(etat, i=None):
    """La dalle telle que le MOTEUR la REÇOIT : la copie de api_web/dalle_positions.pour_moteur (sans les clés du site,
    l'axe des couches posées sur la nappe principale écrit) — ce que le site calcule ; le moteur sur cette copie, c'est
    le chiffre que l'écran doit montrer."""
    return POS.pour_moteur(etat["dalles"][etat["actif"] if i is None else i])


def nouveau():
    """Une note neuve — la réponse complète, comme celle de « nouveau »."""
    r = req(module="dalle", action="nouveau")
    assert r["ok"], r
    return r


# une ligne de vérification (C.ligne_verif) : (gouvernante, nom, pièce, niveau, taux) ; sa comparaison : CMP
LIGNE = re.compile(r'<div class="el vc( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>(.*?)<span class="pct (\w)"[^>]*>([^<]+)</span></div>', re.S)
CMP = re.compile(r'<div class="el vc( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<div class="l2"><span class="cmp">(.*?)</span>', re.S)
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-section="(\d+)">.*?<span class="n">([^<]+)</span><span class="e">([^<]+)</span>', re.S)


def fenetre(volets, ident):
    """La fenêtre de calcul ``ck-<ident>`` seule (jusqu'à la fenêtre suivante)."""
    return re.search(r'id="ck-' + re.escape(ident) + r'".*?(?=<div class="flyout calcul"|$)', volets, re.S).group(0)

# ------------------------------------------------------------------ 1. une note neuve
r = nouveau()
chk("nouveau : la requête aboutit", r["ok"], r.get("erreur"))
attendu = EN.nouvelle_dalle()
chk("la dalle neuve est celle de l'écran Streamlit (bande 100 × 20, C30/37, B500, principale Y, treillis 10/10/100/100)",
    dalle_de(r["etat"]) == attendu and r["etat"]["dalles"][0]["verrou"] is False, dalle_de(r["etat"]))
ref = MO.verifier(attendu)
chk("une note neuve ne porte aucun effort : « Données incomplètes », sans taux, jamais « Conforme » (comme la poutre)",
    not ref["sollicitee"] and r["resume"]["taux"] is None and r["resume"]["niveau"] == "n" and r["resume"]["statut"] == "Données incomplètes"
    and BLOC.findall(r["resultat"]) == [("inc", "0", "Section A", "Données incomplètes")], (r["resume"], BLOC.findall(r["resultat"])))
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
ref = MO.verifier(moteur_de(etat))
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
gov_cmp = [c_ for g_, _, c_ in CMP.findall(r["resultat"]) if g_]
chk("la ligne gouvernante dit sa comparaison « valeur ≤ limite » (ici τ ≤ τ_adm,I), nommée ; le résumé de la note, le taux maximal",
    gv["cle"] == "tau" and gov_cmp == ["τ{0}={0}0,50{0}N/mm² ≤ τ<sub>adm,I</sub>{0}={0}0,56{0}N/mm²".format(C.NBSP)]
    and r["resume"]["taux"] == C.pct(ref["taux_max"]), (gov_cmp, r["resume"]))
cmp_ = CMP.findall(r["resultat"])
chk("une nappe compare ce qu'elle demande et ce qui est posé, nommés : « A_s,néc = 770 ≤ A_s = 1 124 mm² »",
    cmp_[1][1] == "Inférieure Y" and cmp_[1][2] == "A<sub>s,néc</sub>{0}={0}770 ≤ A<sub>s</sub>{0}={0}{1}{0}mm²".format(
        C.NBSP, P._n(ref["sections"][0]["faces"]["inf_y"]["As"], "mm²")), cmp_[1])
chk("un bloc par section, coloré par son état, titré par sa section : A et B conformes",
    BLOC.findall(r["resultat"]) == [("ok", "0", "Section A", "Conforme"), ("ok", "1", "Section B", "Conforme")], BLOC.findall(r["resultat"]))
# (13/11/2026 : « éviter de proposer à nouveau l'édition des armatures dans les résultats » — la nappe y est NOMMÉE, en
#  texte ; son sélecteur s'ouvre depuis la rubrique « Armatures », un seul endroit pour modifier les armatures)
chk("aucune ligne du résultat n'ouvre le sélecteur : la nappe y est nommée en texte ; chaque nappe de la rubrique « Armatures » l'ouvre",
    "data-pk" not in r["resultat"] and 'class="prof"' not in r["resultat"].replace('<span class="prof">', "")
    and sorted(set(re.findall(r'data-pk="([^"]+)"', r["donnees"]))) == sorted(f"{f}-{i}" for i in range(2) for f in EN.FACES),
    (re.findall(r'data-pk="([^"]+)"', r["resultat"])[:4], sorted(set(re.findall(r'data-pk="([^"]+)"', r["donnees"])))))
n_verifs = sum(len(s_["verifications"]) for s_ in ref["sections"])
chk("chaque ligne ouvre SA fenêtre de calcul, l'en-tête d'un bloc celle de toute la section (comme la poutre)",
    set(re.findall(r'data-calcul="([^"]+)"', r["resultat"])) == {f"s{i}-{v['cle']}" for i, s_ in enumerate(ref["sections"]) for v in s_["verifications"]} | {"s0", "s1"}
    and r["volets"].count('class="flyout calcul"') == n_verifs + 2)
chk("le renfort posé à 8 cm se lit sur la ligne : « 10/10/100/100 + 3 Ø12 » (la base, puis le renfort)",
    "<span>10/10/100/100 + 3 Ø12</span>" in lignes[1][2], lignes[1][2])
chk("deux sections : la coupe et le plan de chacune, puis « Toutes » (comme la poutre)",
    [v["cle"] for v in r["vues"]] == ["c0", "p0", "c1", "p1", "toutes"] and r["vues"][-1]["court"] == "Toutes")

# la fenêtre de calcul : une ligne par vérification, les formules du moteur substituées
ck = fenetre(r["volets"], "s0")
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
e2 = nouveau()["etat"]      # le cas ASMAX de la parité : bande de 20 cm, h = 8 cm, Ø25/100
for ch, x in (("b", "20"), ("h", "8"), ("sections/0/Mx_inf", "1"), ("sections/0/My_inf", "1"), ("sections/0/V", "2")):
    e2 = saisir(e2, ch, x)["etat"]
for f, dd, esp in (("inf_x", "25", "100"), ("inf_y", "25", "100"), ("sup_x", "6", "250"), ("sup_y", "6", "250")):
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/type", "barres")["etat"]
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/d", dd)["etat"]
    e2 = saisir(e2, f"sections/0/faces/{f}/couches/0/esp", esp)["etat"]
r2 = req(module="dalle", action="calculer", etat=e2)
ref2 = MO.verifier(moteur_de(e2))
excessif = [v for s in ref2["sections"] for v in s["verifications"] if "acier_excessif" in v["motifs"] and v["taux"] <= 1]
lignes2 = LIGNE.findall(r2["resultat"])
att2 = [v for s in ref2["sections"] for v in s["verifications"]]
chk("acier excessif (A_s > A_s,max) sous 100 % : la ligne est rouge, comme le moteur la déclare",
    bool(excessif) and all(n == "r" for (_, _, _, n, _), v in zip(lignes2, att2) if v in excessif), [x[3:] for x in lignes2[:4]])
# … et SANS AUCUN EFFORT : la section n'est pas « Données incomplètes » mais NON CONFORME (rien ne cache une non-conformité)
e0 = e2
for m_ in EN.MOMENTS + ("V",):
    e0 = saisir(e0, f"sections/0/{m_}", "0")["etat"]
r0_ = req(module="dalle", action="calculer", etat=e0)
m0_ = MO.verifier(moteur_de(e0))
chk("acier excessif sans aucun effort : le bloc dit « Non conforme » (rouge), jamais « Données incomplètes » ; la note aussi",
    not m0_["sollicitee"] and m0_["statut"] == MO.NOK and BLOC.findall(r0_["resultat"]) == [("nok", "0", "Section A", "Non conforme")]
    and r0_["resume"]["niveau"] == "r", (BLOC.findall(r0_["resultat"]), r0_["resume"]))

# ------------------------------------------------------------------ 4. les saisies refusées (la dalle reste intacte)
# (13/11/2026 : « Sans signe » — les colonnes s'appellent M⁺ et M⁻, la valeur se saisit sans le signe ; V de même)
REFUS = [("sections/0/Mx_inf", "-20", "saisir 20, pas -20"), ("sections/0/Mx_sup", "-20", "M⁻ : la face supérieure"),
         ("sections/0/V", "-1", "sans signe : 1, pas -1"), ("b", "15", "entre 20 et 500"),
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
# (inf. X, direction secondaire : son automatique est celui du SITE, posée sur la base Y — treillis Ø10 sur Ø10 à 3,5 cm :
# 3,5 + 0,5 + 0,5 = 4,5 cm ; le moteur seul dirait 3,5)
auto = POS.axe_auto(etat["dalles"][0], etat["dalles"][0]["sections"][0], "inf_x", 0)
r5 = saisir(etat, "sections/0/faces/inf_x/couches/0/dist", C.nb(auto))
chk(f"l'axe automatique affiché ({C.nb(auto)} cm, la direction secondaire posée sur la principale), réécrit, reste automatique "
    "(comme le champ de l'écran)",
    auto == 4.5 and dalle_de(r5["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None)
r5 = saisir(etat, "sections/0/faces/inf_x/couches/0/dist", "5")
chk("un autre axe est imposé : son champ le montre, et se VIDE pour revenir à l'automatique (dit en clair : « auto · 4,5 »)",
    dalle_de(r5["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] == 5.0
    and 'data-chemin="sections/0/faces/inf_x/couches/0/dist" data-sorte="texte" inputmode="decimal" value="5" data-vide="auto" '
        'placeholder="auto · 4,5"' in r5["donnees"])
r5 = saisir(etat, "sections/0/faces/inf_y/yG", "4,2")
chk("un centre de gravité saisi est imposé (y_G = 4,2 cm dans le calcul)", MO.verifier(moteur_de(r5["etat"]))["sections"][0]["grandeurs"]["yG_inf_y"] == 4.2)
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
# (13/11/2026, § 8.2) le VERROU est le cadenas de la ligne de la dalle — plus une case de la rubrique 1 : un bouton à chemin
# « verrou » dont la valeur est celle qu'un clic pose (« false » sur une dalle verrouillée)
chk("dalle verrouillée : le cadre Données est en lecture seule (aucun champ, plus de case), le type dit (« Dalle pleine »), un "
    "cadenas devant le nom dans la LIGNE de la dalle et le CADENAS allumé à droite (un clic déverrouille), la tête des résultats "
    "un simple titre, et aucun sélecteur",
    'data-sorte="nombre"' not in r8["donnees"] and 'type="checkbox"' not in r8["donnees"] and "data-pk=" not in r8["resultat"]
    and '<span class="dl-ro">Dalle pleine</span>' in r8["donnees"]
    and re.search(r'<div class="et-elsel et-elsel-d" data-elements="1"><span class="et-elt-nom lu[^"]*"[^>]*>' + re.escape(C.icon("lock"))
                  + r'<span>Dalle 1</span></span>', r8["donnees"]) is not None
    and re.search(r'<button type="button" class="et-elt-b et-verrou on" data-chemin="verrou" data-valeur="false" aria-pressed="true" '
                  r'title="Déverrouiller la dalle"', r8["donnees"]) is not None
    and '<div class="et-rp"><b>Dalle 1</b>' in r8["resultat"])
chk("dalle verrouillée : son nom se LIT dans la ligne de la dalle (aucun champ « nom » nulle part) ; plus de « Renommer » (un clic sur "
    "le nom suffit) ; déverrouillée, le nom redevient un champ — dans la ligne de la dalle, jamais dans la tête des résultats",
    'data-chemin="nom"' not in r8["donnees"] + r8["resultat"] and "data-renommer" not in r8["donnees"]
    and '<input data-chemin="nom" data-sorte="texte" value="Dalle 1" class="et-elt-nom' in (r9 := saisir(r8["etat"], "verrou", "false"))["donnees"]
    and 'class="et-elt-b et-verrou" data-chemin="verrou" data-valeur="true" aria-pressed="false"' in r9["donnees"]
    and 'data-chemin="nom"' not in r9["resultat"] and "et-elt-nom lu" not in r9["donnees"] and "data-renommer" not in r9["donnees"])
chk("… et déverrouillée, elle se modifie de nouveau", saisir(r8["etat"], "verrou", False)["etat"] == etat)

# ------------------------------------------------------------------ 6. le sélecteur de la nappe de base
s = req(module="dalle", action="selecteur", etat=etat, role="inf_y-0")
# (les candidats du moteur, chacun posé aux niveaux du site : POS.candidats_base, contrôlé contre le moteur par
# tests/test_api_web_dalle_positions.py ; la liste et le tri sont ceux du moteur)
cands = POS.candidats_base(etat["dalles"][0], 0, "inf_y")
rows = re.findall(r'<button class="pk-row( cur)?( opt)?" data-choix="([^"]+)"', s["html"])
chk(f"sélecteur : les {len(cands)} nappes du moteur, triées par section au mètre", [x[2] for x in rows] == [c["nom"] for c in cands]
    and [c["As_pm"] for c in cands] == sorted(c["As_pm"] for c in cands)
    and [c["nom"] for c in cands] == [c["nom"] for c in MO.candidats_base(moteur_de(etat), 0, "inf_y")])
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
    MO.verifier(moteur_de(r9["etat"]))["sections"][0]["grandeurs"]["besoin_inf_y"] / MO.verifier(moteur_de(r9["etat"]))["sections"][0]["grandeurs"]["As_inf_y"]
    == opt["taux"])
chk("la plus petite nappe qui tient (★) se lit dans le sélecteur, et plus dans le résultat (aucune étoile sur ses lignes)",
    'class="leger"' not in r["resultat"] and "la plus petite section qui tient" in s["html"], re.findall(r'class="leger"[^>]*>', r["resultat"])[:2])
r10 = req(module="dalle", action="choisir", etat=etat, role="sup_x-1", nom="Ø12/150")
chk("choisir « Ø12/150 » pose des barres Ø12 tous les 150 mm", {k: dalle_de(r10["etat"])["sections"][1]["faces"]["sup_x"]["couches"][0][k]
                                                               for k in ("type", "d", "esp")} == {"type": "barres", "d": 12, "esp": 150})

# (13/11/2026, § 8.4) le bouton du sélecteur est une LISTE (plus une ampoule) ; les ESPACEMENTS admis des barres proposées :
# un réglage de la note (tous par défaut), des puces dans les options (un appui bascule ; jamais le dernier), que le
# sélecteur suit (les barres aux espacements admis seulement, la ★ choisie parmi elles) ; une note d'avant, sans la clé,
# les a tous ; une liste abîmée est refusée
_n0 = nouveau()
_don0 = _n0["donnees"]
_puces = re.findall(r'<button type="button" class="fchip( on)?" data-chemin="espacements" data-valeur="(\d+)" data-bascule="1"', _don0)
chk("le sélecteur s'ouvre d'un bouton « liste » (plus d'ampoule) ; les options portent une puce par espacement du moteur, toutes allumées "
    "(la note neuve les admet tous)",
    C.icon("list-filter") in _don0 and C.icon("lightbulb") not in _don0 and _n0["etat"]["espacements"] == list(P.ESPACEMENTS) == [100, 125, 150, 200, 250]
    and _puces == [(" on", str(e)) for e in P.ESPACEMENTS], _puces)
_e1 = req(module="dalle", action="modifier", etat=_n0["etat"], chemin=["espacements"], valeur="125")
_e2 = req(module="dalle", action="modifier", etat=_e1["etat"], chemin=["espacements"], valeur="150")
_sel = req(module="dalle", action="selecteur", etat=_e2["etat"], role="inf_y-0", familles=["barres"])
_esp_sel = sorted({int(x) for x in re.findall(r'data-choix="Ø\d+/(\d+)"', _sel["html"])})
_cands_adm = POS.candidats_base(_e2["etat"]["dalles"][0], 0, "inf_y", ("barres",), espacements=[100, 200, 250])
chk("un appui sur « 12,5 » puis « 15 » les retire (la puce s'éteint) ; le sélecteur ne propose plus que des barres à 10, 20 et 25 cm, et le "
    "dit ; sa ★ est la plus petite qui tient PARMI elles (celle de dalle_positions, filtrée)",
    _e1["message"] is None and _e2["etat"]["espacements"] == [100, 200, 250]
    and 'class="fchip" data-chemin="espacements" data-valeur="125"' in _e2["donnees"] and _esp_sel == [100, 200, 250]
    and "espacements admis : 10 · 20 · 25 cm" in _sel["html"]
    and re.findall(r'<button class="pk-row(?: cur)? opt" data-choix="([^"]+)"', _sel["html"]) == [c["nom"] for c in _cands_adm if c["optimal"]],
    (_esp_sel, _e2["etat"]["espacements"]))
_e3 = _e2["etat"]
for _v in ("100", "200"):
    _e3 = req(module="dalle", action="modifier", etat=_e3, chemin=["espacements"], valeur=_v)["etat"]
_dernier = req(module="dalle", action="modifier", etat=_e3, chemin=["espacements"], valeur="250")
_retour = req(module="dalle", action="modifier", etat=_e3, chemin=["espacements"], valeur="150")
chk("le dernier espacement ne se retire pas (« Au moins un espacement »), la note intacte ; un appui sur une puce éteinte la rallume",
    _e3["espacements"] == [250] and "Au moins un espacement" in (_dernier["message"] or "") and _dernier["etat"] == _e3
    and _retour["etat"]["espacements"] == [150, 250], (_e3["espacements"], _dernier.get("message")))
_vieille = req(module="dalle", action="calculer", etat={"version": 3, "dalles": [{"nom": "D"}], "actif": 0})
_abimees = [req(module="dalle", action="calculer", etat={"dalles": [{"nom": "D"}], "espacements": x}) for x in ([], [99], "100", [True])]
chk("une note d'avant (sans la clé) admet tous les espacements ; une liste abîmée (vide, hors du moteur, texte, booléen) est REFUSÉE avec "
    "un message, jamais remplacée en silence ; un espacement inconnu saisi est refusé",
    _vieille["etat"]["espacements"] == list(P.ESPACEMENTS) and all(not x["ok"] and "espacements admis" in (x.get("erreur") or "") for x in _abimees)
    and "Espacement : l'un de" in (req(module="dalle", action="modifier", etat=_n0["etat"], chemin=["espacements"], valeur="175")["message"] or ""),
    [x.get("erreur") for x in _abimees])

# ------------------------------------------------------------------ 7. les dalles de la note
a = req(module="dalle", action="ajouter", etat=etat)
chk("« Ajouter une dalle » : Dalle 2 ouverte, neuve", len(a["etat"]["dalles"]) == 2 and a["etat"]["actif"] == 1
    and dalle_de(a["etat"]) == dict(EN.nouvelle_dalle(), nom="Dalle 2"))
cp = req(module="dalle", action="dupliquer", etat=etat)
chk("« Copier » : la dalle ouverte, nommée « (copie) »", dalle_de(cp["etat"]) == dict(d, nom="Dalle 1 (copie)"))
trois = req(module="dalle", action="ajouter", etat=cp["etat"])["etat"]
cp2 = req(module="dalle", action="dupliquer", etat=trois, index=0)
chk("« Copier » du menu « ⋯ » d'une dalle (index) : CETTE dalle, copiée juste après elle et ouverte (comme la poutre)",
    [x["nom"] for x in cp2["etat"]["dalles"]] == ["Dalle 1", "Dalle 1 (copie)", "Dalle 1 (copie)", "Dalle 3"] and cp2["etat"]["actif"] == 1
    and dalle_de(cp2["etat"], 1) == dict(d, nom="Dalle 1 (copie)"), [x["nom"] for x in cp2["etat"]["dalles"]])
chk("le résumé de la note est sa dalle la plus chargée", cp["resume"]["taux"] == C.pct(ref["taux_max"]))
sp = req(module="dalle", action="supprimer", etat=cp["etat"], index=0)
chk("supprimer la première dalle garde la copie ouverte", len(sp["etat"]["dalles"]) == 1 and sp["etat"]["dalles"][0]["nom"] == "Dalle 1 (copie)")
chk("la dernière dalle ne se supprime pas", len(req(module="dalle", action="supprimer", etat=sp["etat"], index=0)["etat"]["dalles"]) == 1)
lourd = saisir(a["etat"], "sections/0/My_inf", "80")
chk("une seconde dalle plus chargée devient le résumé de la note, nommée dans le gouvernant",
    lourd["resume"]["taux"] == C.pct(MO.verifier(moteur_de(lourd["etat"]))["taux_max"]) and lourd["resume"]["gouvernant"].startswith("Dalle 2"),
    lourd["resume"])

# ------------------------------------------------------------------ 8. « Reprendre une note Streamlit » : retirée
# (le propriétaire, 13/11/2026 : « supprimer définitivement » — le bureau n'a aucune ancienne note à reprendre ; le code
#  d'import est retiré avec l'option : le chemin n'existe plus, il est refusé comme tout champ inconnu, la note intacte)
V1 = {"dalles": [{"id": 1, "nom": "Plancher", "sections": [{"id": 1, "nom": "A"}]}], "values": {"dal1_sec1_M_inf": 28.0}}
f = req(module="dalle", action="modifier", etat=etat, chemin=["fichier_streamlit"], valeur={"nom": "dalle_ba.json", "texte": json.dumps(V1)})
chk("« Reprendre une note Streamlit » n'existe plus : ni dans les options, ni comme chemin (refusé, « champ inconnu », la note intacte)",
    "fichier_streamlit" not in r["donnees"] and "Streamlit" not in r["donnees"].split('data-sec="op"')[-1]
    and f["message"] == "champ inconnu : fichier_streamlit" and f["etat"] == etat, (f.get("message"), f["etat"] == etat))

# ------------------------------------------------------------------ 9. l'échappement
PIEGE = '<img src=x onerror=alert(1)>'
e9 = saisir(etat, "nom", PIEGE)["etat"]
r9 = saisir(e9, "sections/1/nom", PIEGE)
frag = {k: r9[k] for k in ("resultat", "volets", "donnees", "elements", "phrase")}
frag["dessins"] = "".join(v["svg"] for v in r9["vues"])
frag["en-têtes"] = "".join(v["entete"] for v in r9["vues"])   # (titre, court : du TEXTE, que js/etabli.js échappe)
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
    e = nouveau()["etat"]
    for ch, x in (("b", rng.choice(("50", "100", "120", "150"))), ("h", rng.choice(("16", "20", "25"))),
                  ("beton", rng.choice(sorted(MO.classes_beton()))), ("fyk", rng.choice(("400", "500"))),
                  ("dir_principale", rng.choice("XY")), ("enrobage", rng.choice(("0", "2,5", "3")))):
        e = saisir(e, ch, x)["etat"]
    for m in EN.MOMENTS + ("V",):
        e = saisir(e, f"sections/0/{m}", C.nb(round(rng.uniform(0, 60), 2)))["etat"]
    f = rng.choice(EN.FACES)
    e = saisir(e, f"sections/0/faces/{f}/couches/0/type", rng.choice(("treillis", "barres", "n_barres")))["etat"]
    rr = req(module="dalle", action="calculer", etat=e)
    rf = MO.verifier(moteur_de(e))
    li = LIGNE.findall(rr["resultat"])
    att = [v for s_ in rf["sections"] for v in s_["verifications"]]
    if len(li) != len(att) or any(t != f"{C.pc(C.pct(v['taux']))}{C.NBSP}%" or n != niveau_de(v) for (_, _, _, n, t), v in zip(li, att)) \
            or rr["resume"]["taux"] != C.pct(rf["taux_max"]):
        fautes.append(k)
chk("25 dalles tirées au sort : chaque ligne affichée porte le taux et la couleur du moteur", not fautes, fautes)

# ------------------------------------------------------------------ 12. ce que la relecture indépendante a trouvé
# a) une dalle NON CONFORME sous 100 % (acier excessif) : le verdict ne nomme pas une vérification conforme
ex = nouveau()["etat"]
for ch, x in (("sections/0/Mx_inf", "10"), ("sections/0/faces/inf_x/couches/0/type", "barres"),
              ("sections/0/faces/inf_x/couches/0/d", "40"), ("sections/0/faces/inf_x/couches/0/esp", "25")):
    ex = saisir(ex, ch, x)["etat"]
rx = req(module="dalle", action="calculer", etat=ex)
mx = MO.verifier(moteur_de(ex))
vx = {v["cle"]: v for v in mx["sections"][0]["verifications"]}
chk("acier excessif sous 100 % (Ø40/25 : A_s > A_s,max) : la dalle est non conforme, et le moteur nomme la hauteur (la plus chargée)",
    mx["statut"] == MO.NOK and mx["gouvernant"]["cle"] == "hauteur" and vx["inf_x"]["etat"] == MO.NOK and vx["inf_x"]["taux"] < vx["hauteur"]["taux"] <= 1)
lx = LIGNE.findall(rx["resultat"])
cx = CMP.findall(rx["resultat"])
chk("… la ligne gouvernante est la non conforme, rouge sous 100 %, et sa comparaison dit pourquoi (A_s > A_s,max · acier excessif)",
    [(n, niv) for g_, n, _, niv, _ in lx if g_] == [("Inférieure X", "r")]
    and any(n == "Inférieure X" and g_ and "A<sub>s,max</sub>{0}={0}8{1}000{0}mm²".format(C.NBSP, C.fr(1000, 0)[1]) in c_
            and "acier excessif" in c_ for g_, n, c_ in cx), [(x[0], x[1]) for x in lx])
chk("… son bloc dit « Non conforme » (rouge) ; le résumé de la note la nomme, en rouge",
    BLOC.findall(rx["resultat"]) == [("nok", "0", "Section A", "Non conforme")] and rx["resume"]["niveau"] == "r"
    and rx["resume"]["gouvernant"].endswith("inférieure X") and rx["resume"]["taux"] == C.pct(vx["inf_x"]["taux"]), rx["resume"])
ck = fenetre(rx["volets"], "s0")
chk("… et une seule ligne gouvernante dans le tableau de la fenêtre de la section (la non conforme)",
    ck.count('<tr class="gov">') == 1 and re.search(r'<tr class="gov"><td class="nm">Inférieure X', ck) is not None)
neuf = nouveau()
ck = fenetre(neuf["volets"], "s0")
chk("une note neuve (quatre nappes à 38 %) : une seule ligne gouvernante dans le tableau de la section, la première à égalité",
    ck.count('<tr class="gov">') == 1 and re.search(r'<tr class="gov"><td class="nm">Inférieure Y', ck) is not None)

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
ap = nouveau()["etat"]
for ch, x in (("sections/0/My_sup", "30"), ("sections/0/My_inf", "5"), ("sections/0/Mx_inf", "12"),
              ("sections/0/faces/sup_y/couches/0/type", "barres"), ("sections/0/faces/sup_y/couches/0/d", "12")):
    ap = saisir(ap, ch, x)["etat"]
ra = MO.verifier(moteur_de(ap))
ca = DD.svg_coupe("t", ra["sections"][0], ra["dalle"])
coupes = ca.split("Coupe ⊥")
chk("coupe ⊥ Y (M_y,sup > M_y,inf) : d = 160 de la nappe supérieure Y (axe Ø12 à 40 mm), et non 165 de l'inférieure",
    "d = 160" in coupes[1] and "d = 165" not in coupes[1], re.findall(r"d = \d+", ca))
chk("coupe ⊥ X (M_x,inf seul) : d = 155 de la nappe inférieure X — posée sur la base Y (Ø10 sur Ø10 à 35 mm : axe 45 mm)",
    "d = 155" in coupes[2], re.findall(r"d = \d+", coupes[2]))
chk("la légende des couches dit « axe à 35 » (l'espacement s'écrit Ø/e : « Ø12/150 »)", "axe à 35" in ca and "e = 35" not in ca)
fo = saisir(ap, "nom", "Dalle 1")["donnees"]
chk("une couche de treillis dit ses fils porteurs (Ø10 tous les 100 mm), son axe est nommé « e axe » et vide = automatique",
    "porteurs Ø10 tous les 100 mm" in fo and fo.count(">e axe<") == 4
    and 'data-chemin="sections/0/faces/sup_y/couches/0/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" '
        'placeholder="auto · 4"' in fo, re.findall(r'placeholder="auto · [^"]+"', fo))

# ------------------------------------------------------------------ 13. l'établi compact : les fonctions aux places de la poutre
ar = req(module="dalle", action="calculer", etat=etat)
don = ar["donnees"]
# la LIGNE de la dalle ouverte (etabli_compact.arbre_compact) : de son début aux premières rubriques
ligne_ = re.search(r'<div class="et-elsel et-elsel-d" data-elements="\d+">.*?(?=<div class="sec)', don, re.S)
ligne_ = ligne_.group(0) if ligne_ else ""
chk("la colonne de gauche COMPACTE (fusion, § 1, revue le 09/11/2026) : une seule dalle, ni en-tête « Dalles (1) » ni arborescence — "
    "la LIGNE de la dalle (que l'écran pose dans l'en-tête du panneau) puis ses rubriques ; « Ajouter une dalle » dans la liste des "
    "dalles ; la liste « Dans cette note » de l'ancien écran n'existe plus ; aucun taux sur la ligne",
    "et-arbre-h" not in don and not re.search(r'class="et-pt(?: on)?"|class="et-pt-(?:l|corps|nom|add)\b', don)
    and re.search(r'<div class="et-arbre et-arbre-c" data-dalle="0"><div class="et-elsel et-elsel-d" data-elements="1">', don) is not None
    and noms_arbre(ligne_) == [("Dalle 1", "· 100 × 20 cm")] and 'data-ajouter="1"' in ligne_ and "Ajouter une dalle" in ligne_
    and ar["elements"] == "" and ligne_ and "%" not in ligne_, ligne_[:300])
chk("sous la dalle ouverte, ses quatre rubriques dans l'ordre de la poutre : Caractéristiques (le type et les dimensions), "
    "Sections et sollicitations, Armatures, Options",
    re.findall(r'<div class="sec[^"]*" data-sec="([^"]+)"', don) == ["dl", "so", "arm", "op"]
    and re.findall(r'<span class="tt">([^<]+)</span>', don) == ["Caractéristiques", "Sections et sollicitations", "Armatures", "Options"]
    and re.search(r'<div class="sec" data-sec="op" data-secondaire="1">', don) is not None)
chk("la ligne de la dalle : son NOM en champ (vert : la dalle est conforme), la liste des dalles (la seule, cochée ; « Ajouter une "
    "dalle »), puis à droite le CADENAS (ouvert : un clic verrouille), Copier la dalle, Supprimer la dalle (éteint : c'est la seule) — "
    "plus de menu « ⋯ » ni de « Renommer »",
    ligne_.startswith('<div class="et-elsel et-elsel-d" data-elements="1"><input data-chemin="nom" data-sorte="texte" value="Dalle 1" '
                      'class="et-elt-nom et-n-ok"')
    and ligne_.index("et-elt-liste") < ligne_.index('class="et-elt-b et-verrou"') < ligne_.index("data-dupliquer-index")
    and 'data-chemin="verrou" data-valeur="true" aria-pressed="false" title="Verrouiller la dalle (lecture seule)"' in ligne_
    and re.search(r'aria-checked="true" class="et-elt-i on" data-activer="0"', ligne_) is not None
    and 'data-dupliquer-index="0"' in ligne_ and 'aria-label="Copier la dalle"' in ligne_
    and re.search(r'data-supprimer="0" data-confirmer="Supprimer la dalle « Dalle 1 » de la note \?" title="[^"]*" aria-label="Supprimer la '
                  r'dalle" disabled>', ligne_) is not None
    and "data-renommer" not in don and 'title="Actions de la dalle"' not in ligne_, ligne_[:300])
deux_ = req(module="dalle", action="ajouter", etat=etat)
chk("deux dalles : la LISTE de la ligne les nomme dans l'ordre (la seconde ouverte, cochée), le compte « 2 » sur son bouton, le champ "
    "porte le nom de l'ouverte ; les rubriques d'une seule dalle ; une seule liste ; Supprimer allumé",
    re.findall(r'class="et-elt-i( on)?" data-activer="(\d)"', deux_["donnees"]) == [("", "0"), (" on", "1")]
    and re.search(r'<div class="et-arbre et-arbre-c" data-dalle="1"><div class="et-elsel et-elsel-d" data-elements="2"><input data-chemin="nom" '
                  r'data-sorte="texte" value="Dalle 2" class="et-elt-nom et-n-inc"', deux_["donnees"]) is not None
    and '<span class="n">2</span></button>' in deux_["donnees"]
    and deux_["donnees"].count('data-sec="dl"') == 1 and deux_["donnees"].count('class="et-mn et-elt-liste"') == 1
    and re.search(r'<span class="nom et-n-ok">Dalle 1</span>.*<span class="nom et-n-inc">Dalle 2</span>', deux_["donnees"], re.S) is not None
    and re.search(r'data-supprimer="1" data-confirmer="Supprimer la dalle « Dalle 2 » de la note \?" title="Supprimer la dalle" '
                  r'aria-label="Supprimer la dalle">', deux_["donnees"]) is not None)
cibles = set(re.findall(r'data-cible="([^"]+)"', don))
sels = set(re.findall(r'data-sel="([^"]+)"', "".join(v["svg"] for v in ar["vues"])))
couches_ = {f"s{i}.{f}-{j}" for i, s_ in enumerate(d["sections"]) for f in EN.FACES for j in range(len(s_["faces"][f]["couches"]))}
chk("chaque couche de chaque section a sa ligne dans les données (data-cible) et sa pièce sur les dessins (data-sel), sous la "
    "MÊME clé ; le béton et les efforts aussi",
    couches_ <= cibles and couches_ <= sels and {"beton", "s0.efforts", "s1.efforts"} <= cibles and {"s0.beton", "s1.beton"} <= sels,
    (sorted(couches_ - cibles)[:3], sorted(couches_ - sels)[:3]))
for cle_ in ("c0", "p0", "c1", "p1"):
    sv_ = next(v for v in ar["vues"] if v["cle"] == cle_)["svg"]
    i_ = int(cle_[1:])
    chk(f"la vue {cle_} ne porte que les pièces de SA section (s{i_}.…)", sv_.count("data-sel=") > 0
        and all(k.startswith(f"s{i_}.") for k in re.findall(r'data-sel="([^"]+)"', sv_)))
chk("« Toutes » : la coupe principale de chaque section, ses pièces cliquables (s0 et s1)",
    {k.split(".")[0] for k in re.findall(r'data-sel="([^"]+)"', ar["vues"][-1]["svg"])} == {"s0", "s1"})
chk("la section courante : ses armatures (.et-arm), son résumé (.et-r3) et son bloc de résultats (.et-bloc) par section ; le choix "
    "de la section courante dès deux sections",
    re.findall(r'<div class="et-arm( on)?" data-section="(\d)">', don) == [(" on", "0"), ("", "1")]
    and len(re.findall(r'class="et-r3" data-section="\d"', don)) == 2 and len(re.findall(r'data-section-courante="\d"', don)) == 4
    and [b_[1] for b_ in BLOC.findall(ar["resultat"])] == ["0", "1"])
ent = ar["vues"][0]["entete"]
chk("l'en-tête de la vue : la section, la bande, les matériaux et la direction principale ; les nappes inférieures et supérieures "
    "(la principale d'abord) ; la bande des quatre moments et de V, une fois",
    "<b>Section A</b><span>· 100 × 20 cm · C30/37 · B500 · principale Y</span>" in ent
    and ">Inf. Y Treillis 10/10/100/100 + 3 Ø12 · X Treillis 10/10/100/100<" in ent and ">Sup. Y Treillis 10/10/100/100 · X Treillis 10/10/100/100<" in ent
    and ent.count("<span><i>") == 5 and "<sub>x,inf</sub><b>28,0</b>" in ent and "<i>V</i><b>60,0</b>" in ent, ent[:400])
rub1 = re.search(r'<div class="sec[^"]*" data-sec="dl".*?(?=<div class="sec[^"]*" data-sec="so")', don, re.S).group(0)
# (13/11/2026, § 8.2) la rubrique 1 en SYMBOLES : le type en deux segments sans libellé (« Dalle pleine » · « Prédalle »), b
# (son ⓘ : la bande de calcul de 1 m), h, c — le mot au survol du symbole ; plus de case « Verrouiller » (le cadenas de la ligne)
_lab1 = [re.sub(r"<[^>]+>", "", x) for x in re.findall(r'<label><span>(.*?)</span>(?:<span class="aide"|</label>)', rub1)]
chk("le NOM de l'élément ouvert se modifie dans la LIGNE de la dalle (même chemin « nom » qu'avant ; plus dans la tête des résultats) ; "
    "la rubrique 1 ouvre sur les deux segments du type SANS libellé (« Dalle pleine » retenu, « Prédalle »), puis 3 lignes de deux "
    "champs aux libellés COURTS (Béton | Acier, b | h, c | Principale), l'acier en menu déroulant ; ni « Type », ni case à cocher",
    '<input data-chemin="nom" data-sorte="texte" value="Dalle 1" class="et-elt-nom' in ligne_ and 'data-chemin="nom"' not in rub1
    and 'data-chemin="nom"' not in ar["resultat"]
    and re.search(r'<div class="row full et-type" title="[^"]+"><div class="seg full"><button type="button" class="on" data-chemin="type" '
                  r'data-sorte="choix" data-valeur="coulee">Dalle pleine</button><button type="button" class="" data-chemin="type" '
                  r'data-sorte="choix" data-valeur="predalle">Prédalle</button></div></div>', rub1) is not None
    and ">Type<" not in rub1 and "type=\"checkbox\"" not in rub1 and "Verrouiller" not in rub1 and 'data-chemin="h_pre"' not in rub1
    and rub1.count('class="row2 n2"') == 3 and _lab1 == ["Béton", "Acier", "b", "h", "c", "Principale"]
    and all(m not in rub1 for m in (">b bande<", ">h épaisseur<", ">c enrobage<"))
    and re.search(r'<select data-chemin="fyk" data-sorte="choix">', rub1) is not None and 'class="hint"' not in rub1,
    (_lab1, re.findall(r'<div class="row[^"]*"', rub1)))
# (13/11/2026, § 8.3) la rubrique 2 : chaque section en un TABLEAU compact — X et Y en lignes, M⁺ (face inférieure
# tendue) et M⁻ (face supérieure tendue) en colonnes, V (un par section) à côté, sur les deux lignes ; plus de textes
# répétés (« bas tendu », « haut tendu », « effort tranchant ») ; « Dupliquer la section » dans le menu « ⋯ »
rub2 = re.search(r'<div class="sec[^"]*" data-sec="so".*?(?=<div class="sec[^"]*" data-sec="arm")', don, re.S).group(0)
_tab = re.search(r'<div class="et-mom et-mom4">(.*?)(?=</div></div><div class="et-(?:sec |pied))', rub2, re.S)
_tab_txt = re.sub(r"<[^>]+>", "", _tab.group(1)) if _tab else ""
_ordre = re.findall(r'data-chemin="sections/0/(Mx_inf|Mx_sup|V|My_inf|My_sup)"', _tab.group(0)) if _tab else []
chk("la rubrique 2 : un TABLEAU par section — titres « M⁺ · inf. tendue », « M⁻ · sup. tendue », « V » ; lignes X (Mx,inf | Mx,sup) "
    "et Y (My,inf | My,sup) ; V à côté, dans son cadre .et-v (sur les deux lignes) ; les unités dans les champs ; plus de « bas tendu », "
    "« haut tendu », « effort tranchant » écrits ; « Dupliquer la section »",
    _tab is not None and "M+ · inf. tendue" in _tab_txt and "M− · sup. tendue" in _tab_txt and _tab_txt.startswith("M+ · inf. tendueM− · sup. tendueVX")
    and _ordre == ["Mx_inf", "Mx_sup", "V", "My_inf", "My_sup"] and '<div class="et-v">' in _tab.group(0)
    and _tab.group(0).count("kN·m") == 4 and _tab.group(0).count(">kN<") == 1
    and all(t not in rub2 for t in ("bas tendu", "haut tendu", "effort tranchant", "M<sub>inf</sub>", "M<sub>sup</sub>"))
    and "Dupliquer la section" in rub2 and "Copier la section" not in rub2, (_tab_txt[:120], _ordre))
_aides1 = [html.unescape(t) for t in re.findall(r'<span class="aide" title="([^"]*)">', rub1)]
chk("… b porte son ⓘ (« Bande de calcul : 100 cm = 1 m de dalle »), c et la direction principale le leur ; chaque symbole dit son mot "
    "au survol (b : largeur de la bande de calcul, h : épaisseur de la dalle, c : enrobage)",
    _aides1[0].startswith("Bande de calcul : 100 cm = 1 m de dalle") and len(_aides1) == 3
    and all(f'<span class="et-sym" title="{t}">{k}</span>' in rub1
            for k, t in (("b", "Largeur de la bande de calcul"), ("h", "Épaisseur de la dalle"), ("c", "Enrobage"))), _aides1)
aides_ = [html.unescape(t) for t in re.findall(r'<span class="aide" title="([^"]*)">', don)]
chk("condensées, les données ne perdent aucune explication : les paragraphes d'aide d'avant (direction principale, γs et "
    "jeu communs à la note) sont dans les ⓘ de leur champ ; plus d'option « nappes empilées » (décision du 09/11/2026), et "
    "l'aide de l'axe dit la règle (la direction secondaire, base et renforts, posée sur la principale)",
    any(t.startswith("La direction principale ordonne les vérifications") for t in aides_)
    and sum("Commun à toutes les dalles de la note" in t for t in aides_) == 2
    and not any("empil" in t.lower() for t in aides_) and 'data-chemin="empilement"' not in don
    and any("direction secondaire (sa base et ses renforts), posée sur la principale" in t for t in aides_), aides_)
chk("chaque vue dit ce qu'on lit sur son dessin, en dernière ligne de son en-tête (la légende : barres coupées en points, X horizontales…)",
    len(ar["vues"]) == 5 and all(re.findall(r'<div class="lg">([^<]*)</div>(?=</div>)', v_["entete"])
                                 == [P.LEGENDES["p" if v_["cle"][0] == "p" else "c"]] == [v_["legende"]] for v_ in ar["vues"]),
    [(v_["cle"], v_["legende"]) for v_ in ar["vues"]])
chk("le dessin d'une section, dans l'établi, n'écrit plus les efforts (ils sont dans la bande de l'en-tête, une fois, sans rouge)",
    "#9C3341" not in ar["vues"][0]["svg"] and "kN·m" not in ar["vues"][0]["svg"] and "kN·m" in DD.svg_coupe("t", ref["sections"][0], ref["dalle"]))
chk("la tête des résultats : le nom de la dalle (un titre qui se lit : le champ est dans la ligne de la dalle) et ses dimensions "
    "(comme la poutre)",
    ar["resultat"].startswith('<div class="pbody"><div class="et-rp"><b>Dalle 1</b><span>100 × 20 cm · C30/37 · B500</span></div>')
    and "<input" not in ar["resultat"].split('<div class="et-blocs">')[0], ar["resultat"][:300])
_f = P.telecharger(copy.deepcopy(etat), "note_pdf", {"nom": "Essai"}) if hasattr(P, "telecharger") else None
chk("la dalle fait sa note PDF sur le site (S4, 09/11/2026 : api_web/note_dalle.py, contrôlée par tests/test_note_dalle.py) : "
    "« note_pdf » rend le PDF, tout autre fichier rien (le bouton s'allume quand construire.NOTE_PDF la nomme)",
    bool(_f) and _f["type"] == "application/pdf" and _f["nom"] == "Note de calcul - Essai.pdf"
    and P.telecharger(copy.deepcopy(etat), "autre") is None, (_f or {}).get("nom"))
# un axe ou un centre de gravité SAISI se rend à l'automatique d'un clic (« ↺ », comme l'écran d'avant du site), et la
# couche dit ce que serait l'automatique ; automatique, ni bouton ni phrase (le champ vide montre « auto · … »)
RAZ = r'<button type="button" class="et-raz" data-chemin="{}" data-valeur="" title="([^"]+)"'


def raz_chemins(donnees):
    return set(re.findall(r'<button type="button" class="et-raz" data-chemin="([^"]+)"', donnees))


def saisis(el):
    """Les chemins des axes et des centres de gravité SAISIS d'un élément (dalle ou prédalle)."""
    out = set()
    for i_, s_ in enumerate(el["sections"]):
        for f_, fa in s_["faces"].items():
            if fa["yG"] is not None:
                out.add(f"sections/{i_}/faces/{f_}/yG")
            out |= {f"sections/{i_}/faces/{f_}/couches/{j_}/dist" for j_, c_ in enumerate(fa["couches"]) if c_["dist"] is not None}
    return out


ax_ = saisir(saisir(etat, "sections/0/faces/inf_x/couches/0/dist", "5")["etat"], "sections/0/faces/sup_y/yG", "4")
d_ax = ax_["donnees"]
# (l'automatique du SITE : la base inf. X posée sur la base Y — ce que la couche dit, ce que « ↺ » rend)
e_auto = P._calculer(ax_["etat"]["dalles"][0])["sections"][0]["faces"]["inf_x"]["couches"][0]["e_auto"]
t_ax = re.search(RAZ.format("sections/0/faces/inf_x/couches/0/dist"), d_ax)
t_yg = re.search(RAZ.format("sections/0/faces/sup_y/yG"), d_ax)
chk("un axe saisi : « ↺ Revenir à l'axe automatique (… cm) » à côté du champ, et la couche dit l'automatique ; un centre de "
    "gravité imposé : « ↺ Revenir au centre de gravité calculé » ; un bouton par valeur SAISIE, aucun sur un champ automatique",
    t_ax is not None and html.unescape(t_ax.group(1)) == f"Revenir à l'axe automatique ({PC._cm(e_auto)} cm)"
    and f"axe saisi, l'automatique serait {PC._cm(e_auto)} cm" in html.unescape(d_ax)
    and t_yg is not None and html.unescape(t_yg.group(1)).startswith("Revenir au centre de gravité calculé (")
    and all(raz_chemins(r_["donnees"]) == saisis(r_["etat"]["dalles"][0]) for r_ in (ar, ax_)) and len(saisis(ax_["etat"]["dalles"][0])) == 3,
    (t_ax and t_ax.group(1), raz_chemins(d_ax), saisis(ax_["etat"]["dalles"][0])))
raz_ = saisir(ax_["etat"], "sections/0/faces/inf_x/couches/0/dist", "")
chk("… et « ↺ » (la valeur vidée) rend l'axe automatique : la clé du moteur redevient None, le bouton disparaît",
    raz_["etat"]["dalles"][0]["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None
    and re.search(RAZ.format("sections/0/faces/inf_x/couches/0/dist"), raz_["donnees"]) is None)

# ce que le calcul NE vérifie PAS (petits chantiers, 10/11/2026) : la dalle CSP le dit comme la prédalle — le même composant
# (C.aide) aux mêmes places. Sous le résultat, la seule phrase « Non vérifié(e)s » (13/11/2026 : ce qui EST vérifié, les
# lignes le montrent déjà — le paragraphe entier était un message redondant) ; en entier au pied de chaque fenêtre de calcul.
_vp = saisir(saisir(P.nouvel_etat(), "sections/0/My_inf", "28")["etat"], "type", "predalle")
_vc = saisir(P.nouvel_etat(), "sections/0/My_inf", "28")
_court_c, _court_p = C.aide(P.non_verifies(_vc["etat"]["dalles"][0])), C.aide(P.non_verifies(_vp["etat"]["dalles"][0]))
_aide_c, _aide_p = C.aide(P.PERIMETRE_CSP), C.aide(P.texte_perimetre(_vp["etat"]["dalles"][0]))
chk("la dalle CSP dit ce qui n'est PAS vérifié (la flèche, la fissuration), comme la prédalle : une phrase sous le résultat (après les "
    "avertissements, avant le pied), le périmètre entier au pied de chaque fenêtre de calcul ; ni « Vérifié : » sous le résultat, ni « Streamlit »",
    _court_c == C.aide("Non vérifiées : la flèche et la fissuration.") and P.PERIMETRE_CSP.endswith(P.NON_VERIFIEES_CSP)
    and _court_c in _vc["resultat"] and _court_p in _vp["resultat"] and "Non vérifiés : la phase provisoire" in _court_p
    and _vc["resultat"].index(_court_c) > _vc["resultat"].index('<div class="et-blocs">') and _vc["resultat"].index(_court_c) < _vc["resultat"].index('<div class="pfoot">')
    and "Vérifié :" not in _vc["resultat"] and "Vérifié :" not in _vp["resultat"]
    and "Streamlit" not in _vc["resultat"] + _vp["resultat"] + _vc["volets"] + _vp["volets"]
    and _vc["volets"].count(_aide_c) >= 1 and _vp["volets"].count(_aide_p) >= 1, (_vc["volets"].count(_aide_c), _vp["volets"].count(_aide_p)))

# d utile ≤ 0 (13/11/2026) : la LIGNE de la face le dit, quel que soit son taux — et plus un message sous le bloc, qui
# répétait ce que le rouge disait déjà ; au-delà de 100 %, la ligne disait « A_s,néc = 16 667 > A_s = 785 mm² » et la
# raison n'était que dans ce message
_dn = saisir(P.nouvel_etat(), "h", "8")["etat"]
for _ch, _v in (("sections/0/My_inf", "5"), ("sections/0/Mx_inf", "5"), ("sections/0/faces/inf_y/couches/0/dist", "7.5")):
    _dn = saisir(_dn, _ch, _v)["etat"]
_rdn = req(module="dalle", action="calculer", etat=_dn)
_vdn = next(v for v in MO.verifier(moteur_de(_dn))["sections"][0]["verifications"] if v["cle"] == "inf_x")
_cdn = dict((n, c_) for _, n, c_ in CMP.findall(_rdn["resultat"]))
chk("d utile ≤ 0 au-delà de 100 % : la ligne « Inférieure X » dit « d ≤ 0 · nappe hors de l'épaisseur », en rouge ; aucun message ne "
    "le répète sous le bloc",
    "d_utile_negatif" in _vdn["motifs"] and _vdn["taux"] > 1 and "≤ 0 · nappe hors de l'épaisseur" in _cdn.get("Inférieure X", "")
    and _rdn["resultat"].count("hors de l'épaisseur") == 1 and "incompatible avec l'épaisseur" not in _rdn["resultat"]
    and 'class="msg' not in _rdn["resultat"], (_vdn["taux"], _cdn.get("Inférieure X"), _rdn["resultat"].count("hors de l'épaisseur")))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:240])
sys.exit(1 if KO else 0)
