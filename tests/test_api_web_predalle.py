# -*- coding: utf-8 -*-
"""L'API du site pour les prédalles (api_web/predalle.py), vérifiée en Python ordinaire, à travers le routeur (les mêmes
requêtes JSON que le navigateur).

- le résultat affiché est celui du moteur beton/predalle : une ligne par vérification, son taux, sa couleur, la
  vérification gouvernante, section par section, et le résumé de la note (la prédalle la plus chargée) ;
- une vérification que le moteur déclare non conforme est rouge même sous 100 % (acier excessif, d utile ≤ 0) ;
- la fenêtre de calcul : hypothèses (la peau et le coulé en place), une ligne par vérification, et chaque formule
  « symbole = expression = valeurs = résultat » porte le résultat du moteur — l'axe de chaque couche et sa règle ;
- la PEAU : bornes de l'écran (3 cm ; min(15 ; h − 1)), refusée hors bornes (l'écran la ramenait en silence), une
  épaisseur totale qui ne la contiendrait plus refusée aussi ; la peau déplace les couches posées SUR elle, pas la base
  principale (fond de moule) ; le jeu premier lit ne touche que les nappes supérieures ;
- chaque champ pointe un endroit réel de la prédalle ; une saisie hors du domaine de l'écran est REFUSÉE avec un
  message et la prédalle reste intacte ; ajouter / copier / retirer (sections, renforts), axe et centre de gravité
  automatiques ou imposés, γs et jeu communs à la note, le verrou ;
- le sélecteur de la nappe de base : trié par section au mètre, retenue ✓ et plus petite qui tient ★ (celles du
  moteur), axe recalculé par la règle de la prédalle ; le choix garde les renforts ; l'étoile des lignes du résultat ;
- les prédalles de la note s'ajoutent, se copient, se suppriment (jamais la dernière) ;
- la reprise d'un fichier de l'écran Streamlit (predalle_ba.json) et ce qu'il remplaçait en silence ;
- les avertissements (base hors de la peau, couches croisées sur la peau, nappes confondues) ;
- les dessins à l'échelle (peau hachurée, plan de reprise, chaîne de cotes, axes, légende de pose) ; un nom saisi est
  échappé partout.

    python3 tests/test_api_web_predalle.py
"""
import copy
import html
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import dessin_predalle as DP  # noqa: E402
from api_web import plaque_compact as PC  # noqa: E402
from api_web import predalle as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton import treillis as TR  # noqa: E402
from beton.predalle import entree as EN  # noqa: E402
from beton.predalle import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="predalle", **k))))


def saisir(etat, chemin, valeur):
    """Comme l'écran : le chemin lu dans data-chemin, coupé aux « / », la valeur telle que tapée."""
    out = req(action="modifier", etat=etat, chemin=chemin.split("/"), valeur=valeur)
    assert out["ok"], out
    return out


NOMS = re.compile(r'(?:<span class="nom">([^<]+)</span>|<input data-chemin="nom" data-sorte="texte" value="([^"]*)"[^>]*>)'
                  r'<span class="dim">([^<]+)</span>')


def noms_arbre(html):
    """(nom, dimensions) de chaque élément de l'arborescence : le nom d'un élément fermé s'écrit, celui de l'élément
    OUVERT est un champ, sur sa ligne même (11/2026)."""
    return [(a or b, d) for a, b, d in NOMS.findall(html)]


def pd(etat, i=None):
    return etat["predalles"][etat["actif"] if i is None else i]


def moteur(etat, i=None):
    return MO.verifier({k: v for k, v in pd(etat, i).items() if k != "verrou"})


def niveau_de(v):
    return "r" if v["etat"] == MO.NOK else C.niveau(v["taux"])


def texte(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&").strip()


# une ligne de vérification (C.ligne_verif) : (gouvernante, section, nom, niveau, taux) ; l'état d'un bloc de section
LIGNE = re.compile(r'<div class="el vc( gov)?" data-calcul="(s\d+)-\w+"[^>]*><div class="l1"><b>([^<]+)</b>.*?<span class="pct (\w)"[^>]*>([^<]+)</span></div>', re.S)
CMP = re.compile(r'<div class="el vc( gov)?"[^>]*><div class="l1"><b>([^<]+)</b>.*?<div class="l2"><span class="cmp">(.*?)</span>', re.S)
BLOC = re.compile(r'<section class="et-bloc (\w+)" data-section="(\d+)">.*?<span class="n">([^<]+)</span><span class="e">([^<]+)</span>', re.S)


def fenetre(volets, ident):
    """La fenêtre de calcul ``ck-<ident>`` seule (jusqu'à la fenêtre suivante)."""
    return re.search(r'id="ck-' + re.escape(ident) + r'".*?(?=<div class="flyout calcul"|$)', volets, re.S).group(0)


def note_type():
    """La note des tests de l'écran (tests/test_predalle.py) : Mx = My = 30, sup 12 / 9, V 60, renfort 3 Ø12 en inf. Y,
    plus une section d'appui."""
    e = req(action="nouveau")["etat"]
    for k, v in (("Mx_inf", "30"), ("My_inf", "30"), ("Mx_sup", "12"), ("My_sup", "9"), ("V", "60")):
        e = saisir(e, f"sections/0/{k}", v)["etat"]
    e = saisir(e, "sections/0/faces/inf_y/couches/+", 1)["etat"]
    e = saisir(e, "sections/+", 1)["etat"]
    for k, v in (("Mx_sup", "22"), ("My_sup", "30"), ("V", "85")):
        e = saisir(e, f"sections/1/{k}", v)["etat"]
    return e


# ------------------------------------------------------------------ 1. le résultat = le moteur
print("=== 1. Le résultat affiché est celui du moteur ===")
r0 = req(action="nouveau")
e0 = r0["etat"]
chk("une note neuve : une prédalle « Prédalle 1 », 100 × 22 cm dont 6 cm de peau, C30/37, B500, Y principale",
    len(e0["predalles"]) == 1 and {k: pd(e0)[k] for k in ("nom", "b", "h", "h_pre", "beton", "fyk", "dir_principale")}
    == {"nom": "Prédalle 1", "b": 100, "h": 22, "h_pre": 6.0, "beton": "C30/37", "fyk": 500, "dir_principale": "Y"})
E = note_type()
rep = req(action="calculer", etat=E)
m = moteur(E)
lignes = LIGNE.findall(rep["resultat"])
attendu = [(f"s{i}", P.nom_ligne(v), niveau_de(v), f"{C.pc(C.pct(v['taux']))}{C.NBSP}%")
           for i, s in enumerate(m["sections"]) for v in s["verifications"]]
chk(f"chaque vérification du moteur a sa ligne, dans l'ordre de l'écran Streamlit, avec son taux et sa couleur ({len(attendu)} lignes)",
    [(c, n, niv, t) for _, c, n, niv, t in lignes] == attendu, (lignes[:3], attendu[:3]))
g = m["gouvernant"]
gov = [x for x in lignes if x[0]]
chk("la ligne gouvernante est marquée (une seule) : celle du moteur",
    len(gov) == 1 and gov[0][1] == f"s{g['section']}" and gov[0][2] == P.nom_ligne(next(v for v in m["sections"][g["section"]]["verifications"]
                                                                                      if v["cle"] == g["cle"])), gov)
v_g = next(v for v in m["sections"][g["section"]]["verifications"] if v["cle"] == g["cle"])
gov_cmp = [c_ for g_, _, c_ in CMP.findall(rep["resultat"]) if g_]
chk("la ligne gouvernante dit sa comparaison « valeur ≤ limite », nommée ; chaque section a son bloc, coloré par son état",
    len(gov_cmp) == 1 and " ≤ " in gov_cmp[0] and [(c_, n_) for c_, _, n_, _ in BLOC.findall(rep["resultat"])]
    == [("ok", "Section A"), ("ok", "Section B")], (gov_cmp, BLOC.findall(rep["resultat"])))
chk("le résumé de la note : taux, niveau, statut, gouvernant (prédalle · section · vérification)",
    rep["resume"]["taux"] == C.pct(m["taux_max"]) and rep["resume"]["niveau"] == ("r" if m["statut"] == "nok" else C.niveau(m["taux_max"]))
    and rep["resume"]["gouvernant"].startswith("Prédalle 1 · Section"), rep["resume"])
s0 = m["sections"][0]
chk("la ligne « Inférieure Y » : la nappe et le renfort (« 10/10/100/100 + 3 Ø12 »), le sélecteur de la nappe de base",
    'data-pk="inf_y-0"' in rep["resultat"] and "10/10/100/100 + 3 Ø12" in rep["resultat"])
chk("les chiffres de l'écran Streamlit (test_predalle) : As,req 573 (principale, renfort sur la peau) et 645 mm² (secondaire)",
    round(s0["grandeurs"]["As_req_inf_y"]) == 573 and round(s0["grandeurs"]["As_req_inf_x"]) == 645)
# un acier excessif : rouge même si le taux reste sous 100 %
ex = req(action="nouveau")["etat"]
for chemin, v in (("b", "20"), ("h", "8"), ("h_pre", "4"), ("sections/0/Mx_inf", "1"), ("sections/0/faces/inf_x/couches/0/type", "barres"),
                  ("sections/0/faces/inf_x/couches/0/d", "25"), ("sections/0/faces/inf_x/couches/0/esp", "100")):
    o = saisir(ex, chemin, v)
    chk(f"saisie {chemin} = {v} acceptée", not o["message"], o["message"]) if o["message"] else None
    ex = o["etat"]
r_ex = req(action="calculer", etat=ex)
v_ex = next(v for v in moteur(ex)["sections"][0]["verifications"] if v["cle"] == "inf_x")
chk("acier excessif (Ø25/100 sur 20 × 8) : non conforme et ROUGE alors que le taux est sous 100 %",
    v_ex["etat"] == "nok" and v_ex["taux"] < 1 and "acier_excessif" in v_ex["motifs"]
    and re.search(r'<b>Inférieure X</b>.*?<span class="pct r">', r_ex["resultat"]) is not None)

# ------------------------------------------------------------------ 2. la fenêtre de calcul
print("\n=== 2. La fenêtre de calcul : hypothèses, tableau, formules substituées ===")
v = rep["volets"]
n_verifs = sum(len(s_["verifications"]) for s_ in m["sections"])
chk("une fenêtre par vérification, et une par section (ck-s0, ck-s1 : tout le calcul de la section) — comme la poutre",
    'id="ck-s0"' in v and 'id="ck-s1"' in v and v.count('class="flyout calcul"') == n_verifs + 2
    and all(f'id="ck-s{i}-{x["cle"]}"' in v for i, s_ in enumerate(m["sections"]) for x in s_["verifications"]))
chk("les hypothèses disent la peau et le coulé en place (« dont prédalle 6 cm et 16 cm coulés en place »)",
    "dont prédalle <b>6 cm</b> et 16 cm coulés en place" in v and "jeu premier lit 0 cm (nappes supérieures)" in v)
fen0 = fenetre(v, "s0")
chk("l'axe de chaque couche et sa règle : base principale c + ⌈Ø/2⌉ = 3 + 0,5 = 3,5 cm dans la prédalle ; renfort h_p + ⌈Ø/2⌉ = 6 + 1 = 7 cm sur la prédalle",
    re.search(r"e<sub>1</sub></span><span class=\"eq\">=</span><span>c \+ ⌈Ø/2⌉</span><span class=\"eq\">=</span><span>3 \+ 0,5</span>"
              r"<span class=\"eq\">=</span><span class=\"r\">3,5 cm · dans la prédalle", fen0) is not None
    and re.search(r"e<sub>2</sub></span><span class=\"eq\">=</span><span>h<sub>p</sub> \+ ⌈Ø/2⌉</span><span class=\"eq\">=</span><span>6 \+ 1</span>"
                  r"<span class=\"eq\">=</span><span class=\"r\">7,0 cm · sur la prédalle", fen0) is not None)
chk("y_G des deux couches : (785 × 3,5 + 339 × 7) / (785 + 339) = 4,6 cm", "(785 × 3,5 + 339 × 7) / (785 + 339)" in fen0 and "4,6 cm" in fen0)
chk("A_s,req de la nappe principale : 30 × 10⁶ / (333,3 × 0,9 × 174,4) = 573 mm²",
    "30 × 10⁶ / (333,3 × 0,9 × 174,4)" in fen0 and ">573 mm²<" in fen0)
taux_fen = re.findall(r'<td class="n"><span class="chip (\w)">([^<]+)</span></td>', fen0)
chk("le tableau de la section : un taux par vérification, ceux des lignes du résultat",
    [(n, t) for n, t in taux_fen] == [(niveau_de(x), f"{C.pc(C.pct(x['taux']))}{C.NBSP}%") for x in s0["verifications"]], taux_fen)
chk("la hauteur nomme la famille qui gouverne (audit C-1) et la base « sur la prédalle » de la secondaire est dite",
    "famille inf. X" in fen0 and "base sur la prédalle" in fen0 and "base dans la prédalle" in fen0)
axe_saisi = saisir(copy.deepcopy(E), "sections/0/faces/inf_x/couches/0/dist", "8")["etat"]
chk("un axe saisi est dit « saisi » dans la formule", "axe saisi" in req(action="calculer", etat=axe_saisi)["volets"])
chk("le périmètre est dit sous le résultat et dans chaque fenêtre de calcul : la section finale seulement ; phase provisoire "
    "(étaiement), cisaillement à l'interface, flèche et fissuration NON vérifiés",
    all("phase provisoire" in x and "étaiement" in x and "interface prédalle / béton coulé en place" in x and "Non vérifiés" in x
        for x in [rep["resultat"]] + rep["volets"].split('<div class="flyout calcul"')[1:]))

# ------------------------------------------------------------------ 3. la peau et la règle des axes
print("\n=== 3. La peau préfabriquée : bornes, refus, règle des axes ===")
for h, hp, ok in ((22, "3", True), (22, "15", True), (22, "2,9", False), (22, "15,5", False), (10, "9", True), (10, "9,5", False), (5, "4", True)):
    e = req(action="nouveau")["etat"]
    if h != 22:
        o = saisir(e, "h_pre", "3")
        e = saisir(o["etat"], "h", str(h))["etat"]
    o = saisir(e, "h_pre", hp)
    chk(f"peau {hp} cm sur h = {h} cm : {'admise' if ok else 'REFUSÉE avec un message, la prédalle intacte'}",
        (not o["message"] and pd(o["etat"])["h_pre"] == C.lire_nombre(hp)) if ok
        else (bool(o["message"]) and "prédalle" in o["message"] and pd(o["etat"]) == pd(e)), o["message"])
o = saisir(copy.deepcopy(e0), "h", "6")
chk("épaisseur totale 6 cm avec une peau de 6 : REFUSÉE (au moins 1 cm coulé en place), l'écran ramenait la peau sans le dire",
    o["message"] and "au moins 7 cm" in o["message"] and o["etat"] == e0, o["message"])
o = saisir(copy.deepcopy(e0), "h_pre", "abc")
chk("peau illisible : refusée", o["message"] and "illisible" in o["message"] and o["etat"] == e0)
e7 = saisir(copy.deepcopy(E), "h_pre", "7,5")["etat"]
f7 = moteur(e7)["sections"][0]["faces"]
chk("peau 7,5 cm : la base principale reste en fond de moule (3,5), la base secondaire passe à 8,0 et le renfort à 8,5",
    [f7["inf_y"]["couches"][0]["e"], f7["inf_x"]["couches"][0]["e"], f7["inf_y"]["couches"][1]["e"]] == [3.5, 8.0, 8.5])
d7 = req(action="calculer", etat=e7)["donnees"]
chk("… et le cadre Données suit : les axes automatiques disent 3,5 / 8 / 8,5 (le champ vide, « auto · 8 »), le coulé en place 14,5 cm",
    'data-chemin="sections/0/faces/inf_x/couches/0/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 8"' in d7
    and 'data-chemin="sections/0/faces/inf_y/couches/0/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 3,5"' in d7
    and 'data-chemin="sections/0/faces/inf_y/couches/1/dist" data-sorte="texte" inputmode="decimal" value="" data-vide="auto" placeholder="auto · 8,5"' in d7
    and "14,5 cm" in d7)
ej = saisir(copy.deepcopy(E), "jeu_premier_lit", "1")["etat"]
fj = moteur(ej)["sections"][0]["faces"]
chk("jeu premier lit 1 cm : nappes supérieures à 4,5 cm, nappes inférieures inchangées (3,5 / 6,5 / 7,0)",
    fj["sup_x"]["couches"][0]["e"] == 4.5 and fj["sup_y"]["couches"][0]["e"] == 4.5
    and [fj["inf_y"]["couches"][0]["e"], fj["inf_x"]["couches"][0]["e"], fj["inf_y"]["couches"][1]["e"]] == [3.5, 6.5, 7.0])
ex_ = saisir(copy.deepcopy(E), "dir_principale", "X")["etat"]
fx = moteur(ex_)["sections"][0]["faces"]
chk("direction principale X : la base X passe dans la peau (3,5), la base Y sur la peau (6,5)",
    fx["inf_x"]["couches"][0]["e"] == 3.5 and fx["inf_y"]["couches"][0]["e"] == 6.5)
d = req(action="calculer", etat=E)["donnees"]
chk("le cadre Données dit la pose de chaque couche : « dans la prédalle (c + ½ Ø) », « sur la prédalle (h_p + ½ Ø) », « depuis le dessus »",
    "axe auto · dans la prédalle (c + ½ Ø)" in d and "axe auto · sur la prédalle (h<sub>p</sub> + ½ Ø)" in d
    and "axe auto · depuis le dessus (c + ½ Ø + j)" in d)

# ------------------------------------------------------------------ 4. les saisies
print("\n=== 4. Les saisies : chaque champ, les refus, les boutons ===")
REFUS = [("sections/0/Mx_inf", "-20", "positifs"), ("sections/0/V", "x", "illisible"), ("b", "15", "entre 20 et 500"),
         ("b", "100,5", "entier"), ("h", "101", "entre 5 et 100"), ("enrobage", "25", "entre 0 et 20"),
         ("gamma_s", "0,9", "entre 1 et 2"), ("jeu_premier_lit", "-1", "au moins 0"),
         ("sections/0/faces/inf_x/couches/0/esp", "600", "entre 25 et 500"), ("sections/0/faces/inf_x/couches/0/d", "14", "n'est pas proposé"),
         ("sections/0/faces/inf_x/couches/0/treillis", "10/10/abc/100", "Øl/Øt/el/et"), ("sections/0/faces/inf_x/couches/0/type", "grille", "inconnu"),
         ("sections/0/faces/inf_x/couches/0/dist", "-3", "plus de 0"), ("sections/0/faces/inf_x/yG", "0", "plus de 0"),
         ("sections/0/nom", "  ", "vide"), ("beton", "C16/20", "inconnue"), ("fyk", "450", "B400 ou B500"), ("dir_principale", "Z", "X ou Y"),
         ("sections/0/faces/inf_x/couches/0/-", 1, "base ne se retire pas")]
mauvais = []
for chemin, val, mot in REFUS:
    o = saisir(copy.deepcopy(E), chemin, val)
    if not (o["message"] and mot in o["message"] and o["etat"] == E):
        mauvais.append((chemin, val, o["message"]))
chk(f"{len(REFUS)} saisies hors du domaine de l'écran REFUSÉES avec un message clair, la note intacte", not mauvais, mauvais)
e1 = saisir(copy.deepcopy(E), "sections/0/faces/inf_y/couches/1/dist", "9,5")["etat"]
c1 = pd(e1)["sections"][0]["faces"]["inf_y"]["couches"][1]
e2 = saisir(copy.deepcopy(e1), "sections/0/faces/inf_y/couches/1/dist", "")["etat"]
e3 = saisir(copy.deepcopy(E), "sections/0/faces/inf_y/couches/1/dist", "7")["etat"]
chk("un axe saisi (9,5) est imposé ; vidé, il revient à l'automatique ; l'automatique réécrit (7) reste automatique",
    c1["dist"] == 9.5 and pd(e2)["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] is None
    and pd(e3)["sections"][0]["faces"]["inf_y"]["couches"][1]["dist"] is None)
yg = saisir(copy.deepcopy(E), "sections/0/faces/inf_y/yG", "5,25")["etat"]
chk("un centre de gravité saisi remplace Σ As·e / Σ As (son champ le montre), et VIDÉ il revient au calcul",
    moteur(yg)["sections"][0]["grandeurs"]["yG_inf_y"] == 5.25
    and 'data-chemin="sections/0/faces/inf_y/yG" data-sorte="texte" inputmode="decimal" value="5,25" data-vide="auto"' in req(action="calculer", etat=yg)["donnees"]
    and pd(saisir(yg, "sections/0/faces/inf_y/yG", "")["etat"])["sections"][0]["faces"]["inf_y"]["yG"] is None)
ea = copy.deepcopy(E)
for _ in range(3):
    ea = saisir(ea, "sections/0/faces/sup_x/couches/+", 1)["etat"]
o = saisir(copy.deepcopy(ea), "sections/0/faces/sup_x/couches/+", 1)
chk("renforts : 3 Ø12 « n barres » par défaut, quatre couches au plus (le cinquième refusé)",
    [c["type"] for c in pd(ea)["sections"][0]["faces"]["sup_x"]["couches"]] == ["treillis", "n_barres", "n_barres", "n_barres"]
    and o["message"] and "Au plus 4" in o["message"])
er = saisir(copy.deepcopy(ea), "sections/0/faces/sup_x/couches/2/-", 1)["etat"]
chk("retirer un renfort", len(pd(er)["sections"][0]["faces"]["sup_x"]["couches"]) == 3)
ec = saisir(copy.deepcopy(E), "sections/0/copie", 1)["etat"]
chk("copier une section : toutes ses données, la première lettre libre (C)",
    len(pd(ec)["sections"]) == 3 and pd(ec)["sections"][2]["nom"] == "C"
    and {k: v for k, v in pd(ec)["sections"][2].items() if k != "nom"} == {k: v for k, v in pd(ec)["sections"][0].items() if k != "nom"})
es = saisir(copy.deepcopy(E), "sections/1/-", 1)["etat"]
o = saisir(copy.deepcopy(es), "sections/0/-", 1)
chk("retirer une section ; la dernière ne se retire pas", len(pd(es)["sections"]) == 1 and o["message"] and "au moins une" in o["message"])
eg = saisir(copy.deepcopy(E), "gamma_s", "1,15")
eg = req(action="ajouter", etat=eg["etat"])
chk("γs est commun à la note (comme l'écran) : toutes les prédalles le prennent, une prédalle ajoutée aussi",
    all(p["gamma_s"] == 1.15 for p in eg["etat"]["predalles"]) and len(eg["etat"]["predalles"]) == 2)
ev = saisir(copy.deepcopy(E), "verrou", True)["etat"]
o = saisir(copy.deepcopy(ev), "h", "25")
rv = req(action="calculer", etat=ev)
chk("verrou : les données se lisent, rien ne se modifie, aucun sélecteur ; décoché, la prédalle se modifie",
    o["message"] and "verrouillée" in o["message"] and o["etat"] == ev and "data-pk" not in rv["resultat"] and "data-chemin=\"h\"" not in rv["donnees"]
    and not saisir(saisir(ev, "verrou", False)["etat"], "h", "25")["message"])
chk("choisir dans un sélecteur d'une prédalle verrouillée : refusé (erreur lisible)",
    not req(action="choisir", etat=ev, role="inf_y-0", nom="Ø12/150")["ok"])

# ------------------------------------------------------------------ 5. le sélecteur
print("\n=== 5. Le sélecteur de la nappe de base ===")
s = req(action="selecteur", etat=E, role="inf_y-0")
cands = MO.candidats_base({k: v for k, v in pd(E).items() if k != "verrou"}, 0, "inf_y")
noms = re.findall(r'data-choix="([^"]+)"', s["html"])
chk(f"les {len(cands)} nappes du moteur, triées par section au mètre", noms == [c["nom"] for c in cands])
cur = re.findall(r'class="pk-row cur[^"]*" data-choix="([^"]+)"', s["html"])
opt = re.findall(r'class="pk-row[^"]* opt" data-choix="([^"]+)"', s["html"])
chk("la retenue ✓ et la plus petite qui tient ★ : celles du moteur",
    cur == [c["nom"] for c in cands if c["retenue"]] and opt == [c["nom"] for c in cands if c["optimal"]], (cur, opt))
chk("chaque ligne dit son axe, recalculé par la règle de la prédalle (Ø20 dans la peau : 4,0 cm)",
    "axe 4,0 cm (dans la prédalle)" in s["html"] and "couche de base (principale), dans la prédalle" in s["html"])
s2 = req(action="selecteur", etat=E, role="inf_x-0", familles=["barres"])
chk("le filtre « Barres » et la nappe secondaire « sur la prédalle »",
    all(n.startswith("Ø") for n in re.findall(r'data-choix="([^"]+)"', s2["html"])) and "sur la prédalle" in s2["html"])
ch = req(action="choisir", etat=E, role="inf_y-0", nom="Ø12/150")
cs = pd(ch["etat"])["sections"][0]["faces"]["inf_y"]["couches"]
chk("choisir « Ø12/150 » : la base devient des barres Ø12 tous les 150 mm, le renfort est gardé",
    ch["ok"] and (cs[0]["type"], cs[0]["d"], cs[0]["esp"]) == ("barres", 12, 150) and len(cs) == 2)
etoiles = re.findall(r'<span class="leger" title="Plus petite nappe de base qui tient : ([^"]+)">', rep["resultat"])
chk("l'étoile des lignes du résultat : la plus petite nappe qui tient, quand ce n'est pas la retenue", len(etoiles) >= 4, etoiles[:3])

# ------------------------------------------------------------------ 6. les éléments de la note
print("\n=== 6. Les prédalles de la note ===")
a = req(action="ajouter", etat=copy.deepcopy(E))
chk("ajouter : « Prédalle 2 », ouverte", [p["nom"] for p in a["etat"]["predalles"]] == ["Prédalle 1", "Prédalle 2"] and a["etat"]["actif"] == 1)
dpl = req(action="dupliquer", etat=copy.deepcopy(E))
chk("copier : « Prédalle 1 (copie) », mêmes données", dpl["etat"]["predalles"][1]["nom"] == "Prédalle 1 (copie)"
    and {k: v for k, v in dpl["etat"]["predalles"][1].items() if k != "nom"} == {k: v for k, v in pd(E).items() if k != "nom"})
deux = saisir(a["etat"], "sections/0/My_inf", "80")["etat"]
r2 = req(action="calculer", etat=deux)
chk("le résumé de la note : la prédalle la plus chargée, même si l'ouverte l'est moins",
    r2["resume"]["gouvernant"].startswith("Prédalle 2") and r2["resume"]["taux"] == C.pct(max(moteur(deux, 0)["taux_max"], moteur(deux, 1)["taux_max"])))
chk("l'arborescence : chaque prédalle « nom · b × h cm », sans taux (le résultat est à droite) ; la liste de l'ancien écran n'existe plus",
    r2["donnees"].count('class="et-pt-t') == 2 and noms_arbre(r2["donnees"])
    == [("Prédalle 1", "· 100 × 22 cm"), ("Prédalle 2", "· 100 × 22 cm")] and r2["elements"] == "")
sup = req(action="supprimer", etat=copy.deepcopy(deux), index=0)
chk("supprimer : il en reste une, ouverte", len(sup["etat"]["predalles"]) == 1 and sup["etat"]["predalles"][0]["nom"] == "Prédalle 2")

# ------------------------------------------------------------------ 7. le fichier de l'écran Streamlit
print("\n=== 7. La reprise d'un fichier de l'écran Streamlit ===")
fichier = {"version": "predalle-1.0", "predalles": [{"id": 1, "nom": "Plancher", "sections": [{"id": 1, "nom": "A"}, {"id": 2, "nom": "Appui"}]}],
           "values": {"pre1_h": 20, "pre1_h_pre": 5.0, "pre1_sec1_My_inf": 35.0, "pre1_sec1_Mx_inf_raw": "-12,0", "pre1_sec2_My_sup": 18.0,
                      "pre1_sec1_ncouches_inf_y": 2, "pre1_sec1_arm_type_inf_y_c2": "n barres", "gamma_s": 1.5}}
o = saisir(copy.deepcopy(E), "fichier_streamlit", {"nom": "predalle_ba.json", "texte": json.dumps(fichier)})
p0 = pd(o["etat"])
chk("predalle_ba.json : la prédalle, ses deux sections, sa peau de 5 cm et son renfort sont repris",
    p0["nom"] == "Plancher" and [s_["nom"] for s_ in p0["sections"]] == ["A", "Appui"] and p0["h_pre"] == 5.0
    and len(p0["sections"][0]["faces"]["inf_y"]["couches"]) == 2 and p0["sections"][0]["My_inf"] == 35.0)
chk("… et ce que l'écran y remplaçait en silence est dit (le moment négatif lu 0)", o["message"] and "lu 0" in o["message"], o["message"])
o = saisir(copy.deepcopy(E), "fichier_streamlit", {"nom": "x.json", "texte": "{pas du json"})
chk("un fichier qui n'est pas du JSON : refusé, la note intacte", o["message"] and "JSON" in o["message"] and o["etat"] == E)
f_peau = copy.deepcopy(fichier)
f_peau["values"].update({"pre1_h": 8, "pre1_h_pre": 12.0})
o = saisir(copy.deepcopy(E), "fichier_streamlit", {"nom": "p.json", "texte": json.dumps(f_peau)})
chk("une peau de 12 cm sur 8 cm : reprise à 7 cm comme l'écran, et c'est dit", pd(o["etat"])["h_pre"] == 7.0 and "ramenée à 7 cm" in o["message"])

# ------------------------------------------------------------------ 8. les avertissements
print("\n=== 8. Les avertissements ===")
rr = req(action="calculer", etat=E)["resultat"]
bloc0 = re.search(r'<section class="et-bloc \w+" data-section="0">.*?</section>', rr, re.S).group(0)
chk("les avertissements d'une section sont posés DANS son bloc (sous ses vérifications)",
    "posées toutes deux sur la prédalle" in bloc0 and "les bases X et Y supérieures sont calculées au même niveau" in bloc0)
chk("couches croisées sur la peau (renfort Y à 7,0, base X à 6,5) : dit, avec les axes",
    "posées toutes deux sur la prédalle" in rr and "axe 7,0 cm" in rr and "axe 6,5 cm" in rr)
chk("les bases supérieures X et Y au même niveau : dit en une phrase par section", "les bases X et Y supérieures sont calculées au même niveau" in rr)
chk("deux couches croisées à 6,5 et 7,0 cm ne sont pas dites « au même niveau » mais « à des niveaux trop proches pour se croiser »",
    "sur la prédalle, à des niveaux trop proches pour se croiser (il faut au moins 1,1 cm d" in rr, texte(rr)[-900:])
d_ = req(action="calculer", etat=E)["donnees"]
pictos = re.findall(r'<svg class="pd-m pd-(\w+)" viewBox="0 0 18 12" aria-hidden="true">', d_)
chk("chaque couche porte le pictogramme de sa pose (une coupe en miniature : dans la peau, sur la peau, sous le dessus), sans identifiant",
    sorted(set(pictos)) == ["predalle", "superieure", "sur_predalle"] and len(pictos) == sum(len(f["couches"]) for s_ in pd(E)["sections"]
                                                                                         for f in s_["faces"].values())
    and " id=" not in "".join(re.findall(r'<svg class="pd-m.*?</svg>', d_)), pictos)
eh = saisir(saisir(copy.deepcopy(e0), "h_pre", "3")["etat"], "sections/0/faces/inf_y/couches/0/treillis", "12/12/150/150")["etat"]
rh = req(action="calculer", etat=eh)["resultat"]
chk("base Ø12 à 4,0 cm dans une peau de 3 cm : « sort de la peau », message grave", '<div class="msg grave">' in rh and "sort de la peau" in rh)

# ------------------------------------------------------------------ 9. les dessins
print("\n=== 9. Les dessins ===")
vues = rep["vues"]
chk("deux vues par section : la coupe et le plan, puis « Toutes » ; titres en texte, chacune sous son en-tête technique",
    [x["cle"] for x in vues] == ["c0", "p0", "c1", "p1", "toutes"] and all(x["svg"].startswith("<svg") for x in vues)
    and all("<" not in x["titre"] + x["court"] + x["legende"] for x in vues) and all(x["entete"].startswith('<div class="et-ent">') for x in vues)
    and "<b>Section A</b><span>· 100 × 22 cm, prédalle 6 cm · C30/37 · B500 · principale Y</span>" in vues[0]["entete"])
sv = vues[0]["svg"]
chk("la coupe : la peau hachurée (hachures dessinées, aucun motif SVG à identifiant partagé), le plan de reprise, les deux bétons nommés",
    'stroke="#8C97A6"' in sv and "<pattern" not in sv and "prédalle préfabriquée · 60 mm" in sv and "béton coulé en place · 160 mm" in sv)
chk("la chaîne de cotes : h = 220 = 60 + 160 ; axes 35 / 65 ; hauteur utile d = 174,4 (le CDG du renfort sur la peau)",
    ">h = 220<" in sv and ">60<" in sv and ">160<" in sv and ">35<" in sv and ">65<" in sv and ">d = 174,4<" in sv)
chk("la légende dit la pose : « dans la prédalle », « sur la prédalle »", "e = 35 · dans la prédalle" in sv and "e = 70 · sur la prédalle" in sv)
rond = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="5"', sv)
chk("à l'échelle : les fils Ø10 dessinés en cercles de 5 mm de rayon, les Ø12 de 6 mm", len(rond) >= 10 and 'r="6"' in sv)
chk("le plan : les nappes inférieure et supérieure, le fond grisé de la prédalle, le repère", "Nappe inférieure" in vues[1]["svg"]
    and "Nappe supérieure" in vues[1]["svg"] and "fond grisé : la prédalle" in vues[1]["svg"])
eb = saisir(copy.deepcopy(e0), "b", "250")["etat"]
chk("une bande de 2,50 m : seule sa partie centrale est dessinée, et la cote le dit (« 1 200 vus sur b = 2 500 »)",
    "1 200 vus sur b = 2 500" in req(action="calculer", etat=eb)["vues"][0]["svg"])
PIEGE = '<img src=x onerror=alert(1)>'
ep = saisir(saisir(copy.deepcopy(E), "nom", PIEGE)["etat"], "sections/0/nom", PIEGE)["etat"]
rp = req(action="calculer", etat=ep)
fuites = [k for k in ("resultat", "volets", "donnees", "elements", "phrase") if "<img" in rp[k]] + [x["cle"] for x in rp["vues"] if "<img" in x["svg"] + x["entete"]]
chk("un nom piégé (prédalle, section) est échappé dans tous les fragments et dans les dessins", not fuites, fuites)
chk("dessin_predalle.positions : base centrée sur la bande, renfort intercalé d'une demi-maille, « n barres » réparties",
    DP.positions({"type": "barres", "esp": 150}, 1000, 0)[0] == 500 - 3 * 150
    and DP.positions({"type": "barres", "esp": 150}, 1000, 1)[0] == 500 - 3 * 150 + 75
    and DP.positions({"type": "n_barres", "n": 4}, 1000, 0) == [125, 375, 625, 875])

# ------------------------------------------------------------------ 10. robustesse (relecture du 07/11/2026)
print("\n=== 10. Robustesse : l'absurde, la note abîmée, le verdict d'une prédalle non conforme ===")
ABSURDES = [("sections/0/Mx_inf", "1e300", "Au plus 100"), ("sections/0/V", "1e9", "Au plus 100"),
            ("sections/0/faces/inf_x/couches/0/dist", "1e6", "au plus 100 cm"), ("sections/0/faces/inf_x/yG", "500", "au plus 100 cm"),
            ("jeu_premier_lit", "1e5", "au plus 100 cm"), ("nom", "x" * 61, "60 caractères"), ("sections/0/nom", "y" * 61, "60 caractères"),
            ("verrou", "peut-être", "coché ou décoché"), ("sections/9/Mx_inf", "3", "champ introuvable"),
            ("sections/0/faces/zz/yG", "3", "champ introuvable"), ("sections/x/V", "3", "champ introuvable")]
mauvais = []
for chemin, val, mot in ABSURDES:
    o = saisir(copy.deepcopy(E), chemin, val)
    if not (o["message"] and mot in o["message"] and o["etat"] == E and len(o["message"]) < 400):
        mauvais.append((chemin, val, o["message"]))
chk(f"{len(ABSURDES)} saisies absurdes (10³⁰⁰ kN·m, axe de 10 km, nom de 61 caractères, chemin inexistant…) REFUSÉES avec un "
    "message court et lisible, la note intacte (l'écran Streamlit calculait des taux de 10³⁰⁰ %)", not mauvais, mauvais)
chk("le verrou lit « false » comme décoché (et non comme un texte non vide, donc coché)",
    not pd(saisir(copy.deepcopy(E), "verrou", "false")["etat"])["verrou"] and pd(saisir(copy.deepcopy(E), "verrou", "true")["etat"])["verrou"])
ABIMEES = [{"version": 1, "predalles": "abc"}, {"predalles": [{"sections": [{"Mx_inf": 1e300}]}]},
           {"predalles": [{"jeu_premier_lit": 1e300}]}, {"predalles": [{"sections": [{"faces": {"inf_x": {"yG": 1e300}}}]}]},
           {"predalles": [{"nom": "z" * 200}]}]
lues = [req(action="calculer", etat=x) for x in ABIMEES]
chk("une note .stab abîmée (« predalles » qui n'est pas une liste, valeurs absurdes) est REFUSÉE avec une erreur lisible, "
    "jamais remplacée en silence par une note neuve", all(not x["ok"] and x.get("erreur") and len(x["erreur"]) < 300 for x in lues),
    [x.get("erreur") for x in lues])
vide = req(action="calculer", etat={"predalles": [{"sections": [{"nom": " "}, {"nom": "A"}]}]})
chk("un nom de section vide dans un .stab prend la première lettre libre (pas de « Section  » muette)",
    vide["ok"] and [s_["nom"] for s_ in pd(vide["etat"])["sections"]] == ["B", "A"], vide.get("etat", {}).get("predalles"))
f_abs = copy.deepcopy(fichier)
f_abs["values"]["pre1_sec1_My_inf"] = 1e300
o = saisir(copy.deepcopy(E), "fichier_streamlit", {"nom": "absurde.json", "texte": json.dumps(f_abs)})
chk("un fichier Streamlit aux valeurs absurdes : refusé, la note intacte", o["message"] and "refusé" in o["message"] and o["etat"] == E, o["message"])
# le verdict d'une prédalle non conforme nomme une vérification NON conforme (d utile ≤ 0, acier excessif sous 100 %)
ej30 = saisir(copy.deepcopy(e0), "jeu_premier_lit", "30")["etat"]
rj = req(action="calculer", etat=ej30)
cj = CMP.findall(rj["resultat"])
chk("jeu de 30 cm sur 22 cm, SANS AUCUN EFFORT : d ≤ 0 en haut, taux sous 100 % ; le bloc dit « Non conforme » (jamais « Données "
    "incomplètes », qui cacherait la non-conformité), le résumé de la note nomme la nappe supérieure, en rouge",
    BLOC.findall(rj["resultat"]) == [("nok", "0", "Section A", "Non conforme")] and rj["resume"]["niveau"] == "r"
    and "supérieure" in rj["resume"]["gouvernant"], (BLOC.findall(rj["resultat"]), rj["resume"]))
chk("… la ligne gouvernante est la nappe supérieure, rouge sous 100 %, et sa comparaison dit pourquoi (« d = … ≤ 0 »)",
    any(g_ and n.startswith("Supérieure") and "≤ 0" in c_ for g_, n, c_ in cj)
    and re.search(r'<b>Supérieure Y</b>.*?<span class="pct r"', rj["resultat"]) is not None, [(x[0], x[1]) for x in cj])
ax = req(action="nouveau")["etat"]
for chemin, v in (("b", "20"), ("h", "8"), ("h_pre", "4"), ("sections/0/faces/inf_x/couches/0/type", "barres"),
                  ("sections/0/faces/inf_x/couches/0/d", "25"), ("sections/0/faces/inf_x/couches/0/esp", "100")):
    ax = saisir(ax, chemin, v)["etat"]
r_ax = req(action="calculer", etat=ax)
c_ax = CMP.findall(r_ax["resultat"])
m_ax = moteur(ax)["sections"][0]["verifications"]
chk("acier excessif SEUL en défaut (sans moment, Ø25/100 sur 20 × 8 : taux 2 %, hauteur conforme à 69 %) : la ligne gouvernante est "
    "« Inférieure X » et compare A_s > A_s,max — et non la hauteur, plus chargée mais conforme ; le bloc et la note sont rouges",
    [v["cle"] for v in m_ax if v["etat"] == "nok"] == ["inf_x"] and max(m_ax, key=lambda v: v["taux"])["cle"] == "hauteur"
    and [(n, "A<sub>s,max</sub>" in c_ and "acier excessif" in c_) for g_, n, c_ in c_ax if g_] == [("Inférieure X", True)]
    and BLOC.findall(r_ax["resultat"])[0][0] == "nok"
    and r_ax["resume"]["niveau"] == "r" and r_ax["resume"]["gouvernant"].endswith("inférieure X"), (c_ax, r_ax["resume"]))


# ------------------------------------------------------------------ 11. bout à bout : l'écran Streamlit rejoué face au SITE
print("\n=== 11. Bout à bout : ce que le SITE affiche = ce que l'écran Streamlit modules/predalle.py affiche ===")
try:
    import contextlib
    import logging
    import random
    logging.disable(logging.WARNING)
    import streamlit as st  # noqa: F401 — l'écran l'importe
    import modules.predalle as D
    with open(os.path.join(RACINE, "beton_classes.json"), encoding="utf-8") as fh:
        D.BETON_DATA = json.load(fh)
except ImportError as e:  # pragma: no cover — la comparaison EXIGE l'écran
    D = None
    chk("streamlit et modules/predalle.py se chargent (indispensables à la comparaison bout à bout)", False, e)

TYPES_ECRAN = {"treillis": "Treillis", "barres": "Barres", "n_barres": "n barres"}


def charger_ecran(predalles):
    """L'état de l'écran Streamlit pour la note ``predalles`` (format du moteur), puis le début d'un run et les
    synchronisations des champs « Dist. axe » et « CDG » (comme tests/test_predalle_moteur.py)."""
    ss = st.session_state
    ss.clear()
    D._ensure_global_defaults()
    ss["gamma_s"], ss["jeu_enrobage_cm"] = predalles[0]["gamma_s"], predalles[0]["jeu_premier_lit"]
    ss.predalles = [{"id": k + 1, "nom": p["nom"], "sections": [{"id": i + 1, "nom": s_["nom"]} for i, s_ in enumerate(p["sections"])]}
                    for k, p in enumerate(predalles)]
    for k, p in enumerate(predalles):
        did = k + 1
        ss[f"meta_predalle_nom_{did}"] = p["nom"]
        for cle_ecran, cle in (("b", "b"), ("h", "h"), ("h_pre", "h_pre"), ("enrobage_beton", "enrobage"), ("beton", "beton"),
                               ("fyk", "fyk"), ("dir_principale", "dir_principale")):
            ss[D.KD(cle_ecran, did)] = p[cle]
        for i, s_ in enumerate(p["sections"]):
            sid = i + 1
            ss[f"meta_pre{did}_nom_{sid}"] = s_["nom"]
            for x in EN.MOMENTS + ("V",):
                ss[D.KS(x, did, sid)] = s_[x]
            for which in EN.FACES:
                f = s_["faces"][which]
                ss[D.KS(f"ncouches_{which}", did, sid)] = len(f["couches"])
                for j, c in enumerate(f["couches"], start=1):
                    for cle_ecran, v_ in (("arm_type", TYPES_ECRAN[c["type"]]), ("treillis", c["treillis"]), ("ø_barres", c["d"]),
                                          ("esp_barres", c["esp"]), ("n_barres", c["n"])):
                        ss[D.KS(f"{cle_ecran}_{which}_c{j}", did, sid)] = v_
                    if c["dist"] is not None:
                        ss[D.KS(f"dist_axe_{which}_c{j}", did, sid)] = repr(float(c["dist"])).replace(".", ",")
                        ss[D.KS(f"dist_auto_{which}_c{j}", did, sid)] = False
                if f["yG"] is not None:
                    ss[D.KS(f"ycdg_{which}", did, sid)] = repr(float(f["yG"])).replace(".", ",")
                    ss[D.KS(f"ycdg_auto_{which}", did, sid)] = False
    D._ensure_global_defaults()
    D._init_dalles_if_needed()
    for d_ in ss.predalles:
        for s_ in d_["sections"]:
            for which in EN.FACES:
                for j in range(1, D._get_ncouches(int(d_["id"]), int(s_["id"]), which) + 1):
                    D._sync_dist_state(int(d_["id"]), int(s_["id"]), which, j)
                D._sync_ycdg_state(int(d_["id"]), int(s_["id"]), which)


class FauxSt:
    """Un « st » qui ENREGISTRE ce que l'écran afficherait (l'état reste le vrai st.session_state)."""

    def __init__(self):
        self.ev = []
        self.session_state = st.session_state

    def markdown(self, body="", unsafe_allow_html=False, help=None, **_):
        self.ev.append(("md", str(body), help or ""))

    def expander(self, label, expanded=False, **_):
        self.ev.append(("exp", label))
        return contextlib.nullcontext()

    def container(self, **_):
        return contextlib.nullcontext()

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def _widget(self, *_, key=None, **__):
        return self.session_state.get(key) if key else None

    selectbox = text_input = number_input = checkbox = _widget

    def button(self, *_, **__):
        return False


def ecran_affiche(predalles):
    """Ce que render_dimensionnement_right AFFICHE : par prédalle, son état ; par section, son état et ses blocs
    (titre, texte de droite, état, pourcentage exact, lignes et infobulles qui suivent)."""
    charger_ecran(predalles)
    faux = FauxSt()
    vrai_st, vrai_bloc = D.st, D.open_bloc_left_right

    def espion(left, right="", etat="ok", pct=None):
        faux.ev.append(("bloc", left, right, etat, pct))
        return vrai_bloc(left, right, etat, pct)
    D.st, D.open_bloc_left_right = faux, espion
    try:
        D.render_dimensionnement_right(D.BETON_DATA)
    finally:
        D.st, D.open_bloc_left_right = vrai_st, vrai_bloc
    out, exp = [], 0
    for ev in faux.ev:
        if ev[0] == "exp":
            etat = {"🟢": "ok", "🟡": "warn", "🔴": "nok"}[ev[1][:1]]
            if exp == 0 or (out and len(out[-1]["sections"]) == len(predalles[len(out) - 1]["sections"])):
                out.append({"etat": etat, "sections": []})
            else:
                out[-1]["sections"].append({"etat": etat, "blocs": []})
            exp += 1
        elif ev[0] == "bloc":
            out[-1]["sections"][-1]["blocs"].append({"titre": ev[1], "droite": ev[2], "etat": ev[3], "pct": ev[4], "md": []})
        elif out and out[-1]["sections"] and out[-1]["sections"][-1]["blocs"] and not ev[1].startswith(("<div", "</div>")):
            out[-1]["sections"][-1]["blocs"][-1]["md"].append((ev[1], ev[2]))
    return out


def _num(t):
    """« 8 800 », « 0,36 », « 0.36 » → nombre."""
    return float(re.sub(r"[\s  ]", "", t).replace(",", "."))


def site_affiche(etat, k):
    """Ce que le SITE affiche pour la prédalle k : ses lignes de résultat (section, nom, niveau, taux affiché), ses
    fenêtres de calcul (par section et par vérification : symbole → résultat affiché), son taux dans la liste."""
    e = dict(copy.deepcopy(etat), actif=k)
    rep = req(action="calculer", etat=e)
    assert rep["ok"], rep
    lignes = [(int(c[1:]), n, niv, int(t.replace(C.NBSP + "%", ""))) for _, c, n, niv, t in LIGNE.findall(rep["resultat"])]
    groupes = []
    # (les fenêtres de SECTION, « ck-s<i> », dans l'ordre : chacune détaille toutes ses vérifications)
    for i, fen in enumerate(re.findall(r'<div class="flyout calcul" id="ck-s\d+">.*?(?=<div class="flyout calcul"|$)', rep["volets"], re.S)):
        gs = []
        for g_ in fen.split('<div class="ck-g">')[1:]:
            fx = {}
            for ligne in re.findall(r'<div class="fx">(.*?)</div>', g_):
                sym = re.sub(r"<[^>]+>", "", re.match(r'<span class="s">(.*?)</span>', ligne).group(1))
                fx[sym] = re.search(r'<span class="(?:r|chip \w)">([^<]*)</span>$', ligne).group(1)
                fx[sym + "#"] = texte(ligne)          # la ligne entière : « hmin = hu,min + yG = 11,6 + 6,5 = 18,1 cm »
            gs.append(fx)
        groupes.append(gs)
    # l'état de chaque section : la couleur de l'en-tête de son bloc (rouge : « Non conforme »)
    secs = ["r" if c_ == "nok" else c_ for c_, _, _, _ in BLOC.findall(rep["resultat"])]
    return {"lignes": lignes, "groupes": groupes, "sections": secs, "resume": rep["resume"]}


NOMS_BLOCS = {"Vérification de la hauteur": "Hauteur", "Vérification de l'effort tranchant": "Effort tranchant"}


def nom_bloc(titre):
    if titre in NOMS_BLOCS:
        return NOMS_BLOCS[titre]
    m = re.match(r"Armatures (inférieures|supérieures) — dir\. ([XY])", titre)
    return f"{m.group(1)[:-1].capitalize()} {m.group(2)}"


def comparer_bout_a_bout(nom_cas, predalles):
    """Chaque prédalle de la note : l'écran Streamlit rejoué contre le site, bloc par bloc, à l'arrondi d'affichage."""
    ecart = []
    ecr = ecran_affiche(predalles)
    etat = {"version": 1, "predalles": [dict(p, verrou=False) for p in predalles], "actif": 0}
    for k, p in enumerate(predalles):
        site = site_affiche(etat, k)
        e_p = ecr[k]
        blocs = [(i, b) for i, s_ in enumerate(e_p["sections"]) for b in s_["blocs"]]
        if [(i, nom_bloc(b["titre"])) for i, b in blocs] != [(i, n) for i, n, _, _ in site["lignes"]]:
            ecart.append(("vérifications (ordre, nombre)", [(i, nom_bloc(b["titre"])) for i, b in blocs], [(i, n) for i, n, _, _ in site["lignes"]]))
            continue
        for (i, b), (_, n, niv, t), fx in zip(blocs, site["lignes"], [g_ for gs in site["groupes"] for g_ in gs]):
            ou = f"{p['nom']} · section {i} · {n}"
            # le taux : celui de l'écran (pourcentage exact), arrondi par le site sans quitter sa bande de couleur
            if b["pct"] is None or t != C.pct(b["pct"] / 100.0) or abs(t - b["pct"]) > 1.0:
                ecart.append((ou, "taux", b["pct"], t))
            if (b["etat"] == "nok") != (niv == "r"):
                ecart.append((ou, "état", b["etat"], niv))
            md = [x for x, _ in b["md"]]
            aides = " ".join(a for _, a in b["md"])
            if n == "Hauteur":
                hu = re.search(r"\*\*(-?[\d,]+) cm\*\*", md[0]).group(1)
                hmin = re.search(r"= \*\*(-?[\d,]+) cm\*\*", md[1]).group(1)
                d1 = re.search(r"\+ (-?[\d,]+) =", md[1]).group(1)
                vu = {"hu,min": fx.get("hu,min", ""), "hmin": fx.get("hmin", "")}
                if _num(vu["hu,min"].replace("cm", "")) != _num(hu) or _num(vu["hmin"].replace("cm", "")) != _num(hmin):
                    ecart.append((ou, "hauteur", (hu, hmin), vu))
                yG_site = re.search(r"=\s*[\d,]+ \+ ([\d,]+)\s*=", fx["hmin#"])
                if yG_site is None or abs(_num(d1) - _num(yG_site.group(1))) > 0.05 + 1e-9:
                    ecart.append((ou, "d₁ (y_G de la famille qui gouverne)", d1, fx.get("hmin#")))
            elif n == "Effort tranchant":
                tau, adm = re.findall(r"([\d.]+) N/mm²", md[0])
                if (_num(fx["τ"].replace("N/mm²", "")), _num(fx["τadm,I"].replace("N/mm²", ""))) != (float(tau), float(adm)):
                    ecart.append((ou, "τ", (tau, adm), (fx["τ"], fx["τadm,I"])))
            else:
                ecran_v = {k_: _num(re.search(rf"\*\*{re.escape(lib)} = ([\d,]+) mm", " ".join(md)).group(1))
                           for k_, lib in (("As,req", "Aₛ,req"), ("As,min", "Aₛ,min"), ("As,max", "Aₛ,max"), ("As/m", "Aₛ fourni"))}
                ecran_v["As"] = _num(re.search(r"\((\d+) mm²\)$", b["droite"]).group(1))
                ecran_v["d"] = _num(re.search(r"d = (-?[\d,]+) cm", aides).group(1))
                site_v = {"As,req": _num(fx["As,req"].replace("mm²", "")), "As,min": _num(fx["As,min"].replace("mm²", "")),
                          "As,max": _num(fx["As,max"].replace("mm²", "")), "As": _num(fx["As"].replace("mm²", "")),
                          "d": _num(fx["d"].replace("cm", ""))}
                site_v["As/m"] = _num(re.search(r"/ 100\s*=\s*([\d  ]+) ×", fx["As#"]).group(1))
                if ecran_v != site_v:
                    ecart.append((ou, "armatures", ecran_v, site_v))
        for i, s_ in enumerate(e_p["sections"]):
            if (s_["etat"] == "nok") != (site["sections"][i] == "r"):
                ecart.append((p["nom"], f"état de la section {i}", s_["etat"], site["sections"][i]))
        # l'état de la prédalle : rouge dès qu'un de ses blocs l'est (l'arborescence ne porte plus de pastille)
        if (e_p["etat"] == "nok") != ("r" in site["sections"]):
            ecart.append((p["nom"], "état de la prédalle (ses blocs)", e_p["etat"], site["sections"]))
    return ecart


def _couche(t="treillis", des="10/10/100/100", d=12, esp=150, n=3, dist=None):
    return {"type": t, "treillis": des, "d": d, "esp": esp, "n": n, "dist": dist}


def _fa(*couches, yG=None):
    return {"couches": list(couches) or [_couche()], "yG": yG}


def _sec(nom, Mx_inf=0.0, Mx_sup=0.0, My_inf=0.0, My_sup=0.0, V=0.0, **faces):
    return {"nom": nom, "Mx_inf": Mx_inf, "Mx_sup": Mx_sup, "My_inf": My_inf, "My_sup": My_sup, "V": V,
            "faces": {f: faces.get(f) or _fa() for f in EN.FACES}}


def _note(*predalles, gamma_s=1.5, jeu=0.0):
    """Une NOTE : ses prédalles partagent γs et le jeu (paramètres communs de l'écran Streamlit)."""
    return [MO.normaliser(dict(p, gamma_s=gamma_s, jeu_premier_lit=jeu)) for p in predalles]


def _pre(nom, sections, **kw):
    return dict({"nom": nom, "sections": sections}, **kw)


TR_, BA_, NB_ = (lambda des, dist=None: _couche("treillis", des, dist=dist)), (lambda d, e, dist=None: _couche("barres", d=d, esp=e, dist=dist)), \
    (lambda n, d, dist=None: _couche("n_barres", d=d, n=n, dist=dist))
TP = _sec("A", 30, 12, 30, 9, 60)
TP_R = _sec("A", 30, 12, 30, 9, 60, inf_y=_fa(TR_("10/10/100/100"), NB_(3, 12)))
# τ = τ_adm,I À LA LIMITE (C30/37 : 0,016 × 37 / 1,05 ; bande 100 × 22) : 100 % pile, puis un cheveu au-dessus
V_LIM = 0.016 * 37 / 1.05 * 0.75 * 100 * 22 * 100 / 1e3
BOUT = [
    ("note neuve (aucune sollicitation)", _note(MO.nouvelle_predalle())),
    ("test_predalle 1 : Mx = My = 30, sup 12 / 9, V 60", _note(_pre("P1", [TP]))),
    ("renfort principal 3 Ø12 posé SUR la prédalle", _note(_pre("P2", [TP_R]))),
    ("même prédalle, direction principale X", _note(_pre("P3", [TP_R], dir_principale="X"))),
    ("audit C-1 : la secondaire (sur la peau) gouverne, hauteur non conforme", _note(_pre("C1", [_sec("A", 55, 0, 60, 0, 30)], h=20))),
    ("peau minimale 3 cm, base 12/12/150/150 (sort de la peau)", _note(_pre("H3", [_sec("A", 20, 5, 25, 6, 30, inf_y=_fa(TR_("12/12/150/150")))], h_pre=3.0))),
    ("peau maximale 15 cm sur 40 cm, gros moments", _note(_pre("H15", [_sec("A", 60, 20, 80, 25, 90)], h=40, h_pre=15.0))),
    ("peau 9 cm sur 10 cm (1 cm coulé en place)", _note(_pre("H9", [_sec("A", 4, 0, 5, 0, 8)], h=10, h_pre=9.0))),
    ("base Ø20/150 principale, peau 5,5 cm", _note(_pre("H55", [_sec("A", 25, 10, 40, 12, 50, inf_y=_fa(BA_(20, 150)))], h_pre=5.5))),
    ("quatre couches mêlant les trois types, renforts partout", _note(_pre("Q4", [_sec(
        "A", 45, 18, 60, 22, 70, inf_y=_fa(TR_("12/12/150/150"), BA_(10, 200, dist=5.0), NB_(4, 16), TR_("6/6/150/150")),
        inf_x=_fa(TR_("8/8/150/150"), NB_(4, 10)), sup_y=_fa(TR_("8/8/150/150"), BA_(10, 200)), sup_x=_fa(TR_("6/6/150/150"), NB_(2, 12)))], h=25))),
    ("axes saisis (8,0 et 9,5 cm) et centre de gravité saisi (5,25 cm)", _note(_pre("AX", [_sec(
        "A", 28, 12, 35, 9, 60, inf_x=_fa(TR_("10/10/100/100", dist=8.0)), inf_y=_fa(TR_("10/10/100/100"), NB_(3, 12, dist=9.5), yG=5.25),
        sup_x=_fa(yG=3.0))]))),
    ("C25/30, B400, γs 1,15", _note(_pre("B4", [_sec("A", 25, 10, 28, 12, 45)], beton="C25/30", fyk=400), gamma_s=1.15)),
    ("C40/50, γs 2, bande de 150 cm en « n barres »", _note(_pre("B150", [_sec("A", 40, 15, 52, 20, 80, inf_y=_fa(NB_(9, 12)))], beton="C40/50", b=150), gamma_s=2.0)),
    ("jeu premier lit 1 cm, enrobage 2,5 cm", _note(_pre("J1", [TP_R], enrobage=2.5), jeu=1.0)),
    ("enrobage nul : lu 3 cm comme l'écran", _note(_pre("E0", [TP], enrobage=0.0))),
    ("enrobage 20 cm sur 22 : d ≤ 0 en haut (non conforme sous 100 %)", _note(_pre("E20", [_sec("A", 5, 5, 5, 5, 10)], enrobage=20.0))),
    ("acier excessif : Ø25/100 sur 20 × 8 (non conforme sous 100 %)", _note(_pre("AX2", [_sec(
        "A", 1, 0, 1, 0, 2, inf_x=_fa(BA_(25, 100)), inf_y=_fa(BA_(25, 100)), sup_x=_fa(BA_(6, 250)), sup_y=_fa(BA_(6, 250)))], b=20, h=8, h_pre=4.0))),
    ("τ > τ_adm,I", _note(_pre("TAU", [_sec("A", 20, 0, 25, 0, 150)]))),
    ("τ = τ_adm,I à la limite (100 %)", _note(_pre("TL", [_sec("A", 10, 0, 10, 0, V_LIM)]))),
    ("τ un cheveu au-dessus de τ_adm,I", _note(_pre("TL+", [_sec("A", 10, 0, 10, 0, V_LIM * 1.0004)]))),
    ("bande de 500 cm, épaisseur 100 cm", _note(_pre("G", [_sec("A", 300, 120, 450, 200, 600, inf_y=_fa(BA_(25, 100)), inf_x=_fa(BA_(20, 100)))],
                                                       b=500, h=100, h_pre=15.0))),
    ("épaisseur minimale 5 cm, peau 3", _note(_pre("H5", [_sec("A", 2, 0, 3, 0, 5)], h=5, h_pre=3.0))),
    ("trois sections et deux prédalles dans la note", _note(_pre("T1", [TP, _sec("B", 10, 35, 12, 40, 80, sup_y=_fa(TR_("10/10/100/100"), NB_(4, 12))),
                                                                      _sec("C", 5, 0, 8, 0, 15)]),
                                                         _pre("T2", [_sec("A", 80, 0, 90, 0, 40)], h=18, b=50, fyk=400, h_pre=4.5), jeu=0.5)),
]
RNG = random.Random(20261107)
for k in range(1, 9):                                         # des tirages à graine fixe, de part et d'autre de la capacité
    h_ = RNG.choice((12, 16, 20, 22, 25, 30))
    secs_ = [_sec("ABC"[i], *[round(RNG.uniform(0, 70), 1) for _ in range(4)], round(RNG.uniform(0, 130), 1),
                  **{f: _fa(*[_couche(RNG.choice(("treillis", "barres", "n_barres")), RNG.choice(TR.liste_choix()), RNG.choice((8, 10, 12, 16, 20)),
                                      RNG.choice((100, 150, 200)), RNG.randint(2, 8)) for _ in range(RNG.randint(1, 3))]) for f in EN.FACES})
             for i in range(RNG.randint(1, 2))]
    BOUT.append((f"tirage {k} (graine 20261107)", _note(_pre(f"Aléa {k}", secs_, h=h_, h_pre=min(RNG.choice((4.0, 5.0, 6.0, 8.0)), EN.h_pre_max(h_)),
                                                             beton=RNG.choice(sorted(MO.classes_beton())), fyk=RNG.choice((400, 500)),
                                                             dir_principale=RNG.choice("XY"), enrobage=RNG.choice((2.0, 3.0, 3.5))),
                                                        gamma_s=RNG.choice((1.15, 1.5)), jeu=RNG.choice((0.0, 0.5)))))
if D is not None:
    tous, n_blocs, limites = [], 0, set()
    for nom_cas, notes in BOUT:
        ec = comparer_bout_a_bout(nom_cas, notes)
        tous += [(nom_cas,) + tuple(x) for x in ec]
        for p in notes:
            for s_ in MO.verifier(p)["sections"]:
                n_blocs += len(s_["verifications"])
                limites |= {(v["etat"], C.niveau(v["taux"])) for v in s_["verifications"]}
    chk(f"bout à bout, {len(BOUT)} notes ({n_blocs} vérifications) : les taux, états, A_s,req / A_s,min / A_s,max / A_s fourni, d, "
        "h_u,min, h_min, y_G, τ et τ_adm,I AFFICHÉS par le site = ceux de l'écran Streamlit rejoué (à l'arrondi d'affichage, la bande "
        "de couleur gardée), chaque section et chaque prédalle de la même couleur que son icône", not tous, tous[:6])
    chk("… les cas couvrent les trois bandes de couleur et les deux états (dont non conforme sous 100 %)",
        {("ok", "p"), ("ok", "f"), ("nok", "r"), ("nok", "p")} <= limites, limites)
    tl = MO.verifier(BOUT[[n for n, _ in BOUT].index("τ = τ_adm,I à la limite (100 %)")][1][0])["sections"][0]
    v_tau = next(v for v in tl["verifications"] if v["cle"] == "tau")
    chk("à la limite exacte τ = τ_adm,I : 100 % vert foncé (conforme), comme l'écran (✅)", abs(v_tau["taux"] - 1) < 1e-12 and v_tau["etat"] == "ok"
        and C.pct(v_tau["taux"]) == 100 and C.niveau(v_tau["taux"]) == "f", v_tau["taux"])

# ------------------------------------------------------------------ 12. l'établi compact : les fonctions aux places de la poutre
print("\n=== 12. L'établi compact : les fonctions aux places de la poutre ===")
ar = req(action="calculer", etat=E)
don = ar["donnees"]
chk("l'arborescence : « Prédalles » et leur nombre, chaque prédalle « nom · b × h cm » (sans taux), « Ajouter une prédalle »",
    re.search(r'<div class="et-arbre-h"><span>Prédalles</span><span class="n">1</span></div>', don) is not None
    and noms_arbre(don) == [("Prédalle 1", "· 100 × 22 cm")]
    and 'data-ajouter="1"' in don and "Ajouter une prédalle" in don and ar["elements"] == ""
    and "%" not in re.search(r'<div class="et-pt-l">.*?</button>', don, re.S).group(0))
chk("sous la prédalle ouverte, ses quatre rubriques dans l'ordre de la poutre : Prédalle, Sections et sollicitations, Armatures, Options",
    re.findall(r'<div class="sec[^"]*" data-sec="([^"]+)"', don) == ["pd", "so", "arm", "op"]
    and re.findall(r'<span class="tt">([^<]+)</span>', don) == ["Prédalle", "Sections et sollicitations", "Armatures", "Options"]
    and re.search(r'<div class="sec" data-sec="op" data-secondaire="1">', don) is not None)
menu_ = re.search(r'<div class="et-mn-l" role="menu" hidden>(.*?)</div></div>', don, re.S).group(1)
chk("le menu « ⋯ » de la prédalle : Renommer, Copier la prédalle, Supprimer la prédalle (grisé : c'est la seule)",
    'data-renommer="0"' in menu_ and 'data-dupliquer-index="0"' in menu_ and "Copier la prédalle" in menu_
    and re.search(r'data-supprimer="0" data-confirmer="Supprimer la prédalle « Prédalle 1 » de la note \?" disabled>', menu_) is not None)
trois = req(action="ajouter", etat=req(action="ajouter", etat=copy.deepcopy(E))["etat"])["etat"]
cp3 = req(action="dupliquer", etat=trois, index=0)
chk("« Copier » du menu « ⋯ » d'une prédalle (index) : CETTE prédalle, copiée juste après elle et ouverte (comme la poutre)",
    [p["nom"] for p in cp3["etat"]["predalles"]] == ["Prédalle 1", "Prédalle 1 (copie)", "Prédalle 2", "Prédalle 3"] and cp3["etat"]["actif"] == 1
    and re.findall(r'<div class="et-pt( on)?" data-predalle="(\d)">', cp3["donnees"]) == [("", "0"), (" on", "1"), ("", "2"), ("", "3")],
    ([p["nom"] for p in cp3["etat"]["predalles"]], cp3["etat"]["actif"]))
cibles = set(re.findall(r'data-cible="([^"]+)"', don))
sels = set(re.findall(r'data-sel="([^"]+)"', "".join(v["svg"] for v in ar["vues"])))
couches_ = {f"s{i}.{f}-{j}" for i, s_ in enumerate(pd(E)["sections"]) for f in EN.FACES for j in range(len(s_["faces"][f]["couches"]))}
chk("chaque couche de chaque section a sa ligne dans les données (data-cible) et sa pièce sur les dessins (data-sel), sous la "
    "MÊME clé ; le béton et les efforts aussi",
    len(couches_) == 9 and couches_ <= cibles and couches_ <= sels and {"beton", "s0.efforts", "s1.efforts"} <= cibles
    and {"s0.beton", "s1.beton"} <= sels, (sorted(couches_ - cibles)[:3], sorted(couches_ - sels)[:3]))
chk("les vues : les coupes et le plan de chaque section, puis « Toutes » dès deux sections",
    [v["cle"] for v in ar["vues"]] == ["c0", "p0", "c1", "p1", "toutes"])
for cle_ in ("c0", "p0", "c1", "p1"):
    sv_ = next(v for v in ar["vues"] if v["cle"] == cle_)["svg"]
    chk(f"la vue {cle_} ne porte que les pièces de SA section (s{cle_[1:]}.…)", sv_.count("data-sel=") > 0
        and all(k.startswith(f"s{cle_[1:]}.") for k in re.findall(r'data-sel="([^"]+)"', sv_)))
chk("« Toutes » : la coupe principale de chaque section, ses pièces cliquables (s0 et s1)",
    {k.split(".")[0] for k in re.findall(r'data-sel="([^"]+)"', ar["vues"][-1]["svg"])} == {"s0", "s1"})
chk("la section courante : ses armatures (.et-arm), son résumé (.et-r3) et son bloc de résultats (.et-bloc) par section ; le choix "
    "de la section courante dès deux sections",
    re.findall(r'<div class="et-arm( on)?" data-section="(\d)">', don) == [(" on", "0"), ("", "1")]
    and len(re.findall(r'class="et-r3" data-section="\d"', don)) == 2 and len(re.findall(r'data-section-courante="\d"', don)) == 4
    and [b_[1] for b_ in BLOC.findall(ar["resultat"])] == ["0", "1"])
ent = ar["vues"][0]["entete"]
chk("l'en-tête de la vue : la section, la bande et la peau, les matériaux, la direction principale ; les nappes (la principale "
    "d'abord) ; la bande des quatre moments et de V, une fois",
    "<b>Section A</b><span>· 100 × 22 cm, prédalle 6 cm · C30/37 · B500 · principale Y</span>" in ent
    and ">Inf. Y Treillis 10/10/100/100 + 3 Ø12 · X Treillis 10/10/100/100<" in ent and ">Sup. Y Treillis 10/10/100/100 · X Treillis 10/10/100/100<" in ent
    and ent.count("<span><i>") == 5 and "<sub>x,inf</sub><b>30,0</b>" in ent and "<i>V</i><b>60,0</b>" in ent, ent[:400])
tete_on = re.search(r'<div class="et-pt on" data-predalle="0"><div class="et-pt-l">(.*?)<div class="et-mn">', don, re.S).group(1)
rub1 = re.search(r'<div class="sec[^"]*" data-sec="pd".*?(?=<div class="sec[^"]*" data-sec="so")', don, re.S).group(0)
chk("le NOM de l'élément ouvert se modifie sur sa ligne de l'arborescence ; la rubrique 1 n'a plus de ligne Nom et tient en "
    "4 lignes de deux champs (béton | acier, b | h…), l'acier en menu déroulant ; les explications dans les ⓘ",
    '<input data-chemin="nom" data-sorte="texte" value="Prédalle 1" class="et-pt-nom"' in tete_on and 'data-chemin="nom"' not in rub1
    and len(re.findall(r'<div class="row(?:2 n2)?"', rub1)) == 4 == rub1.count('class="row2 n2"')
    and re.search(r'<select data-chemin="fyk" data-sorte="choix">', rub1) is not None and 'class="hint"' not in rub1,
    (tete_on[-300:], re.findall(r'<div class="row[^"]*"', rub1)))
aides_ = [html.unescape(t) for t in re.findall(r'<span class="aide" title="([^"]*)">', don)]
chk("condensées, les données ne perdent aucune explication : les paragraphes d'aide d'avant (la base principale posée "
    "DANS la prédalle, γs et jeu communs à la note) sont dans les ⓘ de leur champ",
    any(t.startswith("La base inférieure de la direction principale est posée DANS la prédalle") for t in aides_)
    and sum("Commun à toutes les prédalles de la note" in t for t in aides_) == 2, aides_)
chk("chaque vue dit ce qu'on lit sur son dessin, en dernière ligne de son en-tête (la légende : barres coupées en points, X horizontales…)",
    len(ar["vues"]) == 5 and all(re.findall(r'<div class="lg">([^<]*)</div>(?=</div>)', v_["entete"])
                                 == [P.LEGENDES["p" if v_["cle"][0] == "p" else "c"]] == [v_["legende"]] for v_ in ar["vues"]),
    [(v_["cle"], v_["legende"]) for v_ in ar["vues"]])
chk("le dessin d'une section, dans l'établi, n'écrit plus les efforts (ils sont dans la bande de l'en-tête, une fois, sans rouge)",
    "#9C3341" not in ar["vues"][0]["svg"] and "kN·m" not in ar["vues"][0]["svg"] and "kN·m" in DP.svg_coupe("t", m["sections"][0], m["predalle"]))
chk("la tête des résultats : le nom de la prédalle et ses dimensions (comme la poutre)",
    '<div class="et-rp"><b>Prédalle 1</b><span>100 × 22 cm, prédalle 6 cm · C30/37 · B500</span></div>' in ar["resultat"])
chk("une prédalle n'a pas de note PDF sur le site : rien à télécharger (le bouton reste « bientôt », construire.NOTE_PDF)",
    not hasattr(P, "telecharger"))
# un axe ou un centre de gravité SAISI se rend à l'automatique d'un clic (« ↺ », comme l'écran d'avant du site) ; la
# pose de la couche dit ce que serait l'automatique (une fois : la ligne de la couche ne le répète pas)
RAZ = r'<button type="button" class="et-raz" data-chemin="{}" data-valeur="" title="([^"]+)"'
ax_ = saisir(saisir(copy.deepcopy(E), "sections/0/faces/inf_x/couches/0/dist", "8")["etat"], "sections/0/faces/inf_y/yG", "4")
d_ax = ax_["donnees"]
e_auto = moteur(ax_["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["e_auto"]
t_ax = re.search(RAZ.format("sections/0/faces/inf_x/couches/0/dist"), d_ax)
chk("un axe saisi : « ↺ Revenir à l'axe automatique (… cm) » à côté du champ ; un centre de gravité imposé aussi ; la pose dit "
    "l'automatique une seule fois ; aucun bouton sur un champ automatique",
    t_ax is not None and html.unescape(t_ax.group(1)) == f"Revenir à l'axe automatique ({PC._cm(e_auto)} cm)"
    and re.search(RAZ.format("sections/0/faces/inf_y/yG"), d_ax) is not None
    and set(re.findall(r'class="et-raz" data-chemin="([^"]+)"', d_ax)) == {"sections/0/faces/inf_x/couches/0/dist", "sections/0/faces/inf_y/yG"}
    and html.unescape(d_ax).count("l'automatique serait") == 1 and 'class="et-raz"' not in don, (t_ax and t_ax.group(1), d_ax.count('class="et-raz"')))
raz_ = saisir(ax_["etat"], "sections/0/faces/inf_x/couches/0/dist", "")
chk("… et « ↺ » (la valeur vidée) rend l'axe automatique : la clé du moteur redevient None, le bouton disparaît",
    pd(raz_["etat"])["sections"][0]["faces"]["inf_x"]["couches"][0]["dist"] is None
    and re.search(RAZ.format("sections/0/faces/inf_x/couches/0/dist"), raz_["donnees"]) is None)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
