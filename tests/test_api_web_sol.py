# -*- coding: utf-8 -*-
"""Le module « Rigidité du sol » du SITE (api_web/sol.py) : ce que l'écran affiche est ce que le moteur (sol/rigidite.py)
calcule ; les saisies, les refus, les effets de bord de l'écran Streamlit ; l'import d'un essai ; l'échappement ; les
formules substituées ; les dessins (SVG bien formés, sans NaN).

    python3 tests/test_api_web_sol.py
"""
import copy
import json
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import routeur as RT  # noqa: E402
from api_web import sol as M  # noqa: E402
from sol import rigidite as R  # noqa: E402

OK, KO = [], []
PIEGE = '<img src=x onerror=alert(1)>'


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:400]}"))


def req(**k):
    return json.loads(RT.traiter(json.dumps(k)))


def neuf():
    return req(module="sol", action="nouveau")


def mod(etat, chemin, valeur):
    return req(module="sol", action="modifier", etat=copy.deepcopy(etat), chemin=[str(c) for c in chemin], valeur=valeur)


def texte(h):
    """Le texte d'un fragment HTML, sans balises ni entités (pour y chercher des nombres)."""
    t = re.sub(r"<[^>]+>", " ", h)
    return re.sub(r"\s+", " ", t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#x27;", "'")
                  .replace("&quot;", '"').replace(C.NBSP, " ")).strip()


def fondation(sondages=None, **kw):
    f = R.nouvelle_fondation("T")
    if sondages is not None:
        f["sondages"] = []
        for i, couches in enumerate(sondages, start=1):
            s = R.nouveau_sondage(i, f"S{i}", premiere_couche=False)
            s["couches"] = [R.nouvelle_couche(h=h, type_sol=t, M=m, gamma=g) for h, t, m, g in couches]
            f["sondages"].append(s)
    f.update(kw)
    f = R.normaliser(f)
    f["trace"] = 0
    return {"version": 1, "fondations": [f], "actif": 0}


def svgs_bien_formes(rep):
    """Chaque vue : un SVG (ou un bloc HTML de tableaux) qui se relit en XML, sans NaN ni infini dans ses nombres."""
    fautes = []
    for v in rep["vues"]:
        h = v["svg"]
        if h.startswith("<svg"):
            try:
                ET.fromstring(h)
            except ET.ParseError as e:
                fautes.append((v["cle"], str(e)))
        if re.search(r'="[^"]*\b(nan|inf)\b', h, re.I):
            fautes.append((v["cle"], "NaN ou infini"))
    return fautes


# ================================================================== 1. la note neuve = le moteur
print("=== 1. La note neuve : ce que l'écran affiche = ce que le moteur calcule")
r0 = neuf()
e0 = r0["etat"]
f0 = e0["fondations"][0]
res = R.resultats_profil({k: v for k, v in f0.items() if k != "trace"})
t0 = texte(r0["resultat"])
x = res["scia"][0]
chk("la note neuve est la note neuve de l'écran Streamlit (moteur : nouvelle_fondation) et dessine son premier sondage",
    {k: v for k, v in f0.items() if k != "trace"} == R.nouvelle_fondation("F1") and f0["trace"] == 0)
chk(f"le k du sondage (centre) est celui du moteur : {C.fr(x['k_bas'], 2)} MN/m³ ; son tassement {C.fr(x['w_bas'], 1)} mm",
    f"{C.fr(x['k_bas'], 2)} MN/m³" in t0 and f"w = {C.fr(x['w_bas'], 1)} mm" in t0, t0[:300])
chk("le zonage de la dalle (centre, milieu du grand et du petit côté, angle) est celui du moteur, en MN/m³ et en kN/m³",
    all(texte(f"{C.fr(z['k_bas'], 2)} MN/m³") in t0 and texte(f"{C.fr(z['k_bas'] * 1000, 0)} kN/m³") in t0 for z in res["zonage"]), res["zonage"])
chk("le résumé de la note est une consultation (pas de taux) qui nomme la fondation et son k",
    r0["resume"]["taux"] is None and r0["resume"]["niveau"] == "i" and f"{C.fr(x['k_bas'], 2)} MN/m³" in r0["resume"]["gouvernant"], r0["resume"])
chk("les vues de la note neuve : diffusion, coupe, k(B), tableau SCIA — SVG bien formés", [v["cle"] for v in r0["vues"]] ==
    ["diffusion", "coupe", "kb", "scia"] and not svgs_bien_formes(r0), svgs_bien_formes(r0))
d = R.donnees_diffusion(res["resultats"][0], res["parametres"]["D"], res["parametres"]["nappe"])
dif = r0["vues"][0]["svg"]
chk(f"le graphique de diffusion porte la profondeur d'influence du moteur ({C.fr(d['z_influence'], 2)} m) et le critère 20 %",
    f"profondeur d'influence {C.fr(d['z_influence'], 2)} m" in texte(dif) and "20 % · σ' v0" in texte(dif)   # v0 en indice (tspan)
    and "σ'<tspan" in dif and "_v0" not in dif)
kb = R.k_fonction_B(res["resultats"][0]["bas"], res["resultats"][0]["haut"], 2.0, 2.0, 150.0, 1.0, None, 0.2)
chk("le graphique k(B) trace les 18 largeurs du moteur (k_fonction_B) : 18 points sur la borne basse",
    r0["vues"][2]["svg"].count('r="2.6"') == 18 and len(kb["B"]) == 18)

# ================================================================== 2. les modes 3 à 6
print("\n=== 2. Les modes 3 à 6")
for m, attendu in (("rapide", lambda f: R.calcul_rapide(**f["rapide"])["k_MNm3"]), ("theories", lambda f: R.theories(**f["theories"])["k_retenu"]),
                   ("plat", lambda f: R.plat(**f["plat"])["keq"] / 1000.0)):
    r = mod(e0, ["mode"], m)
    f = r["etat"]["fondations"][0]
    k = attendu(f)
    nd = 0 if m == "plat" else 2
    chk(f"mode {m} : le verdict porte le k du moteur ({C.fr(k, nd)} MN/m³), une fenêtre de calcul, des vues bien formées",
        texte(C.fr(k, nd)) in texte(r["resultat"]) and r["volets"].count('class="flyout calcul"') == 1 and not svgs_bien_formes(r) and r["vues"],
        texte(r["resultat"])[:200])
r = mod(e0, ["mode"], "abaque")
ab = R.abaque(**r["etat"]["fondations"][0]["abaque"])
chk(f"mode abaque : les {len(ab['lignes'])} sols du moteur, chacun avec son k corrigé, dans le résultat et dans le tableau",
    all(C.esc(x_["type"]) in r["resultat"] and f"{C.fr(x_['k_bas'], 1)}–{C.fr(x_['k_haut'], 1)}" in r["resultat"] for x_ in ab["lignes"])
    and r["vues"][1]["svg"].count("<tr>") == len(ab["lignes"]) + 1)
th = mod(e0, ["mode"], "theories")
tt = R.theories(**th["etat"]["fondations"][0]["theories"])
chk("mode théories : les cinq méthodes du moteur, la retenue en avant, l'ancienne formule E/(2B) marquée « biaisée »",
    all(f"{C.fr(m_['k'], 2)} MN/m³" in texte(th["resultat"]) for m_ in tt["methodes"]) and "biaisée" in th["resultat"])
r = mod(mod(e0, ["mode"], "rapide")["etat"], ["rapide", "q"], "0")
chk("mode k = q / w : q = 0 donne k nul, statut « Calcul impossible »", r["resume"]["niveau"] == "x" and "Calcul impossible" in r["resultat"])

# ================================================================== 3. les saisies, les refus
print("\n=== 3. Les saisies et les refus")
REFUS = [(["B"], "0,1", "Largeur B"), (["L"], "0", "Longueur L"), (["D"], "-1", "Assise D"), (["q"], "0,5", "Pression q"), (["nu"], "0,5", "Poisson"),
         (["nappe"], "-2", "Nappe"), (["sondages", 0, "couches", 0, "gamma"], "30", "γ"), (["sondages", 0, "couches", 0, "h"], "-1", "Épaisseur"),
         (["sondages", 0, "couches", 0, "M"], "-5", "Module M"), (["critere"], "12", "critère"), (["B"], "abc", "illisible"),
         (["sondages", 0, "couches", 0, "type"], "Argile verte", "inconnu"), (["mode"], "7", "inconnue"), (["position"], "milieu", "position")]
fautes = []
for chemin, v, mot in REFUS:
    r = mod(e0, chemin, v)
    if not (r["ok"] and r["message"] and mot.lower() in r["message"].lower() and r["etat"] == e0):
        fautes.append((chemin, v, r.get("message"), r.get("erreur")))
chk(f"{len(REFUS)} saisies hors des bornes de l'écran (que Streamlit ramène EN SILENCE à la borne) ou illisibles sont REFUSÉES avec un "
    "message qui nomme le champ ; la note reste intacte", not fautes, fautes)
r = mod(e0, ["B"], "0,1")
chk(f"le message dit la valeur et la borne : « {r['message']} »", "0,1" in r["message"] and "0,2" in r["message"])
chk("les champs portent les bornes de l'écran pour un refus immédiat dans le navigateur (data-min, data-max)",
    'data-chemin="B" data-sorte="nombre" inputmode="decimal" value="2" data-min="0.2"' in r0["donnees"]
    and 'data-chemin="nu" data-sorte="nombre" inputmode="decimal" value="0,3" data-min="0.0" data-max="0.49"' in r0["donnees"])
r = mod(e0, ["L"], "1")
chk("L < B : B et L échangés pour le calcul, comme l'écran, et dit", "échangés" in r["resultat"] and r["etat"]["fondations"][0]["L"] == 1.0)
r = mod(e0, ["nappe_active"], True)
chk("cocher la nappe fait apparaître son champ (nappe à 3 m, la valeur initiale de l'écran)",
    'data-chemin="nappe"' in r["donnees"] and 'data-chemin="nappe"' not in r0["donnees"] and "Nappe à 3 m" in r["volets"])
# le préremplissage au changement de type
fautes = []
rng = random.Random(7)
for _ in range(40):
    j = rng.randrange(3)
    t = rng.choice(R.TYPES)
    e = copy.deepcopy(e0)
    if rng.random() < 0.5:
        e = mod(e, ["sondages", 0, "couches", j, "qc"], "0")["etat"]
        e = mod(e, ["sondages", 0, "couches", j, "M"], "0")["etat"]
    c = copy.deepcopy(e["fondations"][0]["sondages"][0]["couches"][j])
    R.appliquer_type(c, t)
    r = mod(e, ["sondages", 0, "couches", j, "type"], t)
    if r["etat"]["fondations"][0]["sondages"][0]["couches"][j] != R.normaliser_couche(c):
        fautes.append((t, c, r["etat"]["fondations"][0]["sondages"][0]["couches"][j]))
chk("changer le type d'une couche (40 tirages) : qc, Rf, M préremplis si qc et M sont nuls, γ du type toujours — le moteur (appliquer_type)",
    not fautes, fautes[:2])
# boutons : sondages et couches
r = mod(e0, ["sondages", "+"], 1)
s = r["etat"]["fondations"][0]["sondages"]
chk("« Ajouter un sondage » : CPT02 et le profil de départ profond de l'écran ; la liste et le menu « Dessiné » apparaissent",
    len(s) == 2 and s[1]["nom"] == "CPT02" and len(s[1]["couches"]) == 3 and 'data-chemin="trace"' in r["donnees"])
r2 = mod(r["etat"], ["sondages", 0, "copie"], 1)
s2 = r2["etat"]["fondations"][0]["sondages"]
chk("« Copier ce sondage » : « CPT01 (copie) », mêmes couches, identifiant neuf", len(s2) == 3 and s2[2]["nom"] == "CPT01 (copie)"
    and s2[2]["couches"] == s2[0]["couches"] and s2[2]["id"] == 3)
r3 = mod(mod(r2["etat"], ["trace"], "2")["etat"], ["sondages", 0, "-"], 1)
f3 = r3["etat"]["fondations"][0]
chk("supprimer un sondage avant le sondage dessiné : le dessin suit le même sondage", f3["trace"] == 1 and f3["sondages"][1]["nom"] == "CPT01 (copie)")
chk("le dernier sondage ne se supprime pas (message)", mod(e0, ["sondages", 0, "-"], 1)["message"] == "Il faut au moins un sondage")
r = mod(e0, ["sondages", 0, "couches", "+"], 1)
chk("« Ajouter une couche » : h = 1 m, type « — », M nul, EN BAS du profil — « ignorée (h ou M manquant) »",
    r["etat"]["fondations"][0]["sondages"][0]["couches"][-1] == R.nouvelle_couche() and "ignorée (h ou M manquant)" in r["donnees"])
e1 = fondation([[(5.0, "Sable dense", 60.0, 20.0)]])
chk("la dernière couche ne se supprime pas (message)", mod(e1, ["sondages", 0, "couches", 0, "-"], 1)["message"] == "Il faut au moins une couche")

# ================================================================== 4. l'import d'un essai
print("\n=== 4. L'import d'un essai (fichier, tableau collé), « Remplir le tableau »")
r = mod(e0, ["sondages", 0, "fichier"], {"nom": "CPT-DEMO.gef", "texte": M.ESSAI_DEMO})
s = r["etat"]["fondations"][0]["sondages"][0]
imp = R.lire_essai("CPT-DEMO.gef", M.ESSAI_DEMO, 3.0)
chk("un fichier GEF lu par le moteur (lire_essai) : l'essai en attente, sa nappe (2,40 m du fichier), finesse 0,25, épaisseur 0,50 m",
    s["import"] == imp and imp["nappe"] == 2.4 and imp["seuil"] == 0.25 and imp["h_min"] == 0.5)
ap = R.apercu_import(imp, 0.3)
chk(f"l'aperçu : {len(ap['lignes'])} couches découpées listées, les vues « Import » et « Découpage » en tête, les boutons et les exports",
    all(C.esc(x_["type"]) in r["donnees"] for x_ in ap["lignes"]) and [v["cle"] for v in r["vues"]][:2] == ["import", "import_tab"]
    and "Remplir le tableau" in r["donnees"] and "Abandonner" in r["donnees"] and "CSV des mesures" in r["donnees"] and "CSV des couches" in r["donnees"])
lien = re.search(r'download="CPT-DEMO_mesures.csv" href="data:text/csv;charset=utf-8,([^"]+)"', r["donnees"])
chk("l'export « CSV des mesures » est le CSV du moteur (csv_mesures), encodé dans le lien",
    lien is not None and unquote(lien.group(1)) == R.csv_mesures("CPT-DEMO", imp["points"]))
r4 = mod(r["etat"], ["sondages", 0, "import", "seuil"], "0.1")
chk("la finesse du découpage se règle (0,10) et le découpage suit", r4["etat"]["fondations"][0]["sondages"][0]["import"]["seuil"] == 0.1)
chk("une finesse hors de l'écran (0,7) est refusée", bool(mod(r["etat"], ["sondages", 0, "import", "seuil"], "0.7")["message"]))
rv = mod(r["etat"], ["sondages", 0, "import", "valider"], 1)
g = copy.deepcopy(r["etat"]["fondations"][0])
g.pop("trace")
R.valider_import(g, 0)
fv = rv["etat"]["fondations"][0]
chk("« Remplir le tableau » = le moteur (valider_import) : couches découpées, mesures gardées, nappe du sondage ET de la fondation cochée, "
    "nom pris dans le fichier", {k: v for k, v in fv.items() if k != "trace"} == R.normaliser(g) and fv["nappe_active"] and fv["nappe"] == 2.4
    and fv["sondages"][0]["nom"] == "CPT-DEMO" and fv["sondages"][0]["import"] is None)
chk("après l'import, la vue « CPT » (qc, fs, Ic) apparaît et l'export des mesures brutes est offert",
    "cpt" in [v["cle"] for v in rv["vues"]] and "Exporter les mesures brutes (CSV)" in rv["donnees"] and not svgs_bien_formes(rv))
ra = mod(r["etat"], ["sondages", 0, "import", "annuler"], 1)
chk("« Abandonner » rend le sondage tel qu'il était", ra["etat"] == e0)
rf = mod(e0, ["sondages", 0, "fichier"], {"nom": "faux.gef", "texte": "pas un essai"})
chk(f"un fichier illisible est refusé avec le message de l'analyseur (« {rf['message'][:70]}… ») ; la note reste intacte",
    rf["message"].startswith("Lecture impossible") and rf["etat"] == e0)
rc = mod(e0, ["sondages", 0, "colle"], "0,5\t2,1\t30\n1,0\t2,4\t35\n1,5\t3,0\t40\n2,0\t5,2\t41\n2,5\t6,0\t44\n3,0\t8,0\t50\n")
sc = rc["etat"]["fondations"][0]["sondages"][0]
chk("un tableau collé (copié d'Excel) devient un essai en attente, lu par l'analyseur CSV de l'écran", sc["import"] is not None
    and sc["import"]["origine"] == "colle" and sc["import"]["points"][0] == [0.5, 2.1, 30.0])
rx = mod(e0, ["sondages", 0, "colle"], "rien de lisible")
chk("un tableau collé illisible : le texte est GARDÉ (pas perdu) et l'erreur dite sous la zone", rx["etat"]["fondations"][0]["sondages"][0]["colle"]
    == "rien de lisible" and "Tableau collé illisible" in rx["donnees"] and not rx["message"])
chk("réécrire le même tableau collé ne relance pas le découpage (les réglages de l'aperçu restent)",
    mod(mod(rc["etat"], ["sondages", 0, "import", "seuil"], "0.4")["etat"], ["sondages", 0, "colle"], sc["colle"])["etat"]["fondations"][0]
    ["sondages"][0]["import"]["seuil"] == 0.4)
# l'aperçu qui plante dans l'écran (dernières mesures sans frottement) : refusé avec un message, pas d'exception
pts = "\n".join(f"{z / 10:.2f};{2 + z % 3};{'' if z > 55 else 30 + z}" for z in range(1, 61))
rp = mod(e0, ["sondages", 0, "colle"], "Profondeur (m);qc (MPa);fs (kPa)\n" + pts)
chk("un essai dont les dernières mesures n'ont pas de frottement (l'écran Streamlit PLANTE au découpage) : un message, pas d'exception",
    rp["ok"] and ("découpage automatique échoue" in rp["donnees"] or "Aucune couche" in rp["donnees"] or "Remplir" in rp["donnees"]))

# ================================================================== 5. le fichier de l'écran Streamlit
print("\n=== 5. Reprendre le fichier « Enregistrer » de l'écran Streamlit")
payload = {"version": "v5.0", "soundings": [{"id": 1, "nom": "P1"}, {"id": 4, "nom": "P4"}], "orders": {"1": [1, 2], "4": [3]},
           "values": {"rs_mode": "2. Profil de sol saisi à la main", "rs_B": 3.0, "rs_L": 5.0, "rs_q": 220.0, "rs_nappe_active": True, "rs_nappe": 1.5,
                      "snd1_nom": "P1", "snd1_layer_1_h": 2.0, "snd1_layer_1_type": "Limon (loess)", "snd1_layer_1_M": 20.0,
                      "snd1_layer_2_h": 10.0, "snd1_layer_2_type": "Sable dense", "snd1_layer_2_M": 70.0, "snd4_nom": "P4",
                      "snd4_layer_3_h": 8.0, "snd4_layer_3_type": "Argile ferme / raide", "snd4_layer_3_M": 30.0}}
r = mod(e0, ["streamlit"], {"nom": "raideur_sol.json", "texte": json.dumps(payload)})
att = R.depuis_payload(payload, "F1")
chk("le fichier raideur_sol.json de l'écran remplace la fondation ouverte, relu par le moteur (depuis_payload), son nom gardé",
    {k: v for k, v in r["etat"]["fondations"][0].items() if k != "trace"} == att and r["etat"]["fondations"][0]["mode"] == "profil")
chk("un fichier qui n'est pas celui de l'écran est refusé avec un message ; la note reste intacte",
    mod(e0, ["streamlit"], {"nom": "x.json", "texte": "{\"a\": 1}"})["etat"] == e0
    and "Fichier refusé" in mod(e0, ["streamlit"], {"nom": "x.json", "texte": "pas du json"})["message"])

# ================================================================== 6. les cas de l'écran : avertissements, statuts
print("\n=== 6. Les avertissements de l'écran et le statut de la note")
cas = {
    "écarté": (fondation([[(10.0, "—", 0.0, 19.0)], [(2.0, "Limon (loess)", 15.0, 18.0), (18.0, "Sable dense", 70.0, 20.0)]]), "c", "n'entre pas dans les résultats"),
    "tous écartés": (fondation([[(10.0, "—", 0.0, 19.0)]]), "c", "Renseigner au moins une couche"),
    "incalculable": (fondation([[(2.0, "Limon (loess)", 20.0, 18.0), (2.0, "Personnalisé", 0.0, 19.0), (16.0, "Sable dense", 80.0, 20.0)]]), "c",
                     "k non calculable"),
    "profil court": (fondation([[(3.0, "Argile ferme / raide", 30.0, 19.0)]], B=4.0, L=4.0, D=0.0), "c", "Profil trop court — k surestimé"),
    "pression nette nulle": (fondation(D=4.0, q=40.0), "x", "pression nette nulle"),
    "plafond 6·B": (fondation([[(200.0, "Sable dense", 90.0, 8.0)]], B=2.0, L=2.0, D=0.0, q=5000.0, critere=10), "i", "Précision du site"),
    "encadré (M haut)": (None, "i", "vérifiée aux DEUX bornes"),
}
e_enc = copy.deepcopy(rv["etat"])
cas["encadré (M haut)"] = (e_enc, "i", "vérifiée aux DEUX bornes")
for nom, (e, niv, phrase) in cas.items():
    r = req(module="sol", action="calculer", etat=e)
    chk(f"{nom} : statut « {C.STATUTS[niv][1]} », message « {phrase} », dessins bien formés",
        r["ok"] and r["resume"]["niveau"] == niv and phrase in texte(r["resultat"]) and not svgs_bien_formes(r),
        (r.get("erreur"), r.get("resume"), texte(r.get("resultat", ""))[:300]))
r = req(module="sol", action="calculer", etat=cas["écarté"][0])
chk("plusieurs sondages : la vue « Sondages » (côte à côte, même échelle) et une fenêtre de calcul par sondage calculé, plus le zonage",
    "sondages" in [v["cle"] for v in r["vues"]] and 'id="ck-s1"' in r["volets"] and 'id="ck-zon"' in r["volets"] and 'id="ck-s0"' not in r["volets"])

# ================================================================== 7. les formules substituées
print("\n=== 7. Les formules de la fenêtre de calcul : les valeurs substituées redonnent le résultat du moteur")
rng = random.Random(20261107)
fautes, n = [], 0
for k in range(60):
    couches = [(round(rng.uniform(0.3, 6), 2), rng.choice(["Limon (loess)", "Sable dense", "Argile ferme / raide"]), round(rng.uniform(5, 90), 1),
                round(rng.uniform(15, 22), 1)) for _ in range(rng.randint(1, 4))] + [(30.0, "Sable dense", 60.0, 20.0)]
    e = fondation([couches], B=round(rng.uniform(0.5, 8), 2), L=round(rng.uniform(0.5, 12), 2), D=round(rng.uniform(0, 4), 2), q=round(rng.uniform(60, 400), 1),
                  q_net=rng.random() < 0.8, nappe_active=rng.random() < 0.5, nappe=round(rng.uniform(0, 6), 2))
    f = e["fondations"][0]
    resk = R.resultats_profil({kk: v for kk, v in f.items() if kk != "trace"})
    if not resk["resultats"]:
        continue
    rr = resk["resultats"][0]
    termes, s_aff = M._sigma_d(rr["bas"], resk["parametres"]["D"])
    det = rr["detail"]
    n += 1
    if not math.isclose(s_aff, det["sigma_v0_assise"], rel_tol=1e-12, abs_tol=1e-9):
        fautes.append(("σv0(D)", s_aff, det["sigma_v0_assise"]))
    vol = req(module="sol", action="calculer", etat=e)["volets"]
    if f"{C.fr(det['sigma_v0_assise'], 2)} kPa" not in vol or f"{C.fr(det['q_net_kPa'], 2)} kPa" not in vol:
        fautes.append(("affichage", k))
    if det["w_m"] > 1e-12 and f"{C.fr(det['q_net_kPa'], 1)} / {C.fr(det['w_m'], 5)}" not in vol:
        fautes.append(("k = q_net / w", k))
    if det["w_m"] > 1e-12 and not math.isclose(det["q_net_kPa"] / det["w_m"], det["k_kNm3"], rel_tol=1e-12):
        fautes.append(("k moteur", k))
chk(f"{n} fondations tirées : σ_v0(D) = Σ γ·h écrit terme à terme redonne la valeur du moteur ; q_net, w et k = q_net / w affichés "
    "sont ceux du moteur", not fautes and n > 40, fautes[:3])
p6 = R.plat(200.0, 300.0, 0.5, 30.0, True, 20.0, 20.0)
r = mod(mod(mod(e0, ["mode"], "plat")["etat"], ["plat", "L"], "300")["etat"], ["plat", "mortier"], True)
chk("mode plat : h_c = α·min(B ; L) = 0,5 × min(200 ; 300) = 100 mm, k_c, k_g et k_eq du moteur dans la fenêtre",
    "0,5 × min(200 ; 300)" in r["volets"] and f"{C.fr(p6['keq'] / 1000.0, 0)} MN/m³" in r["volets"] and f"{C.fr(p6['kg'] / 1000.0, 0)} MN/m³" in r["volets"])

# ================================================================== 8. les éléments, l'échappement, la note abîmée
print("\n=== 8. Plusieurs fondations, l'échappement, la note abîmée")
r = req(module="sol", action="ajouter", etat=e0)
r = mod(r["etat"], ["mode"], "plat")
chk("deux fondations dans la note, chacune sa méthode et son k dans la liste", r["elements"].count('data-activer="') == 2
    and "Sondage CPT" in r["elements"] and "Plat en béton" in r["elements"])
r = req(module="sol", action="dupliquer", etat=r["etat"])
chk("copier la fondation ouverte : « F2 (copie) »", r["etat"]["fondations"][-1]["nom"] == "F2 (copie)" and r["etat"]["actif"] == 2)
r = req(module="sol", action="calculer", etat={"version": 1, "fondations": [e0["fondations"][0], cas["tous écartés"][0]["fondations"][0]], "actif": 0})
chk("le résumé de la note signale la fondation à compléter, même si elle n'est pas ouverte", r["resume"]["niveau"] == "c" and "T ·" in r["resume"]["gouvernant"])
fuites = []
for chemin in (["nom"], ["sondages", 0, "nom"]):
    rr = mod(e0, chemin, PIEGE)
    rr = mod(rr["etat"], ["sondages", 0, "fichier"], {"nom": PIEGE + ".gef", "texte": M.ESSAI_DEMO})
    for k_, h in (("resultat", rr["resultat"]), ("volets", rr["volets"]), ("donnees", rr["donnees"]), ("elements", rr["elements"]),
                  ("vues", "".join(v["svg"] for v in rr["vues"])), ("phrase", rr["phrase"])):
        if "<img" in h:
            fuites.append((chemin, k_))
chk("un nom piégé (fondation, sondage, fichier d'essai) est échappé partout, liens d'export compris", not fuites, fuites)
for abime, mot in (({"version": 1, "fondations": [{"B": -1}]}, "Largeur B"), ({"version": 1, "fondations": "x"}, "illisible"),
                   ({"version": 1, "fondations": [{"sondages": [{"id": 1, "couches": [{"type": "Argile verte"}]}]}]}, "type de sol")):
    rr = req(module="sol", action="calculer", etat=abime)
    chk(f"une note abîmée ({mot}) rend une erreur lisible, pas une exception", rr["ok"] is False and mot.lower() in rr["erreur"].lower(), rr.get("erreur"))
chk("pas de sélecteur dans ce module : aucune ligne n'en annonce, la brique répond vide",
    "data-pk" not in r0["resultat"] and "pk-vide" in req(module="sol", action="selecteur", etat=e0, role="x")["html"])

# ================================================================== 9. la relecture indépendante (07/11/2026)
print("\n=== 9. Relecture : vraisemblance, fichier Streamlit abîmé, légendes de l'écran, indices, disposition")
from api_web import dessin_sol as DS  # noqa: E402

for chemin, v in ((["B"], "1e308"), (["q"], "99999999"), (["sondages", 0, "couches", 0, "M"], "2e6")):
    rr = mod(e0, chemin, v)
    chk(f"{'/'.join(map(str, chemin))} = {v} : refusé (hors de toute vraisemblance), la note reste intacte",
        rr["ok"] and "vraisemblance" in (rr["message"] or "") and rr["etat"] == e0, rr.get("message") or rr.get("erreur"))
e_r = mod(e0, ["mode"], "rapide")["etat"]
rr = mod(e_r, ["rapide", "q"], "1e308")
chk("k = q / w : q = 1e308 refusé avec un message (l'écran Streamlit affichait « inf », le site levait une OverflowError)",
    rr["ok"] and "vraisemblance" in (rr["message"] or ""), rr.get("message") or rr.get("erreur"))
rr = req(module="sol", action="calculer", etat={"version": 1, "fondations": [{"q": 1e308, "B": 1e308, "L": 1e308}]})
chk("une note .stab aux nombres démesurés : erreur lisible (ValueError nommant le champ), plus de ZeroDivisionError",
    rr["ok"] is False and rr["erreur"].startswith("ValueError") and "vraisemblance" in rr["erreur"], rr.get("erreur"))
rr = req(module="sol", action="calculer", etat={"version": 1, "fondations": [{"sondages": [{"couches": [{"h": 1}], "points": [[1, 1e300, 2]]}]}]})
chk("des mesures CPT démesurées dans une note : refusées (« mesure illisible »)", rr["ok"] is False and "mesure illisible" in rr["erreur"],
    rr.get("erreur"))
for payload, mot in (({"version": "v5.0", "soundings": [None], "values": {}}, "sondage illisible"),
                     ({"version": "v5.0", "soundings": [{"id": "x"}], "values": {}}, "sondage illisible"),
                     ({"version": "v5.0", "soundings": [{"id": 1}], "orders": {"1": "abc"}, "values": {}}, "ordre des couches"),
                     ({"version": "v5.0", "soundings": "abc", "values": {}}, "illisible")):
    rr = mod(e0, ["streamlit"], {"nom": "raideur_sol.json", "texte": json.dumps(payload)})
    chk(f"fichier Streamlit abîmé ({mot}) : refusé en français, la fondation reste intacte",
        rr["ok"] and (rr["message"] or "").startswith("Fichier refusé") and mot in rr["message"] and "NoneType" not in rr["message"]
        and rr["etat"] == e0, rr.get("message"))
ok_ent = R.depuis_payload({"version": "v5.0", "soundings": [{"id": 1.0, "nom": "A"}], "orders": {"1": [1]}, "values": {"snd1_layer_1_h": 4.0}})
chk("un identifiant de sondage écrit 1.0 (entier en flottant) reste accepté, comme int() de l'écran", ok_ent["sondages"][0]["id"] == 1
    and ok_ent["sondages"][0]["couches"][0]["h"] == 4.0)
vues0 = {v["cle"]: v["svg"] for v in r0["vues"]}
chk("les légendes de l'écran Streamlit sous ses graphiques : diffusion (« À gauche le profil… ») et k(B) (« À sol identique… »)",
    "À gauche le profil, à droite les contraintes" in texte(vues0["diffusion"]) and "À sol identique, k varie fortement" in texte(vues0["kb"]))
fautes = []
for m in ("cpt", "rapide", "theories", "abaque", "plat"):
    e_m = mod(e0, ["mode"], m)["etat"]
    if m == "plat":
        e_m = mod(e_m, ["plat", "mortier"], True)["etat"]
    for v in mod(e_m, ["nom"], "F1")["vues"]:
        txt = " ".join(re.findall(r">([^<>]+)<", v["svg"]))
        for brut in ("q_net", "h_c", "E_c", "E_g", "t_g", "σ'v0", "_v0"):
            if brut in txt:
                fautes.append((m, v["cle"], brut))
chk("aucun indice écrit « à la développeur » dans les dessins (q_net, h_c, E_c, t_g, σ'v0 → indices en tspan)", not fautes, fautes)
chk("les nombres des dessins et des tableaux : espace fine insécable des milliers (un « 1 000 » ne se coupe pas)",
    DS.nombre(1234.5, 1) == "1 234,5" and DS.nombre(-12000, 0) == "-12 000" and DS.nombre(0.25) == "0,25")
sec = {m.group(1): m.group(2) for m in re.finditer(r'data-sec="(\w+)".*?<div class="rows">(.*?)(?=<div class="sec|$)', r0["donnees"], re.S)}
chk("la position de référence SCIA se choisit dans « Semelle et charge » (ouverte), plus dans les options repliées",
    'data-chemin="position"' in sec.get("se", "") and 'data-chemin="position"' not in sec.get("op", "x"), list(sec))
chk("les libellés du menu Méthode tiennent dans leur colonne (27 caractères au plus)", max(len(t) for _, t in M.METHODES) <= 27)
court = fondation([[(4.0, "Argile ferme / raide", 30.0, 19.0)]], B=2.0, L=2.0, D=2.0)
rr = req(module="sol", action="calculer", etat=court)
h = rr["resultat"]
chk("profil trop court sous une assise profonde : la précision « D + 2,5·B = 7,0 m sous le TN » (le conseil de l'écran, 5 m, ne dit pas "
    "d'où il se compte) ; le message GRAVE est posé sous le verdict, avant les lignes",
    "D + 2,5·B = 7,0 m sous le TN" in texte(h) and h.find("Profil trop court") < h.find('class="els"') and rr["resume"]["niveau"] == "c",
    texte(h)[:300])
deux = fondation([[(2.0, "Limon (loess)", 22.0, 18.0), (30.0, "Sable dense", 70.0, 20.0)]] * 4, nappe_active=True, nappe=1.2, D=1.0)
for k_, s_ in enumerate(deux["fondations"][0]["sondages"]):
    s_["id"], s_["nom"] = k_ + 1, f"S{k_ + 1}"
rr = req(module="sol", action="calculer", etat=deux)
sv = next(v["svg"] for v in rr["vues"] if v["cle"] == "sondages")
bords = [float(a) + float(b) for a, b in re.findall(r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" height="[\d.]+" fill="#[0-9a-f]+" stroke="#000"', sv)]
chk("vue « Sondages » : les colonnes restent à gauche de la marge des repères (assise, nappe ne les chevauchent plus)",
    bords and max(bords) <= 580 and "assise D = 1 m" in texte(sv) and "nappe" in texte(sv), (max(bords) if bords else None))

inc = fondation([[(2.0, "Limon (loess)", 20.0, 18.0), (2.0, "—", 0.0, 19.0), (16.0, "Sable dense", 80.0, 20.0)]])
rr = req(module="sol", action="calculer", etat=inc)
tout = rr["resultat"] + rr["volets"]
chk("le motif d'arrêt du moteur (« profil incomplet : 2.00 m … zone d'influence ») écrit à la française (2,00 m), sans double "
    "échappement (« &amp;#x27; »)", "2,00 m sans module" in tout and "2.00 m" not in tout and "&amp;" not in tout, texte(tout)[:200])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
