# -*- coding: utf-8 -*-
"""L'écran du module « Taux d'armature » refait (api_web/taux.py, docs/site/LOT_Q_QUATRE_MODULES.md § 3.5), vérifié en
Python ordinaire : CE QUE LE SITE AFFICHE = CE QUE LE MOTEUR CALCULE (le moteur lui-même est prouvé par
tests/test_taux_moteur.py).

1. la poutre TYPE de la note neuve, recalculée À LA MAIN (math seul) : chaque repère, le nombre d'étriers, le taux
   ≈ 85,8 kg/m³ — et l'écran l'affiche ;
2. sur dix états (type, riche à cadre intérieur et épingle, deux talons, talon gauche, 14 m à recouvrements, peau et
   retours, sans barres, sans étriers, reçue, deux poutres) : le verdict, chaque ligne des blocs « Acier par
   famille », « Par diamètre », « Béton » (et « Note entière »), le ratio au mètre, les cartes des données (repère,
   bilan des zones et des types), le bordereau de la vue, l'en-tête des vues — chaque nombre est celui du moteur ;
3. les FENÊTRES DE CALCUL : chaque formule du moteur écrite « symbole = expression = valeurs = résultat » ; les valeurs
   ÉCRITES, relues et recalculées ici, retombent sur le résultat écrit ; chaque fenêtre annoncée existe ;
4. les REFUS : chaque saisie hors domaine est refusée avec un message en français, la poutre intacte ;
5. les GESTES : + Zone propose la suite, une limite déplacée fait suivre la zone jointive, la longueur fait suivre la
   moitié droite — et le résultat le DIT ; un type renommé est suivi par ses zones ; l'ancienne note (v1) : noms
   gardés, avis dit une fois ;
6. le CONTRAT DE SÉLECTION : chaque data-sel des dessins a sa carte dans les données (data-cible) ;
7. la SÉCURITÉ : un nom piégé (poutre, type, provenance) est échappé partout ; aucun script ;
8. le BORDEREAU CSV : type text/csv, nom sûr, il se relit et redonne les nombres du moteur et de l'écran ;
9. de BOUT EN BOUT : une poutre du module « Poutres béton armé » avec ses longueurs, envoyée par le routeur (action
   « transmettre », vers « taux »), donne une note que cet écran relit et rend, provenance comprise ; sans longueur,
   le refus est en français ;
10. des MUTATIONS de l'écran (quinze), chacune vue rouge.

    /usr/bin/python3.12 tests/test_api_web_taux.py
"""
import copy
import csv
import io
import json
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import commun as C  # noqa: E402
from api_web import poutre as PO  # noqa: E402
from api_web import routeur as R  # noqa: E402
from api_web import taux as T  # noqa: E402
from beton.taux import moteur as MO  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(dict(module="taux", **k))))


def lire_fr(s):
    s = str(s).replace(C.NBSP, "").replace(C.FINE, "").replace(" ", "").replace("−", "-").replace(",", ".")
    return float(re.match(r"-?\d+(?:\.\d+)?", s).group(0))


def note(**champs):
    e = T.nouvel_etat()
    e["poutres"][0].update(copy.deepcopy(champs))
    return T.normaliser(e)


def barre(fam, lit, n, d, de=None, a=None, retours=0, l_retour=None):
    return {"famille": fam, "lit": lit, "n": n, "d": d, "de": de, "a": a, "retours": retours, "l_retour": l_retour}


# ====================================================================== les états
DEUX = T.nouvel_etat()
T.gerer_elements(DEUX, "ajouter")
DEUX["poutres"][1].update(b=40.0, h=80.0, L=8.0)
DEUX = T.normaliser(DEUX)
ETATS = {
    "type (note neuve)": T.nouvel_etat(),
    "riche (cadre intérieur, épingle, trou)": note(**copy.deepcopy(T._RICHE)),
    "deux talons": note(forme="deux_talons", talon_gauche={"b": 20.0, "h": 25.0}, talon_droit={"b": 15.0, "h": 20.0}),
    "talon gauche sans cadre de talon": note(forme="talon_gauche", types_etriers=[{"nom": "E1", "lignes": [{"forme": "cadre", "d": 8, "largeur": None}],
                                                                                  "cadre_talon": False}]),
    "14 m à recouvrements": note(L=14.0, zones=[{"de": 0.0, "a": 14.0, "s": 20.0, "type": "E1"}],
                                 barres=[barre("inf", 1, 3, 20), barre("sup", 1, 2, 12)]),
    "peau et retours": note(barres=[barre("inf", 1, 3, 16, retours=2), barre("peau", None, 2, 10, retours=2, l_retour=20.0),
                                    barre("sup", 1, 2, 12, a=2.0, retours=1)]),
    "sans barres": note(barres=[]),
    "sans étriers": note(zones=[]),
    "reçue de la poutre V1": note(origine={"module": "poutre", "nom": "V1", "date": "10/11/2026"}),
    "deux poutres (la seconde ouverte)": DEUX,
}


# ====================================================================== 1. la poutre type, à la main
def m_lin(d):
    return math.pi * (d / 1000.0) ** 2 / 4.0 * 7850.0


L, c = 6.0, 0.03
l_fil = L - 2 * c                                   # une barre filante : L − 2c
m1 = 3 * l_fil * m_lin(20)                          # 3 Ø 20 inférieures
m2 = 2 * (4.8 - 1.2) * m_lin(16)                    # renfort 2 Ø 16 de 1,20 à 4,80 m
m3 = 2 * l_fil * m_lin(12)                          # 2 Ø 12 supérieures
l_cadre = 2 * ((0.30 - 2 * c) + (0.60 - 2 * c)) + 2 * max(10 * 8, 70) / 1000.0
n_cadres = math.ceil(1.5 / 0.15) + math.ceil(3.0 / 0.25) + math.ceil(1.5 / 0.15) + 1   # une suite jointive : + 1 de fermeture
m4 = n_cadres * l_cadre * m_lin(8)
M_s = m1 + m2 + m3 + m4
M_maj = M_s * 1.05
V_c = 0.30 * 0.60 * L
taux_main = M_maj / V_c
r_type = MO.calculer(T.nouvel_etat()["poutres"][0])
chk(f"poutre type à la main : 33 cadres de {l_cadre:.2f} m (10 + 12 + 10 + 1 de fermeture)", n_cadres == 33 and abs(l_cadre - 1.72) < 1e-12)
chk(f"poutre type à la main : M_s = {M_s:.3f} kg, majorée {M_maj:.3f} kg, V_c = {V_c:.3f} m³, taux = {taux_main:.2f} kg/m³ (≈ 85,8) = le moteur",
    abs(r_type["totaux"]["taux_majore"] - taux_main) < 1e-9 and abs(r_type["totaux"]["masse"] - M_s) < 1e-9 and 85.7 < taux_main < 85.9)
chk("poutre type à la main : chaque repère (masse) = le moteur",
    all(abs(r["masse"] - m) < 1e-9 for r, m in zip(r_type["reperes"], (m1, m2, m3, m4))), [r["masse"] for r in r_type["reperes"]])
out = req(action="nouveau")
chk("la note neuve AFFICHE le taux de la main (85,8 kg/m³) et 92,7 kg d'acier pour 1,080 m³",
    f"{C.fr(taux_main, 1)}<small>kg/m³</small>" in out["resultat"] and "92,7" in out["resultat"] and "1,080" in out["resultat"])
chk("la note neuve : statut « Consultation » (un quantitatif n'a pas de conformité), niveau « i »",
    out["resume"]["niveau"] == "i" and "Consultation" in out["statut_html"], out["resume"])


# ====================================================================== 2. l'écran = le moteur
LIGNE = re.compile(r'<div class="el vc[^"]*"(?: data-calcul="(\w+)" title="[^"]*")?><div class="l1"><b>([^<]*)</b>'
                   r'(?:<span class="prof"><span>(.*?)</span></span>)?<span class="(?:pct|et-mot) c"[^>]*>(.*?)</span></div>'
                   r'<div class="l2"><span class="cmp">(.*?)</span>')


def blocs(html):
    """{indice du bloc : [(calcul, nom, pièce, droite, comparaison)]}."""
    out = {}
    for m in re.finditer(r'<section class="et-bloc [^"]*" data-theme="(\d+)">(.*?)</section>', html, re.S):
        out[int(m.group(1))] = LIGNE.findall(m.group(2))
    return out


def q(x, u, nd):
    return C.qte(x, u, nd)


def controle_resultat(mod, etat):
    """Les fautes de l'écran ``mod`` sur ``etat`` : chaque nombre du résultat, des données, des vues = le moteur."""
    f = []
    e = mod.normaliser(copy.deepcopy(etat))
    rep = mod.resultat(e)
    p = e["poutres"][e["actif"]]
    r = MO.calculer(p)
    t, b = r["totaux"], r["beton"]
    h = rep["resultat"]
    # le verdict
    if f'<span class="chip xl c" style="flex:none">{C.fr(t["taux_majore"], 1)}<small>kg/m³</small></span>' not in h:
        f.append(("verdict : taux majoré", C.fr(t["taux_majore"], 1)))
    if f"{q(t['masse_majoree'], 'kg', 1)} d'acier (+{C.NBSP}{C.nb(t['majoration'])}{C.NBSP}%) pour {q(b['volume'], 'm³', 3)} de béton" not in h:
        f.append(("verdict : comparaison", t["masse_majoree"], b["volume"]))
    if f"sans majoration : {q(t['taux'], 'kg/m³', 1)}" not in h:
        f.append(("verdict : taux sans majoration", t["taux"]))
    bl = blocs(h)
    # acier par famille : une ligne par famille présente (masse, part entière, longueur)
    fams = [x for x in r["familles"] if x["reperes"]]
    lignes = bl.get(0, [])
    if len(lignes) != len(fams):
        f.append(("familles : nombre de lignes", len(lignes), len(fams)))
    for x, (cal, nom, piece, droite, cmp_) in zip(fams, lignes):
        if nom != x["libelle"] or piece != q(x["masse"], "kg", 1) or lire_fr(droite) != round(x["part"] * 100) \
                or q(x["longueur"], "m", 2) not in cmp_ or cal != ("s3" if x["cle"] == "etriers" else "s0"):
            f.append(("famille", x["cle"], nom, piece, droite, cal))
    # par diamètre
    lignes = bl.get(1, [])
    if len(lignes) != len(r["diametres"]):
        f.append(("diamètres : nombre de lignes", len(lignes)))
    for d, (cal, nom, piece, droite, cmp_) in zip(r["diametres"], lignes):
        if nom != f"Ø {d['d']}" or piece != q(d["masse"], "kg", 1) or lire_fr(droite) != round(d["part"] * 100) \
                or q(d["longueur"], "m", 2) not in cmp_ or q(d["masse_lineique"], "kg/m", 3) not in cmp_:
            f.append(("diamètre", d["d"], nom, piece, droite))
    # béton
    lignes = bl.get(2, [])
    att = [("Section", b["aire_cm2"], "cm²"), ("Volume", b["volume"], "m³"), ("Masse", b["masse"], "kg")]
    for (nom, v, u), (cal, nom_l, piece, droite, cmp_) in zip(att, lignes):
        if nom_l != nom or abs(lire_fr(droite) - v) > 0.5 * 10 ** -(3 if u == "m³" else 0) + 1e-9 or cal != "s2":
            f.append(("béton", nom, droite, v))
    if len(lignes) != 3:
        f.append(("béton : trois lignes", len(lignes)))
    # la note entière (deux poutres)
    if len(e["poutres"]) > 1:
        sn = MO.synthese_note([MO.calculer(x) for x in e["poutres"]])
        if 3 not in bl or q(sn["taux_majore"], "kg/m³", 1).replace(C.NBSP, " ") not in h:
            f.append(("note entière", sn["taux_majore"]))
    elif 3 in bl:
        f.append(("note entière pour une seule poutre",))
    # le ratio au mètre, le bouton
    if f"<b>{q(t['kg_m_majore'], 'kg/m', 1)}</b>" not in h:
        f.append(("kg/m", t["kg_m_majore"]))
    if 'data-telecharger="bordereau"' not in h:
        f.append(("bouton Bordereau CSV",))
    # les avertissements : tous, le grave en tête
    for a in r["avertissements"]:
        if C.esc(a["message"]) not in h:
            f.append(("avertissement", a["code"]))
    # la provenance
    o = p.get("origine")
    if o and f"reçue de la poutre {o['nom']}" not in h:
        f.append(("provenance dans le résultat",))
    # les données : chaque carte de barres porte son repère et sa masse ; chaque zone son nombre ; chaque type ses cours
    d = rep["donnees"]
    for x in r["reperes"]:
        if x["sorte"] == "barre":
            m = re.search(rf'data-cible="{x["cle"]}">.*?<button type="button" class="et-tag"[^>]*>(\d+)</button>', d)
            if not m or int(m.group(1)) != x["repere"] or q(x["masse"], "kg", 2) not in d:
                f.append(("carte de barres", x["cle"]))
    for z in r["zones"]:
        m = re.search(rf'data-cible="z{z["index"]}">.*?<span class="tx-bilan">(\d+)', d)
        if not m or int(m.group(1)) != z["n"] or f"→ {z['cours']}{C.NBSP}cours" not in d:
            f.append(("carte de zone", z["nom"], m and m.group(1), z["n"]))
    for ty in r["types"]:
        m = re.search(rf'data-cible="t{ty["index"]}">.*?<span class="tx-bilan[^"]*">([^<]+)', d)
        if not m or (ty["employe"] and lire_fr(m.group(1)) != ty["n"]):
            f.append(("carte de type", ty["nom"]))
    # les vues : le bordereau (chaque repère, chaque Ø, le total majoré), l'en-tête (la bande, la provenance)
    vues = {v["cle"]: v for v in rep["vues"]}
    bord = vues["bordereau"]["svg"]
    for x in r["reperes"]:
        ligne = re.search(rf'<tr class="pc" data-sel="[^"]+"><td class="rep"><span class="tx-bul">{x["repere"]}</span></td>(.*?)</tr>', bord)
        if not ligne:
            f.append(("bordereau : repère absent", x["repere"]))
            continue
        cases = re.findall(r"<td(?: class=\"\w+\")?>(.*?)</td>", ligne.group(1))[1:]
        if [lire_fr(z) for z in cases] != [x["d"], x["n"], round(x["l_unitaire"], 3), round(x["l_totale"], 2), round(x["masse_lineique"], 3), round(x["masse"], 2)]:
            f.append(("bordereau : repère", x["repere"], cases))
    if f'<td class="kg">{C.fr(t["masse_majoree"], 2)}</td></tr></tfoot>' not in bord:
        f.append(("bordereau : total majoré", t["masse_majoree"]))
    for v in rep["vues"]:
        ent = v.get("entete", "")
        if f"<b>{C.fr(t['taux_majore'], 1)}</b><small>kg/m³</small>" not in ent:
            f.append(("en-tête : taux", v["cle"]))
        if o and f"Reçue de la poutre {o['nom']}" + (f", le {o['date']}" if o["date"] else "") not in ent:
            f.append(("en-tête : provenance", v["cle"]))
    # le résumé de la note
    if rep["resume"]["niveau"] != "i" or "kg/m³" not in rep["resume"]["gouvernant"]:
        f.append(("résumé", rep["resume"]))
    return f


for nom, etat in ETATS.items():
    fautes = controle_resultat(T, etat)
    chk(f"écran = moteur : {nom}", not fautes, fautes[:4])


# ====================================================================== 3. les fenêtres de calcul
def evaluer(valeurs: str) -> float:
    """Les valeurs ÉCRITES d'une formule (« 2 × [(0,3 − 2 × 0,03) + …] », « π × 0,02² / 4 × 7850 », « ⌈10⌉ ») relues et
    calculées ici."""
    s = valeurs.replace(C.NBSP, " ").replace(" m²", "").replace("×", "*").replace("−", "-").replace(",", ".").replace("π", "pi")
    s = s.replace("²", "**2").replace("⌈", "ceil(").replace("⌉", ")").replace("[", "(").replace("]", ")").replace(" ; ", ", ")
    if not re.fullmatch(r"[0-9.+\-*/() ,a-z]*", s):
        raise ValueError(s)
    return eval(s, {"__builtins__": {}}, {"pi": math.pi, "ceil": math.ceil, "max": max, "min": min})  # noqa: S307 — nos chiffres


def formules_de(res):
    out = list(res["beton"]["formules"]) + list(res["totaux"]["formules"])
    for bloc in res["reperes"] + res["zones"] + res["diametres"]:
        out += bloc["formules"]
    for ty in res["types"]:
        for li in ty["lignes"]:
            out += li["formules"]
        if ty["talon"]:
            out += ty["talon"]["formules"]
    return out


def controle_formules(mod, etat):
    f, n = [], 0
    e = mod.normaliser(copy.deepcopy(etat))
    rep = mod.resultat(e)
    r = MO.calculer(e["poutres"][e["actif"]])
    for fx in formules_de(r):
        n += 1
        li = mod._fx(fx)
        parts = li["parts"]
        if len(parts) != 3:
            f.append(("formule sans valeurs", fx["cle"]))
            continue
        try:
            calc = evaluer(parts[2])
        except Exception as ex:  # noqa: BLE001
            f.append(("valeurs illisibles", fx["cle"], parts[2], ex))
            continue
        ecrit = lire_fr(li["res"])
        nd = len(re.sub(r"[^0-9,]", "", li["res"].replace(C.NBSP, " ").split(" ")[0]).partition(",")[2])
        if abs(calc - ecrit) > 0.51 * 10 ** -nd + 3e-3 * abs(ecrit):
            f.append((fx["cle"], parts, li["res"], calc))
        if abs(ecrit - fx["resultat"]) > 0.51 * 10 ** -nd + 1e-9:
            f.append(("résultat écrit ≠ moteur", fx["cle"], li["res"], fx["resultat"]))
    # chaque formule est dans une fenêtre, chaque fenêtre annoncée existe
    nfx = rep["volets"].count('<div class="fx">')
    if nfx != n:
        f.append(("formules dans les fenêtres", nfx, n))
    for cal in set(re.findall(r'data-calcul="([^"]+)"', rep["resultat"])):
        if f'id="ck-{cal}"' not in rep["volets"]:
            f.append(("fenêtre absente", cal))
    return f, n


total = 0
for nom, etat in ETATS.items():
    fautes, n = controle_formules(T, etat)
    total += n
    chk(f"fenêtres : chaque formule réévaluée à partir des valeurs écrites ({n}) — {nom}", not fautes, fautes[:3])
chk(f"fenêtres : {total} formules réévaluées en tout (au moins 300)", total >= 300, total)
v0 = req(action="nouveau")["volets"]
chk("fenêtres : quatre (repères et taux, diamètres, béton, étriers par zone), colonnes d'un quantitatif",
    all(f'id="ck-s{i}"' in v0 for i in range(4)) and "<th>Repère</th>" in v0 and "<th>Zone</th>" in v0 and "Vérification" not in v0)


# ====================================================================== 4. les refus
E0 = T.nouvel_etat()
REFUS = [
    (["b"], "5", "largeur b"), (["L"], "0,2", "longueur"), (["enrobage"], "40", "enrobage"), (["h"], "abc", "illisible"),
    (["barres", 0, "n"], "0", "nombre de barres"), (["barres", 0, "d"], "13", "Ø"), (["barres", 1, "de"], "5", "doit dépasser"),
    (["barres", 0, "a"], "pas un nombre", "n'est pas un nombre"), (["barres", 7, "n"], "2", "il n'y en a que"),
    (["zones", 0, "a"], "5", "doit dépasser"), (["zones", 1, "s"], "3", "pas"), (["zones", 0, "type"], "E9", "n'est pas un type"),
    (["types", 0, "-"], "1", "l'emploient"), (["types", 0, "lignes", 0, "-"], "1", "au moins une ligne"),
    (["zones", "+"], "1", "couvrent déjà toute la poutre"), (["majoration"], "60", "majoration"),
    (["nom"], "x" * 80, "trop long"), (["recouvrements", "l_com"], "30", "longueur commerciale"),
    (["inconnu"], "1", "Champ inconnu"), (["barres", 0, "zzz"], "1", "Champ inconnu"),
]
mal = []
for chemin, valeur, mot in REFUS:
    o = req(action="modifier", etat=copy.deepcopy(E0), chemin=chemin, valeur=valeur)
    if not (o.get("ok") and o.get("message") and mot.lower() in o["message"].lower() and o["etat"] == E0):
        mal.append((chemin, valeur, o.get("message") or o.get("erreur")))
chk(f"refus : {len(REFUS)} saisies hors domaine refusées en français, en nommant le champ, la poutre intacte", not mal, mal[:3])
riche = ETATS["riche (cadre intérieur, épingle, trou)"]
o = req(action="modifier", etat=copy.deepcopy(riche), chemin=["types", 0, "lignes", 1, "largeur"], valeur="40")
chk("refus : un cadre intérieur plus large que le cadre de l'âme (b − 2c)", o["message"] and "dépasse le cadre" in o["message"] and o["etat"] == riche,
    o.get("message"))
o = req(action="modifier", etat=copy.deepcopy(riche), chemin=["zones", 1, "de"], valeur="1,5")
chk("refus : deux zones qui se recouvrent sont nommées", o["message"] and "Z1" in o["message"] and "se recouvrent" in o["message"]
    and o["etat"] == riche, o.get("message"))


# ====================================================================== 5. les gestes, ce qui suit, l'ancienne note
o = req(action="modifier", etat=copy.deepcopy(riche), chemin=["zones", "+"], valeur="1")
z = o["etat"]["poutres"][0]["zones"]
chk("+ Zone propose la suite : la portion sans étriers (2 à 3 m), au pas et du type de la précédente",
    len(z) == 3 and z[1] == {"de": 2.0, "a": 3.0, "s": 10.0, "type": "E1"}, z)
o = req(action="modifier", etat=copy.deepcopy(E0), chemin=["zones", 0, "a"], valeur="2")
zs = o["etat"]["poutres"][0]["zones"]
chk("une limite de zones déplacée : la zone jointive suit (Z2 commence à 2 m) — et le résultat le DIT",
    zs[0]["a"] == 2.0 and zs[1]["de"] == 2.0 and "Ont suivi : Z2 commence à 2 m" in o["resultat"] and "suivi" not in o["etat"], o.get("message"))
o = req(action="modifier", etat=copy.deepcopy(E0), chemin=["L"], valeur="8")
pz = o["etat"]["poutres"][0]
chk("la longueur change : la zone d'about droit et la fin du renfort gardent leur distance à l'about droit, et c'est dit",
    pz["zones"][2] == {"de": 6.5, "a": 8.0, "s": 15.0, "type": "E1"} and pz["barres"][1]["a"] == 6.8 and "Ont suivi" in o["resultat"], pz["zones"])
o = req(action="modifier", etat=copy.deepcopy(E0), chemin=["types", 0, "nom"], valeur="C8")
chk("un type renommé : ses zones le suivent", {z["type"] for z in o["etat"]["poutres"][0]["zones"]} == {"C8"} and not o["message"])
o = req(action="modifier", etat=copy.deepcopy(E0), chemin=["types", 0, "lignes", "+"], valeur="1")
o2 = req(action="modifier", etat=o["etat"], chemin=["types", 0, "lignes", 1, "forme"], valeur="cadre_interieur")
li = o2["etat"]["poutres"][0]["types_etriers"][0]["lignes"][1]
chk("+ Ligne (une épingle), puis « cadre intérieur » : la largeur proposée apparaît (la moitié du cadre de l'âme)",
    li == {"forme": "cadre_interieur", "d": 8, "largeur": 12.0} and 'data-chemin="types/0/lignes/1/largeur"' in o2["donnees"], li)
o = req(action="modifier", etat=copy.deepcopy(E0), chemin=["barres", "+"], valeur="renfort")
chk("+ Renfort : 2 Ø 16 en second lit inférieur, du cinquième aux quatre cinquièmes de L",
    o["etat"]["poutres"][0]["barres"][-1] == barre("inf", 2, 2, 16, 1.2, 4.8))
o = req(action="calculer", etat=copy.deepcopy(T.NOTE_V1))
chk("ancienne note (v1) : le nom est gardé, la poutre est la poutre type, l'avis est dit",
    o["ok"] and o["etat"]["poutres"][0]["nom"] == "Poutre RDC" and o["etat"]["poutres"][0]["barres"] == MO.nouvelle_poutre()["barres"]
    and o.get("avis") and "ancien module" in o["resultat"] and o["etat"]["version"] == 2, o.get("erreur"))
o2 = req(action="calculer", etat=o["etat"])
chk("ancienne note : l'avis n'est dit qu'une fois (l'état relu n'en porte plus)", not o2.get("avis") and "ancien module" not in o2["resultat"])
o = req(action="dupliquer", etat=copy.deepcopy(DEUX), index=0)
chk("copier une poutre : la copie s'insère juste après et s'ouvre", [p["nom"] for p in o["etat"]["poutres"]] == ["P1", "P1 (copie)", "P2"]
    and o["etat"]["actif"] == 1)
chk("note abîmée : « poutres » qui n'est pas une liste est refusée en clair",
    not req(action="calculer", etat={"version": 2, "poutres": "x"})["ok"])


# ====================================================================== 6. le contrat de sélection
mal = []
for nom, etat in ETATS.items():
    o = T.resultat(T.normaliser(copy.deepcopy(etat)))
    sels = set(re.findall(r'data-sel="([^"]+)"', "".join(v["svg"] for v in o["vues"])))
    cibles = set(re.findall(r'data-cible="([^"]+)"', o["donnees"]))
    if not sels or sels - cibles:
        mal.append((nom, sorted(sels - cibles)))
chk("contrat de sélection : chaque pièce des dessins (data-sel) a sa carte dans les données (data-cible), sur chaque état", not mal, mal[:3])


# ====================================================================== 7. la sécurité
PIEGE = '<img src=x onerror=alert(1)>'
e = note(nom=PIEGE, types_etriers=[{"nom": "<b>E1</b>", "lignes": [{"forme": "cadre", "d": 8, "largeur": None}], "cadre_talon": True}],
         zones=[{"de": 0.0, "a": 6.0, "s": 20.0, "type": "<b>E1</b>"}], origine={"module": "poutre", "nom": PIEGE, "date": "<i>x</i>"})
o = req(action="calculer", etat=e)
frag = "".join([o["resultat"], o["volets"], o["donnees"], o["elements"], o["statut_html"]] + [v["svg"] + v.get("entete", "") for v in o["vues"]])
chk("sécurité : un nom de poutre, de type et de provenance piégés sont échappés partout", o["ok"] and "<img" not in frag and "<b>E1" not in frag
    and "<i>x" not in frag, re.findall(r".{20}<(?:img|b>E1|i>x).{20}", frag)[:3])
o = req(action="calculer", etat=copy.deepcopy(riche))
frag = "".join([o["resultat"], o["volets"], o["donnees"]] + [v["svg"] + v.get("entete", "") for v in o["vues"]])
chk("sécurité : aucun script, aucun « on…= » dans les fragments", not re.search(r"<script|\son[a-z]+\s*=", frag, re.I))


# ====================================================================== 8. le bordereau CSV
mal = []
for nom, etat in ETATS.items():
    o = req(action="telecharger", etat=copy.deepcopy(etat), quoi="bordereau")
    if not o.get("ok"):
        mal.append((nom, o.get("erreur")))
        continue
    fi = o["fichier"]
    ee = T.normaliser(copy.deepcopy(etat))
    p = ee["poutres"][ee["actif"]]
    r = MO.calculer(p)
    rows = list(csv.reader(io.StringIO(fi["texte"]), delimiter=";"))
    if fi["type"] != "text/csv" or not fi["nom"].endswith(".csv") or C.refus_fichier(fi) or rows[0] != list(MO.EN_TETE_CSV):
        mal.append((nom, "type, nom ou en-tête", fi["nom"]))
        continue
    reps = rows[1:1 + len(r["reperes"])]
    for x, row in zip(r["reperes"], reps):
        if int(row[0]) != x["repere"] or abs(lire_fr(row[5]) - x["l_totale"]) > 5e-4 or abs(lire_fr(row[7]) - x["masse"]) > 5e-4:
            mal.append((nom, "repère", row))
    tot = rows[-1]
    if tot[0] != "Total majoré" or abs(lire_fr(tot[7]) - r["totaux"]["masse_majoree"]) > 5e-4:
        mal.append((nom, "total majoré", tot))
    # les mêmes nombres que l'écran : le total majoré du bordereau affiché
    bord = next(v["svg"] for v in T.resultat(ee)["vues"] if v["cle"] == "bordereau")
    ecran = re.search(r'<td class="kg">([^<]+)</td></tr></tfoot>', bord)
    if not ecran or abs(lire_fr(ecran.group(1)) - lire_fr(tot[7])) > 0.0051:
        mal.append((nom, "CSV ≠ écran", tot[7], ecran and ecran.group(1)))
    if sum(lire_fr(row[7]) for row in reps) - r["totaux"]["masse"] > 5e-4 * (len(reps) + 1):
        mal.append((nom, "Σ repères ≠ total"))
chk(f"bordereau CSV : text/csv, nom sûr, il se relit et redonne les nombres du moteur et de l'écran ({len(ETATS)} états)", not mal, mal[:3])
o = req(action="telecharger", etat=note(nom="P1/2 : RDC"), quoi="bordereau")
chk("bordereau CSV : le nom du fichier est sûr (« Bordereau P1-2 - RDC.csv »)", o["fichier"]["nom"] == "Bordereau P1-2 - RDC.csv", o["fichier"]["nom"])
chk("bordereau CSV : un autre fichier demandé est refusé en clair", req(action="telecharger", etat=E0, quoi="note_pdf")["ok"] is False)


# ====================================================================== 9. de bout en bout : la poutre reçue
def etat_poutre(longueurs):
    e = PO.nouvel_etat()
    p = e["poutres"][0]
    s = p["sections"][0]
    a = copy.deepcopy(s)
    b = copy.deepcopy(s)
    b["nom"], b["inf"], b["pas"] = "B", [{"n": 4, "d": 16}], 20.0
    cc = copy.deepcopy(s)
    cc["nom"] = "C"
    p["sections"] = [a, b, cc]
    for sec, lg in zip(p["sections"], longueurs):
        if lg is not None:
            sec["longueur"] = lg
    return e


env = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "transmettre", "etat": etat_poutre([1.5, 3.0, 1.5]), "vers": "taux",
                                       "date": "10/11/2026"})))
chk("bout en bout : la poutre V1 et ses trois longueurs, envoyée par le routeur, donne une note du module « taux »",
    env.get("ok") and env["note"]["module"] == "taux", env.get("erreur"))
if env.get("ok"):
    et = env["note"]["etat"]
    pt = et["poutres"][0]
    o = req(action="calculer", etat=et)
    date = pt["origine"]["date"]
    chk("bout en bout : l'écran relit la note reçue (L = 6 m, une zone par section, 2 Ø 16 filants et un renfort de 2 Ø 16 au milieu)",
        o.get("ok") and pt["L"] == 6.0 and len(pt["zones"]) == 3 and any(b["de"] == 1.5 and b["a"] == 4.5 and b["n"] == 2 for b in pt["barres"]),
        o.get("erreur") or pt["barres"])
    attendu = "reçue de la poutre V1" + (f", le {date}" if date else "")
    chk(f"bout en bout : la provenance est dite dans le résultat et l'en-tête de chaque vue (« {attendu} »)",
        o.get("ok") and attendu in o["resultat"] and all(attendu[0].upper() + attendu[1:] in v["entete"] for v in o["vues"]))
    if not date:
        print("     (le routeur et la poutre ne transmettent pas encore la date de l'envoi : provenance sans date — besoin commun)")
    chk("bout en bout : la note reçue se recalcule sans perte (écran = moteur)", not controle_resultat(T, et))
ref = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "transmettre", "etat": etat_poutre([1.5, None, 1.5]), "vers": "taux"})))
chk("bout en bout : une poutre sans longueur de section est refusée, en français, en disant où les saisir",
    ref.get("ok") is False and "longueur" in ref.get("erreur", "").lower() and "Traceback" not in ref.get("erreur", ""), ref.get("erreur"))


# ====================================================================== 10. les mutations : chacune vue rouge
def mutant(ancien, nouveau):
    src = open(T.__file__, encoding="utf-8").read()
    assert src.count(ancien) == 1, (ancien, src.count(ancien))
    m = types.ModuleType("api_web.taux_mutant")
    m.__package__ = "api_web"
    m.__file__ = T.__file__
    exec(compile(src.replace(ancien, nouveau, 1), T.__file__, "exec"), m.__dict__)  # noqa: S102 — le module lui-même, muté
    return m


def rouge(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — un mutant qui plante est vu rouge aussi
        return True


def refuse_tout(m):
    e = m.normaliser(copy.deepcopy(E0))
    return m.modifier(e, ["b"], "5") is not None and e == E0


E_TYPE = ETATS["type (note neuve)"]
E_RECUE = ETATS["reçue de la poutre V1"]
MUTATIONS = [
    ("le verdict montre le taux SANS majoration", 'C.verdict_valeur(t["taux_majore"]', 'C.verdict_valeur(t["taux"]',
     lambda m: controle_resultat(m, E_TYPE)),
    ("la jauge d'une famille rapporte sa masse au total MAJORÉ", '_part(f["part"]), "c", calcul="s3"',
     '_part(f["masse"] / r["totaux"]["masse_majoree"]), "c", calcul="s3"', lambda m: controle_resultat(m, E_TYPE)),
    ("la masse d'un diamètre est sa longueur", 'piece=C.qte(d["masse"], "kg", 1))', 'piece=C.qte(d["longueur"], "kg", 1))',
     lambda m: controle_resultat(m, E_TYPE)),
    ("le volume de béton affiché est la masse", 'etat_txt=C.qte(b["volume"], "m³", 3).replace(C.NBSP, " "))',
     'etat_txt=C.qte(b["masse"], "m³", 3).replace(C.NBSP, " "))', lambda m: controle_resultat(m, E_TYPE)),
    ("le contour du cadre écrit 3 × au lieu de 2 × dans les valeurs", "f\"2 × [({v('b', 100.0)}", "f\"3 × [({v('b', 100.0)}",
     lambda m: controle_formules(m, E_TYPE)[0]),
    ("la masse linéique écrit Ø²/2 au lieu de Ø²/4", "² / 4 × {v('ρ_s')}", "² / 2 × {v('ρ_s')}", lambda m: controle_formules(m, E_TYPE)[0]),
    ("le quotient d'une zone oublie le passage en m du pas", "{v('l_z')} / {v('s', 100.0)}", "{v('l_z')} / {v('s')}",
     lambda m: controle_formules(m, E_TYPE)[0]),
    ("la provenance n'est plus dite dans l'en-tête des vues", "    if o:\n        lignes.append(", "    if False:\n        lignes.append(",
     lambda m: controle_resultat(m, E_RECUE)),
    ("ce qui a suivi une saisie n'est plus dit", "    if suivis:\n        msgs +=", "    if False:\n        msgs +=",
     lambda m: (lambda e: (m.modifier(e, ["zones", 0, "a"], "2") or True) and "Ont suivi" not in m.resultat(e)["resultat"])(m.normaliser(copy.deepcopy(E0)))),
    ("une saisie refusée n'est plus dite (rendue sans message)", 'return f"Saisie refusée : {e}"', "return None", lambda m: not refuse_tout(m)),
    ("le bordereau part en texte brut", '"type": "text/csv",', '"type": "text/plain",',
     lambda m: m.telecharger(m.normaliser(copy.deepcopy(E0)), "bordereau")["type"] != "text/csv"),
    ("la carte d'une zone change de clé (le dessin ne la trouve plus)", 'data-cible="z{i}"', 'data-cible="zone{i}"',
     lambda m: (lambda o: set(re.findall(r'data-sel="(z\d+)"', "".join(v["svg"] for v in o["vues"])))
                - set(re.findall(r'data-cible="([^"]+)"', o["donnees"])))(m.resultat(m.normaliser(copy.deepcopy(E0))))),
    ("le nom de la poutre n'est plus échappé dans le résultat", 'nom = C.esc(p.get("nom") or "Poutre")', 'nom = (p.get("nom") or "Poutre")',
     lambda m: "<img" in m.resultat(m.normaliser(note(nom=PIEGE)))["resultat"]),
    ("l'ancienne note perd le nom de ses poutres", "MO.nouvelle_poutre(_nom_v1(p, i))", "MO.nouvelle_poutre()",
     lambda m: m.normaliser(copy.deepcopy(T.NOTE_V1))["poutres"][0]["nom"] != "Poutre RDC"),
    ("le nombre d'étriers d'une zone n'est plus celui du moteur (sans la fermeture)",
     'f\'<span class="et-txt">{C.esc(txt)}</span><span class="tx-bilan">{rz["n"]}',
     'f\'<span class="et-txt">{C.esc(txt)}</span><span class="tx-bilan">{rz["cours"]}', lambda m: controle_resultat(m, E_TYPE)),
]
for titre, ancien, nouveau, controle in MUTATIONS:
    try:
        mm = mutant(ancien, nouveau)
        vu = rouge(lambda: controle(mm))
    except AssertionError as ex:
        vu = False
        print("     (mutation introuvable :", ex, ")")
    chk(f"mutation vue rouge : {titre}", vu)
chk("témoin : le module tel quel passe les mêmes contrôles",
    not controle_resultat(T, E_TYPE) and not controle_formules(T, E_TYPE)[0] and refuse_tout(T) and not controle_resultat(T, E_RECUE))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
