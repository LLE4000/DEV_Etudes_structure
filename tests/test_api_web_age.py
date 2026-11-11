# -*- coding: utf-8 -*-
"""L'écran « Maturité du béton » du site (api_web/age.py), refait le 10/11/2026 : l'écran = le moteur.

  1. La note neuve se lit d'un coup d'œil (« Décoffrage · 21 MPa : J + 10,7 ») ; les quatre rubriques, la colonne
     compacte, la première rubrique seule ouverte.
  2. Chaque nombre et chaque date affichés = le moteur (``beton.maturite.moteur.calculer``), sur des coulages variés :
     le verdict, chaque ligne des blocs « Objectifs », « Températures », « Essai », le tableau jour par jour, la
     courbe (ses repères), le sélecteur de classe.
  3. Les fenêtres de calcul : chaque ligne « symbole = expression = valeurs = résultat » est RÉÉVALUÉE à partir des
     valeurs affichées (petit interpréteur : exp, ln, √, max).
  4. Les saisies : chaque champ relié à l'état ; chaque refus avec un message qui nomme le champ, l'état intact ; les
     effets de bord (mode de la cible, durcissement, objectifs ajoutés / retirés, modèles).
  5. Le sélecteur de classe (✓ la classe du coulage, ★ la plus petite qui tient l'âge prévu, aucune sans âge prévu).
  6. La note d'avant (version 1) : noms gardés, coulages neufs, l'avis dit une fois ; l'échappement d'un nom piégé.
  7. Pureté : le module se charge et calcule sans numpy, streamlit, matplotlib, pandas, requests.
  8. La météo du lieu (le faux Open-Meteo : aucun réseau) : le plan du bouton, l'import gardé, les refus.
  9. MUTATIONS de l'écran : une mise en forme fautive (l'échéance tronquée au jour, nominal et prudent confondus, un
     taux recopié…) fait rougir les contrôles du § 2.

    python3.12 tests/test_api_web_age.py
"""
import copy
import datetime as _dt
import html
import json
import math
import os
import re
import subprocess
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faux_open_meteo as FO  # noqa: E402  (tests/ : le faux Open-Meteo)
from api_web import age as A  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.maturite import entree as EN  # noqa: E402
from beton.maturite import lois as L  # noqa: E402
from beton.maturite import meteo as MT  # noqa: E402
from beton.maturite import moteur as M  # noqa: E402

OK, KO = [], []
NB = C.NBSP


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def note(*coulages, actif=0):
    return {"version": 2, "coulages": [EN.normaliser(c) for c in coulages], "actif": actif}


def coulage(nom="Coulage 1", **k):
    """Un coulage neuf modifié : « temperature__T » écrit dans le groupe « temperature »."""
    e = EN.nouvel_element(nom)
    for cle, v in k.items():
        a, _, b = cle.partition("__")
        if b:
            e[a][b] = v
        else:
            e[a] = v
    return EN.normaliser(e)


def objectif(nom, **k):
    o = EN.nouvel_objectif(nom)
    o.update(k)
    return o


def calc(etat):
    out = req(module="age", action="calculer", etat=copy.deepcopy(etat))
    assert out["ok"], out
    return out


def saisir(etat, chemin, valeur):
    out = req(module="age", action="modifier", etat=copy.deepcopy(etat), chemin=chemin.split("/") if isinstance(chemin, str) else chemin,
              valeur=valeur)
    assert out["ok"], out
    return out


def texte(h):
    """Le texte d'un fragment HTML, sans balises ni entités, les espaces insécables gardées."""
    return html.unescape(re.sub(r"<[^>]+>", "", h))


def plat(h):
    """… et ses espaces (insécables comprises) ramenées à une espace simple."""
    return re.sub(r"[\s  ]+", " ", texte(h)).strip()


AUJ = _dt.date(2026, 11, 7)
A.aujourdhui = lambda: AUJ                                   # le jour de l'écran, figé (le plan du bouton en dépend)
A.maintenant = lambda: "2026-11-07T10:30"                   # l'heure d'une récupération, figée


# ------------------------------------------------------------------ la mise en forme attendue, écrite ici
JOURS = ("lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.")


def fr(x, nd=1):
    """Un nombre à la française, arrondi demi vers l'extérieur sur sa valeur décimale (comme l'écran)."""
    return C.fr(x, nd).replace("-", "−")


def taux_txt(t):
    """Le taux tel que la pastille l'écrit (C.chip) : « 87 % », « > 999 % »."""
    return html.unescape(C._taux_txt(C.pct(t))[0])


def jplus(t):
    return f"J{NB}+{NB}{C.fr(t, 1)}"


def instant(iso, annee=None, mot="vers"):
    d = _dt.datetime.fromisoformat(iso)
    an = "" if annee is None or d.year == annee else f"/{d.year}"
    return f"{JOURS[d.weekday()]}{NB}{d:%d/%m}{an}{NB}{mot}{NB}{d.hour}{NB}h"


# ------------------------------------------------------------------ 1. la note neuve
print("=== 1. La note neuve : « Décoffrage · 21 MPa : J + 10,7 » ===")
r0 = req(module="age", action="nouveau")
m0 = M.calculer(EN.nouvel_element("Coulage 1"))
o0 = m0["objectifs"][0]
v0 = re.search(r'<div class="verdict age-v"[^>]*>.*?</div></div>', r0["resultat"], re.S)
chk("la note neuve se calcule : C30/37, béton à durcissement moyen, 20 °C constante, « Décoffrage » à 70 % de f_ck non daté",
    r0["ok"] and r0["etat"]["coulages"][0]["classe"] == "C30/37" and r0["etat"]["coulages"][0]["durcissement"]["beton"] == "moyen"
    and r0["etat"]["coulages"][0]["objectifs"][0]["age"] is None, r0.get("erreur"))
chk("le verdict se lit d'un coup d'œil : « J + 10,7 j », « Décoffrage · 21 MPa », la variante prudente du moteur (J + 12,2)",
    v0 and "10,7" in texte(v0.group(0)) and "Décoffrage · 21 MPa" in texte(v0.group(0)) and abs(o0["t"] - 10.73) < 0.01
    and f"prudent (−{NB}3{NB}°C) : {jplus(o0['prudent']['t'])}" in texte(v0.group(0)) and abs(o0["prudent"]["t"] - 12.19) < 0.01,
    v0 and texte(v0.group(0)))
d0 = r0["donnees"]
secs = re.findall(r'<div class="sec( open)?" data-sec="(\w+)"( data-secondaire="1")?>', d0)
chk("quatre rubriques : Béton (ouverte), Températures, Objectifs, Hypothèses (secondaire) ; la colonne compacte (data-elements)",
    [(o, k, s) for o, k, s in secs] == [(" open", "bt", ""), ("", "tp", ""), ("", "ob", ""), ("", "hy", ' data-secondaire="1"')]
    and '<div class="et-elsel" data-elements="1">' in d0, secs)
chk("trois vues — Courbe, Températures, Tableau — hors échelle, chacune sous son en-tête",
    [v["cle"] for v in r0["vues"]] == ["courbe", "temperatures", "tableau"] and all(v["echelle"] is False and "et-ent" in v["entete"] for v in r0["vues"]))
chk("la note neuve est une consultation (aucun objectif daté) : le statut le dit, le résumé nomme l'échéance",
    "Consultation" in r0["statut_html"] and r0["resume"]["gouvernant"] == "Coulage 1 · Décoffrage 21 MPa : J + 10,7", r0["resume"])
# J + 6,5 en ciment N (conception § 2.5)
rN = saisir(r0["etat"], "durcissement/choix", "ciment:N")
chk("le ciment N (au lieu du béton moyen) : J + 6,5 — et la ligne discrète dit le r supposé (50 %) et s = 0,25",
    "6,5" in texte(re.search(r'<div class="verdict.*?</div></div>', rN["resultat"], re.S).group(0))
    and f"f<sub>cm</sub>(2{NB}j) = 50{NB}% de f<sub>cm</sub>(28{NB}j) · s = 0,25" in rN["donnees"], texte(rN["resultat"])[:200])

# ------------------------------------------------------------------ 2. l'écran = le moteur
print("\n=== 2. Chaque nombre et chaque date affichés = le moteur ===")
RELEVE = ("2026-11-02;9,5;mesuré\n2026-11-03;8,0;mesuré\n2026-11-04;—;mesuré\n2026-11-05;6,5;mesuré\n2026-11-06;-2,0;mesuré\n"
          "2026-11-07;4,0;mesuré\n2026-11-08;6,0;prévision\n2026-11-09;7,5;prévision\n2026-11-10;8,5;prévision\n2026-11-11;7,0;prolongé")
FROID = "\n".join(["2,0"] * 7 + ["20,0"] * 7)                 # le défaut n° 1 de l'ancien écran : 7 j à 2 °C puis 7 j à 20 °C
CAS = {
    "neuf": coulage(),
    "ciment N, daté le 02/11 à 16 h, décoffrage à J + 3": coulage(durcissement__mode="ciment", date="2026-11-02", heure=16,
                                                                 objectifs=[objectif("Décoffrage", age=3.0)]),
    "relevé de Namur, trois objectifs (dont un modèle en MPa sur la moyenne)": coulage(
        "Voile R+1", date="2026-11-02", heure=7, temperature__mode="releve", temperature__releve=RELEVE, temperature__lieu="Namur",
        objectifs=[objectif("Décoffrage", age=6.0), M.modele_objectif("hors_gel", 30.0), objectif("Mise en charge", mode="mpa", mpa=25.0, base="fck")]),
    "défaut n° 1 : 7 j à 2 °C puis 7 j à 20 °C, ciment N, EC2, 15 MPa caractéristique": coulage(
        durcissement__mode="ciment", loi__mode="ec2", temperature__mode="releve", temperature__releve=FROID,
        objectifs=[objectif("Décoffrage", mode="mpa", mpa=15.0, base="fck")]),
    "non atteignable (40 MPa sur un C30/37) et jamais (−15 °C en loi EC2)": coulage(
        temperature__T=-15.0, loi__mode="ec2", objectifs=[objectif("Trop haut", mode="mpa", mpa=40.0), objectif("Gelé", age=5.0)]),
    "essai cube calé, Arrhenius 45 kJ/mol, C25/30 lent": coulage(
        classe="C25/30", durcissement__beton="lent", loi__mode="perso", loi__Ea_kJ=45.0, temperature__T=12.0, date="2026-12-28", heure=18,
        essai__actif=True, essai__age=7.0, essai__mpa=18.0, essai__caler=True, objectifs=[objectif("Décoffrage", age=10.0), objectif("100 %", pct=100.0)]),
    "r connu 0,40, objectif daté non tenu (C20/25, 5 °C)": coulage(
        classe="C20/25", durcissement__mode="r", durcissement__r=0.40, temperature__T=5.0, objectifs=[objectif("Désétaiement", pct=80.0, age=4.0)]),
    "météo importée (Bruxelles, décoffrage à J + 3)": A._etat_meteo(True)["coulages"][0],
}


def bloc(h, i):
    m = re.search(rf'<section class="et-bloc [^"]*" data-theme="{i}">(.*?)</section>', h, re.S)
    return m.group(1) if m else ""


def lignes_bloc(h, i):
    return re.findall(r'<div class="el vc[^"]*"[^>]*>(.*?)</div></div>', bloc(h, i), re.S)


def controle_ecran(nom, c, ecran=None):
    """Les fautes de l'écran face au moteur pour un coulage : chaque nombre, chaque date."""
    fautes = []
    m = M.calculer(c)
    rep = ecran or calc({"version": 2, "coulages": [c], "actif": 0})
    res, an = rep["resultat"], (int(m["temps"]["date"][:4]) if m["temps"]["date"] else None)
    v = re.search(r'<div class="verdict[^"]*"[^>]*>.*?</div></div>', res, re.S)
    v = texte(v.group(0)) if v else ""
    o = m["objectifs"][0]
    # le verdict : le premier objectif
    if o["age_prevu"] is not None:
        if o["taux"] is not None and taux_txt(o["taux"]) not in v:
            fautes.append(f"{nom} : verdict sans le taux {o['taux']}")
        if f"{fr(o['f_prevu'])}{NB}MPa" not in v or f"{fr(o['f_prevu_prudent'])}{NB}MPa" not in v:
            fautes.append(f"{nom} : verdict sans f(t_prévu) = {o['f_prevu']} ou sa variante prudente")
        if o["instant_prevu"] and instant(o["instant_prevu"], an, "à") not in v:
            fautes.append(f"{nom} : verdict sans la date de l'opération {o['instant_prevu']}")
    elif o["t"] is not None:
        if f"J +{C.fr(o['t'], 1)}j" not in v:
            fautes.append(f"{nom} : verdict sans l'âge {o['t']}")
        if o["instant"] and instant(o["instant"], an) not in v:
            fautes.append(f"{nom} : verdict sans l'instant {o['instant']}")
        if m["synthese"]["prudent_distinct"] and o["prudent"]["t"] != o["t"] and jplus(o["prudent"]["t"]) not in v:
            fautes.append(f"{nom} : verdict sans la variante prudente {o['prudent']['t']}")
    # le bloc « Objectifs » : une ligne par objectif, dans l'ordre
    ls = lignes_bloc(res, 0)
    if len(ls) != len(m["objectifs"]):
        fautes.append(f"{nom} : {len(ls)} lignes d'objectifs pour {len(m['objectifs'])}")
    for x, h in zip(m["objectifs"], ls):
        t = texte(h)
        if not t.startswith(x["nom"]):
            fautes.append(f"{nom} : ligne « {t[:30]} » pour « {x['nom']} »")
        if f"{C.nb(x['cible'], 1)}{NB}MPa" not in t:
            fautes.append(f"{nom}/{x['nom']} : cible {x['cible']} absente")
        if x["age_prevu"] is not None:
            if x["taux"] is not None and taux_txt(x["taux"]) not in t:
                fautes.append(f"{nom}/{x['nom']} : taux {x['taux']} absent")
            if f"{fr(x['f_prevu'])}{NB}MPa" not in t:
                fautes.append(f"{nom}/{x['nom']} : f_prévu {x['f_prevu']} absent")
        elif x["t"] is not None:
            if jplus(x["t"]) not in t:
                fautes.append(f"{nom}/{x['nom']} : âge {x['t']} absent")
            if x["instant"] and instant(x["instant"], an) not in t:
                fautes.append(f"{nom}/{x['nom']} : instant {x['instant']} absent")
            if m["synthese"]["prudent_distinct"] and x["prudent"]["t"] != x["t"] and f"prudent {jplus(x['prudent']['t'])}" not in t:
                fautes.append(f"{nom}/{x['nom']} : prudent {x['prudent']['t']} absent")
        elif x["etat"] == "non_atteignable" and "non atteignable" not in t:
            fautes.append(f"{nom}/{x['nom']} : « non atteignable » absent")
        elif x["etat"] == "jamais" and "jamais" not in t:
            fautes.append(f"{nom}/{x['nom']} : « jamais » absent")
    # le bloc « Températures »
    bt = texte(bloc(res, 1))
    sy = m["synthese"]
    for q in (sy["T_moy"], sy["T_min"], sy["T_max"]):
        if f"{fr(q)}{NB}°C" not in bt:
            fautes.append(f"{nom} : température {q} absente du bloc")
    if f"{fr(sy['t_T_28'])}{NB}j" not in bt:
        fautes.append(f"{nom} : t_T(28 j) = {sy['t_T_28']} absent")
    g = m["gel"]
    if g["t_5MPa"] is not None and jplus(g["t_5MPa"]) not in bt:
        fautes.append(f"{nom} : 5 MPa à {g['t_5MPa']} absent")
    if g["alerte"] != ("protéger" in bt):
        fautes.append(f"{nom} : l'alerte de gel ({g['alerte']}) n'est pas celle du bloc")
    # le bloc « Essai »
    x = m["essai"]
    be = texte(bloc(res, 2))
    if (x is None) != (be == ""):
        fautes.append(f"{nom} : bloc « Essai » présent sans essai, ou l'inverse")
    if x:
        for q in (x["f_cyl"], x["f_prevu"]):
            if f"{fr(q)}{NB}MPa" not in be:
                fautes.append(f"{nom} : essai {q} absent")
        if x["ecart"] is not None and f"{C.fr(abs(x['ecart']), 0)} %" not in be:
            fautes.append(f"{nom} : écart de l'essai {x['ecart']} absent")
        if x["cale"] and f"s = {C.nb(x['s_cale'], 3)}" not in be:
            fautes.append(f"{nom} : s calé absent")
    # les avertissements : chacun écrit
    tr = plat(res)
    for a in m["avertissements"]:
        if plat(C.symb_math(a["message"])) not in tr:
            fautes.append(f"{nom} : avertissement « {a['code']} » absent")
    # le tableau jour par jour
    tab = next(vv["svg"] for vv in rep["vues"] if vv["cle"] == "tableau")
    rows = re.findall(r"<tr[^>]*><th>(\d+)</th>(.*?)</tr>", tab, re.S)
    if len(rows) != len(m["tableau"]["lignes"]):
        fautes.append(f"{nom} : {len(rows)} lignes au tableau pour {len(m['tableau']['lignes'])}")
    for (j, h), li in zip(rows, m["tableau"]["lignes"]):
        cases = [texte(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", h)]
        att = ([_dt.date.fromisoformat(li["date"]).strftime("%d/%m/%Y")] if m["temps"]["date"] else [])
        att += [fr(li["T"]) if li["T"] is not None else "—", A.MOTS_SOURCE[li["source"]]]
        if m["tableau"]["horaire"]:
            att += [fr(li["T_min_h"]) if li["T_min_h"] is not None else "—", fr(li["T_max_h"]) if li["T_max_h"] is not None else "—"]
        att += [C.fr(li["k"], 3), C.fr(li["t_T"], 2), C.fr(li["beta"], 3), C.fr(li["fcm"], 1), C.fr(li["fck"], 1)]
        if int(j) != li["jour"] or cases != att:
            fautes.append(f"{nom} : jour {j} du tableau {cases} ≠ {att}")
            break
    # la courbe : chaque objectif atteint dans l'horizon porte son étiquette
    svg = next(vv["svg"] for vv in rep["vues"] if vv["cle"] == "courbe")
    for x in m["courbes"]["objectifs"]:
        if x["t"] is not None and x["t"] <= m["courbes"]["horizon"] and f"J+{C.fr(x['t'], 1)}" not in html.unescape(svg):
            fautes.append(f"{nom} : la courbe n'étiquette pas « {x['nom']} » à J+{x['t']}")
    return fautes


ECRANS = {}
for nom_cas, c in CAS.items():
    ECRANS[nom_cas] = calc({"version": 2, "coulages": [c], "actif": 0})
    f = controle_ecran(nom_cas, c, ECRANS[nom_cas])
    chk(f"écran = moteur — {nom_cas}", not f, f[:4])

# des valeurs attendues, en clair
m_d1 = M.calculer(CAS["défaut n° 1 : 7 j à 2 °C puis 7 j à 20 °C, ciment N, EC2, 15 MPa caractéristique"])
chk("le défaut n° 1 de l'ancien écran : atteint dans le 8ᵉ jour civil (J + 6,9 : le jour 7 commence à J + 6,5), plus « 4,8 j »",
    6.5 < m_d1["objectifs"][0]["t"] < 7.5 and "4,8" not in texte(ECRANS["défaut n° 1 : 7 j à 2 °C puis 7 j à 20 °C, ciment N, EC2, 15 MPa caractéristique"]["resultat"]),
    m_d1["objectifs"][0]["t"])
e_rel = ECRANS["relevé de Namur, trois objectifs (dont un modèle en MPa sur la moyenne)"]
chk("un relevé avec un jour à −2 °C le 5ᵉ jour : l'avertissement de gel (s'il précède 5 MPa) ou « hors gel », comme le moteur ; la ligne "
    "« sans donnée » du 04/11 est dite", "1 jour sans température" in texte(e_rel["resultat"]), texte(e_rel["resultat"])[-800:])
m_md = M.calculer(CAS["météo importée (Bruxelles, décoffrage à J + 3)"])
chk("météo importée, décoffrage à J + 3 : le verdict porte la date de l'opération « ven. 30/10 à 15 h » et le taux du moteur",
    instant(m_md["objectifs"][0]["instant_prevu"], 2026, "à") == f"ven.{NB}30/10{NB}à{NB}15{NB}h"
    and taux_txt(m_md["objectifs"][0]["taux"]) in texte(ECRANS["météo importée (Bruxelles, décoffrage à J + 3)"]["resultat"]))
r_an = ECRANS["essai cube calé, Arrhenius 45 kJ/mol, C25/30 lent"]
m_an = M.calculer(CAS["essai cube calé, Arrhenius 45 kJ/mol, C25/30 lent"])
chk("une échéance de l'année suivante (bétonnage le 28/12/2026) s'écrit avec son année (« …/2027 »)",
    m_an["objectifs"][1]["instant"][:4] == "2027" and "/2027" in texte(r_an["resultat"]), m_an["objectifs"][1]["instant"])
chk("la courbe calée sur l'essai : s calé du moteur, l'avis « Courbe CALÉE » et « (calé) » dans le bloc",
    m_an["materiau"]["cale"] and f"s = {C.nb(m_an['materiau']['s'], 3)} (calé)" in texte(r_an["resultat"]) and "Courbe CALÉE" in texte(r_an["resultat"]))

# ------------------------------------------------------------------ 3. les fenêtres de calcul, réévaluées
print("\n=== 3. Les fenêtres de calcul : chaque ligne réévaluée ===")
FX = re.compile(r'<div class="fx">(.*?)</div>', re.S)
SPAN = re.compile(r'<span(?: class="([^"]*)")?>(.*?)</span>', re.S)


def evaluer(vals):
    """« exp(0,4391 × (1 − √(28 / 3))) » → le nombre (exp, ln, √, max(a ; b), ², les parenthèses)."""
    s = html.unescape(re.sub(r"<[^>]+>", "", vals))
    s = s.replace(NB, "").replace(C.FINE, "").replace(" ", "").replace("−", "-")
    s = re.sub(r"(\d),(\d)", r"\1.\2", s).replace(";", ",")
    s = re.sub(r"√(\d+(?:\.\d+)?)", r"sqrt(\1)", s).replace("√(", "sqrt(")
    s = s.replace("×", "*").replace("²", "**2").replace("ln(", "log(")
    return eval(s, {"__builtins__": {}}, {"sqrt": math.sqrt, "exp": math.exp, "log": math.log, "max": max})  # noqa: S307


def lire_res(t):
    t = texte(t).replace(NB, " ").replace(C.FINE, "")
    mt = re.match(r"([-−]?\d+(?:,(\d+))?)", t)
    x = float(mt.group(1).replace("−", "-").replace(",", "."))
    return x, len(mt.group(2) or ""), "%" in t


def controle_formules(rep, nom):
    fautes, n, lues = [], 0, 0
    for ligne in FX.findall(rep["volets"]):
        spans = SPAN.findall(ligne)
        parts = [t for cls, t in spans if cls in ("s", "")]
        res = next((t for cls, t in spans if cls == "r" or cls.startswith("chip")), None)
        if res is None or len(parts) not in (2, 3):
            fautes.append(f"{nom} : ligne mal formée « {texte(ligne)[:60]} »")
            continue
        if len(parts) == 2:
            lues += 1
            continue
        n += 1
        try:
            val = evaluer(parts[2])
        except Exception as e:  # noqa: BLE001 — une expression illisible est une faute
            fautes.append(f"{nom} {texte(parts[0])} : illisible ({e}) : {texte(parts[2])}")
            continue
        if texte(res).startswith(">"):                      # au-delà du plafond de la pastille : « > 999 % »
            if not val * 100.0 > C.PLAFOND_TAUX:
                fautes.append(f"{nom} {texte(parts[0])} = {val} écrit « {texte(res)} »")
            continue
        att, nd, pct = lire_res(res)
        if pct and texte(parts[0]) == "η":
            val *= 100.0
        tol = 0.5 * 10 ** -nd + 2e-3 * abs(att) + (0.5 if pct else 0)
        if abs(val - att) > tol:
            fautes.append(f"{nom} {texte(parts[0])} = {texte(parts[2])} → {val} ≠ {att}")
    return fautes, n, lues


total = 0
for nom_cas, rep in ECRANS.items():
    f, n, lues = controle_formules(rep, nom_cas)
    total += n
    chk(f"fenêtres : chaque ligne réévaluée ({n}, et {lues} valeurs posées par la loi) — {nom_cas}", not f and n >= 8, f[:3])
cles_vues = {f["cle"] for nom_cas, c in CAS.items() for x in [M.calculer(c)]
             for f in x["materiau"]["formules"] + x["gel"]["formules"] + [x["temperature"]["formule_k"]]
             + [g for o in x["objectifs"] for g in o["formules"]] + (x["essai"]["formules"] if x["essai"] else [])}
chk(f"chaque formule du moteur a son gabarit ou son texte (moteur.EXPRESSIONS, moteur.LUES) — {len(cles_vues)} sortes vues ici, {total} lignes réévaluées",
    all(k in M.EXPRESSIONS or k in M.LUES for k in cles_vues) and len(cles_vues) >= 18 and total > 120, sorted(cles_vues - set(M.EXPRESSIONS) - set(M.LUES)))
ids = set(re.findall(r'id="ck-([^"]+)"', ECRANS["essai cube calé, Arrhenius 45 kJ/mol, C25/30 lent"]["volets"]))
annonces = {x for nom_cas, rep in ECRANS.items() for x in re.findall(r'data-calcul="([^"]+)"', rep["resultat"])}
chk("chaque ligne qui annonce un calcul a sa fenêtre (o0, o1, s0, s1, gel, s2)",
    {"o0", "o1", "s0", "s1", "gel", "s2"} <= ids and all(any(f'id="ck-{a}"' in rep["volets"] for rep in ECRANS.values()) for a in annonces), (ids, annonces))

# ------------------------------------------------------------------ 4. les saisies
print("\n=== 4. Les saisies : chaque champ relié, les refus, les effets de bord ===")
e0 = r0["etat"]
RELIES = [("classe", "C35/45", lambda c: c["classe"] == "C35/45"),
          ("durcissement/choix", "beton:rapide", lambda c: c["durcissement"]["mode"] == "beton" and c["durcissement"]["beton"] == "rapide"),
          ("durcissement/choix", "ciment:S", lambda c: c["durcissement"]["mode"] == "ciment" and c["durcissement"]["ciment"] == "S"),
          ("durcissement/choix", "r", lambda c: c["durcissement"]["mode"] == "r"),
          ("date", "02/11/2026", lambda c: c["date"] == "2026-11-02"), ("date", "", lambda c: c["date"] is None),
          ("heure", "7", lambda c: c["heure"] == 7), ("temperature/T", "-5", lambda c: c["temperature"]["T"] == -5.0),
          ("temperature/mode", "releve", lambda c: c["temperature"]["mode"] == "releve"), ("marge", "4,5", lambda c: c["marge"] == 4.5),
          ("loi/mode", "perso", lambda c: c["loi"]["mode"] == "perso"), ("loi/Ea_kJ", "45", lambda c: c["loi"]["Ea_kJ"] == 45.0),
          ("essai/actif", True, lambda c: c["essai"]["actif"] is True), ("essai/age", "3,5", lambda c: c["essai"]["age"] == 3.5),
          ("essai/mpa", "21,6", lambda c: c["essai"]["mpa"] == 21.6), ("essai/eprouvette", "cylindre", lambda c: c["essai"]["eprouvette"] == "cylindre"),
          ("essai/caler", True, lambda c: c["essai"]["caler"] is True), ("nom", "Voile R+1", lambda c: c["nom"] == "Voile R+1"),
          ("objectifs/0/nom", "Désétaiement", lambda c: c["objectifs"][0]["nom"] == "Désétaiement"),
          ("objectifs/0/pct", "50", lambda c: c["objectifs"][0]["pct"] == 50.0), ("objectifs/0/age", "3", lambda c: c["objectifs"][0]["age"] == 3.0),
          ("objectifs/0/mode", "mpa", lambda c: c["objectifs"][0]["mode"] == "mpa" and c["objectifs"][0]["mpa"] == 21.0),
          ("objectifs/+", "nouveau", lambda c: [o["nom"] for o in c["objectifs"]] == ["Décoffrage", "Objectif 2"] and c["objectifs"][1]["pct"] == 70.0),
          ("objectifs/+", "reprise", lambda c: c["objectifs"][1]["mpa"] == 10.0 and c["objectifs"][1]["base"] == "fcm"),
          ("durcissement/r", "0,4", lambda c: c["durcissement"]["r"] == 0.4)]
mal = []
for ch, v, cond in RELIES:
    out = saisir(e0, ch, v)
    if out["message"] or not cond(out["etat"]["coulages"][0]):
        mal.append((ch, v, out["message"]))
chk(f"{len(RELIES)} saisies reliées à l'état (nombres à la française, segments, cases, menus, objectifs, modèles) sans message", not mal, mal)
e2 = saisir(saisir(e0, "objectifs/+", "hors_gel")["etat"], "objectifs/0/age", "3")["etat"]
chk("un objectif en MPa est fixe, un objectif en % suit la classe (C30/37 → C40/50 : 21 → 28 MPa)",
    M.calculer(saisir(e2, "classe", "C40/50")["etat"]["coulages"][0])["objectifs"][0]["cible"] == 28.0
    and M.calculer(saisir(e2, "classe", "C40/50")["etat"]["coulages"][0])["objectifs"][1]["cible"] == 5.0)
e_mpa = saisir(saisir(e0, "objectifs/0/mode", "mpa")["etat"], "objectifs/0/mpa", "18")["etat"]
chk("de la cible en MPa à la cible en % : 18 MPa du C30/37 → 60 % ; en %, la base est forcée à la caractéristique",
    saisir(e_mpa, "objectifs/0/mode", "pct")["etat"]["coulages"][0]["objectifs"][0]["pct"] == 60.0
    and saisir(saisir(e_mpa, "objectifs/0/base", "fcm")["etat"], "objectifs/0/mode", "pct")["etat"]["coulages"][0]["objectifs"][0]["base"] == "fck")
ret = saisir(e2, "objectifs/0/-", "1")
chk("retirer un objectif : le second devient le premier (le verdict le suit)", [o["nom"] for o in ret["etat"]["coulages"][0]["objectifs"]] == ["Hors gel"]
    and "Hors gel · 5 MPa" in texte(ret["resultat"]))
REFUS = [("temperature/T", "-25", "Température : −25 est sous le minimum −20 (°C)"), ("marge", "11", "Marge sur les prévisions"),
         ("loi/Ea_kJ", "70", "Énergie d'activation"), ("durcissement/r", "0,1", "Rapport r"), ("objectifs/0/pct", "4", "Objectif 1, cible"),
         ("objectifs/0/age", "400", "Objectif 1, âge prévu"), ("objectifs/0/age", "trois", "Objectif 1, âge prévu"),
         ("essai/mpa", "200", "Résistance mesurée"), ("essai/age", "0", "Âge de l'essai"), ("heure", "25", "Heure de fin du bétonnage"),
         ("heure", "7,5", "heure entière"), ("date", "32/13/2026", "date illisible"), ("temperature/T", "douze", "nombre illisible"),
         ("nom", "x" * 81, "Nom trop long"), ("objectifs/0/nom", "y" * 41, "Objectif 1, nom"), ("objectifs/0/-", "1", "au moins un objectif"),
         ("objectifs/+", "inconnu", "modèle d'objectif inconnu"), ("objectifs/7/nom", "x", "objectif inconnu"), ("classe", "C99/99", "Classe"),
         ("durcissement/choix", "beton:tres_lent", "Durcissement"), ("inconnu", "1", "champ inconnu"), ("essai/actif", "peut-être", "case à cocher")]
mal = []
for ch, v, mot in REFUS:
    out = req(module="age", action="modifier", etat=copy.deepcopy(e0), chemin=ch.split("/"), valeur=v)
    if not out.get("ok") or not out.get("message") or mot.lower() not in out["message"].lower() or out["etat"] != e0:
        mal.append((ch, v, out.get("message") or out.get("erreur")))
chk(f"{len(REFUS)} saisies refusées : un message qui nomme le champ, l'état intact, jamais une exception", not mal, mal)
e8 = e0
for k in range(7):
    e8 = saisir(e8, "objectifs/+", "nouveau")["etat"]
o9 = req(module="age", action="modifier", etat=copy.deepcopy(e8), chemin=["objectifs", "+"], valeur="nouveau")
chk("huit objectifs au plus : le neuvième est refusé, et la rubrique ne propose plus d'en ajouter",
    len(e8["coulages"][0]["objectifs"]) == 8 and "au plus 8" in (o9["message"] or "") and o9["etat"] == e8 and 'data-chemin="objectifs/+"' not in o9["donnees"])
r_rel = saisir(saisir(e0, "temperature/mode", "releve")["etat"], "temperature/releve", "12\nabc\n13")
chk("un relevé illisible reste à l'écran avec son erreur et sa ligne ; le calcul garde le précédent",
    "Relevé refusé" in r_rel["message"] and "abc" in r_rel["donnees"] and r_rel["etat"]["coulages"][0]["temperature"]["releve"] == ""
    and r_rel["etat"]["coulages"][0][A.BROUILLON]["texte"] == "12\nabc\n13")
r_f = saisir(e0, "temperature/fichier", {"nom": "releve.csv", "texte": RELEVE})
chk("un fichier lu devient le relevé (mode « Relevé », son nom écrit)", r_f["message"] is None
    and r_f["etat"]["coulages"][0]["temperature"]["mode"] == "releve" and "releve.csv" in r_f["donnees"])
chk("le champ r n'apparaît qu'en « r connu » ; E_a qu'en « Eₐ saisie » ; l'essai qu'une fois coché",
    'data-chemin="durcissement/r"' not in r0["donnees"] and 'data-chemin="durcissement/r"' in saisir(e0, "durcissement/choix", "r")["donnees"]
    and 'data-chemin="loi/Ea_kJ"' not in r0["donnees"] and 'data-chemin="loi/Ea_kJ"' in saisir(e0, "loi/mode", "perso")["donnees"]
    and 'data-chemin="essai/mpa"' not in r0["donnees"] and 'data-chemin="essai/mpa"' in saisir(e0, "essai/actif", True)["donnees"])
chk("le menu « Durcissement » a deux groupes : le bon de livraison (rapide, moyen, lent, r connu) puis le ciment (R, N, S)",
    re.findall(r'<optgroup label="([^"]+)">', r0["donnees"]) == ["Bon de livraison · NBN EN 206", "Ciment · EN 1992-1-1"]
    and re.findall(r'<option value="(beton:\w+|r|ciment:\w)"', r0["donnees"]) == ["beton:rapide", "beton:moyen", "beton:lent", "r", "ciment:R", "ciment:N", "ciment:S"])
chk("les six modèles d'objectif, chacun avec sa source au survol",
    all(f'data-valeur="{k}" title="{C.esc(src)}"' in r0["donnees"] for k, *_x, src in M.MODELES_OBJECTIF))
# L'aide de la loi k(T) ne dit que ce qui est VRAI (relecture du 11/11/2026) : la loi belge est plus prudente que (B.10)
# par temps froid, PAS « la plus prudente » — sous 0 °C elle compte encore un peu là où (B.10) compte 0, et une E_a haute
# (ciments au laitier) donne moins qu'elle. Recalculé ici : à 5 °C, 0,45 (belge) contre 0,479 (B.10) et 0,369 (45 kJ/mol).
aide_loi = html.unescape(A._aide_loi())
k5 = {"nbn": L.k_nbn(5.0), "ec2": L.k_ec2(5.0), "e45": L.k_arrhenius(5.0, 45.0)}
chk("l'aide de la loi k(T) est exacte : « plus prudente que (B.10) » (0,45 < 0,479 à 5 °C), jamais « la plus prudente » (0,369 pour 45 kJ/mol) ; "
    "elle dit que la loi belge compte encore sous 0 °C (0,3 à 0 °C), où le tableau de la norme s'arrête (−5 °C) et jusqu'où le module le "
    "prolonge (−10 °C), et qu'une E_a haute rend le tableau belge optimiste — ses nombres sont ceux des tables du moteur",
    abs(k5["nbn"] - 0.45) < 1e-12 and abs(k5["ec2"] - 0.4787) < 5e-5 and abs(k5["e45"] - 0.3691) < 5e-5 and k5["e45"] < k5["nbn"] < k5["ec2"]
    and L.k_nbn(-3.0) > 0.0 == L.k_ec2(-3.0)
    and "plus prudente que (B.10) par temps froid" in aide_loi and "la plus prudente" not in aide_loi
    and abs(L.k_nbn(0.0) - 0.30) < 1e-12 and L.K_NBN_NORME[0][0] == -5.0 and L.T_NUL_NBN == -10.0 and (L.EA_MIN, L.EA_MAX) == (20.0, 60.0)
    and "(0,3 à 0 °C ; le tableau de la norme s'arrête à −5 °C, le module le prolonge jusqu'à 0 à −10 °C)" in aide_loi
    and "le tableau belge peut alors être optimiste" in aide_loi and "(20 à 60 kJ/mol)" in aide_loi, aide_loi[-520:])

# La relecture des captures du 11/11/2026 — six points de finition, chacun figé ici.
e_rel = note(coulage("Voile", temperature__mode="releve", temperature__releve=RELEVE, objectifs=[objectif("Décoffrage"), objectif("Daté", age=6.0)]))
r_rel, r_dat = calc(e_rel), calc(note(coulage("Voile", date="2026-11-02", temperature__mode="releve", temperature__releve=RELEVE)))
m_rel = M.calculer(e_rel["coulages"][0])
def titres(rep):
    """Les cinq titres du tableau de chaque fenêtre de calcul : {identifiant: [titres]} (texte, sans balises)."""
    out = {}
    for bloc in re.split(r'(?=<[^>]*\bid="ck-)', rep["volets"]):
        ident, tete = re.search(r'\bid="ck-([^"]+)"', bloc), re.search(r"<thead><tr>(.*?)</tr></thead>", bloc, re.S)
        if ident and tete:
            out[ident.group(1)] = [texte(t) for t in re.findall(r"<th[^>]*>(.*?)</th>", tete.group(1), re.S)]
    return out


T_OBJ = ["Objectif", "Comparée à", "Âge équivalent requis", "Atteint", "Résultat"]
t0, t_rel = titres(r0), titres(r_rel)
chk("fenêtres : aucune colonne « Taux » sur ce qui n'en est pas un — les objectifs finissent par « Résultat » (l'ÉCHÉANCE « J + … » d'un objectif "
    "non daté, le TAUX d'un objectif daté), le gel par « Verdict » (« hors gel », « protéger ») entre « 5 MPa atteints » et « Premier jour de gel »",
    set(t0) == {"o0", "s0", "s1", "gel"} and set(t_rel) == {"o0", "o1", "s0", "s1", "gel"}
    and all(t[k] == T_OBJ for t in (t0, t_rel) for k in t if k == "s0" or k.startswith("o"))
    and all(t["gel"] == ["Vérification", "Condition", "5 MPa atteints", "Premier jour de gel", "Verdict"] for t in (t0, t_rel))
    and not any("Taux" in x or "Valeur" in x or "Limite" in x for t in (t0, t_rel) for v in t.values() for x in v)
    and m_rel["objectifs"][0]["age_prevu"] is None and m_rel["objectifs"][1]["taux"] is not None
    and f">J + {C.fr(m_rel['objectifs'][0]['t'], 1)}<" in r_rel["volets"] and taux_txt(m_rel["objectifs"][1]["taux"]) in texte(r_rel["volets"]),
    (t0, t_rel))
chk("fenêtres : la variante prudente s'écrit en indice — T_prud, k_prud, t_prud —, plus aucun « k,p » ni « t,p » ; t_p reste l'âge prévu",
    all(f"{x}<sub>prud</sub>" in r0["volets"] for x in "Tkt") and not re.search(r"\b[kt],p\b", texte(r0["volets"]) + texte(r_rel["volets"]))
    and "(t<sub>p</sub>)" in r_rel["volets"] and "<sub>prud</sub>" in r_rel["volets"])
chk("vues : la légende de chaque vue est du TEXTE lisible — aucun symbole écrit avec son tiret bas (« f_cm(t) », « t_T ») —, et l'en-tête non plus",
    all(not re.search(r"[A-Za-zα-ω]_[A-Za-z]", v["legende"] + " " + texte(v["entete"])) for rep in (r0, r_rel, r_dat) for v in rep["vues"])
    and "valeurs indicatives (grisé)" in r0["vues"][0]["legende"] and f"avant {C.nb(L.SEUIL_INDICATIF_J, 0)} j équivalents" in r0["vues"][0]["legende"])
tab_rel, tab_0 = r_rel["vues"][2]["svg"], r0["vues"][2]["svg"]
j_gel = [li["jour"] for li in m_rel["tableau"]["lignes"] if li["gel"]]
chk("tableau : sa légende dit ce que l'œil voit — la LIGNE surlignée est le jour où le premier objectif est atteint, le NUMÉRO en bleu un jour de "
    "gel (le 06/11 à −2,0 °C : jour 4), et elle ne parle du gel que s'il y en a",
    j_gel == [4] and re.search(r'<tr class="[^"]*\bgel\b[^"]*"><th>4</th>', tab_rel) and len(re.findall(r'<tr class="[^"]*\bgel\b', tab_rel)) == 1
    and "ligne surlignée : le jour où le premier objectif est atteint" in tab_rel and "numéro du jour en bleu : jour de gel" in tab_rel
    and "moyenne du jour à 0 °C ou moins" in tab_rel and "en bleu, le jour où" not in tab_rel + tab_0
    and "ligne surlignée" in tab_0 and "jour de gel" not in tab_0 and len(re.findall(r'<tr class="[^"]*\bon\b', tab_rel)) == 1, j_gel)
tp_rel = m_rel["temperature"]
chk("résultats : « Jours » compte les jours par source sans phrase coupée, et ce que le calcul suppose APRÈS le dernier jour connu a sa ligne — "
    "« Jours suivants · 6,6 °C · la moyenne des 5 derniers jours connus · k = 0,50 » (le moteur) ; rien de tel à température constante",
    abs(tp_rel["T_fin"] - (4.0 + 6.0 + 7.5 + 8.5 + 7.0) / 5.0) < 1e-12 and abs(tp_rel["k_fin"] - (0.45 + 0.15 * 1.6 / 5.0)) < 1e-12
    and f"5{NB}j mesurés · 3{NB}j de prévision · 1{NB}j prolongé · 1{NB}j sans donnée" in texte(r_rel["resultat"])
    and re.search(r"Jours suivants.{0,400}?" + re.escape(f"6,6{NB}°C") + r".{0,400}?" + re.escape(f"la moyenne des 5 derniers jours connus · k{NB}={NB}0,50"),
                  texte(r_rel["resultat"]), re.S)
    and "au-delà :" not in texte(r_rel["resultat"]) and "Jours suivants" not in texte(r0["resultat"]) and "température constante" in texte(r0["resultat"]),
    texte(r_rel["resultat"])[:400])
champ = r'<input data-chemin="date" data-sorte="texte" value="{}" placeholder="{}">'
chk("données : une date de fin du bétonnage LUE SUR LE RELEVÉ se voit dans le champ vide (en grisé) et se dit dessous ; saisie, ou sans relevé daté, "
    "le champ reprend son gabarit « JJ/MM/AAAA » et la phrase disparaît",
    m_rel["temps"]["date_origine"] == "releve" and m_rel["temps"]["date"] == "2026-11-02"
    and champ.format("", "02/11/2026") in r_rel["donnees"] and "Date lue sur la première ligne du relevé (02/11/2026)" in r_rel["donnees"]
    and champ.format("02/11/2026", "JJ/MM/AAAA") in r_dat["donnees"] and "Date lue sur" not in r_dat["donnees"]
    and champ.format("", "JJ/MM/AAAA") in r0["donnees"] and "Date lue sur" not in r0["donnees"])

# ------------------------------------------------------------------ 5. le sélecteur de classe
print("\n=== 5. Le sélecteur de classe ===")
sel0 = req(module="age", action="selecteur", etat=e0, role="classe")["html"]
ca0 = M.candidats_classes(e0["coulages"][0])
lignes0 = re.findall(r'<button class="pk-row([^"]*)"[^>]*data-choix="([^"]+)"[^>]*>.*?<span class="kg">([^<]*)</span>', sel0, re.S)
chk("sans âge prévu : les huit classes, l'âge où le premier objectif est atteint (le moteur), ✓ la classe du coulage, aucune étoile",
    [x[1] for x in lignes0] == list(L.CLASSES) and [x[2] for x in lignes0] == [C.fr(c["t"], 1) if c["t"] is not None else "—" for c in ca0]
    and [" cur" in x[0] for x in lignes0] == [c["classe"] == "C30/37" for c in ca0] and " opt" not in "".join(x[0] for x in lignes0), lignes0)
e_d = saisir(saisir(saisir(e0, "objectifs/0/mode", "mpa")["etat"], "objectifs/0/mpa", "15")["etat"], "objectifs/0/age", "7")["etat"]
sel1 = req(module="age", action="selecteur", etat=e_d, role="classe")["html"]
ca1 = M.candidats_classes(e_d["coulages"][0])
opt = [c["classe"] for c in ca1 if c["optimal"]]
chk("avec un âge prévu (15 MPa caractéristiques à J + 7) : ★ la plus petite classe qui tient (le moteur), les taux en pastille",
    len(opt) == 1 and re.search(rf'class="pk-row[^"]* opt"[^>]*data-choix="{re.escape(opt[0])}"', sel1) and "chip" in sel1, opt)
ch1 = req(module="age", action="choisir", etat=copy.deepcopy(e_d), role="classe", nom=opt[0])
chk("un clic retient la classe ; l'objectif en % la suit", ch1["ok"] and ch1["etat"]["coulages"][0]["classe"] == opt[0], ch1.get("erreur"))
chk("la ligne du premier objectif ouvre le sélecteur (data-pk « classe »)", 'data-pk="classe"' in bloc(r0["resultat"], 0))

# ------------------------------------------------------------------ 6. la note d'avant, l'échappement
print("\n=== 6. La note d'avant, plusieurs coulages, l'échappement ===")
v1 = calc(A.NOTE_V1)
chk("une note d'avant (version 1) : les noms des coulages gardés, les coulages neufs, l'avis dit une fois (puis retiré de l'état)",
    [c["nom"] for c in v1["etat"]["coulages"]] == ["Voile R+1", "Dalle R+2"] and v1["etat"]["coulages"][0]["classe"] == "C30/37"
    and v1["etat"]["coulages"][1]["classe"] == "C30/37" and "avis" in v1 and "refait sur la norme" in v1["avis"][0] and A.CLE_AVIS not in v1["etat"]
    and "avis" not in calc(v1["etat"]), v1.get("avis"))
chk("une note d'une version plus récente est refusée avec un message lisible",
    not req(module="age", action="calculer", etat={"version": 3, "coulages": [{}]})["ok"])
deux = req(module="age", action="ajouter", etat=e0)
chk("un coulage ajouté (« Coulage 2 ») ; la ligne de l'élément le dit ; le dernier ne se supprime pas",
    [c["nom"] for c in deux["etat"]["coulages"]] == ["Coulage 1", "Coulage 2"] and 'data-elements="2"' in deux["donnees"]
    and len(req(module="age", action="supprimer", etat=e0, index=0)["etat"]["coulages"]) == 1)
PIEGE = '<img src=x onerror=alert(1)>'
p1 = saisir(saisir(saisir(e0, "nom", PIEGE)["etat"], "objectifs/0/nom", PIEGE[:40])["etat"], "temperature/mode", "releve")
p1 = saisir(p1["etat"], "temperature/lieu", PIEGE)
frag = {k: p1[k] for k in ("resultat", "volets", "donnees", "elements", "phrase")}
frag["vues"] = "".join(v["svg"] + v["entete"] for v in p1["vues"])
fuites = [k for k, h in frag.items() if "<img" in h]
chk("un nom de coulage, un nom d'objectif et un lieu piégés sont échappés dans tous les fragments (et sur les dessins)", not fuites, fuites)
chk("la réponse est du JSON strict (ni NaN ni infini) pour chaque cas", all(json.loads(json.dumps(rep, allow_nan=False)) for rep in ECRANS.values()))

# ------------------------------------------------------------------ 7. pureté
print("\n=== 7. Pureté : sans numpy, streamlit, matplotlib, pandas, requests ===")
SONDE = r"""
import importlib.abc, json, sys
INTERDITS = ("numpy", "streamlit", "matplotlib", "pandas", "requests", "scipy")
class Refus(importlib.abc.MetaPathFinder):
    def find_spec(self, nom, chemin, cible=None):
        if nom.split(".")[0] in INTERDITS:
            raise ImportError("interdit : " + nom)
sys.meta_path.insert(0, Refus())
sys.path.insert(0, sys.argv[1])
from api_web import routeur as R
r = json.loads(R.traiter(json.dumps({"module": "age", "action": "nouveau"})))
print(json.dumps({"ok": r["ok"], "interdits": sorted(m for m in sys.modules if m.split(".")[0] in INTERDITS)}))
"""
p = subprocess.run([sys.executable, "-c", SONDE, RACINE], capture_output=True, text=True, timeout=120)
try:
    pur = json.loads(p.stdout.strip().splitlines()[-1])
except (ValueError, IndexError):
    pur = {"ok": False, "sortie": p.stdout[-300:] + p.stderr[-300:]}
chk("le module du site et le moteur se chargent et calculent quand numpy, streamlit, matplotlib, pandas, requests et scipy sont REFUSÉS à l'import",
    pur.get("ok") and not pur.get("interdits"), pur)

# ------------------------------------------------------------------ 8. la météo du lieu
print("\n=== 8. La météo du lieu (Open-Meteo, importée à la demande ; faux service au format documenté) ===")
BOUTON = re.compile(r'<button type="button" class="btn meteo( sm)?" data-meteo="([^"]*)" data-meteo-chemin="([^"]+)"( disabled)?>')


def boutons(r):
    return [(ch, json.loads(html.unescape(pl)), bool(gr), bool(sm)) for sm, pl, ch, gr in BOUTON.findall(r["donnees"])]


def navigateur(r, chemin="temperature/meteo", service=None):
    b = next(x for x in boutons(r) if x[0] == chemin)
    return FO.recuperer(b[1], service or FO.FauxService(AUJ))


e_m = saisir(e0, "temperature/mode", "meteo")
c_m = e_m["etat"]["coulages"][0]
bt = boutons(e_m)
chk("« Météo du lieu » : le lieu par défaut, le bouton porte le PLAN du moteur (géocodage, archive puis prévisions, coulage n° 0)",
    c_m["temperature"]["lieu"] == MT.LIEU_DEFAUT and len(bt) == 1 and not bt[0][2]
    and bt[0][1] == dict(MT.plan(MT.LIEU_DEFAUT, AUJ - _dt.timedelta(days=14), AUJ, AUJ), coulage=0)
    and "Météo pas encore récupérée" in e_m["resultat"], bt and bt[0][1])
chk("la date de fin du bétonnage est le début par défaut de la météo",
    boutons(saisir(e_m["etat"], "date", "02/11/2026"))[0][1]["debut"] == "2026-11-02")
v = navigateur(e_m)
o = saisir(e_m["etat"], "temperature/meteo", v)
c_o = o["etat"]["coulages"][0]
p_att = MT.plan(MT.LIEU_DEFAUT, "2026-10-24", "2026-11-07", AUJ)
chk("la récupération est lue par le moteur et GARDÉE : le relevé (15 jours), le lieu trouvé, la date de récupération",
    o["message"] is None and c_o["temperature"]["releve"] == MT.releve(MT.lire_journalier(p_att, v["reponses"]))
    and c_o["temperature"]["meteo"]["recupere"] == "2026-11-07T10:30" and c_o["temperature"]["mode"] == "meteo", o["message"])
t_o = texte(o["donnees"])
chk("l'écran dit d'où viennent les températures : le badge, la date de récupération et la position, T min / T moy / T max",
    "Bruxelles, Bruxelles-Capitale, Belgique — 15 j mesurés" in t_o and "Open-Meteo · récupérée le 07/11/2026 à 10:30 · 50,85° N · 4,35° E" in t_o
    and all(x in t_o for x in ("T min", "T moy", "T max")) and "Récupérer à nouveau la météo" in t_o, t_o[:300])
chk("la vue « Météo du lieu » dit Open-Meteo ; le tableau a une ligne par jour du profil, les 15 premières « mesuré »",
    o["vues"][1]["titre"] == "Météo du lieu" and "Open-Meteo" in o["vues"][1]["legende"]
    and o["vues"][2]["svg"].count(">mesuré<") == 15, [x["titre"] for x in o["vues"]])
chk("rechargée (.stab), la note se recalcule HORS LIGNE à l'identique", calc(json.loads(json.dumps(o["etat"])))["resultat"] == o["resultat"])
o3 = saisir(o["etat"], "temperature/releve", c_o["temperature"]["releve"].replace(";", ";1", 1))
chk("une température corrigée à la main : le calcul suit, la récupération reste, « corrigées à la main »",
    o3["message"] is None and o3["etat"]["coulages"][0]["temperature"]["meteo"] == c_o["temperature"]["meteo"] and "corrigées à la main" in o3["donnees"])
bh = [b for b in boutons(o) if b[0] == "temperature/meteo_horaire"]
oh = saisir(o["etat"], "temperature/meteo_horaire", navigateur(o, "temperature/meteo_horaire"))
tab_h = next(x["svg"] for x in oh["vues"] if x["cle"] == "tableau")
mh = M.calculer(oh["etat"]["coulages"][0])
chk("le détail horaire (petit bouton, la position sans géocodage) : gardé, T min h / T max h au tableau, la MINIMALE horaire sert au gel",
    len(bh) == 1 and bh[0][3] and "geocodage" not in bh[0][1] and oh["message"] is None and "T min h" in tab_h and mh["gel"]["base"] == "minimale"
    and "minimale horaire" in texte(oh["resultat"]), oh["message"])
chk("écran = moteur avec le détail horaire", not controle_ecran("détail horaire", oh["etat"]["coulages"][0], oh))
deux = req(module="age", action="dupliquer", etat=o3["etat"])
v_1 = navigateur(req(module="age", action="activer", etat=deux["etat"], index=0))
o_x = req(module="age", action="modifier", etat=copy.deepcopy(deux["etat"]), chemin=["temperature", "meteo"], valeur=v_1)
chk("une récupération revenue alors qu'un autre coulage est ouvert est refusée ; la copie garde ses corrections",
    o_x["ok"] and "Un autre coulage a été ouvert" in (o_x["message"] or "") and o_x["etat"] == deux["etat"], o_x.get("message"))
REFUS_M = [("lieu introuvable", lambda: (saisir(e_m["etat"], "temperature/lieu", "Atlantide")["etat"],), "Lieu introuvable : « Atlantide »"),
           ("une réponse de moins", lambda: (e_m["etat"], dict(v, reponses=v["reponses"][:1])), "Météo refusée"),
           ("le plan d'un autre lieu", lambda: (e_m["etat"], dict(v, plan=dict(v["plan"], lieu="Namur"))), "modifiés pendant la récupération"),
           ("sans plan", lambda: (e_m["etat"], {"reponses": []}), "Météo illisible"),
           ("le détail horaire avant la météo", lambda: (e_m["etat"], dict(v, plan=bh[0][1])), "Récupérez d'abord la météo")]
mal = []
for nom, fab, mot in REFUS_M:
    x = fab()
    base = x[0]
    val = x[1] if len(x) > 1 else FO.recuperer(boutons(calc(base))[0][1], FO.FauxService(AUJ))
    ch = ["temperature", "meteo_horaire" if "horaire" in nom else "meteo"]
    out = req(module="age", action="modifier", etat=copy.deepcopy(base), chemin=ch, valeur=val)
    if not out.get("ok") or mot not in (out.get("message") or "") or out["etat"] != base:
        mal.append((nom, out.get("message") or out.get("erreur")))
chk(f"{len(REFUS_M)} récupérations refusées : un message en français, la note intacte, jamais une exception", not mal, mal)
b6 = boutons(saisir(saisir(e_m["etat"], "temperature/debut", "01/12/2026")["etat"], "temperature/fin", "05/12/2026"))
chk("dates hors de portée : le bouton grisé, le message (horizon des prévisions)", b6 and b6[0][2] and "horizon des prévisions" in b6[0][1].get("erreur", ""), b6)
journal = [req(module="age", **dict(q, etat=copy.deepcopy(q.get("etat") or e0))) for q in A.PARCOURS]
chk(f"le parcours propre du module ({len(journal)} requêtes) : aucune erreur", all(x["ok"] for x in journal), [x.get("erreur") for x in journal if not x["ok"]])

# ------------------------------------------------------------------ 9. les mutations de l'écran
print("\n=== 9. Mutations de l'écran : une mise en forme fautive fait rougir les contrôles ===")


def rejoue():
    fautes = []
    for nom_cas, c in CAS.items():
        fautes += controle_ecran(nom_cas, c)
        fautes += controle_formules(calc({"version": 2, "coulages": [c], "actif": 0}), nom_cas)[0]
    return fautes


MUTATIONS = [
    ("l'échéance tronquée au jour", "_jplus", lambda f: (lambda t: f(None if t is None else math.floor(t)))),
    ("nominal et prudent confondus", "_prudent_txt", lambda f: (lambda r, o: f(r, dict(o, prudent={"t": o["t"], "instant": o["instant"]})))),
    ("l'instant arrondi à l'heure inférieure", "_instant", lambda f: (lambda iso, mot="vers": f(None if not iso else
                                                                       (_dt.datetime.fromisoformat(iso) - _dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M"), mot))),
    ("le taux recopié au lieu d'arrondi dans sa bande", "_taux", lambda f: (lambda o: None if o["taux"] is None else round(o["taux"] * 100 + 0.5))),
    ("une valeur substituée arrondie à deux chiffres", "_v", lambda f: (lambda x: f(float(f"{x:.2g}")))),
    ("la résistance prévue lue sur la mauvaise base", "_mpa", lambda f: (lambda x, nd=1: f(None if x is None else x + 8.0, nd))),
    ("les températures en °F", "_deg", lambda f: (lambda x, nd=1: f(None if x is None else x * 1.8 + 32, nd))),
]
for nom, cible, fab in MUTATIONS:
    orig = getattr(A, cible)
    setattr(A, cible, fab(orig))
    try:
        rouges = rejoue()
    except Exception as e:  # noqa: BLE001 — une exception compte comme un rouge
        rouges = [f"exception : {e}"]
    finally:
        setattr(A, cible, orig)
    chk(f"mutation « {nom} » : les contrôles rougissent ({len(rouges)} fautes)", bool(rouges), "aucune faute vue")
chk("sans mutation, les contrôles restent verts", not rejoue())

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
