# -*- coding: utf-8 -*-
"""Maturité du béton — le MOTEUR : à quel âge, et quel jour, un béton coulé par ce temps atteint-il une résistance ?

Refait le 10/11/2026 sur la norme (``docs/site/LOT_Q_QUATRE_MODULES.md`` § 2 ; méthode, sources et preuves :
``docs/beton/MATURITE_MOTEUR.md``) : il ne reproduit plus l'écran Streamlit ``modules/age_beton.py``. Les lois et les
tables sont dans ``lois.py`` (une source par valeur de norme), l'entrée et ses bornes dans ``entree.py``, la lecture du
relevé et de la météo dans ``releve.py`` et ``meteo.py``. Bibliothèque standard seule.

API
---
    nouvel_element(nom) -> dict            le coulage neuf
    normaliser(coulage) -> dict            copie complétée et contrôlée (ValueError hors domaine, le champ nommé)
    calculer(coulage) -> dict              tout ce que l'écran montre
    candidats_classes(coulage) -> list     chaque classe de béton face au PREMIER objectif (le sélecteur)
    MODELES_OBJECTIF                       les six modèles d'objectif, chacun avec sa source

LE TEMPS : t = 0 à la fin du bétonnage (date et heure). Le jour 0 est le jour du bétonnage ; il ne compte que pour sa
fraction restante. Le relevé commence ce jour-là (sa première ligne) ; s'il est daté et que la date du bétonnage l'est
aussi, c'est le CALENDRIER qui les accorde : les jours du relevé d'avant le bétonnage ne comptent pas, un jour sans
température (absent du relevé, ou « — ») ne compte AUCUN durcissement, les jours d'après le relevé sont PROLONGÉS à la
moyenne de ses cinq derniers jours. Sans date de bétonnage, celle de la première ligne d'un relevé daté la donne.

SORTIE de ``calculer`` — JSON strict (ni NaN ni infini), nombres non arrondis (l'écran formate)
------
    {"entree": <le coulage normalisé>,
     "materiau": {"classe", "fck", "fck_cube", "fcm", "s" (retenu), "r" (qu'il suppose), "cale" (s vient de l'essai),
                  "durcissement": {"mode", "cle", "s", "r", "libelle", "source", "detail"}   (le choix, avant calage),
                  "loi": {"mode", "libelle", "source", "Ea_kJ" | None}, "formules": [...]},
     "temps": {"date", "date_origine": "saisie" | "releve" | None, "heure", "f0", "origine" (AAAA-MM-JJTHH:MM | None)},
     "temperature": {"mode", "source": "constante" | "releve", "T" (la constante lue), "lieu", "fichier", "marge",
                     "releve": <releve.synthese> | None, "meteo": <meteo.synthese> | None,
                     "T_fin" (au-delà des jours connus), "source_fin": "constante" | "prolonge", "k_fin"},
     "profil": [<jour>, …]      un par jour civil, du jour 0 à l'horizon :
                {"jour", "date", "T_brut" (du relevé), "T" (de calcul ; None : sans donnée), "T_prudent", "source":
                 "mesure" | "prevision" | "prolonge" | "constante" | "sans_donnee", "k", "k_prudent", "debut", "fin"
                 (âges réels), "t_T", "t_T_prudent" (cumulés en FIN de jour), "gel", "T_gel", "base_gel",
                 "T_min_h", "T_max_h" (détail horaire, sinon None)},
     "synthese": {"T_moy", "T_min", "T_max", "n_jours" (connus), "n": {source: nombre}, "n_avant", "n_manquants",
                  "fin_connus" (âge réel) , "date_fin_connus", "t_T_28", "t_T_28_prudent", "t_3" (âge réel où t_T = 3 j),
                  "prudent_distinct"},
     "objectifs": [{"nom", "mode", "pct", "mpa", "cible", "base", "atteignable", "t_T" (requis), "t", "instant",
                    "indicatif", "prudent": {"t", "instant"}, "age_prevu", "instant_prevu", "t_T_prevu", "f_prevu",
                    "f_prevu_prudent", "indicatif_prevu", "taux" (cible / f à l'âge prévu), "etat": "ok" | "attention" |
                    "nok" | "date" | "jamais" | "non_atteignable", "prudent_ok", "au_dela" (atteint après les jours
                    connus), "formules": [...]}, …],
     "gel": {"seuil", "base": "minimale" | "moyenne", "premier_jour", "date", "T", "t", "t_5MPa", "alerte",
             "prudent": {"premier_jour", "date", "T", "t", "t_5MPa", "alerte"}, "formules": [...]},
     "temps_froid": {"seuil", "heures", "rappel", "jour", "date", "T", "base"},
     "essai": None | {"age", "mpa", "eprouvette", "rapport_cube", "f_cyl", "t_T", "f_prevu", "ecart" (%), "indicatif",
                      "s_cale", "r_cale", "calage_refuse" (raison | None), "caler", "cale", "formules": [...]},
     "courbes": {"horizon", "t", "fcm", "fck", "fcm_prudent", "fck_prudent" (None si la variante ne diffère pas),
                 "t_3", "t_28", "date", "heure", "objectifs": [...], "essai": {"t", "f"} | None},
     "tableau": {"horaire", "lignes": [<jour> + {"beta", "fcm", "fck"}, …]},
     "avertissements": [{"code", "niveau": "grave" | "attention" | "info", "message"}, …],
     "non_couvert": [texte, …],
     "resume": {"date" (un objectif daté au moins), "taux", "etat", "objectif" (rang), "nom"}}

  formules = [{"cle", "symbole", "termes": [{"symbole", "valeur", "unite"}, …], "resultat", "unite"}, …] : chaque
  résultat se retrouve à partir de ses seuls termes (tests/test_maturite_moteur.py le prouve, clé par clé).
"""
import datetime as _dt
import math

from . import entree as EN
from . import lois as L
from . import meteo as MT
from . import releve as RL
from .entree import normaliser, nouvel_element  # noqa: F401 — l'API

HORIZON_MIN_J = 28.0        # la courbe couvre au moins 28 jours réels…
HORIZON_MAX_J = 90.0        # … et 90 au plus
MARGE_HORIZON = 1.2         # … ou jusqu'au dernier objectif + 20 %
POINTS_COURBE = 240         # le pas de la courbe : horizon / 240 (les cassures de jour et les repères s'y ajoutent)
SEUIL_GEL_MPA = 5.0         # EN 13670:2009 §8.5(12) : pas de température sous 0 °C en surface avant 5 MPa
T_GEL_C = 0.0
SEUIL_FROID_C = 5.0         # Buildwise NIT 285 §9.4 : mesures de bétonnage par temps froid si T < 5 °C…
HEURES_FROID = 72.0         # … au bétonnage ou dans les 72 heures
GRAVE, ATTENTION, INFO = "grave", "attention", "info"
SOURCES = ("mesure", "prevision", "prolonge", "constante", "sans_donnee")
NON_MESURES = ("prevision", "prolonge", "constante")       # les jours que la variante prudente abaisse de la marge

# Les MODÈLES d'objectif (un clic les ajoute) : (clé, nom, mode, valeur, base, source). Les seuils en MPa sont des seuils
# de chantier, comparés à la résistance MOYENNE — ce que mesurent des cubes conservés sur chantier ; les pourcentages de
# f_ck sont des exigences de calcul, comparées à la caractéristique, à fixer par le bureau d'études.
MODELES_OBJECTIF = (
    ("hors_gel", "Hors gel", "mpa", 5.0, "fcm",
     "EN 13670:2009 §8.5(12) : la surface du béton ne descend pas sous 0 °C avant d'avoir 5 MPa"),
    ("decoffrage_vertical", "Décoffrage vertical", "mpa", 5.0, "fcm",
     "annexe informative de l'EN 13670 (décoffrage sans effort), citée par le CSTC et Infociments"),
    ("reprise", "Reprise de bétonnage", "mpa", 10.0, "fcm", "Infociments, bétonnage par temps froid (phase de bétonnage suivante)"),
    ("pct50", "50 % de f_ck", "pct", 50.0, "fck", "exigence de calcul, à fixer par le bureau d'études selon la charge au décoffrage"),
    ("pct70", "70 % de f_ck", "pct", 70.0, "fck", "exigence de calcul, à fixer par le bureau d'études selon la charge au décoffrage"),
    ("pct100", "100 % de f_ck", "pct", 100.0, "fck", "la résistance de calcul entière (mise en charge, désétaiement complet)"),
)
NON_COUVERT = (
    "les délais minimaux de décoffrage par type de coffrage de la NBN B 15-400 et les durées de cure (tableau F.8-ANB) : "
    "leurs valeurs n'ont pas pu être relues sur le texte",
    "les critères en proportion des charges (charge au décoffrage rapportée à la charge de service)",
    "les flèches au jeune âge (module E_cm(t)) et le ré-étaiement sur plusieurs niveaux",
    "l'EN 1992-1-1:2023 (annexe B.4, classes CS, CN, CR) : pas encore d'annexe belge",
    "la chaleur d'hydratation et la température propre d'un élément massif ou isolé : la température est celle qui est saisie",
)


def modele_objectif(cle: str, fck: float) -> dict:
    """L'objectif d'un modèle (``MODELES_OBJECTIF``) ; ValueError si la clé n'en est pas une."""
    for k, nom, mode, valeur, base, _source in MODELES_OBJECTIF:
        if k == cle:
            o = EN.nouvel_objectif(nom, fck)
            o.update(mode=mode, base=base, **{mode: valeur})
            return o
    raise ValueError(f"modèle d'objectif inconnu : {str(cle)[:40]!r}")


# ------------------------------------------------------------------ petits outils
def _f(cle, symbole, termes, resultat, unite=""):
    return {"cle": cle, "symbole": symbole, "termes": [{"symbole": s, "valeur": v, "unite": u} for s, v, u in termes],
            "resultat": resultat, "unite": unite}


def _c(x: float) -> str:
    """Une constante d'une expression, à la française et sans zéro inutile (8 ; 8,314 ; 273)."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


# Les EXPRESSIONS des formules, par leur clé (``formules[k]["cle"]``) : le GABARIT où ``{i}`` est le i-ième terme de la
# formule (dans l'ordre de ``termes``). L'écran en tire « symbole = expression = valeurs = résultat » : l'expression,
# chaque ``{i}`` remplacé par le SYMBOLE du terme et « × » par « · » ; les valeurs, chaque ``{i}`` par sa VALEUR. Les
# constantes viennent des tables de ``lois.py`` (une source). Un gabarit à None : une valeur posée par la loi, dite par
# son seul texte (``LUES``). tests/test_api_web_age.py réévalue chaque ligne affichée à partir de ses valeurs.
_RAC = f"√{_c(L.AGE_REFERENCE_J / L.AGE_RAPPORT_J)}"                           # √14 : (3.2) écrite à 2 jours
_BETA = f"exp({{0}} × (1 − √({_c(L.AGE_REFERENCE_J)} / {{1}})))"
_ARR = f"(1 / ({_c(L.T_KELVIN)} + {{0}}) − 1 / {_c(L.T_KELVIN + L.T_REFERENCE_C)})"
EXPRESSIONS = {
    "cible_pct": "{0} × {1} / 100",
    "fcm": f"{{0}} + {_c(L.DELTA_FCM)}",
    "beta_req_fck": f"({{0}} + {_c(L.DELTA_FCM)}) / {{1}}",
    "beta_req_fcm": "{0} / {1}",
    "t_T_req": f"{_c(L.AGE_REFERENCE_J)} / (1 − ln({{0}}) / {{1}})²",
    "T_prudent": "{0} − {1}",
    "k_nbn": "{2} + ({4} − {2}) × ({0} − {1}) / ({3} − {1})",
    "k_ec2": f"exp(−{_c(L.EA_SUR_R_EC2)} × {_ARR})",
    "k_perso": f"exp(−({{1}} × 1000 / {_c(L.R_GAZ)}) × {_ARR})",
    "t_atteint": "{0} + ({1} − {2}) / {3}",
    "t_T_a": "{0} + {1} × ({2} − {3})",
    "beta": _BETA,
    "fcm_t": "{0} × {1}",
    "fck_t": f"max(0 ; {{0}} − {_c(L.DELTA_FCM)})",
    "taux": "{0} / {1}",
    "rapport_cube": "{0} / {1}",
    "f_cyl": "{0} × {1}",
    "ecart": "({0} − {1}) / {1} × 100",
    "s_cale": f"ln({{0}} / {{1}}) / (1 − √({_c(L.AGE_REFERENCE_J)} / {{2}}))",
    "r_de_s": f"exp({{0}} × (1 − {_RAC}))",
    "s_de_r": f"−ln({{0}}) / ({_RAC} − 1)",
}
LUES = {
    "t_T_req_28": f"{_c(L.AGE_REFERENCE_J)} (la cible demande la résistance à {_c(L.AGE_REFERENCE_J)} j : β_req ≥ 1)",
    "beta_28": f"1 (t_T ≥ {_c(L.AGE_REFERENCE_J)} j)",
    "beta_nul": "0 (aucun durcissement compté : t_T = 0)",
    "k_nbn_nul": f"0 (T ≤ {_c(L.T_NUL_NBN).replace('-', '−')} °C)",
    "k_nbn_plafond": f"1 (T ≥ {_c(L.T_REFERENCE_C)} °C : aucun bonus compté)",
    "k_sous_zero": f"0 (sous {_c(L.T_MIN_EC2)} °C : hors du domaine de la loi)",
    "k_sans_donnee": "0 (jour sans température)",
}


def _fr(x: float, nd: int = 1) -> str:
    """Un nombre d'un message, à la française (le vrai signe moins)."""
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−").replace(".", ",")


def _jj(iso) -> str:
    """« 2026-11-02 » → « 02/11/2026 »."""
    return _dt.date.fromisoformat(iso).strftime("%d/%m/%Y")


def _pluriel(n: int, mot: str) -> str:
    return f"{n} {mot}{'s' if n > 1 else ''}"


def _date_du_jour(date0, j: int):
    if date0 is None:
        return None
    try:
        return (_dt.date.fromisoformat(date0) + _dt.timedelta(days=j)).isoformat()
    except (OverflowError, ValueError):
        return None


def _instant(date0, heure: int, t, plafond: bool = True):
    """L'instant « AAAA-MM-JJTHH:MM » de l'âge réel t, à l'heure près : arrondi à l'heure SUPÉRIEURE (``plafond`` —
    une échéance n'est jamais annoncée plus tôt que le calcul), ou à l'heure la plus proche (une opération prévue).
    None sans date."""
    if date0 is None or t is None:
        return None
    heures = t * L.HEURES_PAR_JOUR
    n = math.ceil(heures - 1e-6) if plafond else math.floor(heures + 0.5)
    try:
        x = _dt.datetime.combine(_dt.date.fromisoformat(date0), _dt.time(hour=heure)) + _dt.timedelta(hours=n)
    except (OverflowError, ValueError):
        return None
    return x.strftime("%Y-%m-%dT%H:%M")


# ------------------------------------------------------------------ le profil des températures
SYM_PRUDENT = "_prud"       # l'indice des symboles de la variante prudente : T_prud, k_prud, t_prud (t_p est l'âge PRÉVU)


def _k_formule(T, loi: dict, symbole: str = "k", sym_T: str = "T") -> dict:
    """La formule du coefficient k à la température T, dans la loi choisie (None : un jour sans température).
    ``sym_T`` : le symbole de la température dans la formule (« T_prud » quand elle est abaissée de la marge)."""
    if T is None:
        return _f("k_sans_donnee", symbole, [], 0.0)
    mode = loi["mode"]
    if mode == "nbn":
        seg = L.segment_nbn(T)
        if seg is None:
            return _f("k_nbn_nul" if T <= L.T_NUL_NBN else "k_nbn_plafond", symbole, [(sym_T, T, "°C")], L.k_nbn(T))
        (Ta, ka), (Tb, kb) = seg
        return _f("k_nbn", symbole, [(sym_T, T, "°C"), ("T_a", Ta, "°C"), ("k_a", ka, ""), ("T_b", Tb, "°C"), ("k_b", kb, "")], L.k_nbn(T))
    if T < L.T_MIN_EC2:
        return _f("k_sous_zero", symbole, [(sym_T, T, "°C")], 0.0)
    if mode == "ec2":
        return _f("k_ec2", symbole, [(sym_T, T, "°C")], L.k_ec2(T))
    return _f("k_perso", symbole, [(sym_T, T, "°C"), ("E_a", loi["Ea_kJ"], "kJ/mol")], L.k_arrhenius(T, loi["Ea_kJ"]))


def _profil(e: dict, avert: list) -> dict:
    """Les jours CONNUS du coulage (jour 0 = le bétonnage) et ce qui les suit : la température, sa source et les deux
    coefficients k (nominal, prudent) de chaque jour ; la température d'après (constante, ou prolongée)."""
    t, loi, marge = e["temperature"], e["loi"], e["marge"]
    out = {"f0": L.fraction_premier_jour(e["heure"]), "date0": e["date"], "date_origine": "saisie" if e["date"] else None,
           "source": "constante", "T_fin": t["T"], "source_fin": "constante", "connus": [], "releve": None, "meteo": None,
           "n_avant": 0, "n_manquants": 0, "minimales": None, "extremes": None}
    if t["mode"] in EN.MODES_RELEVE:
        if t["mode"] == "meteo" and t["meteo"]:
            out["meteo"] = MT.synthese(t["meteo"], t["releve"], t["lieu"], t["debut"], t["fin"])
        rel = RL.lire(t["releve"])
        syn = RL.synthese(rel) if rel["temps"] else None
        if syn is None:
            # pas de relevé exploitable : le calcul se fait à la température CONSTANTE saisie, et c'est dit
            T = f"{_fr(t['T'])} °C"
            if t["mode"] == "meteo" and not rel["temps"] and t["meteo"]:
                avert.append({"code": "meteo_effacee", "niveau": ATTENTION, "message":
                              f"Températures de la météo effacées : le calcul se fait à la température constante ({T}). « Récupérer à "
                              "nouveau la météo » remet celles du service (ou saisissez-les, une ligne par jour)."})
            elif t["mode"] == "meteo" and not rel["temps"]:
                avert.append({"code": "meteo_absente", "niveau": ATTENTION, "message":
                              f"Météo pas encore récupérée : le calcul se fait à la température constante ({T}). Renseignez le lieu "
                              "et les dates, puis « Récupérer la météo »."})
            elif not rel["temps"]:
                avert.append({"code": "releve_vide", "niveau": ATTENTION, "message":
                              f"Aucun relevé : le calcul se fait à la température constante ({T}). Collez les températures "
                              "journalières (ou lisez un fichier)."})
            else:
                avert.append({"code": "releve_inexploitable", "niveau": GRAVE, "message":
                              f"Le relevé ne porte aucune température : le calcul se fait à la température constante ({T})."})
        else:
            out.update(source="releve", releve=syn, source_fin="prolonge", T_fin=L.temperature_prolongee(rel["temps"]))
            n, d0 = len(rel["temps"]), 0
            if rel["dates"]:
                if out["date0"] is None:
                    out.update(date0=rel["dates"][0], date_origine="releve")
                else:
                    d0 = (_dt.date.fromisoformat(out["date0"]) - _dt.date.fromisoformat(rel["dates"][0])).days
            out["n_avant"] = min(max(d0, 0), n)
            out["n_manquants"] = max(-d0, 0)
            out["connus"] += [{"T": None, "source": "sans_donnee"} for _ in range(out["n_manquants"])]
            for i in range(out["n_avant"], n):
                T = rel["temps"][i]
                if math.isnan(T):
                    out["connus"].append({"T": None, "source": "sans_donnee"})
                else:
                    out["connus"].append({"T": T, "source": "prolonge" if rel["extra"][i] else ("prevision" if rel["prev"][i] else "mesure")})
            m = out["meteo"]
            if m and m["horaire"] is not None and out["date0"]:
                # le détail horaire : les extrêmes de chaque jour (le tableau), et la minimale comptée à partir de
                # l'heure de fin du bétonnage le jour 0 (le gel)
                rec = t["meteo"]
                h = MT.lire_horaire(MT.plan_horaire(rec, rec["horaire"]["aujourdhui"]), rec["horaire"]["reponses"])
                out["extremes"] = m["horaire"]
                out["minimales"] = MT.minimales_apres(h, f"{out['date0']}T{e['heure']:02d}:00")
    for j, c in enumerate(out["connus"]):
        T = c["T"]
        c["T_prudent"] = None if T is None else (T - marge if c["source"] in NON_MESURES else T)
        c["k"] = 0.0 if T is None else L.k_T(T, loi)
        c["k_prudent"] = 0.0 if T is None else L.k_T(c["T_prudent"], loi)
    out["T_fin_prudent"] = out["T_fin"] - marge
    out["k_fin"] = L.k_T(out["T_fin"], loi)
    out["k_fin_prudent"] = L.k_T(out["T_fin_prudent"], loi)
    out["ks"] = [c["k"] for c in out["connus"]]
    out["ks_prudent"] = [c["k_prudent"] for c in out["connus"]]
    # les avertissements du relevé posé sur le calendrier du coulage
    if out["n_avant"]:
        avert.append({"code": "releve_avant", "niveau": INFO, "message":
                      f"{_pluriel(out['n_avant'], 'jour')} du relevé avant la fin du bétonnage ({_jj(out['date0'])}) : non "
                      f"compté{'s' if out['n_avant'] > 1 else ''}."})
    if out["n_manquants"]:
        avert.append({"code": "releve_tardif", "niveau": GRAVE, "message":
                      f"Le relevé commence {_pluriel(out['n_manquants'], 'jour')} après la fin du bétonnage ({_jj(out['date0'])}) : "
                      "ces jours, sans température, ne comptent AUCUN durcissement. Complétez le relevé (une ligne par jour, depuis "
                      "le jour du bétonnage) ou corrigez la date."})
    n_nan = sum(1 for c in out["connus"][out["n_manquants"]:] if c["T"] is None)
    if n_nan:
        avert.append({"code": "sans_donnee", "niveau": ATTENTION, "message":
                      f"{_pluriel(n_nan, 'jour')} sans température dans le relevé : compté{'s' if n_nan > 1 else ''} pour AUCUN "
                      "durcissement (du côté sûr). Complétez la ligne pour le compter."})
    if loi["mode"] != "nbn":
        froids = sum(1 for c in out["connus"] if c["T"] is not None and c["T"] < L.T_MIN_EC2) + (1 if not out["connus"] and out["T_fin"] < L.T_MIN_EC2 else 0)
        if froids:
            quoi = "la température est" if not out["connus"] else f"{_pluriel(froids, 'jour')}"
            avert.append({"code": "loi_sous_zero", "niveau": ATTENTION, "message":
                          f"{quoi[0].upper() + quoi[1:]} sous 0 °C : hors du domaine de la loi {L.LOIS[loi['mode']]['libelle']} "
                          "(0 à 80 °C), aucun durcissement n'y est compté. La loi NBN B 15-400 couvre le temps froid."})
    return out


def _jour(p: dict, j: int) -> dict:
    """Le jour j du profil : connu, ou d'après (constante, prolongé)."""
    if j < len(p["connus"]):
        c = p["connus"][j]
        return {"T": c["T"], "T_prudent": c["T_prudent"], "source": c["source"], "k": c["k"], "k_prudent": c["k_prudent"], "connu": True}
    return {"T": p["T_fin"], "T_prudent": p["T_fin_prudent"], "source": p["source_fin"], "k": p["k_fin"], "k_prudent": p["k_fin_prudent"],
            "connu": False}


def _debut_jour(p: dict, j: int) -> float:
    """L'âge réel au début du jour j (0 pour le jour du bétonnage)."""
    return 0.0 if j == 0 else p["f0"] + (j - 1)


def _te(p: dict, t: float, prudent: bool = False) -> float:
    return L.age_equivalent(t, p["ks_prudent" if prudent else "ks"], p["k_fin_prudent" if prudent else "k_fin"], p["f0"])


def _t(p: dict, t_T, prudent: bool = False):
    if t_T is None:
        return None
    return L.age_reel(t_T, p["ks_prudent" if prudent else "ks"], p["k_fin_prudent" if prudent else "k_fin"], p["f0"])


def _fin_connus(p: dict):
    """L'âge réel à la fin du dernier jour connu du relevé (0 quand tout le relevé précède le bétonnage) ; None à
    température constante."""
    if p["connus"]:
        return p["f0"] + len(p["connus"]) - 1
    return 0.0 if p["source"] == "releve" else None


def _source_a(p: dict, t) -> str:
    """La source du jour civil où tombe l'âge réel t (None : jamais atteint — la source des jours d'après)."""
    if t is None:
        return p["source_fin"]
    return _jour(p, 0 if t <= p["f0"] else int(math.ceil(t - p["f0"] - 1e-12)))["source"]


# ------------------------------------------------------------------ le gel, le temps froid
def _T_gel(p: dict, j: int, prudent: bool = False):
    """La température qui sert au contrôle du gel le jour j : (T, base) — la MINIMALE horaire du jour quand le détail
    horaire la donne (comptée dès l'heure de fin du bétonnage le jour 0), sinon la MOYENNE du jour ; (None, base) pour un
    jour sans température. La variante prudente abaisse de la marge les jours qui ne sont pas mesurés."""
    jr = _jour(p, j)
    if jr["T"] is None:
        return None, "moyenne"
    T, base = jr["T"], "moyenne"
    date = _date_du_jour(p["date0"], j)
    if jr["connu"] and p["minimales"] and date in p["minimales"]:
        T, base = p["minimales"][date], "minimale"
    if prudent and jr["source"] in NON_MESURES:
        T -= p["marge"]
    return T, base


def _gele(T, base: str) -> bool:
    """Le critère de gel : une minimale horaire SOUS 0 °C, ou une moyenne journalière à 0 °C ou moins."""
    return T is not None and (T < T_GEL_C if base == "minimale" else T <= T_GEL_C)


def _premier_gel(p: dict, prudent: bool = False):
    """Le premier jour de gel : (jour, T, base) ; (None, None, base disponible) s'il n'y en a pas."""
    for j in range(len(p["connus"]) + 1):                          # les jours connus, puis le premier jour d'après
        T, base = _T_gel(p, j, prudent)
        if _gele(T, base):
            return j, T, base
    return None, None, ("minimale" if p["minimales"] else "moyenne")


def _gel(p: dict, fck: float, s: float) -> dict:
    """Le gel avant 5 MPa (EN 13670 §8.5(12)) : le premier jour de gel face à l'âge où la résistance MOYENNE atteint
    5 MPa — alerte si le gel vient avant (le jour de gel est pris à son début : le gel est un fait du petit matin)."""
    tT5 = L.t_T_requis(SEUIL_GEL_MPA, "fcm", fck, s)

    def variante(prudent):
        j, T, base = _premier_gel(p, prudent)
        t5 = _t(p, tT5, prudent)
        tg = None if j is None else _debut_jour(p, j)
        return {"premier_jour": j, "date": None if j is None else _date_du_jour(p["date0"], j), "T": T, "t": tg, "t_5MPa": t5,
                "alerte": j is not None and (t5 is None or tg < t5)}, base
    nominal, base = variante(False)
    prudent, _ = variante(True)
    fx = [_f("beta_req_fcm", "β_5", [("f_gel", SEUIL_GEL_MPA, "MPa"), ("f_cm", fck + L.DELTA_FCM, "MPa")], L.beta_requis(SEUIL_GEL_MPA, "fcm", fck)),
          _f("t_T_req", "t_T,5", [("β_req", L.beta_requis(SEUIL_GEL_MPA, "fcm", fck), ""), ("s", s, "")], tT5, "j")]
    seg = L.segment_inverse(tT5, p["ks"], p["k_fin"], p["f0"])
    if seg is not None:
        fx.append(_f("t_atteint", "t_5", [("t_d", seg["debut"], "j"), ("t_T,req", tT5, "j"), ("t_T,d", seg["t_T_debut"], "j"), ("k", seg["k"], "")],
                     nominal["t_5MPa"], "j"))
    return dict(nominal, seuil=SEUIL_GEL_MPA, base=base, prudent=prudent, formules=fx)


def _temps_froid(p: dict) -> dict:
    """Le rappel « bétonnage par temps froid » (Buildwise, NIT 285 §9.4) : une température sous 5 °C au bétonnage ou
    dans les 72 premières heures (la minimale horaire quand elle est connue, sinon la moyenne du jour)."""
    out = {"seuil": SEUIL_FROID_C, "heures": HEURES_FROID, "rappel": False, "jour": None, "date": None, "T": None, "base": None}
    j = 0
    while _debut_jour(p, j) < HEURES_FROID / L.HEURES_PAR_JOUR - 1e-12:
        T, base = _T_gel(p, j)
        if T is not None and T < SEUIL_FROID_C:
            return dict(out, rappel=True, jour=j, date=_date_du_jour(p["date0"], j), T=T, base=base)
        j += 1
    return out


# ------------------------------------------------------------------ l'essai sur éprouvettes
def _essai(e: dict, p: dict, fck: float, s: float) -> dict:
    """La moyenne des éprouvettes de chantier face à la prévision f_cm(t) (avec le s du durcissement CHOISI), et le s
    qui ferait passer la courbe par la mesure. Un cube de 150 mm est ramené au cylindre par f_ck/f_ck,cube de la classe."""
    x = e["essai"]
    fcm = fck + L.DELTA_FCM
    cube = x["eprouvette"] == "cube"
    rapport = L.rapport_cube(e["classe"]) if cube else 1.0
    f_cyl = x["mpa"] * rapport
    seg = L.segment(x["age"], p["ks"], p["k_fin"], p["f0"])
    tT = _te(p, x["age"])
    beta = L.beta_retenu(tT, s)
    f_prevu = beta * fcm
    out = {"age": x["age"], "mpa": x["mpa"], "eprouvette": x["eprouvette"], "rapport_cube": rapport, "f_cyl": f_cyl, "t_T": tT,
           "instant": _instant(p["date0"], e["heure"], x["age"], plafond=False),
           "f_prevu": f_prevu, "ecart": (f_cyl - f_prevu) / f_prevu * 100.0 if f_prevu > 0 else None, "indicatif": L.indicatif(tT),
           "s_cale": None, "r_cale": None, "calage_refuse": None, "caler": x["caler"], "cale": False}
    fx = []
    if cube:
        fx += [_f("rapport_cube", "f_ck/f_ck,cube", [("f_ck", fck, "MPa"), ("f_ck,cube", L.CLASSES[e["classe"]][1], "MPa")], rapport),
               _f("f_cyl", "f_mes", [("f_cube", x["mpa"], "MPa"), ("f_ck/f_ck,cube", rapport, "")], f_cyl, "MPa")]
    fx += [_f("t_T_a", "t_T", [("t_T,d", seg["t_T_debut"], "j"), ("k", seg["k"], ""), ("t", x["age"], "j"), ("t_d", seg["debut"], "j")], tT, "j"),
           _f("beta" if tT < L.AGE_REFERENCE_J else "beta_28", "β_cc", [("s", s, ""), ("t_T", tT, "j")], beta),
           _f("fcm_t", "f_cm(t)", [("β_cc", beta, ""), ("f_cm", fcm, "MPa")], f_prevu, "MPa")]
    if out["ecart"] is not None:
        fx.append(_f("ecart", "écart", [("f_mes", f_cyl, "MPa"), ("f_cm(t)", f_prevu, "MPa")], out["ecart"], "%"))
    try:
        sc = L.s_cale(f_cyl, fck, tT)
    except ValueError as err:
        out["calage_refuse"] = str(err)
    else:
        out["s_cale"] = sc
        fx.append(_f("s_cale", "s_cal", [("f_mes", f_cyl, "MPa"), ("f_cm", fcm, "MPa"), ("t_T", tT, "j")], sc))
        if L.S_CALE_MIN <= sc <= L.S_CALE_MAX:
            out["r_cale"] = L.r_de_s(sc)
            fx.append(_f("r_de_s", "r_cal", [("s", sc, "")], out["r_cale"]))
        else:
            out["calage_refuse"] = (f"s calé = {_fr(sc, 3)}, hors de [{_fr(L.S_CALE_MIN, 2)} ; {_fr(L.S_CALE_MAX, 2)}] : la mesure "
                                    "s'écarte trop de la loi de l'EC2 pour caler la courbe sur elle")
    out["cale"] = bool(x["caler"] and out["calage_refuse"] is None)
    out["formules"] = fx
    return out


# ------------------------------------------------------------------ un objectif
def _objectif(i: int, o: dict, e: dict, p: dict, fck: float, s: float) -> dict:
    """Un objectif : sa cible, l'âge équivalent qu'elle demande (forme fermée), l'âge réel et l'instant où elle est
    atteinte (le profil inversé exactement), leur variante prudente ; et, s'il est daté, la résistance à l'âge prévu."""
    fcm = fck + L.DELTA_FCM
    pct = o["mode"] == "pct"
    cible = fck * o["pct"] / 100.0 if pct else o["mpa"]
    base = o["base"]
    fx = []
    if pct:
        fx.append(_f("cible_pct", "f_cible", [("p", o["pct"], "%"), ("f_ck", fck, "MPa")], cible, "MPa"))
    fx.append(_f("fcm", "f_cm", [("f_ck", fck, "MPa")], fcm, "MPa"))
    b_req = L.beta_requis(cible, base, fck)
    fx.append(_f(f"beta_req_{base}", "β_req", [("f_cible", cible, "MPa"), ("f_cm", fcm, "MPa")], b_req))
    tT = L.t_T_requis(cible, base, fck, s)
    out = {"nom": o["nom"] or f"Objectif {i + 1}", "mode": o["mode"], "pct": o["pct"], "mpa": o["mpa"], "cible": cible, "base": base,
           "atteignable": tT is not None, "t_T": tT, "t": None, "instant": None, "indicatif": False,
           "prudent": {"t": None, "instant": None}, "age_prevu": o["age"], "instant_prevu": None, "t_T_prevu": None, "f_prevu": None,
           "f_prevu_prudent": None, "indicatif_prevu": False, "taux": None, "prudent_ok": None, "source": None, "au_dela": False}
    if tT is not None:
        fx.append(_f("t_T_req" if b_req < 1.0 else "t_T_req_28", "t_T,req", [("β_req", b_req, ""), ("s", s, "")], tT, "j"))
        out["indicatif"] = L.indicatif(tT)
        for prudent in (False, True):
            ks, kf = (p["ks_prudent"], p["k_fin_prudent"]) if prudent else (p["ks"], p["k_fin"])
            t = L.age_reel(tT, ks, kf, p["f0"])
            seg = L.segment_inverse(tT, ks, kf, p["f0"])
            inst = _instant(p["date0"], e["heure"], t)
            if prudent:
                out["prudent"] = {"t": t, "instant": inst}
            else:
                out["t"], out["instant"] = t, inst
            if seg is not None:
                jr = _jour(p, seg["jour"])
                suf = SYM_PRUDENT if prudent else ""
                abaisse = prudent and jr["source"] in NON_MESURES   # un jour mesuré n'est pas abaissé : sa température reste T
                if abaisse:
                    fx.append(_f("T_prudent", "T" + suf, [("T", jr["T"], "°C"), ("marge", p["marge"], "°C")], jr["T_prudent"], "°C"))
                fx.append(_k_formule(jr["T_prudent"] if prudent else jr["T"], e["loi"], "k" + suf, "T" + suf if abaisse else "T"))
                fx.append(_f("t_atteint", "t" + suf, [("t_d", seg["debut"], "j"), ("t_T,req", tT, "j"), ("t_T,d", seg["t_T_debut"], "j"),
                                                    ("k" + suf, seg["k"], "")], t, "j"))
        out["source"] = _source_a(p, out["t"])
    if o["age"] is not None:
        a = o["age"]
        out["instant_prevu"] = _instant(p["date0"], e["heure"], a, plafond=False)
        for prudent in (False, True):
            ks, kf = (p["ks_prudent"], p["k_fin_prudent"]) if prudent else (p["ks"], p["k_fin"])
            te = L.age_equivalent(a, ks, kf, p["f0"])
            f = L.resistance(te, base, fck, s)
            if prudent:
                out["f_prevu_prudent"] = f
                continue
            seg = L.segment(a, ks, kf, p["f0"])
            beta = L.beta_retenu(te, s)
            out.update(t_T_prevu=te, f_prevu=f, indicatif_prevu=L.indicatif(te))
            fx += [_f("t_T_a", "t_T(t_p)", [("t_T,d", seg["t_T_debut"], "j"), ("k", seg["k"], ""), ("t", a, "j"), ("t_d", seg["debut"], "j")], te, "j"),
                   _f("beta" if te < L.AGE_REFERENCE_J else "beta_28", "β_cc", [("s", s, ""), ("t_T", te, "j")], beta),
                   _f("fcm_t", "f_cm(t_p)", [("β_cc", beta, ""), ("f_cm", fcm, "MPa")], beta * fcm, "MPa")]
            if base == "fck":
                fx.append(_f("fck_t", "f_ck(t_p)", [("f_cm(t)", beta * fcm, "MPa")], f, "MPa"))
        out["taux"] = cible / out["f_prevu"] if out["f_prevu"] > 0 else None
        if out["taux"] is not None:
            fx.append(_f("taux", "η", [("f_cible", cible, "MPa"), ("f(t_p)", out["f_prevu"], "MPa")], out["taux"]))
        ok = out["f_prevu"] >= cible
        out["prudent_ok"] = out["f_prevu_prudent"] >= cible
        out["etat"] = "nok" if not ok else ("ok" if out["prudent_ok"] else "attention")
        out["source"] = _source_a(p, a)                            # une opération datée : la source du jour PRÉVU
    elif tT is None:
        out["etat"] = "non_atteignable"
    else:
        out["etat"] = "date" if out["t"] is not None else "jamais"
    # le résultat repose sur des températures PROLONGÉES (au-delà du relevé, ou les jours « prolongé » d'une météo)
    out["au_dela"] = out["source"] == "prolonge"
    out["formules"] = fx
    return out


# ------------------------------------------------------------------ le calcul
def _base(e: dict):
    """Ce dont tout dépend : (avertissements, f_ck, le durcissement choisi, le profil, l'essai, le s RETENU — celui du
    durcissement, ou celui que l'essai donne quand on cale la courbe sur lui)."""
    avert = []
    fck = L.fck_classe(e["classe"])
    dur = L.durcissement(e["durcissement"])
    p = _profil(e, avert)
    p["marge"] = e["marge"]
    essai = _essai(e, p, fck, dur["s"]) if e["essai"]["actif"] else None
    return avert, fck, dur, p, essai, (essai["s_cale"] if essai and essai["cale"] else dur["s"])


def calculer(coulage: dict) -> dict:
    """Tout ce que l'écran montre pour un coulage (voir la docstring du module)."""
    e = EN.normaliser(coulage)
    avert, fck, dur, p, essai, s = _base(e)
    fck_cube = L.CLASSES[e["classe"]][1]
    fcm = fck + L.DELTA_FCM
    fx_m = [_f("fcm", "f_cm", [("f_ck", fck, "MPa")], fcm, "MPa")]
    if dur["mode"] == "ciment":
        fx_m.append(_f("r_de_s", "r", [("s", dur["s"], "")], dur["r"]))
    else:
        fx_m.append(_f("s_de_r", "s", [("r", dur["r"], "")], dur["s"]))
    materiau = {"classe": e["classe"], "fck": fck, "fck_cube": fck_cube, "fcm": fcm, "s": s, "r": L.r_de_s(s), "cale": bool(essai and essai["cale"]),
                "durcissement": dur,
                "loi": {"mode": e["loi"]["mode"], "libelle": L.LOIS[e["loi"]["mode"]]["libelle"], "source": L.LOIS[e["loi"]["mode"]]["source"],
                        "Ea_kJ": e["loi"]["Ea_kJ"] if e["loi"]["mode"] == "perso" else None},
                "formules": fx_m}
    # ---- les objectifs
    objectifs = [_objectif(i, o, e, p, fck, s) for i, o in enumerate(e["objectifs"])]
    gel = _gel(p, fck, s)
    froid = _temps_froid(p)
    # ---- l'horizon : 28 jours au moins, le dernier objectif + 20 %, 90 jours au plus ; les jours connus du relevé y entrent
    ages = [x for o in objectifs for x in (o["t"], o["prudent"]["t"], o["age_prevu"]) if x is not None] + ([essai["age"]] if essai else [])
    horizon = min(max([HORIZON_MIN_J] + [MARGE_HORIZON * x for x in ages]), HORIZON_MAX_J)
    fin = _fin_connus(p)
    if fin is not None:
        horizon = max(horizon, min(fin, HORIZON_MAX_J))
    n_jours = int(math.ceil(horizon - p["f0"] - 1e-9)) + 1
    # ---- le profil jour par jour (jusqu'à l'horizon)
    profil, tT, tTp = [], 0.0, 0.0
    for j in range(n_jours):
        jr = _jour(p, j)
        debut = _debut_jour(p, j)
        fin_j = p["f0"] + j
        tT += jr["k"] * (fin_j - debut)
        tTp += jr["k_prudent"] * (fin_j - debut)
        Tg, base_g = _T_gel(p, j)
        date = _date_du_jour(p["date0"], j)
        ext = (p["extremes"] or {}).get(date) if jr["connu"] else None
        brut = jr["T"] if jr["connu"] else None
        profil.append({"jour": j, "date": date, "T_brut": brut, "T": jr["T"], "T_prudent": jr["T_prudent"], "source": jr["source"],
                       "k": jr["k"], "k_prudent": jr["k_prudent"], "debut": debut, "fin": fin_j, "t_T": tT, "t_T_prudent": tTp,
                       "gel": _gele(Tg, base_g), "T_gel": Tg, "base_gel": base_g,
                       "T_min_h": ext[0] if ext else None, "T_max_h": ext[1] if ext else None})
    # ---- la synthèse des températures (les jours connus ; la constante sinon)
    Ts = [c["T"] for c in p["connus"] if c["T"] is not None] or [p["T_fin"]]
    comptes = {k: sum(1 for c in p["connus"] if c["source"] == k) for k in SOURCES}
    prudent_distinct = e["marge"] > 0 and (p["k_fin_prudent"] != p["k_fin"] or p["ks_prudent"] != p["ks"])
    synthese = {"T_moy": sum(Ts) / len(Ts), "T_min": min(Ts), "T_max": max(Ts), "n_jours": len(p["connus"]),
                "n": {k: v for k, v in comptes.items() if v}, "n_avant": p["n_avant"], "n_manquants": p["n_manquants"],
                "fin_connus": fin, "date_fin_connus": _date_du_jour(p["date0"], len(p["connus"]) - 1) if p["connus"] else None,
                "t_T_28": _te(p, HORIZON_MIN_J), "t_T_28_prudent": _te(p, HORIZON_MIN_J, True),
                "t_3": _t(p, L.SEUIL_INDICATIF_J), "t_28": _t(p, L.AGE_REFERENCE_J), "prudent_distinct": prudent_distinct}
    # ---- les avertissements
    for o in objectifs:
        if o["etat"] == "non_atteignable" or (o["age_prevu"] is not None and not o["atteignable"]):
            quoi = "la résistance caractéristique f_ck" if o["base"] == "fck" else "la résistance moyenne f_cm"
            lim = fck if o["base"] == "fck" else fcm
            avert.append({"code": "non_atteignable", "niveau": GRAVE, "message":
                          f"« {o['nom']} » : la cible ({_fr(o['cible'])} MPa) dépasse {quoi} = {_fr(lim, 0)} MPa du {e['classe']} — non "
                          "atteignable avec cette classe."})
        elif o["etat"] == "jamais":
            avert.append({"code": "jamais", "niveau": GRAVE, "message":
                          f"« {o['nom']} » n'est jamais atteint : à cette température, la loi ne compte aucun durcissement (k = 0)."})
    indicatifs = [o["nom"] for o in objectifs if (o["indicatif"] and o["t"] is not None) or o["indicatif_prevu"]]
    if indicatifs:
        avert.append({"code": "indicatif", "niveau": ATTENTION, "message":
                      f"{'« ' + ' », « '.join(indicatifs) + ' »'} : avant 3 jours équivalents, hors du domaine où l'EC2 donne f_ck(t) "
                      "(§3.1.2(5)) — valeur INDICATIVE, à confirmer par essais."})
    if gel["alerte"]:
        quand = f"le {_jj(gel['date'])}" if gel["date"] else f"au jour {gel['premier_jour']}"
        t5 = f"atteints à J + {_fr(gel['t_5MPa'])}" if gel["t_5MPa"] is not None else "jamais atteints à cette température"
        avert.append({"code": "gel", "niveau": GRAVE, "message":
                      f"Gel {quand} ({'minimale horaire' if gel['base'] == 'minimale' else 'moyenne du jour'} {_fr(gel['T'])} °C), avant que le "
                      f"béton ait {_fr(SEUIL_GEL_MPA, 0)} MPa ({t5}) : PROTÉGER DU GEL (EN 13670 §8.5)."})
    elif gel["prudent"]["alerte"]:
        avert.append({"code": "gel_prudent", "niveau": ATTENTION, "message":
                      f"Gel possible avant {_fr(SEUIL_GEL_MPA, 0)} MPa si les températures prévues baissent de {_fr(e['marge'], 0)} °C "
                      "(variante prudente) : prévoir la protection."})
    if froid["rappel"]:
        avert.append({"code": "temps_froid", "niveau": ATTENTION, "message":
                      f"Température sous {_fr(SEUIL_FROID_C, 0)} °C dans les {_fr(HEURES_FROID, 0)} premières heures "
                      f"({'minimale horaire' if froid['base'] == 'minimale' else 'moyenne du jour'} {_fr(froid['T'])} °C"
                      + (f", le {_jj(froid['date'])}" if froid["date"] else "") + ") : mesures de bétonnage par temps froid (Buildwise, NIT 285 §9.4)."})
    au_dela = [o["nom"] for o in objectifs if o["au_dela"]]
    if au_dela:
        suite = (" Reculez la fin de la météo pour recevoir les prévisions, et recalculez à l'approche de l'échéance."
                 if e["temperature"]["mode"] == "meteo" else " Complétez le relevé à l'approche de l'échéance.")
        avert.append({"code": "prolonge", "niveau": ATTENTION, "message":
                      f"{'« ' + ' », « '.join(au_dela) + ' »'} : au-delà des températures connues, le relevé est PROLONGÉ à "
                      f"{_fr(p['T_fin'])} °C (la moyenne de ses cinq derniers jours)." + suite})
    if essai:
        if essai["cale"]:
            avert.append({"code": "cale", "niveau": INFO, "message":
                          f"Courbe CALÉE sur l'essai : s = {_fr(s, 3)} (r = {_fr(materiau['r'], 2)}) au lieu de {_fr(dur['s'], 3)} "
                          f"({dur['libelle']}) ; toutes les échéances suivent."})
        elif essai["caler"]:
            avert.append({"code": "calage_refuse", "niveau": GRAVE, "message":
                          f"Calage sur l'essai refusé — {essai['calage_refuse']}. La courbe garde s = {_fr(dur['s'], 3)}."})
        if essai["ecart"] is not None and essai["ecart"] < 0 and not essai["cale"]:
            avert.append({"code": "essai_sous", "niveau": ATTENTION, "message":
                          f"Essai : {_fr(essai['f_cyl'])} MPa mesurés pour {_fr(essai['f_prevu'])} MPa prévus ({_fr(essai['ecart'], 0)} %) — le "
                          "béton durcit moins vite que la loi : caler la courbe sur l'essai, ou revoir le durcissement."})
    # ---- la courbe, le tableau, le résumé
    out = {"entree": e, "materiau": materiau,
           "temps": {"date": p["date0"], "date_origine": p["date_origine"], "heure": e["heure"], "f0": p["f0"],
                     "origine": _instant(p["date0"], e["heure"], 0.0)},
           "temperature": {"mode": e["temperature"]["mode"], "source": p["source"], "T": e["temperature"]["T"], "lieu": e["temperature"]["lieu"],
                           "fichier": e["temperature"]["fichier"], "marge": e["marge"], "releve": p["releve"], "meteo": p["meteo"],
                           "T_fin": p["T_fin"], "source_fin": p["source_fin"], "k_fin": p["k_fin"],
                           "formule_k": _k_formule(p["T_fin"], e["loi"])},
           "profil": profil, "synthese": synthese, "objectifs": objectifs, "gel": gel, "temps_froid": froid, "essai": essai,
           "avertissements": avert, "non_couvert": list(NON_COUVERT)}
    out["courbes"] = _courbes(p, e, fck, s, horizon, objectifs, essai, synthese)
    out["tableau"] = {"horaire": p["extremes"] is not None,
                      "lignes": [dict(j, beta=L.beta_retenu(j["t_T"], s), fcm=L.fcm_t(j["t_T"], fck, s), fck=L.fck_t(j["t_T"], fck, s),
                                      indicatif=L.indicatif(j["t_T"])) for j in profil]}
    out["resume"] = _resume(objectifs)
    return out


def _resume(objectifs: list) -> dict:
    """Le résumé d'un coulage : l'objectif DATÉ le plus chargé (son taux, son état) ; sans objectif daté, le premier
    objectif (une consultation : son échéance)."""
    dates = [(o["taux"] if o["taux"] is not None else float("inf"), -i) for i, o in enumerate(objectifs) if o["age_prevu"] is not None]
    if not dates:
        return {"date": False, "taux": None, "etat": objectifs[0]["etat"], "objectif": 0, "nom": objectifs[0]["nom"]}
    _, moins_i = max(dates)
    o = objectifs[-moins_i]
    # l'état de la note : le pire des objectifs datés (non conforme, puis attention, puis conforme)
    rang = {"nok": 2, "attention": 1, "ok": 0}
    pire = max((x["etat"] for x in objectifs if x["age_prevu"] is not None), key=lambda k: rang[k])
    return {"date": True, "taux": o["taux"], "etat": pire, "objectif": -moins_i, "nom": o["nom"]}


def _courbes(p: dict, e: dict, fck: float, s: float, horizon: float, objectifs: list, essai, synthese: dict) -> dict:
    """Les points de la courbe : f_cm(t) et f_ck(t) de 0 à l'horizon — un pas régulier, plus les changements de jour
    (la pente y casse) et les âges équivalents repères (3 et 28 j) —, leur variante prudente, les objectifs, l'essai."""
    ts = {0.0, horizon}
    ts.update(horizon * i / POINTS_COURBE for i in range(1, POINTS_COURBE))
    j = 0
    while p["f0"] + j < horizon:
        ts.add(p["f0"] + j)
        j += 1
    for prudent in (False, True):
        for x in (L.SEUIL_INDICATIF_J, L.AGE_REFERENCE_J):
            t = _t(p, x, prudent)
            if t is not None and t < horizon:
                ts.add(t)
    ts = sorted(ts)
    te = [_te(p, t) for t in ts]
    c = {"horizon": horizon, "t": ts, "fcm": [L.fcm_t(x, fck, s) for x in te], "fck": [L.fck_t(x, fck, s) for x in te],
         "fcm_prudent": None, "fck_prudent": None, "t_3": synthese["t_3"], "t_28": synthese["t_28"], "date": p["date0"], "heure": e["heure"],
         "objectifs": [{"nom": o["nom"], "cible": o["cible"], "base": o["base"], "t": o["t"], "t_prudent": o["prudent"]["t"],
                        "age_prevu": o["age_prevu"], "f_prevu": o["f_prevu"], "etat": o["etat"]} for o in objectifs],
         "essai": {"t": essai["age"], "f": essai["f_cyl"]} if essai else None}
    if synthese["prudent_distinct"]:
        tep = [_te(p, t, True) for t in ts]
        c["fcm_prudent"] = [L.fcm_t(x, fck, s) for x in tep]
        c["fck_prudent"] = [L.fck_t(x, fck, s) for x in tep]
    return c


# ------------------------------------------------------------------ le sélecteur de classe
def candidats_classes(coulage: dict) -> list:
    """Chaque classe de béton (C16/20 à C50/60), avec le MÊME durcissement (le s retenu du coulage, calé ou non) et les
    mêmes températures, face au PREMIER objectif : l'âge où il est atteint (None : non atteignable, ou jamais), son taux
    à l'âge prévu s'il est daté ; ``retenue`` : la classe du coulage ; ``optimal`` : la plus petite classe qui tient l'âge
    prévu (aucune sans âge prévu : il n'y a alors rien à tenir)."""
    e = EN.normaliser(coulage)
    _avert, _fck, _dur, p, _essai_, s = _base(e)
    date = e["objectifs"][0]["age"] is not None
    out = []
    for classe, (fck, _cube) in L.CLASSES.items():
        o = _objectif(0, e["objectifs"][0], e, p, fck, s)
        out.append({"classe": classe, "fck": fck, "cible": o["cible"], "base": o["base"], "t": o["t"], "t_T": o["t_T"],
                    "atteignable": o["atteignable"], "instant": o["instant"], "indicatif": o["indicatif"], "prudent": o["prudent"],
                    "age_prevu": o["age_prevu"], "f_prevu": o["f_prevu"], "taux": o["taux"], "etat": o["etat"],
                    "ok": (o["etat"] in ("ok", "attention")) if date else o["t"] is not None,
                    "retenue": classe == e["classe"], "optimal": False})
    if date:
        for c in out:
            if c["ok"]:
                c["optimal"] = True
                break
    return out


__all__ = ["nouvel_element", "normaliser", "calculer", "candidats_classes", "modele_objectif", "MODELES_OBJECTIF", "NON_COUVERT", "EXPRESSIONS",
           "LUES"]
