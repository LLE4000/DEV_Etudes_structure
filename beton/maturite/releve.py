# -*- coding: utf-8 -*-
"""Le RELEVÉ de températures journalières d'un coulage : sa lecture (un tableau collé ou un fichier texte), son
écriture, et ce qu'il contient (``synthese``).

Les températures arrivent de trois façons : une constante ; ce relevé, collé ou lu dans un fichier ; la météo du lieu
(``meteo.py`` lit les réponses d'Open-Meteo et les écrit en relevé, que l'on peut ensuite corriger à la main). Le calcul
reste hors ligne, sur le relevé que la note garde. C'est le moteur (``moteur.py``) qui pose le relevé sur le calendrier
du coulage — sa première ligne est le JOUR du bétonnage — et qui le prolonge.

Format accepté (une ligne par jour, du jour du bétonnage au dernier jour) :

    18,5                                  la température moyenne du jour (°C), virgule ou point
    2026-11-02 18,5                       précédée de sa date (AAAA-MM-JJ ou JJ/MM/AAAA)
    02/11/2026;18,5;prévision             suivie de sa source : mesuré (défaut), prévision, prolongé
    Date;T moy (°C);Source                une ligne d'en-tête (la première) est ignorée

Séparateurs : point-virgule, tabulation ou espaces (une virgule seule sépare aussi les colonnes d'un CSV dont les
nombres ont un point). Une température « — », « nan » ou vide = jour sans donnée. Les dates, si elles sont données,
le sont à chaque ligne et se suivent jour après jour. Le tableau « Données météo » de l'écran Streamlit (Date, T moy,
Source, T min h, T max h) se colle tel quel.

Bibliothèque standard seule.
"""
import datetime as _dt
import math
import re
import unicodedata

MAX_JOURS = 400
MAX_TEXTE = 200_000
SOURCES = ("mesure", "prevision", "prolonge")
_MOTS = {"mesure": "mesure", "mesuree": "mesure", "mes": "mesure", "m": "mesure", "obs": "mesure", "observe": "mesure",
         "prevision": "prevision", "prevu": "prevision", "prevue": "prevision", "prev": "prevision", "p": "prevision",
         "prolonge": "prolonge", "prolongee": "prolonge", "prol": "prolonge", "x": "prolonge", "extrapole": "prolonge"}
_VIDE = {"", "—", "–", "-", "nan", "na", "n/a", "none", "null"}
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_FR = re.compile(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{4})$")
_DATEE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")       # ce qui a la forme d'une date (valide ou non)


def _sans_accent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def lire_date(s: str):
    """« 2026-11-02 » ou « 02/11/2026 » → date ; None si ce n'est pas une date."""
    m = _ISO.match(s) or _FR.match(s)
    if not m:
        return None
    a, b, c = (int(x) for x in m.groups())
    try:
        return _dt.date(a, b, c) if _ISO.match(s) else _dt.date(c, b, a)
    except ValueError:
        return None


def _nombre(s: str):
    """Un nombre écrit à la française ou à l'anglaise ; None sinon."""
    t = s.replace(" ", "").replace("\xa0", "").replace(",", ".")
    try:
        x = float(t)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _morceaux(ligne: str) -> list:
    if ";" in ligne or "\t" in ligne:
        return [m.strip() for m in re.split(r"[;\t]", ligne)]
    mots = ligne.split()
    if len(mots) == 1 and "," in mots[0] and _nombre(mots[0]) is None:
        return [m.strip() for m in mots[0].split(",")]           # un CSV à virgules (nombres à point)
    return mots


def lire(texte: str) -> dict:
    """Le relevé collé → {"dates": [AAAA-MM-JJ…] | None, "temps": [°C | nan…], "prev": [bool…], "extra": [bool…]}.
    Un texte vide rend un relevé vide ; une ligne illisible lève ValueError (le numéro de la ligne et ce qu'elle porte)."""
    if not isinstance(texte, str):
        raise ValueError("le relevé doit être un texte")
    if len(texte) > MAX_TEXTE:
        raise ValueError(f"relevé trop long (plus de {MAX_TEXTE // 1000} ko)")
    jours = []
    lignes = [(i + 1, x.strip()) for i, x in enumerate(texte.splitlines())]
    lignes = [(n, x) for n, x in lignes if x and not x.startswith("#")]
    for k, (n, ligne) in enumerate(lignes):
        mots = _morceaux(ligne)
        if k == 0 and len(lignes) > 1 and not any(_nombre(m) is not None or _DATEE.match(m) for m in mots):
            continue                                    # la ligne d'en-tête du tableau (ni nombre ni date)
        if mots and _DATEE.match(mots[0]) and lire_date(mots[0]) is None:
            raise ValueError(f"ligne {n} : date impossible « {mots[0]} »")
        date = lire_date(mots[0]) if mots else None
        reste = mots[1:] if date else mots
        if not reste:
            raise ValueError(f"ligne {n} : température manquante (« {ligne} »)")
        brut = reste[0]
        t = _nombre(brut)
        if t is None and _sans_accent(brut) not in _VIDE:
            raise ValueError(f"ligne {n} : température illisible « {brut} »")
        source = "mesure"
        apres = reste[1:]
        if apres and apres[0] != "":
            mot = _sans_accent(apres[0]).strip(".")
            if mot in _MOTS:
                source = _MOTS[mot]
                apres = apres[1:]
            elif _nombre(apres[0]) is not None or _sans_accent(apres[0]) in _VIDE:
                raise ValueError(f"ligne {n} : une seule température par jour (« {ligne} ») — la source se dit "
                                 "« mesuré », « prévision » ou « prolongé »")
            else:
                raise ValueError(f"ligne {n} : source inconnue « {apres[0]} » (mesuré, prévision ou prolongé)")
        elif apres:
            apres = apres[1:]
        # au-delà de la source : les colonnes « T min h », « T max h » du tableau de l'écran Streamlit (ignorées)
        if any(x and _nombre(x) is None and _sans_accent(x) not in _VIDE for x in apres):
            raise ValueError(f"ligne {n} : colonnes en trop « {ligne} »")
        if t is not None and not -60.0 <= t <= 70.0:
            raise ValueError(f"ligne {n} : {brut} °C n'est pas une température de cure plausible")
        jours.append((n, date, float("nan") if t is None else t, source))
    if len(jours) > MAX_JOURS:
        raise ValueError(f"au plus {MAX_JOURS} jours ({len(jours)} lignes)")
    datees = [j for j in jours if j[1] is not None]
    if datees and len(datees) != len(jours):
        sans = next(j for j in jours if j[1] is None)
        raise ValueError(f"ligne {sans[0]} : sans date, alors que d'autres lignes en ont une")
    for a, b in zip(datees, datees[1:]):
        if (b[1] - a[1]).days != 1:
            raise ValueError(f"ligne {b[0]} : le {b[1].strftime('%d/%m/%Y')} ne suit pas le {a[1].strftime('%d/%m/%Y')} "
                             "(un jour par ligne, sans trou)")
    return {"dates": [j[1].isoformat() for j in jours] if datees else None,
            "temps": [j[2] for j in jours],
            "prev": [j[3] != "mesure" for j in jours],
            "extra": [j[3] == "prolonge" for j in jours]}


def ecrire(rel: dict) -> str:
    """Le relevé réécrit dans le format qu'il accepte (une ligne par jour) — lire(ecrire(r)) = r."""
    out = []
    for i, t in enumerate(rel["temps"]):
        src = "prolongé" if rel["extra"][i] else ("prévision" if rel["prev"][i] else "mesuré")
        tt = "—" if math.isnan(t) else repr(float(t)).replace(".", ",")
        d = rel["dates"][i] if rel.get("dates") else ""
        out.append(";".join(x for x in (d, tt, src) if x))
    return "\n".join(out)


def synthese(rel: dict):
    """Ce que le relevé CONTIENT, tel qu'il est écrit : {"n_jours", "n_mes", "n_prev", "n_extra", "n_nan" (jours sans
    donnée), "T_moy", "T_min", "T_max" (des jours qui portent une température)} ; None si aucun jour n'en porte.
    (Le calcul, lui, se fait sur le PROFIL que le moteur en tire : aligné sur le jour du bétonnage, prolongé.)"""
    temps = rel["temps"]
    prev = rel.get("prev", [False] * len(temps))
    extra = rel.get("extra", [False] * len(temps))
    valides = [t for t in temps if not math.isnan(t)]
    if not valides:
        return None
    n_extra = sum(1 for x in extra if x)
    n_prev = sum(1 for p, x in zip(prev, extra) if p and not x)
    return {"n_jours": len(temps), "n_mes": len(temps) - n_prev - n_extra, "n_prev": n_prev, "n_extra": n_extra,
            "n_nan": len(temps) - len(valides), "T_moy": sum(valides) / len(valides), "T_min": float(min(valides)),
            "T_max": float(max(valides))}
