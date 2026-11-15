# -*- coding: utf-8 -*-
"""Les DURÉES d'une grue (cahier § 6, « Durées ») :

- mois = mois complets + fraction du mois entamé (du 01/09/2025 au 01/05/2026 = 8,0 mois : la date de fin est le jour où la
  grue n'est plus louée) ;
- jours facturés = mois × jours par mois (20, réglage) ; semaines = mois × semaines par mois (4) ;
- jours du grutier = mois × 20 par défaut, ou le CALENDRIER RÉEL : lundi à vendredi, moins les jours fériés belges, moins
  les congés du bâtiment saisis dans les réglages (sur la période du test T1 : 170 jours contre 160 comptés) ;
- une grue mobile louée à la journée (logique MA) se compte en JOURS (les dates, si on les donne, comptent les jours
  ouvrables, la date de fin comprise)."""
import calendar
import datetime as dt
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

JOURS_PAR_MOIS = 20
SEMAINES_PAR_MOIS = 4


def lire_date(v) -> Optional[dt.date]:
    """« 2025-09-01 » (ou une date) → la date ; ``None`` si illisible."""
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if not isinstance(v, str) or not v.strip():
        return None
    t = v.strip()[:10]
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(t, f).date()
        except ValueError:
            pass
    return None


def plus_mois(d: dt.date, n: int) -> dt.date:
    """La date ``n`` mois après ``d`` (le 31 janvier + 1 mois = le dernier jour de février)."""
    m = d.month - 1 + n
    a, m = d.year + m // 12, m % 12 + 1
    return dt.date(a, m, min(d.day, calendar.monthrange(a, m)[1]))


def mois_entre(debut: dt.date, fin: dt.date) -> float:
    """Mois complets + fraction du mois entamé, du début (compris) à la fin : la fraction est la part des jours du mois
    entamé (du 01/09 au 16/10 : 1 mois + 15/31)."""
    if fin <= debut:
        return 0.0
    n = 0
    while plus_mois(debut, n + 1) <= fin:
        n += 1
    a, b = plus_mois(debut, n), plus_mois(debut, n + 1)
    return n + (fin - a).days / (b - a).days


def paques(annee: int) -> dt.date:
    """Le dimanche de Pâques (calendrier grégorien, méthode de Meeus)."""
    a, b, c = annee % 19, annee // 100, annee % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mois = (h + l - 7 * m + 114) // 31
    jour = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(annee, mois, jour)


def feries_belges(annee: int) -> Set[dt.date]:
    """Les dix jours fériés légaux belges : Nouvel An, lundi de Pâques, 1er mai, Ascension, lundi de Pentecôte, 21 juillet,
    15 août, Toussaint, 11 novembre, Noël."""
    p = paques(annee)
    return {dt.date(annee, 1, 1), p + dt.timedelta(days=1), dt.date(annee, 5, 1), p + dt.timedelta(days=39),
            p + dt.timedelta(days=50), dt.date(annee, 7, 21), dt.date(annee, 8, 15), dt.date(annee, 11, 1),
            dt.date(annee, 11, 11), dt.date(annee, 12, 25)}


def _conges(conges: Iterable) -> List[Tuple[dt.date, dt.date]]:
    out = []
    for c in conges or ():
        try:
            a, b = lire_date(c[0]), lire_date(c[1])
        except (TypeError, IndexError, KeyError):
            continue
        if a and b and b >= a:
            out.append((a, b))
    return out


def jours_ouvrables(debut: dt.date, fin: dt.date, conges: Iterable = (), fin_comprise: bool = False) -> int:
    """Les jours du lundi au vendredi du début (compris) à la fin (exclue, sauf ``fin_comprise``), moins les jours fériés
    belges et les congés du bâtiment (périodes [du, au], bornes comprises)."""
    if fin_comprise:
        fin = fin + dt.timedelta(days=1)
    if fin <= debut:
        return 0
    feries: Set[dt.date] = set()
    for a in range(debut.year, fin.year + 1):
        feries |= feries_belges(a)
    cg = _conges(conges)
    n, d = 0, debut
    while d < fin:
        if d.weekday() < 5 and d not in feries and not any(a <= d <= b for a, b in cg):
            n += 1
        d += dt.timedelta(days=1)
    return n


def conges_de(reglages: Dict) -> List:
    """Les congés du bâtiment de toutes les années saisies (réglage ``conges`` : {année: [[du, au], …]})."""
    out = []
    for v in (reglages.get("conges") or {}).values():
        out += list(v or [])
    return out


def _nombre(v, defaut=None):
    try:
        x = float(v)
        return x if x == x else defaut
    except (TypeError, ValueError):
        return defaut


def periode(saisie: Dict, reglages: Dict, logique: str, jours_par_mois: Optional[float] = None) -> Dict:
    """La période d'une grue : {mode, debut, fin, mois, jours, semaines, jours_grutier, grutier_reel, annees, texte}.

    ``saisie`` : {"mode": "dates" | "mois" | "jours", "debut", "fin", "mois", "jours"} ; ``reglages`` : les réglages de la
    base (``base.reglages``) ; ``jours_par_mois`` : celui du loueur, s'il en a un. Une logique MA compte des jours."""
    jpm = jours_par_mois or _nombre(reglages.get("jours_par_mois"), JOURS_PAR_MOIS)
    spm = _nombre(reglages.get("semaines_par_mois"), SEMAINES_PAR_MOIS)
    jpm_g = _nombre(reglages.get("jours_par_mois_grutier"), JOURS_PAR_MOIS)
    reel = reglages.get("jours_grutier") == "calendrier réel"
    mode = saisie.get("mode") or ("jours" if logique == "MA" else "dates")
    debut, fin = lire_date(saisie.get("debut")), lire_date(saisie.get("fin"))
    out = {"mode": mode, "debut": debut.isoformat() if debut else None, "fin": fin.isoformat() if fin else None,
           "mois": None, "jours": None, "semaines": None, "jours_grutier": None, "grutier_reel": False, "annees": [],
           "jours_par_mois": jpm, "semaines_par_mois": spm}
    if logique == "MA":
        if mode == "dates" and debut and fin:
            j = jours_ouvrables(debut, fin, conges_de(reglages), fin_comprise=True)
        else:
            j = _nombre(saisie.get("jours"), None)
        j = max(0.0, j or 0.0)
        out.update(jours=j, jours_grutier=j, mois=None, semaines=j / 5.0 if j else 0.0)
    else:
        if mode == "dates" and debut and fin:
            m = mois_entre(debut, fin)
        elif mode == "jours":
            j = _nombre(saisie.get("jours"), 0.0) or 0.0
            m = j / jpm if jpm else 0.0
        else:
            m = _nombre(saisie.get("mois"), 0.0) or 0.0
        m = max(0.0, m)
        jours = m * jpm if mode != "jours" else (_nombre(saisie.get("jours"), 0.0) or 0.0)
        out.update(mois=m, jours=jours, semaines=m * spm, jours_grutier=m * jpm_g)
        if reel and debut and fin and mode == "dates":
            out.update(jours_grutier=float(jours_ouvrables(debut, fin, conges_de(reglages))), grutier_reel=True)
    if debut and fin:
        out["annees"] = list(range(debut.year, (fin - dt.timedelta(days=1)).year + 1)) if fin > debut else [debut.year]
    return out
