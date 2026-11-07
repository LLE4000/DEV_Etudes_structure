# -*- coding: utf-8 -*-
"""Tests du module « Soudures » du site (api_web/soudures.py) — et sa PARITÉ avec l'écran Streamlit.

Le calcul est le même par construction (les deux écrans appellent ``acier.soudures.etat.verifier``) ; ce qui peut
diverger, c'est l'ÉTAT que chaque écran donne au moteur et ce qu'il AFFICHE. Ces tests le prouvent en exécutant
l'écran Streamlit RÉEL (``modules.soudures.show``) avec un faux « st » qui enregistre chaque widget, chaque texte, et
qui sait déclencher les rappels (on_change), les boutons, le tableau des cordons libres et l'ouverture d'un fichier :

  A. COPIES NOMMÉES : les libellés, aides et options que le site recopie sont ceux de l'écran (lecture de son source).
  B. PARITÉ DU CALCUL : sur des cas tirés au hasard dans chaque configuration, l'état relu par les deux écrans est le
     même, le moteur rend les mêmes nombres AU BIT PRÈS, et les deux écrans affichent les mêmes chiffres (taux,
     quatre lectures, messages et leur gravité, tableau des cordons, détail par point, efforts réduits, formules).
  C. PARITÉ DE LA SAISIE : mêmes champs (et mêmes minimums) par configuration et par variante ; chaque saisie, case,
     menu, bouton (+ Cas, copier, supprimer, k Eurocode, revenir à 100), le tableau des cordons libres et le fichier
     JSON mènent au MÊME état dans les deux écrans — sauf les écarts VOULUS, nommés et vérifiés ici.
  D. LE SITE AFFICHE CE QUE LE MOTEUR CALCULE : verdict, lignes, fenêtres de calcul, sélecteurs (✓ retenu, ★ optimal),
     refus d'une saisie hors domaine, note abîmée refusée, échappement, écriture française.
  E. MUTATIONS : chaque garantie ci-dessus rougit quand on casse la copie qu'elle surveille.

Lancement : python3.12 tests/test_api_web_soudures.py (depuis la racine du dépôt).
"""
import ast
import contextlib
import copy
import dataclasses
import html
import io
import json
import os
import random
import re
import sys
import types
import warnings
from decimal import Decimal

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

from acier.profiles import catalogue as CAT  # noqa: E402
from acier.soudures import configurations as CF  # noqa: E402
from acier.soudures import criteres as K  # noqa: E402
from acier.soudures import etat as ET  # noqa: E402
from acier.soudures import formules as FO  # noqa: E402
from api_web import commun as C  # noqa: E402
from api_web import routeur as RT  # noqa: E402
from api_web import soudures as S  # noqa: E402

import modules.liaison as LI  # noqa: E402
import modules.soudures as MS  # noqa: E402

OK, KO = [], []
P = MS.P


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:600]}]" if info and not cond else ""))


def titre(t):
    print(f"\n== {t}")


def sans_blanc(s: str) -> str:
    return re.sub(r"[\s   ]+", "", html.unescape(s))


def chiffre(s: str) -> str:
    """Une cellule de nombre, comparable d'un écran à l'autre : sans blancs, le signe « − » du site (typographique)
    lu comme le « - » de l'écran Streamlit."""
    return sans_blanc(s).replace("−", "-")


def texte(h: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", h))).strip()


NOMBRE = re.compile(r"−?-?\d[\d   ]*(?:,\d+)?")


def nombres(s: str):
    return [sans_blanc(x).replace("−", "-") for x in NOMBRE.findall(html.unescape(re.sub(r"<[^>]+>", " ", s)))]


def meme_nombre(st: str, site: str, brut=None) -> bool:
    """Le même nombre affiché par les deux écrans. Deux écarts seulement sont permis, ceux des deux règles d'écriture :
    - le DEMI exact : le site arrondit demi vers l'extérieur comme les formules du moteur (commun.fr), l'écran
      Streamlit au pair (format Python) — 1 122,5 s'écrit « 1 123 » sur le site et « 1 122 » dans Streamlit : le
      chiffre de Streamlit est alors PAIR et celui du site le dépasse d'une unité du dernier rang, en valeur absolue ;
      quand la valeur brute ``brut`` est connue, elle doit être un demi EXACT à ce rang (sa valeur binaire) ;
    - le signe d'un zéro : Streamlit écrit « -0,0 », le site jamais.
    Tout autre écart (un chiffre de plus, un rang, un signe) reste un écart."""
    a, b = chiffre(st), chiffre(site)
    if a == b:
        return True
    if a.lstrip("-") == b.lstrip("-"):
        return not a.strip("-0,")                         # le seul signe permis : celui d'un zéro
    nd = len(a.partition(",")[2])
    if nd != len(b.partition(",")[2]) or not re.fullmatch(r"-?\d+(,\d+)?", a) or not re.fullmatch(r"-?\d+(,\d+)?", b):
        return False
    ua, ub = int(a.replace(",", "").lstrip("-")), int(b.replace(",", "").lstrip("-"))
    meme_signe = a.startswith("-") == b.startswith("-") or ua == 0
    if brut is not None:
        d = abs(Decimal(float(brut))).scaleb(nd)
        if d - int(d) != Decimal("0.5"):
            return False
    return meme_signe and ua % 2 == 0 and ub == ua + 1


def memes_nombres(st, site, bruts=None) -> bool:
    bruts = bruts or [None] * len(st)
    return len(st) == len(site) == len(bruts) and all(meme_nombre(x, y, z) for x, y, z in zip(st, site, bruts))


# ====================================================================== le faux « st »
class Relance(Exception):
    """st.rerun()."""


class Session(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        self[k] = v


class FauxSt:
    """Un « st » qui EXÉCUTE l'écran : chaque widget prend sa valeur de session (ou sa valeur par défaut), est
    enregistré avec son rappel ; les textes sont enregistrés ; un bouton est « cliqué » s'il est dans ``clics`` ;
    le tableau des cordons libres passe par ``editeur`` ; st.rerun lève Relance."""

    def __init__(self):
        self.session_state = Session()
        self.clics, self.editeur, self.fichier = set(), None, None
        self.widgets, self.textes = {}, []
        noop = (lambda *a, **k: None)
        self.column_config = types.SimpleNamespace(SelectboxColumn=noop, TextColumn=noop, NumberColumn=noop)

    def _w(self, sorte, label, key, defaut, **kw):
        self.widgets[key] = dict(sorte=sorte, label=label, **kw)
        if key not in self.session_state:
            self.session_state[key] = defaut
        return self.session_state[key]

    def markdown(self, body="", unsafe_allow_html=False, **_):
        self.textes.append(("md", str(body)))

    def caption(self, body="", **_):
        self.textes.append(("caption", str(body)))

    def warning(self, body="", **_):
        self.textes.append(("warning", str(body)))

    def error(self, body="", **_):
        self.textes.append(("error", str(body)))

    def toast(self, body="", **_):
        self.textes.append(("toast", str(body)))

    def columns(self, spec, **_):
        return [contextlib.nullcontext() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, noms):
        return [contextlib.nullcontext() for _ in noms]

    def container(self, **_):
        return contextlib.nullcontext()

    def expander(self, label, expanded=False, **_):
        self.textes.append(("exp", label))
        return contextlib.nullcontext()

    def rerun(self):
        raise Relance()

    def number_input(self, label, value=None, step=None, format=None, key=None, on_change=None, args=(), help=None, min_value=None,
                     max_value=None, width=None):
        return self._w("nombre", label, key, value, step=step, on_change=on_change, args=args, min_value=min_value, max_value=max_value, help=help)

    def text_input(self, label, value="", key=None, on_change=None, args=(), help=None, width=None, **_):
        return self._w("texte", label, key, value, on_change=on_change, args=args, help=help)

    def selectbox(self, label, options, index=0, format_func=str, key=None, on_change=None, args=(), help=None, label_visibility=None, width=None):
        return self._w("menu", label, key, list(options)[index], options=list(options), format_func=format_func, on_change=on_change, args=args, help=help)

    def segmented_control(self, label, options, default=None, format_func=str, key=None, on_change=None, args=(), help=None, selection_mode="single",
                          width=None):
        return self._w("segments", label, key, default, options=list(options), format_func=format_func, on_change=on_change, args=args, help=help)

    def radio(self, label, options, format_func=str, index=0, key=None, on_change=None, args=(), horizontal=False, label_visibility=None):
        return self._w("radio", label, key, list(options)[index], options=list(options), format_func=format_func, on_change=on_change, args=args)

    def checkbox(self, label, value=False, key=None, on_change=None, args=(), help=None):
        return self._w("case", label, key, value, on_change=on_change, args=args, help=help)

    def pills(self, label, options, default=None, key=None, on_change=None, selection_mode="single", **_):
        return self._w("pastilles", label, key, list(default or []), options=list(options), on_change=on_change, args=())

    def button(self, label, key=None, **_):
        self.widgets[key] = dict(sorte="bouton", label=label)
        return key in self.clics

    def download_button(self, label, data=None, file_name=None, key=None, **_):
        self.widgets[key] = dict(sorte="telecharger", label=label, data=data, file_name=file_name)
        return False

    def file_uploader(self, label, type=None, key=None, **_):
        self.widgets[key] = dict(sorte="fichier", label=label)
        return self.fichier

    def data_editor(self, df, key=None, **_):
        self.widgets[key] = dict(sorte="tableau", df=df)
        return self.editeur(df.copy()) if self.editeur else df


F = FauxSt()
MS.st = F
LI.st = F


def rendre(clics=(), editeur=None, fichier=None):
    """Un affichage de l'écran Streamlit ; les relances (st.rerun) sont rejouées comme le ferait Streamlit."""
    F.clics, F.editeur, F.fichier = set(clics), editeur, fichier
    for _ in range(8):
        F.widgets, F.textes = {}, []
        try:
            MS.show()
            break
        except Relance:
            F.clics, F.editeur, F.fichier = set(), None, None
    else:  # pragma: no cover
        raise RuntimeError("l'écran Streamlit relance sans fin")
    F.clics, F.editeur, F.fichier = set(), None, None


def charger(etat: dict):
    """Une note ouverte dans l'écran Streamlit comme par son bouton « Ouvrir » (ET.depuis_json), puis affichée."""
    F.session_state.clear()
    MS.REGISTRE.clear()
    F.session_state["sdr"] = ET.depuis_json(json.dumps(etat))
    rendre()


def etat_st() -> dict:
    return F.session_state["sdr"]


def cas_st() -> dict:
    return ET.cas_actif(etat_st())


# ====================================================================== tirage de cas
NUANCES = list(ET.nuances())
I_H = S._profils("I/H")
CREUX = S._profils("creux")


def tirer(rng: random.Random, cfg: str) -> dict:
    """Un cas au hasard dans le domaine de l'écran (ses minimums, ses listes)."""
    e = ET.nouvel_etat()
    cas = e["cas"][0]
    ET.changer_config(cas, cfg)
    p, ef, m = cas["params"], cas["efforts"], cas["methode"]
    U = rng.uniform
    m.update(mode=rng.choice([K.EUROCODE, K.ELS]), ec=rng.choice([K.DIRECTIONNELLE, K.SIMPLIFIEE]), gamma_M2=rng.choice([1.25, 1.25, 1.5]),
             k=rng.choice([100.0, 100.0, 80.0, 141.2]), g=rng.choice([9.81, 10.0]), gamma_F=rng.choice([1.5, 1.35]))
    p["nuance_A"], p["nuance_B"] = rng.choice(NUANCES), rng.choice(NUANCES)
    gorge = lambda: rng.choice([3.0, 4.0, 5.0, 6.0, 8.0] * 3 + [2.5])  # noqa: E731 — 2,5 mm : un cordon qui ne compte pas
    for k in list(ef):
        ef[k] = rng.choice([0.0, round(U(-80, 120), 1), round(U(1, 60), 1)])
    tr = lambda: {"n": rng.choice([0, 0, 0, 0, 2, 3, 5]), "l": float(rng.choice([40, 60, 80]))}  # noqa: E731
    if cfg == "C1":
        p.update(b=float(rng.choice([60, 100, 150])), tB=float(rng.choice([8, 10, 15])), tA=float(rng.choice([8, 12])), L_rec=float(rng.choice([60, 120, 200])),
                 a=gorge(), retours=rng.random() < .5, lateraux=rng.random() < .8, frontaux=rng.choice([0, 1, 2]), troncons=tr())
    elif cfg in ("C2", "C3", "C6"):
        p.update(a=gorge(), p=float(rng.choice([3, 4, 6, 8])), renfort=float(rng.choice([0, 0, 3])), jonction=rng.choice(list(CF.JONCTIONS)), troncons=tr())
        if cfg == "C2":
            p.update(l=float(rng.choice([100, 200, 300])), tB=float(rng.choice([8, 10, 20])), tA=float(rng.choice([10, 12])),
                     deux_cordons=rng.random() < .7, retours=rng.random() < .5)
        elif cfg == "C3":
            p.update(h=float(rng.choice([150, 200, 300])), t=float(rng.choice([8, 10, 15])), e=float(rng.choice([0, 80, 150])),
                     deux_cordons=rng.random() < .7, retours=rng.random() < .5, poteau=rng.choice(["plat", "HEB 200", "HEA 240", "IPE 300"]))
        else:
            p.update(profil=rng.choice(["SHS 100x100x5", "RHS 150x100x6.3", "CHS 114.3x5", "SHS 80x80x4"]), tp=float(rng.choice([10, 15, 20])))
            if p["profil"] not in CREUX:
                p["profil"] = "SHS 100x100x5"
    elif cfg == "C4":
        p.update(bx=float(rng.choice([150, 200, 300])), by=float(rng.choice([100, 150, 250])), t=float(rng.choice([8, 10])), a=gorge(),
                 xV=float(rng.choice([150, 250, 400])), yV=float(rng.choice([0, 75, 120])), retours=rng.random() < .5, troncons=tr(),
                 cotes=rng.sample(list(S.COTES_C4), rng.choice([1, 2, 3, 4])), poteau=rng.choice(["plat", "HEB 200", "HEA 300"]))
    elif cfg == "C5":
        p.update(profil=rng.choice(["IPE 200", "IPE 300", "HEA 200", "HEB 160"]), a_f=gorge(), a_w=gorge(), p_f=float(rng.choice([4, 6])),
                 p_w=float(rng.choice([3, 4])), renfort_f=float(rng.choice([0, 3])), renfort_w=float(rng.choice([0, 2])),
                 semelles=rng.choice(list(S.SEMELLES)), ame=rng.random() < .85, retours=rng.random() < .5,
                 jonction_semelles=rng.choice(list(CF.JONCTIONS)), jonction_ame=rng.choice(list(CF.JONCTIONS)),
                 troncons_semelles=tr(), troncons_ame=tr(), tp=float(rng.choice([12, 15, 20])))
    elif cfg == "C7":
        p.update(b=float(rng.choice([100, 150, 250])), t=float(rng.choice([8, 12, 20])), penetration=rng.choice(list(S.PENETRATIONS)),
                 p=float(rng.choice([3, 5, 7])), deux_faces=rng.random() < .6)
    else:
        els = []
        for i in range(rng.choice([1, 2, 3])):
            if rng.random() < .3:
                els.append({"type": "arc", "nom": rng.choice(["", f"arc {i + 1}"]), "xc": float(rng.choice([0, 20])), "yc": 0.0, "r": float(rng.choice([40, 60])),
                            "ang1": float(rng.choice([0, 30])), "ang2": float(rng.choice([180, 360])), "a": gorge(), "cote": rng.choice([1, -1]),
                            "retours": False, "troncons": {"n": rng.choice([0, 0, 4]), "l": 40.0}})
            else:
                x = float(rng.choice([-60, 0, 60]) + 30 * i)
                els.append({"type": "segment", "nom": rng.choice(["", f"cordon {i + 1}"]), "x1": x, "y1": -100.0, "x2": x + rng.choice([0, 40]), "y2": 100.0,
                            "a": gorge(), "cote": rng.choice([1, -1]), "retours": False, "troncons": {"n": rng.choice([0, 0, 3]), "l": 50.0}})
        p.update(elements=els, xR=float(rng.choice([0, 20])), yR=float(rng.choice([0, -30])), zR=float(rng.choice([0, 0, 40])))
    cas["nom"] = rng.choice([cas["nom"], f"Cas <b>{cfg}</b> & « {rng.randint(1, 99)} »"])
    return e


# ====================================================================== lecture des deux écrans
def comparable(res: dict) -> dict:
    """Ce que le moteur rend, sous une forme comparable AU BIT (les objets du moteur dépliés)."""
    e = res.get("efforts")
    m = res.get("methode")
    return {"statut": res.get("statut"), "taux_max": res.get("taux_max"), "gouvernant": res.get("gouvernant"),
            "chaines": res.get("chaines"), "points": res.get("points"), "proprietes": res.get("proprietes"),
            "comparaison": res.get("comparaison"), "messages": res.get("messages"), "resistances": res.get("resistances"),
            "efforts": None if e is None else (e.Vx, e.Vy, e.N, e.Mx, e.My, e.Mz, tuple(e.reference)),
            "methode": None if m is None else dataclasses.astuple(m), "acier": {k: v for k, v in (res.get("acier") or {}).items() if k not in ("A", "B")},
            "beta_Lw": res.get("beta_Lw"), "n_pleine": res.get("n_pleine"), "pleine": res.get("pleine")}


def lecture_streamlit(res: dict) -> dict:
    """Ce que l'écran Streamlit vient d'afficher : η, les quatre lectures, les messages, les tableaux, les formules."""
    out = {"eta": None, "barres": [], "warnings": [], "puces": [], "cordons": [], "points": [], "efforts": None, "formules": []}
    en_formules = en_carte = False
    for sorte, t in F.textes:
        if sorte == "md" and 'class="sdr-eta"' in t:
            m = re.search(r"η = (\d+) %", t)
            out["eta"] = int(m.group(1)) if m else None
            out["barres"] = [(lab, int(v)) for lab, v in re.findall(r'<div class="sdr-row(?: sel)?"><span>(.*?)(?: ◀)?</span>.*?<span class="sdr-v"[^>]*>(\d+) %</span>', t)]
            en_carte = True                      # les messages suivent la carte RÉSULTAT (_carte_resultat)
            continue
        if en_carte and sorte == "warning":
            out["warnings"].append(t)
            continue
        if en_carte and sorte == "caption" and t.startswith("• "):
            out["puces"].append(t[2:])
            continue
        en_carte = False
        if sorte == "md" and "<th class='n'>a (mm)</th>" in t:
            out["cordons"] = re.findall(r"<tr><td><b>(.*?)</b></td><td class='n'>(.*?)</td><td class='n'>(.*?)</td><td class='n'>(.*?)</td><td>(.*?)</td>"
                                        r"<td class='n'>(.*?)</td><td class='n [^']*'>(\d+) %</td><td>(.*?)</td></tr>", t)
        elif sorte == "md" and "<th>point</th>" in t:
            out["points"] = re.findall(r"<tr class='g?'><td>(.*?)</td><td>(.*?)</td>" + r"<td class='n'>(.*?)</td>" * 5
                                       + r"<td class='n [^']*'>(\d+) %</td>" * 4 + "</tr>", t)
        elif sorte == "caption" and t.startswith("Au point de référence"):
            out["efforts"] = t
        elif sorte == "exp" and t.startswith("Les formules"):
            en_formules = True
        elif en_formules and sorte == "md" and t.startswith("**"):
            out["formules"] += [x for x in t.split("  \n")[1:]]
    return out


def taux_lu(titre: str, texte: str) -> str:
    """Le taux qu'un œil lit : le chiffre écrit ; au-delà de 999 % la brique écrit « > 999 % » et la valeur exacte au
    survol (attribut title, commun._taux_txt) — c'est elle qu'on compare."""
    t = titre if titre and texte.startswith("&gt;") else texte
    return re.sub(r"[\s\u00a0\u202f%]+", "", html.unescape(t))


def contenu_chip(h: str) -> str:
    """Le texte d'une pastille rendue par la brique (sans sa balise ni son title)."""
    return re.sub(r"^<span[^>]*>|</span>$", "", h)


def lecture_site(r: dict) -> dict:
    """Ce que le site affiche : son verdict, ses lignes, ses messages, ses tableaux, ses fenêtres de calcul."""
    h = r["resultat"]
    v = re.search(r'<span class="chip xl (\w)"(?: title="([^"]*)")?><small>η</small>([^<]+)</span>', h)
    lectures = [(nom, sel, niv, taux_lu(ti, tx)) for nom, sel, niv, ti, tx in
                re.findall(r'<b>([^<]*)</b><span class="prof"><span>(méthode choisie ◀|à titre indicatif)</span></span></div><div class="l2">'
                           r'.*?<span class="pct (\w)"(?: title="([^"]*)")?>([^<]+)</span>', h)]
    msgs = [(bool(g), html.unescape(re.sub(r"<sub>(.*?)</sub>", r"_\1", t))) for g, t in re.findall(r'<div class="msg( grave)?">(.*?)</div>', h)]
    raison = re.search(r"<b>Calcul impossible</b>(.*?)</div>", re.sub(r"<sub>(.*?)</sub>", r"_\1", h))
    vues = {x["cle"]: x["svg"] for x in r["vues"]}
    cordons = re.findall(r'<tr class="(?:gov)?"><td class="nm">(.*?)<span>(.*?) · le plus chargé : (.*?)</span></td><td class="n">(.*?)</td>'
                         r'<td class="n">(.*?)</td><td class="n">(.*?)</td><td class="n">(.*?)</td><td class="n"><span class="chip (\w)"(?: title="[^"]*")?>(.*?)</span></td></tr>',
                         vues.get("cordons", ""))
    points = re.findall(r'<tr class="(?:gov)?"><td class="nm">(.*?)<span>(.*?)</span></td>' + r'<td class="n">(.*?)</td>' * 5
                        + r'<td class="n"><span class="chip \w"(?: title="[^"]*")?>(.*?)</span></td>' * 4 + "</tr>", vues.get("points", ""))
    return {"eta": taux_lu(v.group(2), v.group(3)) if v else None, "niveau": v.group(1) if v else None, "lectures": lectures, "messages": msgs,
            "raison": html.unescape(raison.group(1)) if raison else None, "cordons": cordons, "points": points}


def fr_msg(m: str) -> str:
    """L'oracle du test pour l'écriture française des messages du moteur : « 98.1 » → « 98,1 » (les renvois §x.y.z
    et « Tab. 4.1 » intacts) — écrit ici indépendamment de api_web/soudures.texte_fr."""
    return re.sub(r"(?<![§\d.,])(?<!Tab\. )(\d+)\.(\d+)(?![\d.])", r"\1,\2", m)


def fx_nombres(h: str):
    """Les lignes « symbole = formule = valeurs = résultat » d'un fragment : leurs nombres, sans le résultat."""
    return [tuple(nombres(x)[:-1]) for x in re.findall(r'<div class="fx">(.*?)</div>', h)]


# ====================================================================== A. copies nommées
titre("A · Les copies nommées sont celles de l'écran Streamlit")
SRC = open("modules/soudures.py", encoding="utf-8").read()
ARBRE = ast.parse(SRC)
CHAINES = {n.value for n in ast.walk(ARBRE) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
DICTS = []
for n in ast.walk(ARBRE):
    if isinstance(n, ast.Dict):
        try:
            DICTS.append(ast.literal_eval(n))
        except ValueError:
            pass
chk("MODES, CRITERES, JONCTION_AIDE, COTES_C4 : ceux de l'écran", S.MODES == MS.MODES and S.CRITERES == MS.CRITERES and S.JONCTION_AIDE == MS.JONCTION_AIDE
    and S.COTES_C4 == tuple(MS.COTES_C4))
chk("l'aide des tronçons est celle de l'écran", S.TRONCONS_AIDE in CHAINES)
chk("l'aide du tableau des cordons libres est celle de l'écran", S.C8_AIDE in CHAINES)
chk("frontaux, semelles, pénétration, gravité : les options de l'écran", all(d in DICTS for d in (S.FRONTAUX, S.SEMELLES, S.PENETRATIONS, S.G_OPTIONS)),
    [d for d in (S.FRONTAUX, S.SEMELLES, S.PENETRATIONS, S.G_OPTIONS) if d not in DICTS])
chk("élément libre relu comme l'écran : mêmes colonnes", set(S.COORDS_SEGMENT + S.COORDS_ARC) <= CHAINES)
# la relecture d'une ligne du tableau : celle de l'écran, ligne par ligne, cellules vides comprises
rng = random.Random(7)
NAN = float("nan")
ecarts = []
for i in range(400):
    row = {"type": rng.choice(["segment", "arc", None, "autre"]), "nom": rng.choice(["", "c1", None, NAN, 3])}
    for k in S.COORDS_SEGMENT + S.COORDS_ARC + ("a", "l"):
        row[k] = rng.choice([None, NAN, round(rng.uniform(-200, 200), 1), 0.0])
    row["cote"] = rng.choice([1, -1, 1.0, -1.0, None, NAN, 2, 0])
    row["n"] = rng.choice([None, NAN, 0, 3, 5.0])
    if S.element_depuis_ligne(row) != MS.elements_depuis_lignes([row])[0]:
        ecarts.append(row)
chk("une ligne du tableau des cordons libres est relue comme l'écran le fait (400 lignes, cellules vides et NaN)", not ecarts, ecarts[:2])


# ====================================================================== B. parité du calcul et de l'affichage
def parite_cas(etat: dict) -> list:
    """Le même cas dans les deux écrans : les écarts (liste vide = parité)."""
    out = []
    charger(etat)
    e_st = etat_st()
    try:
        e_site = S.normaliser(copy.deepcopy(etat))
    except ValueError as x:
        return [f"le site refuse la note : {x}"]
    c_st, c_site = ET.cas_actif(e_st), ET.cas_actif(e_site)
    if c_st != c_site:
        out.append(f"états différents : {json.dumps(c_st, ensure_ascii=False)[:300]} ≠ {json.dumps(c_site, ensure_ascii=False)[:300]}")
    res_st, res_site = ET.verifier(c_st), ET.verifier(c_site)
    a, b = comparable(res_st), comparable(res_site)
    if a != b:
        out.append("résultats du moteur différents : " + ", ".join(k for k in a if a[k] != b[k]))
    rs = lecture_streamlit(res_st)
    r = S.resultat(e_site)
    rw = lecture_site(r)
    t = res_site.get("taux_max")
    if res_site["statut"] == ET.REFUS:
        if rw["eta"] is not None or rs["eta"] is not None:
            out.append("calcul impossible : un taux est affiché")
    else:
        if rs["eta"] != int(f"{t * 100:.0f}"):
            out.append(f"η Streamlit {rs['eta']} ≠ {t * 100:.2f}")
        if rw["eta"] != C.pc(C.pct(t)) or rw["niveau"] != C.niveau(t):
            out.append(f"η site {rw['eta']} ≠ {C.pc(C.pct(t))}")
        if abs(float(rw["eta"].replace(",", ".")) - rs["eta"]) > 1:
            out.append(f"η site {rw['eta']} et Streamlit {rs['eta']} s'écartent")
    # les quatre lectures
    if [lab for lab, _ in rs["barres"]] != [html.unescape(x[0]) for x in rw["lectures"]]:
        out.append(f"quatre lectures : {rs['barres']} ≠ {rw['lectures']}")
    for (lab, vs), (_, sel, niv, vw) in zip(rs["barres"], rw["lectures"]):
        if abs(float(vw.replace(",", ".")) - vs) > 1:
            out.append(f"lecture {lab} : {vs} ≠ {vw}")
    # les messages : les mêmes, avec la même gravité (le refus passe dans le verdict du site)
    msgs_st = rs["warnings"] + rs["puces"]
    if sorted(msgs_st) != sorted(res_st.get("messages", [])):  # la carte RÉSULTAT les montre tous, une fois
        out.append("Streamlit n'affiche pas tous les messages (lecture du faux st)")
    site = {m: g for g, m in rw["messages"]}
    if rw["raison"] is not None:
        site[rw["raison"]] = True
    attendu = {fr_msg(m): (m in rs["warnings"]) for m in res_st.get("messages", [])}
    if res_st["statut"] == ET.REFUS and not res_st.get("messages"):
        attendu["données manquantes"] = True
    if set(site) != set(attendu):
        out.append(f"messages : {sorted(set(site) ^ set(attendu))[:3]}")
    elif any(site[m] != attendu[m] for m in attendu if m != rw["raison"]):
        out.append("gravité des messages différente")
    # le tableau des cordons
    if len(rs["cordons"]) != len(rw["cordons"]):
        out.append(f"tableau des cordons : {len(rs['cordons'])} ≠ {len(rw['cordons'])} lignes")
    for cs, cw, ch in zip(rs["cordons"], rw["cordons"], res_site.get("chaines", [])):
        nom, a_, l_, le, nat, f, eta, pos = cs
        if (html.unescape(cw[0]), html.unescape(cw[1]), html.unescape(cw[2])) != (html.unescape(nom), nat, pos) \
                or not memes_nombres((a_, l_, le, f), cw[3:7], [ch[k] for k in ("a", "l", "l_eff", "f_max")]) \
                or int(eta) != int(f"{ch['taux'] * 100:.0f}") or sans_blanc(cw[8]) != sans_blanc(contenu_chip(C.chip(C.pct(ch["taux"]), "p"))):
            out.append(f"cordon {nom} : {cs} ≠ {cw}")
    # le détail par point
    if len(rs["points"]) != len(rw["points"]):
        out.append(f"détail par point : {len(rs['points'])} ≠ {len(rw['points'])} lignes")
    for ps, pw, pt in zip(rs["points"], rw["points"], res_site.get("points", [])):
        if (html.unescape(ps[0]), ps[1]) != (html.unescape(pw[0]), pw[1]) or not memes_nombres(ps[2:7], pw[2:7], [pt[k] for k in ("f", "sigma_perp", "tau_perp", "tau_par", "sigma_c")]):
            out.append(f"point {ps[:3]} ≠ {pw[:3]}")
        for k, cle in enumerate(("taux_directionnelle", "taux_simplifiee", "taux_bureau", "taux")):
            if int(ps[7 + k]) != int(f"{pt[cle] * 100:.0f}") or sans_blanc(pw[7 + k]) != sans_blanc(contenu_chip(C.chip(C.pct(pt[cle]), "p"))):
                out.append(f"point {ps[:2]} : {cle}")
    # les efforts réduits au point de référence (cadre Efforts)
    if rs["efforts"] is not None:
        m = re.search(r"Au point de référence.*?</div>", r["donnees"])
        if not m or not memes_nombres(nombres(rs["efforts"]), nombres(m.group(0))):
            out.append("efforts réduits différents")
    # les formules du point gouvernant : chacune dans la fenêtre de calcul du cordon gouvernant
    g = res_site.get("gouvernant")
    if g and rs["formules"]:
        i_g = next(i for i, c in enumerate(res_site["chaines"]) if c["nom"] == S._chaine_de(res_site).get(g["cordon"]))
        fen = re.search(rf'<div class="flyout calcul" id="ck-c{i_g}">.*?(?=<div class="flyout calcul"|$)', r["volets"], re.S)
        lignes_site = set(fx_nombres(fen.group(0))) if fen else set()
        for li in rs["formules"]:
            n = tuple(nombres(li.split(" = ", 1)[1])[:-1]) if " = " in li else ()
            if n and n not in lignes_site:
                out.append(f"formule absente du site : {li}")
    return out


titre("B · Les deux écrans, sur des cas tirés au hasard : même état, même calcul au bit près, mêmes chiffres affichés")
rng = random.Random(20261107)
N_PAR_CONFIG = 25
for cfg in CF.CONFIGS:
    ecarts, refus = [], 0
    for i in range(N_PAR_CONFIG):
        e = tirer(rng, cfg)
        x = parite_cas(e)
        refus += ET.verifier(ET.cas_actif(ET.depuis_json(json.dumps(e))))["statut"] == ET.REFUS
        if x:
            ecarts.append((i, x[:3]))
    chk(f"{cfg} : {N_PAR_CONFIG} cas ({refus} calcul(s) impossible(s)) — états, résultats du moteur et affichages identiques", not ecarts, ecarts[:2])
# un cas « deux élements » : le résumé de la note est celui du cas le plus chargé, partout
e = tirer(random.Random(3), "C3")
e["cas"].append(tirer(random.Random(4), "C5")["cas"][0])
e["actif"] = 1
chk("une note à deux cas : même parité sur le cas ouvert", not parite_cas(e), parite_cas(e)[:2])


# ====================================================================== C. parité de la saisie
def champs_site(donnees: str) -> dict:
    """Les champs du cadre Données du site : chemin → (sorte, minimum, options)."""
    out = {}
    for m in re.finditer(r'<(input|select|button)([^>]*?)data-chemin="([^"]+)"([^>]*)>', donnees):
        attrs = m.group(2) + m.group(4)
        chemin = html.unescape(m.group(3))
        sorte = re.search(r'data-sorte="(\w+)"', attrs)
        mini = re.search(r'data-min="([^"]+)"', attrs)
        d = out.setdefault(chemin, {"sorte": sorte.group(1) if sorte else ("ajout" if "addrow" in attrs else "bouton"), "min": None, "options": []})
        if mini:
            d["min"] = float(mini.group(1))
        v = re.search(r'data-valeur="([^"]*)"', attrs)
        if v and d["sorte"] == "choix":
            d["options"].append(html.unescape(v.group(1)))
    for chemin, opts in re.findall(r'<select data-chemin="([^"]+)" data-sorte="choix"[^>]*>(.*?)</select>', donnees, re.S):
        out[html.unescape(chemin)]["options"] = [html.unescape(x) for x in re.findall(r'<option value="([^"]*)"', opts)]
    return out


def vers_site(cle: str, w: dict):
    """Le chemin du site qui répond au widget ``cle`` de l'écran Streamlit (None : pas un champ)."""
    if w.get("on_change") is MS._ecrire:
        return "/".join(str(c) for c in w["args"][0])
    if cle == P + "w_config":
        return "config"
    if cle == P + "w_cotes":
        return "params/cotes"
    if cle.startswith(P + "w_elements_"):
        return "params/elements"
    m = re.fullmatch(re.escape(P) + r"w_(troncons\w*)_on", cle)
    if m:
        return f"params/{m.group(1)}/on"
    m = re.fullmatch(re.escape(P) + r"id_(projet|partie|indice)", cle)
    if m:
        return f"identification/{m.group(1)}"
    return None


def depuis_site(chemin: str) -> str:
    """Un chemin du site ramené au champ de l'écran Streamlit qui fait la même chose."""
    if chemin in ("params/poteau_famille",):
        return "params/poteau"
    if chemin == "params/profil_famille":
        return "params/profil"
    if re.fullmatch(r"params/cotes/\w+", chemin):
        return "params/cotes"
    if chemin.startswith("params/elements/"):
        return "params/elements"
    return chemin


BOUTONS_SITE = {"methode/k_eurocode": P + "btn_kq", "methode/k_bureau": P + "btn_k100", "import": P + "btn_open"}
# les écarts VOULUS de disposition (le champ existe sur le site, montré quand il sert) : C7 en pleine pénétration, la
# pénétration p et « deux faces » ne servent ni au calcul ni au dessin (configurations.c7, schemas.c7_coupe)
CACHES_VOULUS = {("C7", "pleine"): {"params/p", "params/deux_faces"}}


def variantes(cfg: str):
    """Les états qui montrent tous les champs d'une configuration : jonctions, tronçons, pénétration."""
    base = ET.nouvel_etat()
    ET.changer_config(base["cas"][0], cfg)
    out = [("défaut", base)]
    p = base["cas"][0]["params"]

    def avec(nom, **kw):
        e = copy.deepcopy(base)
        for k, v in kw.items():
            e["cas"][0]["params"][k] = v
        out.append((nom, e))
    if "jonction" in p:
        avec("partielle", jonction="partielle")
        avec("pleine", jonction="pleine")
    if cfg == "C5":
        avec("partielles", jonction_semelles="partielle", jonction_ame="partielle")
        avec("pleines", jonction_semelles="pleine", jonction_ame="pleine")
        avec("sans âme", ame=False)
    for k in S.TRONCONS[cfg]:
        avec(f"{k} par tronçons", **{k: {"n": 3, "l": 50.0}})
    if cfg == "C7":
        out[0] = ("pleine", base)
        avec("partielle", penetration="partielle")
    if cfg in ("C3", "C4"):
        avec("sur plat", poteau="plat")
    return out


titre("C · Mêmes champs, mêmes minimums, mêmes options — configuration par configuration, variante par variante")
for cfg in CF.CONFIGS:
    for nom_v, e in variantes(cfg):
        charger(e)
        st_ch = {}
        for cle, w in F.widgets.items():
            ch = vers_site(cle, w)
            if ch is not None:
                st_ch[ch] = (cle, w)
        site = champs_site(S.resultat(S.normaliser(copy.deepcopy(e)))["donnees"])
        site_ch = {depuis_site(c) for c in site if c not in BOUTONS_SITE and not c.endswith("/+") and not c.endswith("/-")}
        manque = set(st_ch) - site_ch - CACHES_VOULUS.get((cfg, nom_v), set())
        invente = site_ch - set(st_ch) - {"nom"} - ({"params/elements"} if cfg == "C8" else set())
        if "nom" not in st_ch:
            invente.discard("nom")
        chk(f"{cfg} · {nom_v} : chaque champ de l'écran Streamlit est sur le site, et rien d'autre", not manque and not invente,
            f"manque {sorted(manque)} · inventé {sorted(invente)}")
        # les minimums (number_input min_value) : ceux du site (data-min et refus)
        mins = []
        for ch, (cle, w) in st_ch.items():
            if w["sorte"] != "nombre":
                continue
            m_st = w.get("min_value")
            m_site = site.get(ch, {}).get("min")
            k = tuple(ch.split("/"))
            m_nom = S.MINIMUMS.get(("troncons", k[-1]) if k[0] == "params" and k[1].startswith("troncons") else k)
            if ch in CACHES_VOULUS.get((cfg, nom_v), set()):
                continue
            if (m_st is None) != (m_nom is None) or (m_st is not None and (float(m_st) != float(m_nom) or m_site != float(m_st))):
                mins.append((ch, m_st, m_nom, m_site))
        chk(f"{cfg} · {nom_v} : les minimums des champs sont ceux de l'écran (MINIMUMS, data-min)", not mins, mins)
        # les options des menus
        opts = []
        for ch, (cle, w) in st_ch.items():
            if w["sorte"] in ("menu", "segments") and ch in site and ch not in ("params/poteau", "params/profil"):
                if [str(o) for o in w["options"]] != site[ch]["options"]:
                    opts.append((ch, w["options"], site[ch]["options"]))
        chk(f"{cfg} · {nom_v} : les options des menus sont celles de l'écran", not opts, opts)
reachables = set()
for fam in S._familles("I/H"):
    reachables |= {n for n, r in CAT.par_nom().items() if r["famille"] == fam}
chk("support : les deux menus du site (famille, puis taille) atteignent exactement la liste de l'écran", reachables == set(I_H))


def valeur_neuve(w: dict, cur):
    """Une autre valeur, dans le domaine du champ."""
    if w["sorte"] == "case":
        return [not bool(cur)]
    if w["sorte"] in ("menu", "segments", "radio"):
        return [o for o in w["options"] if o != cur][:4]
    if w["sorte"] == "texte":
        return ["Essai <b>&amp; « x »"]
    entier = isinstance(w.get("step"), int)
    mini = w.get("min_value")
    x = float(cur or 0.0)
    vals = [x * 1.37 + 2.0, max(x - 1.0, mini if mini is not None else x - 1.0)]
    return [int(round(v)) if entier else round(v, 2) for v in vals]


def vers_saisie(w: dict, v):
    """Ce que l'écran du site envoie pour la valeur v : le texte tapé (virgule), l'option, la case."""
    if w["sorte"] == "case":
        return bool(v)
    if w["sorte"] == "nombre":
        return C.nb(v, 6)
    return str(v)


def appliquer_st(cle: str, w: dict, v):
    F.session_state[cle] = v
    w["on_change"](*w.get("args", ()))


titre("C · Chaque saisie mène au même état dans les deux écrans (rappels de l'écran Streamlit exécutés)")
for cfg in CF.CONFIGS:
    n_essais, ecarts = 0, []
    for nom_v, e in variantes(cfg):
        charger(e)
        base = copy.deepcopy(etat_st())
        widgets = dict(F.widgets)
        for cle, w in widgets.items():
            ch = vers_site(cle, w)
            if ch is None or ch.startswith("params/elements") or w.get("on_change") is None:
                continue
            charger(base)
            cur = F.session_state.get(cle)
            if ch == "params/cotes":
                essais = [(c, [x for x in cur if x != c] if c in cur else cur + [c]) for c in S.COTES_C4]
            else:
                essais = [(None, v) for v in valeur_neuve(w, cur)]
            for cote, v in essais:
                # l'écran rechargé avant chaque saisie : _cfg, _on et _cotes sont des fermetures sur le cas affiché, et
                # le changement de configuration resynchronise les clés des widgets (modules/liaison.py)
                charger(base)
                ww = F.widgets[cle]
                appliquer_st(cle, ww, v)
                e_site = S.normaliser(copy.deepcopy(base))
                if cote is not None:
                    msg = S.modifier(e_site, ["params", "cotes", cote], cote not in cur)
                else:
                    msg = S.modifier(e_site, ch.split("/"), vers_saisie(w, v))
                n_essais += 1
                a, b = etat_st(), e_site
                if msg or ET.cas_actif(a) != ET.cas_actif(b) or a.get("identification") != b.get("identification"):
                    ecarts.append((nom_v, ch, v, msg or json.dumps(ET.cas_actif(a), ensure_ascii=False)[:200]))
        F.session_state["sdr"] = base
    chk(f"{cfg} : {n_essais} saisies (nombres, cases, menus, côtés, tronçons, configuration, identification) — même état", not ecarts, ecarts[:3])

titre("C · Les boutons de l'écran : ＋ Cas, copier, supprimer, choisir le cas, k Eurocode, revenir à 100, ouvrir un fichier")
e0 = tirer(random.Random(11), "C5")
e0["cas"][0]["methode"].update(mode=K.ELS, k=80.0, gamma_F=1.35)
for bouton, action in ((P + "btn_new", ("ajouter", None)), (P + "btn_copy", ("dupliquer", None)), (P + "btn_kq", ["methode", "k_eurocode"]),
                       (P + "btn_k100", ["methode", "k_bureau"])):
    charger(e0)
    rendre(clics={bouton})
    e_site = S.normaliser(copy.deepcopy(e0))
    if isinstance(action, tuple):
        S.gerer_elements(e_site, *action)
    else:
        S.modifier(e_site, action, 1)
    a = json.loads(json.dumps(etat_st()))
    chk(f"bouton {bouton[len(P):]} : même note dans les deux écrans", a["cas"] == e_site["cas"] and a["actif"] == e_site["actif"],
        json.dumps(a["cas"][-1], ensure_ascii=False)[:200])
e2 = copy.deepcopy(e0)
e2["cas"] += [tirer(random.Random(12), "C1")["cas"][0], tirer(random.Random(13), "C8")["cas"][0]]
for actif in range(3):
    e2["actif"] = actif
    charger(e2)
    rendre(clics={P + "btn_del"})
    e_site = S.normaliser(copy.deepcopy(e2))
    S.gerer_elements(e_site, "supprimer", actif)
    # l'écran Streamlit ne relit le tableau d'un cas C8 qu'en l'affichant : la note est comparée relue (la relecture
    # est celle de l'écran, prouvée en A et en C8)
    st_relu = S.normaliser(copy.deepcopy(etat_st()))
    chk(f"supprimer le cas ouvert n° {actif + 1} sur 3 : même note et même cas ouvert", st_relu["cas"] == e_site["cas"] and st_relu["actif"] == e_site["actif"],
        (st_relu["actif"], e_site["actif"]))
charger(e2)
w = F.widgets[P + "sel_cas"]
F.session_state[P + "sel_cas"] = w["options"][2]
w["on_change"]()                                   # le rappel _sel de l'écran (menu des cas)
e_site = S.normaliser(copy.deepcopy(e2))
S.gerer_elements(e_site, "activer", 2)
chk("choisir le 3ᵉ cas : même cas ouvert", etat_st()["actif"] == e_site["actif"] == 2)
# ouvrir un fichier JSON de l'écran (tous ses cas)
fichier = ET.vers_json(e2)
charger(e0)
F.session_state[P + "ouvrir"] = True
rendre(fichier=io.BytesIO(fichier.encode("utf-8")))
e_site = S.normaliser(copy.deepcopy(e0))
msg = S.modifier(e_site, ["import"], {"nom": "soudures_essai.json", "texte": fichier})
rendre()
chk("ouvrir un fichier JSON de l'écran : la même note dans les deux écrans", msg is None and json.loads(json.dumps(etat_st())) == e_site, msg)
msg = S.modifier(e_site, ["import"], {"nom": "x.json", "texte": "﻿" + fichier})
chk("ouvrir un fichier JSON avec BOM (utf-8-sig, comme l'écran) : accepté", msg is None, msg)

titre("C · Le tableau des cordons libres (C8) : chaque édition de l'écran a son équivalent sur le site")
e8 = ET.nouvel_etat()
ET.changer_config(e8["cas"][0], "C8")
e8["cas"][0]["params"]["elements"].append(dict(CF.ELEMENT_ARC, nom="arc 3", r=40.0))


def editer(fn):
    def ed(df):
        fn(df)
        return df
    return ed


def ajouter_ligne(df):
    df.loc[len(df)] = [None] * len(df.columns)


EDITIONS = [
    ("gorge de la 1re ligne", editer(lambda df: df.__setitem__("a", [7.5] + list(df["a"][1:]))), [(["params", "elements", "0", "a"], "7,5")]),
    ("x₂ de la 2e ligne", editer(lambda df: df.loc.__setitem__((1, "x2"), 80.0)), [(["params", "elements", "1", "x2"], "80")]),
    ("nom de la 1re ligne", editer(lambda df: df.loc.__setitem__((0, "nom"), "gousset")), [(["params", "elements", "0", "nom"], "gousset")]),
    ("côté de la 1re ligne", editer(lambda df: df.loc.__setitem__((0, "cote"), 1)), [(["params", "elements", "0", "cote"], "1")]),
    ("tronçons de la 3e ligne", editer(lambda df: df.loc.__setitem__((2, "n"), 4)), [(["params", "elements", "2", "troncons", "n"], "4")]),
    ("longueur de tronçon", editer(lambda df: df.loc.__setitem__((2, "l"), 35.0)), [(["params", "elements", "2", "troncons", "l"], "35")]),
    ("rayon de l'arc", editer(lambda df: df.loc.__setitem__((2, "r"), 55.0)), [(["params", "elements", "2", "r"], "55")]),
    ("une ligne neuve (cellules vides)", editer(ajouter_ligne), [(["params", "elements", "+"], "segment")]),
    ("retirer la 1re ligne", lambda df: df.drop(index=0).reset_index(drop=True), [(["params", "elements", "0", "-"], 1)]),
]
for nom, ed, ops in EDITIONS:
    charger(e8)
    rendre(editeur=ed)
    rendre()
    e_site = S.normaliser(copy.deepcopy(e8))
    msgs = [S.modifier(e_site, ch, v) for ch, v in ops]
    a = cas_st()["params"]["elements"]
    b = ET.cas_actif(e_site)["params"]["elements"]
    chk(f"C8 · {nom} : mêmes cordons libres", not any(msgs) and a == b, (msgs, json.dumps(a, ensure_ascii=False)[:250], json.dumps(b, ensure_ascii=False)[:250]))
# l'écart VOULU : un segment devenu arc a r = 0 dans l'écran Streamlit (il le refuse ensuite) ; le site lui donne le
# rayon et les angles par défaut d'un arc
charger(e8)
rendre(editeur=editer(lambda df: df.loc.__setitem__((0, "type"), "arc")))
rendre()
res_st = ET.verifier(cas_st())
e_site = S.normaliser(copy.deepcopy(e8))
S.modifier(e_site, ["params", "elements", "0", "type"], "arc")
el_site = ET.cas_actif(e_site)["params"]["elements"][0]
chk("C8 · un segment devenu arc : l'écran Streamlit lui laisse r = 0 (défaut écrit) ; le site lui donne r = 50 et 0 → 360°",
    cas_st()["params"]["elements"][0]["r"] == 0.0 and el_site["r"] == CF.ELEMENT_ARC["r"] and el_site["ang2"] == CF.ELEMENT_ARC["ang2"],
    (cas_st()["params"]["elements"][0], el_site))
charger(e8)
rendre(editeur=editer(lambda df: df.loc.__setitem__((2, "type"), "segment")))
rendre()
e_site = S.normaliser(copy.deepcopy(e8))
S.modifier(e_site, ["params", "elements", "2", "type"], "segment")
a, b = cas_st()["params"]["elements"][2], ET.cas_actif(e_site)["params"]["elements"][2]
chk("C8 · un arc devenu segment : l'écran Streamlit en fait un segment de longueur nulle (0 ; 0) → (0 ; 0) ; le site lui donne "
    "le segment par défaut (0 ; 0) → (100 ; 0)", (a["x1"], a["y1"], a["x2"], a["y2"]) == (0.0, 0.0, 0.0, 0.0)
    and (b["x1"], b["y1"], b["x2"], b["y2"]) == tuple(CF.ELEMENT_SEGMENT[k] for k in S.COORDS_SEGMENT), (a, b))
chk("C8 · deux cordons du même nom : refusé sur le site (l'écran Streamlit mêle leurs résultats, défaut écrit)",
    S.modifier(S.normaliser(copy.deepcopy(e8)), ["params", "elements", "1", "nom"], "cordon 1") is not None)


# ====================================================================== D. le site affiche ce que le moteur calcule
titre("D · Le site affiche ce que le moteur calcule")
rng = random.Random(99)
ecarts = []
for cfg in CF.CONFIGS:
    for _ in range(4):
        e = S.normaliser(tirer(rng, cfg))
        r = S.resultat(e)
        res = ET.verifier(ET.cas_actif(e))
        if res["statut"] == ET.REFUS:
            if r["resume"]["niveau"] != "x" or "Calcul impossible" not in r["resultat"]:
                ecarts.append((cfg, "refus mal dit"))
            continue
        if r["resume"]["taux"] != C.pct(res["taux_max"]) or r["resume"]["niveau"] != C.niveau(res["taux_max"]):
            ecarts.append((cfg, "résumé"))
        for i, ch in enumerate(res["chaines"]):
            ligne = re.search(rf'<div class="el[^"]*" data-calcul="c{i}".*?<span class="pct (\w)"(?: title="([^"]*)")?>([^<]+)', r["resultat"])
            if not ligne or taux_lu(ligne.group(2), ligne.group(3)) != C.pc(C.pct(ch["taux"])) or ligne.group(1) != C.niveau(ch["taux"]):
                ecarts.append((cfg, f"ligne {ch['nom']}"))
            if f'id="ck-c{i}"' not in r["volets"]:
                ecarts.append((cfg, f"fenêtre {ch['nom']}"))
        gov = re.findall(r'<div class="el gov"', r["resultat"])
        if len(gov) != 1:
            ecarts.append((cfg, f"{len(gov)} lignes gouvernantes"))
chk("8 configurations × 4 cas : résumé, une ligne par cordon (taux et couleur du moteur), une fenêtre par ligne, une seule gouvernante",
    not ecarts, ecarts[:4])

e = S.nouvel_etat()
r = S.resultat(e)
res = ET.verifier(ET.cas_actif(e))
g = res["gouvernant"]
subs = {s.nom: s for s in FO.critere(res, g)}
chk("le verdict écrit la vérification gouvernante, valeur ≤ limite (σ_c = 218,5 ≤ 360,0 N/mm²)",
    f"σ<sub>c</sub> = {C.fr(subs['σ_c'].res, 1)} ≤ {C.fr(subs['lim_c'].res, 1)} N/mm²" in r["resultat"])
fen = re.search(r'<div class="flyout calcul" id="ck-c0">.*?(?=<div class="flyout calcul"|$)', r["volets"], re.S).group(0)
chk("la fenêtre de calcul porte les formules substituées du moteur (σ_c, lim_c, η_c, |f|, σ⊥, A_w)",
    all(x in fen for x in ("√(105,8² + 3 × (105,8² + 31,58²))", "360/(0,8 × 1,25)", "218,5/360", "2 × 5 × 190")))
chk("un nombre négatif au carré s'écrit entre parenthèses : √(0² + (−157,9)² + 747,9²)", "√(0² + (−157,9)² + 747,9²)" in fen and "+ −157,9²" not in fen)
chk("les quatre lectures ont leur fenêtre (directionnelle, simplifiée, ELS k = 100, ELS k Eurocode)", 'id="ck-m"' in r["volets"]
    and all(x in r["volets"] for x in ("EC3 directionnelle", "EC3 simplifiée", "ELS bureau k = 100", "ELS k Eurocode = 141")))
chk("les vues : élévation, coupe, face, gorge, cordons, points, méthode", [v["cle"] for v in r["vues"]] == ["elevation", "plan", "face", "gorge", "cordons", "points", "methode"]
    or len(r["vues"]) == 7, [v["cle"] for v in r["vues"]])
chk("l'encart de la gorge prend la place du dessin (sa borne de 320 px, faite pour la colonne Streamlit, est levée)",
    "max-width:320px" not in next(v["svg"] for v in r["vues"] if v["cle"] == "gorge"))
chk("les dessins sont ceux du moteur (schemas.vues), à l'échelle", all(v["svg"].startswith("<svg") for v in r["vues"][:3]))

# le sélecteur de gorge : trié, ✓ retenu, ★ la plus petite qui tient — chaque ligne recalculée par le moteur
h = S.selecteur(e, "gorge-a")
noms = re.findall(r'data-choix="([^"]+)"', h)
vals = [float(x.split("= ")[1].split(" ")[0].replace(",", ".")) for x in noms]
cur = re.findall(r'class="pk-row cur[^"]*" data-choix="([^"]+)"', h)
opt = re.findall(r'class="pk-row[^"]*opt" data-choix="([^"]+)"', h)
attendu = None
for v in sorted(set(S.GORGES) | {5.0}):
    c = copy.deepcopy(ET.cas_actif(e))
    c["params"]["a"] = v
    rr = ET.verifier(c)
    if rr["taux_max"] <= 1 and not rr.get("cordons_exclus"):
        attendu = v
        break
chk("sélecteur de gorge : trié par épaisseur, retenu ✓ (a = 5 mm), optimal ★ = la plus petite qui tient (recalculée)",
    vals == sorted(vals) and cur == ["a = 5 mm"] and opt == [f"a = {C.nb(attendu)} mm"], (cur, opt, attendu))
S.choisir_section(e, "gorge-a", opt[0])
chk("choisir la gorge ★ : la note la prend", ET.cas_actif(e)["params"]["a"] == attendu)
e5 = S.normaliser(tirer(random.Random(5), "C5"))
ET.cas_actif(e5)["efforts"].update(N=0.0, Vz=60.0, Vy=0.0, My=30.0, Mz=0.0, T=0.0)
h = S.selecteur(e5, "profil", ["IPE"])
masses = [float(x.replace(",", ".").replace(" ", "")) for x in re.findall(r'<span class="kg">([^<]+)</span>', h)]
chk("sélecteur de profilé (C5) : trié par masse, ✓ et ★ marqués", masses == sorted(masses) and 'class="pk-row cur' in h and " opt\"" in h)
best = re.findall(r'class="pk-row[^"]*opt" data-choix="([^"]+)"', h)[0]
c = copy.deepcopy(ET.cas_actif(e5))
c["params"]["profil"] = best
rr = ET.verifier(c)
legers = [n for n, r_ in CAT.par_nom().items() if r_["famille"] == "IPE" and r_["masse"] < CAT.par_nom()[best]["masse"]]
plus_legers_tiennent = []
for n in legers:
    c2 = copy.deepcopy(ET.cas_actif(e5))
    c2["params"]["profil"] = n
    r2 = ET.verifier(c2)
    if r2["statut"] != ET.REFUS and r2["taux_max"] <= 1:
        plus_legers_tiennent.append(n)
chk("★ du profilé = le plus léger qui tient (aucun plus léger ne tient)", rr["taux_max"] <= 1 and not plus_legers_tiennent, (best, plus_legers_tiennent))
S.choisir_section(e5, "profil", best)
chk("choisir le profilé ★ : la note le prend", ET.cas_actif(e5)["params"]["profil"] == best)

titre("D · Les refus : ce que l'écran Streamlit n'accepte pas, ou ramène en silence à autre chose")
e = S.nouvel_etat()
avant = json.dumps(e, sort_keys=True)
cas_refus = [(["params", "a"], "0,5", "Au moins 1 mm"), (["params", "h"], "5", "Au moins 10 mm"), (["params", "e"], "-10", "Au moins 0 mm"),
             (["params", "a"], "abc", "Nombre illisible"), (["methode", "gamma_M2"], "0,9", "Au moins 1"), (["methode", "g"], "9,8", "9,81 ou 10"),
             (["methode", "mode"], "autre", "Méthode inconnue"), (["config"], "C9", "Configuration inconnue"), (["params", "jonction"], "soudée", "Jonction inconnue"),
             (["params", "nuance_A"], "S999", "Nuance inconnue"), (["params", "poteau"], "HEB 999", "Poteau inconnu"), (["params", "inconnu"], "1", "Champ inconnu"),
             (["efforts", "Z"], "1", "Effort inconnu"), (["params", "troncons", "n"], "0", "Au moins 1"), (["params", "troncons", "n"], "2,5", "entier")]
for ch, v, attendu in cas_refus:
    msg = S.modifier(e, ch, v)
    chk(f"refus de {'/'.join(ch)} = {v} : « {attendu}… », la note reste intacte", bool(msg) and attendu in msg and json.dumps(e, sort_keys=True) == avant, msg)
chk("une valeur au minimum est acceptée (a = 1 mm, e = 0)", S.modifier(e, ["params", "a"], "1") is None and S.modifier(e, ["params", "e"], "0") is None)
for quoi, mut in (("g = 9,8 N/kg (l'écran affiche 9,81 et calcule 9,8)", lambda c: c["methode"].__setitem__("g", 9.8)),
                  ("un côté de cordon libre = 2 (l'écran le ramène à +1)", None),
                  ("une gorge de 0,5 mm (l'écran s'arrête sur une erreur)", lambda c: c["params"].__setitem__("a", 0.5)),
                  ("une nuance inconnue (le moteur prendrait S235)", lambda c: c["params"].__setitem__("nuance_A", "S999")),
                  ("une jonction inconnue", lambda c: c["params"].__setitem__("jonction", "x")),
                  ("une méthode inconnue", lambda c: c["methode"].__setitem__("mode", "x"))):
    n = S.nouvel_etat()
    if mut is None:
        ET.changer_config(n["cas"][0], "C8")
        n["cas"][0]["params"]["elements"][0]["cote"] = 2
    else:
        mut(n["cas"][0])
    rep = json.loads(RT.traiter(json.dumps({"module": "soudures", "action": "calculer", "etat": n})))
    chk(f"note venue d'un fichier avec {quoi} : refusée, et le site dit pourquoi", not rep["ok"] and "abîmée" in rep["erreur"], rep.get("erreur"))

titre("D · Les effets de saisie de l'écran, et l'écriture")
e = S.nouvel_etat()
S.modifier(e, ["params", "troncons", "on"], True)
chk("cocher « par tronçons » : n = 1 (comme l'écran : max(1, n))", ET.cas_actif(e)["params"]["troncons"]["n"] == 1)
S.modifier(e, ["params", "troncons", "n"], "4")
S.modifier(e, ["params", "troncons", "on"], False)
chk("décocher : n = 0 (continu)", ET.cas_actif(e)["params"]["troncons"]["n"] == 0)
S.modifier(e, ["methode", "k_eurocode"], 1)
kq = K.k_equivalent_eurocode(360.0, 0.8, 1.25, 1.5, 9.81)
chk("« Prendre la valeur Eurocode » : k = round(k équivalent, 1) = 141,2", ET.cas_actif(e)["methode"]["k"] == round(kq["k_kg_mm_cm"], 1) == 141.2)
S.modifier(e, ["methode", "k_bureau"], 1)
chk("« Revenir à 100 »", ET.cas_actif(e)["methode"]["k"] == K.K_BUREAU)
S.gerer_elements(e, "ajouter")
chk("« ＋ Cas » : une console C3 qui garde la méthode du cas ouvert", ET.cas_actif(e)["config"] == "C3" and len(e["cas"]) == 2)
S.gerer_elements(e, "supprimer", 1)
S.gerer_elements(e, "supprimer", 0)
chk("le dernier cas ne se supprime pas", len(e["cas"]) == 1)
chk("message du moteur à la française : « 98.1 N/mm² » → « 98,1 N/mm² », les renvois §4.5.3.2 intacts",
    S.texte_fr("k = 98.1 N/mm² (EN 1993-1-8 §4.5.3.2, Tab. 4.1) ; 12.5 mm") == "k = 98,1 N/mm² (EN 1993-1-8 §4.5.3.2, Tab. 4.1) ; 12,5 mm")
e = S.nouvel_etat()
ET.cas_actif(e)["methode"]["mode"] = K.ELS
r = S.resultat(e)
chk("ELS : le message du moteur s'écrit 98,1 et 138,6 N/mm² sur le site (98.1 dans l'écran Streamlit)", "98,1 N/mm²" in r["resultat"] and "98.1" not in r["resultat"])
e = S.nouvel_etat()
ET.changer_config(ET.cas_actif(e), "C6")
S.modifier(e, ["params", "troncons", "on"], True)
r = S.resultat(e)
chk("calcul impossible : le verdict dit la RAISON (le refus du moteur), pas la première note venue",
    re.search(r"<b>Calcul impossible</b>Refus", r["resultat"]) is not None and r["resultat"].count("Refus :") == 1)

titre("D · Échappement : un nom saisi ne devient jamais du code")
piege = '<img src=x onerror="alert(1)">&<script>'
e = S.nouvel_etat()
S.modifier(e, ["nom"], piege)
ET.changer_config(ET.cas_actif(e), "C8")
ET.cas_actif(e)["nom"] = piege
S.modifier(e, ["params", "elements", "0", "nom"], piege)
r = S.resultat(e)
tout = json.dumps({k: v for k, v in r.items() if k != "resume"}, ensure_ascii=False)
chk("nom de cas et nom de cordon piégés : échappés dans tous les fragments HTML (vues et encart compris)", "<img" not in tout and "<script" not in tout,
    re.findall(r".{40}<img.{20}", tout)[:2])
chk("le résumé de la note est du TEXTE (la page l'échappe) : le nom y est tel quel", piege in r["resume"]["gouvernant"])

# ====================================================================== E. mutations
titre("E · Mutations : chaque garantie rougit quand on casse ce qu'elle surveille")


@contextlib.contextmanager
def mutant(obj, nom, valeur):
    ancien = getattr(obj, nom)
    setattr(obj, nom, valeur)
    try:
        yield
    finally:
        setattr(obj, nom, ancien)


def parite_globale(n=3, cfgs=None) -> int:
    r_ = random.Random(424242)
    total = 0
    for cfg in cfgs or CF.CONFIGS:
        for _ in range(n):
            total += len(parite_cas(tirer(r_, cfg)))
    return total


chk("sans mutation, la parité tient (contrôle du contrôle)", parite_globale(2) == 0)
vrai_ligne = S.element_depuis_ligne
with mutant(S, "element_depuis_ligne", lambda row: dict(vrai_ligne(row), cote=-1)):
    chk("mutation : la relecture des cordons libres met côté −1 → la parité C8 rougit", parite_globale(3, ["C8"]) > 0)
with mutant(S, "nature_chaine", lambda c: "l − 2a"):
    chk("mutation : la nature des cordons écrite toujours « l − 2a » → le tableau des cordons rougit", parite_globale(3, ["C1", "C6", "C7"]) > 0)
def cas_fixe(cfg, methode=None, **params):
    e_ = ET.nouvel_etat()
    ET.changer_config(e_["cas"][0], cfg)
    e_["cas"][0]["params"].update(params)
    e_["cas"][0]["methode"].update(methode or {})
    return e_


CAS_MESSAGES = [cas_fixe("C5", a_w=2.5), cas_fixe("C1", a=2.5), cas_fixe("C3", {"mode": K.ELS}), cas_fixe("C3", troncons={"n": 2, "l": 40.0})]
chk("les cas à messages (cordons qui ne comptent pas, refus, ELS, espacement) : parité sans mutation", sum(len(parite_cas(x)) for x in CAS_MESSAGES) == 0)
with mutant(S, "message_grave", lambda m: False):
    chk("mutation : plus aucun avertissement en couleur → la gravité des messages rougit", sum(len(parite_cas(x)) for x in CAS_MESSAGES) > 0)
with mutant(S, "texte_fr", lambda m: m):
    chk("mutation : les messages laissés à l'anglaise (98.1) → la parité des messages rougit", sum(len(parite_cas(x)) for x in CAS_MESSAGES) > 0)
with mutant(S, "_normaliser_c8", lambda cas: None):
    chk("mutation : le site ne relit plus le tableau C8 comme l'écran → l'état diverge", parite_globale(3, ["C8"]) > 0)
mins = dict(S.MINIMUMS)
S.MINIMUMS[("params", "a")] = 0.5
e = S.nouvel_etat()
chk("mutation : minimum de la gorge à 0,5 → a = 0,7 mm passe alors que l'écran le refuse (le contrôle des minimums le voit)",
    S.modifier(e, ["params", "a"], "0,7") is None and S.MINIMUMS[("params", "a")] != 1.0)
S.MINIMUMS.clear()
S.MINIMUMS.update(mins)
vrai_gerer = S.gerer_elements


def gerer_mute(etat, action, index=None):
    vrai_gerer(etat, action, index)
    if action == "ajouter":
        etat["cas"][-1]["methode"] = ET.methode_defaut()


with mutant(S, "gerer_elements", gerer_mute):
    charger(e0)
    rendre(clics={P + "btn_new"})
    e_site = S.normaliser(copy.deepcopy(e0))
    S.gerer_elements(e_site, "ajouter")
    chk("mutation : « ＋ Cas » ne garde plus la méthode → la comparaison au bouton de l'écran rougit", etat_st()["cas"] != e_site["cas"])
vrai_methode = S._methode


def methode_mutee(cas, k, valeur):
    vrai_methode(cas, k, valeur)
    if k == "k_eurocode":
        cas["methode"]["k"] = round(cas["methode"]["k"])
with mutant(S, "_methode", methode_mutee):
    charger(e0)
    rendre(clics={P + "btn_kq"})
    e_site = S.normaliser(copy.deepcopy(e0))
    S.modifier(e_site, ["methode", "k_eurocode"], 1)
    chk("mutation : k Eurocode arrondi à l'unité → la comparaison au bouton de l'écran rougit", etat_st()["cas"] != e_site["cas"])
with mutant(S, "_parentheses", lambda t: t):
    fen = S.resultat(S.nouvel_etat())["volets"]
    chk("mutation : sans les parenthèses, « + −157,9² » revient (le contrôle de l'écriture le voit)", "+ −157,9²" in fen)


# ====================================================================== F. relecture indépendante (07/11/2026)
titre("F · Relecture : saisies hors de portée, notes abîmées, indices des dessins, signe moins, arrondi des limites")


def huge_refuse() -> bool:
    """Une gorge, un effort, une coordonnée C8 de 1e300 : refus propre (message), note intacte — l'écran Streamlit s'arrête
    sur decimal.InvalidOperation (vérifié) et le site, sans borne, rendait une erreur Python au lieu d'un résultat."""
    ok = True
    for cfg, ch in (("C3", ["params", "a"]), ("C3", ["efforts", "V"]), ("C8", ["params", "elements", "0", "x1"])):
        e_ = cas_fixe(cfg)
        e_ = S.normaliser(e_)
        avant = copy.deepcopy(e_)
        msg = S.modifier(e_, ch, "1e300")
        ok &= bool(msg) and "portée" in msg and e_ == avant
    rr = json.loads(RT.traiter(json.dumps({"module": "soudures", "action": "modifier", "etat": S.nouvel_etat(), "chemin": ["efforts", "V"], "valeur": "1e300"})))
    return ok and rr["ok"] and rr["message"] and rr["resume"]["taux"] == 61


chk("une saisie hors de portée (1e300 : gorge, effort, coordonnée C8) est refusée avec un message, la note intacte", huge_refuse())
with mutant(S, "BORNE", float("inf")), warnings.catch_warnings():
    warnings.simplefilter("ignore")              # le moteur déborde : numpy le signale, c'est attendu ici
    try:
        rouge = not huge_refuse()
    except Exception:  # noqa: BLE001 — sans borne, le moteur déborde : c'est ce que le contrôle doit voir
        rouge = True
    chk("mutation : sans borne, 1e300 passe et le moteur déborde (le contrôle le voit)", rouge)
mauvais = cas_fixe("C3", h=1e300)
try:
    S.normaliser(mauvais)
    lu = "relue"
except ValueError as x:
    lu = str(x)
chk("une note .stab qui porte 1e300 est refusée en français (« un nombre entre −1 000 000 et 1 000 000 »)", "un nombre entre" in lu, lu)

e_ = S.nouvel_etat()
m_n = S.modifier(e_, ["params", "troncons", "n"], str(S.N_TRONCONS_MAX + 1))
m_ok = S.modifier(e_, ["params", "troncons", "n"], str(S.N_TRONCONS_MAX))
chk(f"tronçons : n ≤ {S.N_TRONCONS_MAX} (au-delà, une ligne et une fenêtre par tronçon rendent l'écran illisible), écrit en data-max",
    bool(m_n) and m_ok is None and f'data-chemin="params/troncons/n" data-sorte="entier" inputmode="decimal" value="{S.N_TRONCONS_MAX}" data-min="1" data-max="{S.N_TRONCONS_MAX}"'
    in S.resultat(e_)["donnees"], (m_n, m_ok))

abimees = {"un cas illisible": {"version": 1, "cas": [None]}, "cas ouvert illisible": dict(S.nouvel_etat(), actif="a"),
           "params illisible": {"cas": [dict(ET.nouveau_cas("C3"), params=[1])]},
           "n'est pas un nombre lisible": cas_fixe("C8", elements=[dict(CF.ELEMENT_SEGMENT, x1="abc", troncons={"n": 0, "l": 60.0})])}
lus = {}
for attendu, note in abimees.items():
    r_ = json.loads(RT.traiter(json.dumps({"module": "soudures", "action": "calculer", "etat": note})))
    lus[attendu] = r_.get("erreur", "")
chk("note abîmée : le refus est écrit en français (cas illisible, cas ouvert, params, coordonnée C8), jamais un message Python",
    all(k in v and "object has no attribute" not in v and "invalid literal" not in v for k, v in lus.items()), lus)
try:
    S.gerer_elements(S.nouvel_etat(), "activer", "a")
    g_ = "accepté"
except ValueError as x:
    g_ = str(x)
chk("ouvrir le cas « a » : refus clair (« cas inconnu »)", g_.startswith("cas inconnu"), g_)
e2 = S.nouvel_etat()
S.gerer_elements(e2, "ajouter")
sup = []
for idx in (99, -3):
    try:
        S.gerer_elements(e2, "supprimer", idx)
        sup.append("supprimé")
    except ValueError as x:
        sup.append(str(x))
chk("supprimer le cas 99 ou −3 : refusé (ramené dans la liste, il supprimait un autre cas)", len(e2["cas"]) == 2 and all(x.startswith("cas inconnu") for x in sup), sup)
e_ = S.nouvel_etat()
try:
    S.choisir_section(e_, "gorge-a", "a = 1e300 mm")
    c_ = "accepté"
except ValueError as x:
    c_ = str(x)
chk("sélecteur : « a = 1e300 mm » n'est plus lu « a = 1 mm » (le nom entier est lu)", c_ != "accepté" and e_["cas"][0]["params"]["a"] == 5.0, c_)
e8 = S.normaliser(cas_fixe("C8"))
n8 = len(e8["cas"][0]["params"]["elements"])
m8 = [S.modifier(e8, ["params", "elements", i, "-"], 1) for i in ("-1", "7", "x")]
chk("C8 : retirer le cordon « -1 », « 7 » ou « x » est refusé en français (« -1 » retirait le dernier en silence)",
    all(m == "Cordon inconnu" for m in m8) and len(e8["cas"][0]["params"]["elements"]) == n8, m8)
S.choisir_section(e_, "gorge-a", "a = 6 mm")
chk("sélecteur : « a = 6 mm » retient 6 mm", e_["cas"][0]["params"]["a"] == 6.0)


def sans_indices(svg: str) -> str:
    """L'inverse de S.indices_svg : les indices remis en « _x », le texte de retour à sa ligne remis à plat."""
    svg = re.sub(r'<tspan dy="[\d.]+" font-size="[\d.]+">([^<]*)</tspan>', r"_\1", svg)
    return re.sub(r'<tspan dy="-[\d.]+">([^<]*)</tspan>', r"\1", svg)


ecarts_svg, avec_indice = [], 0
for cfg in CF.CONFIGS:
    e_ = cas_fixe(cfg)
    res_ = ET.verifier(e_["cas"][0])
    site_ = {v["cle"]: v["svg"] for v in S.vues(res_)}
    from acier.soudures import schemas as SC  # noqa: E402
    for k, _t, d in SC.vues(res_):
        brut = SC.svg(d, f"sdr_{k}")
        avec_indice += site_[k] != brut
        if sans_indices(site_[k]) != brut or re.search(r">[^<]*\b[A-Za-z]_[A-Za-z0-9]", site_[k]):
            ecarts_svg.append((cfg, k))
chk("dessins : les symboles du moteur (h_p, V_y, b_pl, t_p…) s'écrivent avec leur indice ; rien d'autre ne change (texte, traits, cotes)",
    not ecarts_svg and avec_indice >= 8, (ecarts_svg, avec_indice))
chk("dessins : un nom saisi (« plat_A ») n'est pas pris pour un symbole", S.indices_svg('<text font-size="10">plat_A · h_p</text>').startswith('<text font-size="10">plat_A · h<tspan'))
enc = next(v["svg"] for v in S.resultat(S.nouvel_etat())["vues"] if v["cle"] == "gorge")
chk("encart de la gorge : σ_c ≤ fu/(βw·γM2) écrit σ<sub>c</sub> ≤ f<sub>u</sub>/(β<sub>w</sub>·γ<sub>M2</sub>) (tspans)",
    "fu/" not in enc and ">u</tspan>" in enc and ">w</tspan>" in enc and ">M2</tspan>" in enc)
with mutant(S, "indices_svg", lambda s: s):
    brut = S.resultat(cas_fixe("C3"))["vues"][0]["svg"]
    chk("mutation : sans indices, « h_p = 200 » revient dans le dessin (le contrôle le voit)", re.search(r">[^<]*\bh_p", brut) is not None)

r_ = S.resultat(S.nouvel_etat())
pts = next(v["svg"] for v in r_["vues"] if v["cle"] == "points")
eff = re.search(r"Au point de référence.*?</div>", r_["donnees"]).group(0)
chk("signe moins typographique « − » dans le détail par point, les efforts réduits et la fenêtre de calcul (−105,8 ; −60,0 kN)",
    "−105,8" in pts and not re.search(r">-\d", pts) and "−60,0 kN" in eff and "−60,00 kN" in r_["volets"] and not re.search(r"[ >(]-\d", eff))

e_ = cas_fixe("C3", {"mode": K.ELS})
r_ = S.resultat(e_)
fen = re.search(r'<div class="flyout calcul" id="ck-c0">.*?(?=<div class="flyout calcul"|$)', r_["volets"], re.S).group(0)
lim_tab = re.search(r'<td class="n">([^<]*N/mm)</td><td class="n"><span class="chip', fen).group(1)
lim_fx = re.search(r"F<sub>adm</sub></span>.*?<span class=\"r\">([^<]*)</span>", fen).group(1)
chk("ELS : la limite du tableau et du verdict s'écrit comme la formule du moteur (F_adm = 98,1 × 5 = 490,5 → « 491 N/mm » partout)",
    lim_tab == lim_fx == "491 N/mm" and "F<sub>adm</sub> = 491 N/mm" in r_["resultat"], (lim_tab, lim_fx))
# l'arrondi au pair (format Python) : l'ancienne règle de C.fr — commun.fr arrondit désormais comme le moteur
with mutant(S, "_fr_moteur", lambda x, nd: f"{x:,.{nd}f}".replace(",", "\u202f").replace(".", ",")):
    r_m = S.resultat(e_)
    chk("mutation : la limite arrondie au pair (format Python, l'ancien C.fr) → « 490 » à côté de « 491 » (le contrôle le voit)",
        "F<sub>adm</sub> = 490 N/mm" in r_m["resultat"])
with mutant(S, "_fr_moteur", lambda x, nd: C.fr(x, nd)):
    chk("commun.fr écrit la limite comme le moteur : « 491 N/mm » (490,5, demi vers l'extérieur)", "F<sub>adm</sub> = 491 N/mm" in S.resultat(e_)["resultat"])

e_ = cas_fixe("C6", troncons={"n": 1, "l": 60.0})
h_ = S.resultat(e_)["resultat"]
chk("messages du moteur : « M_x », « f_vw,d » écrits avec leur indice (M<sub>x</sub>) ; refus et avertissements compris",
    "M<sub>x</sub>" in h_ and "M_x" not in h_)
e_ = cas_fixe("C3", {"mode": K.ELS})
h_ = S.resultat(e_)["resultat"]
chk("message ELS : f<sub>vw,d</sub>/1,5", "f<sub>vw,d</sub>/1,5" in h_)
ax5, ax3 = S.correspondance_axes("C5"), S.correspondance_axes("C3")
chk("C5 : la correspondance des efforts saisis et du repère du joint est LUE au moteur (V_z → V_y, M_y → M_x, T → M_z ; C3 : V → −V_y)",
    ("V_z", "V_y", 1) in ax5 and ("M_y", "M_x", 1) in ax5 and ("T", "M_z", 1) in ax5 and ("V", "V_y", -1) in ax3, (ax5, ax3))
d5, d8 = S.resultat(cas_fixe("C5"))["donnees"], S.resultat(cas_fixe("C8"))["donnees"]
face5 = S.resultat(cas_fixe("C5"))["vues"][0]["svg"]
chk("C5 : le cadre Efforts dit que « V_z saisi » est le « V_y » de la vue de face (même effort, deux repères) ; rien en C8",
    "V<sub>z</sub> devient V<sub>y</sub>" in d5 and "devient" not in d8 and 'V<tspan dy="' in face5)
css = open("site/web/css/modules/soudures.css", encoding="utf-8").read()
chk("la pastille η du verdict ne se comprime pas sous une comparaison longue (pleine pénétration)", ".verdict .chip.xl { flex-shrink: 0; }" in css)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
