# -*- coding: utf-8 -*-
"""Le GABARIT DU BUREAU, côté moteur (api_web/gabarit.py, lot 6 — décision du propriétaire du 13/11/2026 : « dans les
options du site, les options par défaut de chaque module, que le responsable fixe ; chaque note peut changer une valeur
au cas par cas »).

  1. les modules qui ont un gabarit : la poutre, les dalles et prédalles — et eux seuls ;
  2. les valeurs d'USINE sont LUES du moteur (R2 : jamais recopiées) — ``lire_gabarit(nouvel_etat())`` égale les tables
     de défauts des moteurs (beton/poutre/entree.py, beton/dalle/entree.py, beton/predalle/entree.py, api_web/dalle.py) ;
  3. une note NEUVE (« nouveau ») reçoit le gabarit, chaque valeur écrite par le module (le contrôle de la rubrique
     Options) — et le CALCUL la suit (l'axe d'une dalle au jeu de 0,5 cm, la longueur de couture au pas de 10 cm, le
     deuxième lit d'une poutre au jeu de 3 cm) ; sans gabarit : la note d'usine, à l'identique ;
  4. les refus : une valeur que le moteur n'admet pas, une option inconnue, un gabarit qui n'est pas un dictionnaire —
     chacun DIT, la valeur d'usine reste ; une note EXISTANTE ne reçoit jamais le gabarit (« calculer », « modifier ») ;
  5. l'action « gabarit » de la page « Paramètres » : la rubrique Options du module (le même composant que dans la
     note, la ligne « Couture » des prédalles comprise), les valeurs, l'usine, les ÉCARTS (ce que la page garde) ; une
     saisie (arrondi, une puce d'espacement, l'option « Taux d'armature ») ; une saisie hors du gabarit refusée ;
  6. mutations : chaque règle retirée fait rougir le contrôle qui la garde.

    /usr/bin/python3.12 tests/test_gabarit.py
"""
import copy
import json
import os
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import dalle as PD  # noqa: E402
from api_web import gabarit as G  # noqa: E402
from api_web import poutre as PP  # noqa: E402
from api_web import registre  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.dalle import entree as ED  # noqa: E402
from beton.poutre import entree as EP  # noqa: E402
from beton.poutre import moteur as MOP  # noqa: E402
from beton.poutre import taux as TA  # noqa: E402
from beton.predalle import entree as EPD  # noqa: E402

sys.path.insert(0, os.path.join(RACINE, "site", "web"))
import construire as SW  # noqa: E402  (le générateur du site : les cartes de la page « Paramètres »)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**q):
    return json.loads(R.traiter(json.dumps(q)))


GAB_D = {"jeu_premier_lit": 0.5, "arrondi": 1.0, "pas_couture": 10.0, "espacements": [150, 200]}
GAB_P = {"jeu_entre_lits": 3.0, "taux/actif": True, "taux/majoration": 15.0}


# ---------------------------------------------------------------- les cas, en fonctions (rejoués par les mutations)
def cas_usine():
    return {"d": PD.lire_gabarit(PD.nouvel_etat()), "p": PP.lire_gabarit(PP.nouvel_etat())}


def cas_neuves():
    nd = req(module="dalle", action="nouveau", gabarit=GAB_D)
    np_ = req(module="poutre", action="nouveau", gabarit=GAB_P)
    # le calcul suit : la dalle neuve devient prédalle (le pas de la note) ; la poutre reçoit un deuxième lit inférieur
    pd = req(module="dalle", action="modifier", etat=nd["etat"], chemin=["type"], valeur="predalle")
    p2 = req(module="poutre", action="modifier", etat=np_["etat"], chemin=["sections", "0", "inf", "+"], valeur=1)
    sans = req(module="dalle", action="nouveau")
    return {"nd": nd, "np": np_, "pd": pd, "p2": p2, "sans": sans}


def cas_refus():
    mauvais = req(module="poutre", action="nouveau", gabarit={"jeu_entre_lits": -2, "gamma_s": 1.15, "x": 1})
    pas2 = req(module="dalle", action="nouveau", gabarit={"pas_couture": 2, "espacements": [175], "arrondi": 1})
    illisible = req(module="dalle", action="nouveau", gabarit=[1, 2])
    existante = req(module="dalle", action="calculer", etat=PD.nouvel_etat(), gabarit=GAB_D)
    modif = req(module="dalle", action="modifier", etat=PD.nouvel_etat(), chemin=["nom"], valeur="D", gabarit=GAB_D)
    return {"mauvais": mauvais, "pas2": pas2, "illisible": illisible, "existante": existante, "modif": modif}


def cas_page():
    d0 = req(module="dalle", action="gabarit", gabarit={})
    d1 = req(module="dalle", action="gabarit", gabarit=GAB_D, chemin=["arrondi"], valeur="0.5")
    d2 = req(module="dalle", action="gabarit", gabarit=d1["ecarts"], chemin=["espacements"], valeur="250")
    hors = req(module="dalle", action="gabarit", gabarit=GAB_D, chemin=["h"], valeur="30")
    p1 = req(module="poutre", action="gabarit", gabarit=GAB_P, chemin=["taux", "actif"], valeur=False)
    sans = req(module="soudures", action="gabarit", gabarit={})
    stable = req(module="dalle", action="gabarit", gabarit=d2["ecarts"])
    return {"d0": d0, "d1": d1, "d2": d2, "hors": hors, "p1": p1, "sans": sans, "stable": stable}


def cas(fonction):
    try:
        return fonction()
    except Exception as e:  # noqa: BLE001 — une mutation peut faire lever : ses contrôles rougissent
        return {"_erreur": repr(e)}


# ---------------------------------------------------------------- les contrôles
def controles(prefixe=""):
    out = []

    def c(nom, f):
        try:
            out.append((prefixe + nom, bool(f())))
        except Exception:  # noqa: BLE001
            out.append((prefixe + nom, False))

    c("1. ont un gabarit : la poutre, les dalles et prédalles — et eux seuls ; la page « Paramètres » a une carte pour chacun "
      "(site/web/construire.MODULES_GABARIT : une copie nommée, liée ici)",
      lambda: sorted(k for k in registre.cles() if G.a_un_gabarit(registre.module(k))) == ["dalle", "poutre"]
      and sorted(SW.MODULES_GABARIT) == ["dalle", "poutre"])

    u = cas(cas_usine)
    c("2. dalles : l'usine est LUE du moteur — γs, jeu de pose, arrondi (beton/dalle/entree.DEFAUTS_DALLE), pas de couture "
      "(beton/predalle/entree.DEFAUTS_PREDALLE : 5 cm), espacements (tous ceux du moteur)",
      lambda: u["d"] == {"gamma_s": ED.DEFAUTS_DALLE["gamma_s"], "jeu_premier_lit": ED.DEFAUTS_DALLE["jeu_premier_lit"],
                         "arrondi": ED.DEFAUTS_DALLE["arrondi"], "pas_couture": EPD.DEFAUTS_PREDALLE["pas_couture"],
                         "espacements": list(PD.ESPACEMENTS)} and u["d"]["pas_couture"] == 5.0)
    c("2. poutre : l'usine est LUE du moteur — les jeux (1 et 2,5 cm), l'arrondi, γs (DEFAUTS_POUTRE), le granulat et la peau "
      "(DEFAUTS_DETAIL), l'option « Taux d'armature » éteinte et ses réglages (beton/poutre/taux.DEFAUTS)",
      lambda: u["p"] == {"jeu_premier_lit": EP.DEFAUTS_POUTRE["jeu_premier_lit"], "jeu_entre_lits": EP.DEFAUTS_POUTRE["jeu_entre_lits"],
                         "arrondi": EP.DEFAUTS_POUTRE["arrondi"], "gamma_s": EP.DEFAUTS_POUTRE["gamma_s"],
                         "dg": EP.DEFAUTS_DETAIL["dg"], "peau_d": EP.DEFAUTS_DETAIL["peau_d"], "peau_smax": EP.DEFAUTS_DETAIL["peau_smax"],
                         "taux/actif": False, "taux/note": False, "taux/majoration": TA.DEFAUTS["majoration"],
                         "taux/retour": TA.DEFAUTS["retour"], "taux/arrondi": TA.DEFAUTS["arrondi"]}
      and u["p"]["jeu_entre_lits"] == 2.5)
    c("2. les clés du gabarit d'un module sont exactement celles qu'il lit (GABARIT ↔ lire_gabarit)",
      lambda: list(u["d"]) == list(PD.GABARIT) and list(u["p"]) == list(PP.GABARIT))

    n = cas(cas_neuves)
    nd, np_ = n.get("nd") or {}, n.get("np") or {}
    c("3. dalles : une note NEUVE reçoit le gabarit — jeu de pose 0,5 cm, arrondi 1 cm, pas de couture 10 cm, espacements 15 et "
      "20 cm ; aucun message",
      lambda: nd["ok"] and nd["message"] is None and PD.lire_gabarit(nd["etat"]) == dict(u["d"], **GAB_D))
    c("3. … et le CALCUL suit : la fenêtre de calcul dit « jeu de pose 0,5 cm · arrondi 1 cm » et l'axe de la base inférieure Y "
      "« c + Ø/2 + jeu = 3 + 0,5 + 0,5 » ; devenue prédalle, la longueur de couture au pas de 10 cm (⌈53,6⌉ → 60 cm, 120 cm en tout)",
      lambda: "jeu de pose 0,5 cm · arrondi 1 cm" in nd["volets"] and "<span>c + Ø/2 + jeu</span><span class=\"eq\">=</span><span>3 + 0,5 + 0,5</span>" in nd["volets"]
      and n["pd"]["ok"] and "60 cm · 120 cm en tout" in n["pd"]["volets"]
      and "⌈l₀⌉ au 10 cm" in n["pd"]["volets"])
    c("3. poutre : une note NEUVE reçoit le gabarit — jeu entre lits 3 cm, l'option « Taux d'armature » allumée, majoration 15 % ; "
      "un deuxième lit Ø16 se pose à 0,8 + 3 + 0,8 = 4,6 cm du premier (réels 48 et 94 mm : 30 mm libres)",
      lambda: np_["ok"] and np_["message"] is None and PP.lire_gabarit(np_["etat"]) == dict(u["p"], **GAB_P)
      and [round(x["reel"] * 10, 9) for x in MOP.verifier(n["p2"]["etat"]["poutres"][0])["sections"][0]["geometrie"]["inf"]["lits"]] == [48.0, 94.0]
      and "Jeu entre lits insuffisant" not in n["p2"]["donnees"])
    c("3. sans gabarit, « nouveau » rend la note d'usine, à l'identique",
      lambda: n["sans"]["ok"] and n["sans"]["etat"] == PD.nouvel_etat() and n["sans"]["message"] is None)

    f = cas(cas_refus)
    c("4. une valeur refusée par le moteur (jeu entre lits −2) : le message du module et « la valeur d'usine est gardée » ; les "
      "autres valeurs passent (γs 1,15) ; une option inconnue est dite et ignorée",
      lambda: f["mauvais"]["ok"] and "jeu_entre_lits" in f["mauvais"]["message"] and "la valeur d'usine est gardée" in f["mauvais"]["message"]
      and "option(s) inconnue(s) de ce module (x)" in f["mauvais"]["message"]
      and f["mauvais"]["etat"]["poutres"][0]["jeu_entre_lits"] == EP.DEFAUTS_POUTRE["jeu_entre_lits"]
      and f["mauvais"]["etat"]["poutres"][0]["gamma_s"] == 1.15)
    c("4. dalles : un pas de couture de 2 cm et un espacement de 17,5 cm refusés (l'usine reste : 5 cm, tous les espacements), "
      "l'arrondi de 1 cm passe",
      lambda: f["pas2"]["ok"] and f["pas2"]["etat"]["pas_couture"] == 5.0 and f["pas2"]["etat"]["espacements"] == list(PD.ESPACEMENTS)
      and f["pas2"]["etat"]["dalles"][0]["arrondi"] == 1.0 and f["pas2"]["message"].count("la valeur d'usine est gardée") == 2)
    c("4. un gabarit qui n'est pas un dictionnaire : « Gabarit du bureau illisible », la note d'usine",
      lambda: f["illisible"]["ok"] and "Gabarit du bureau illisible" in f["illisible"]["message"] and f["illisible"]["etat"] == PD.nouvel_etat())
    c("4. une note EXISTANTE ne reçoit jamais le gabarit (« calculer », « modifier ») : ses options restent les siennes",
      lambda: PD.lire_gabarit(f["existante"]["etat"]) == u["d"] and PD.lire_gabarit(f["modif"]["etat"]) == u["d"]
      and f["existante"]["message"] is None)

    g = cas(cas_page)
    d0, d1, d2 = g.get("d0") or {}, g.get("d1") or {}, g.get("d2") or {}
    c("5. la page : la rubrique Options des dalles, rendue par le module (le même composant que dans la note) — γs, jeu de pose, "
      "arrondi, la ligne « Couture » des prédalles (1 · 5 · 10 cm), les espacements ; usine = valeurs, aucun écart",
      lambda: d0["ok"] and all(f'data-chemin="{k}"' in d0["html"] for k in ("gamma_s", "jeu_premier_lit", "arrondi", "pas_couture", "espacements"))
      and d0["valeurs"] == d0["usine"] == u["d"] and d0["ecarts"] == {} and d0["refus"] == [] and d0["message"] is None)
    c("5. une saisie de la page (l'arrondi rendu à 0,5 cm, la valeur d'usine) : elle sort des ÉCARTS — le gabarit ne garde que ce "
      "qui diffère de l'usine",
      lambda: d1["ok"] and d1["message"] is None and d1["ecarts"] == {"jeu_premier_lit": 0.5, "pas_couture": 10.0, "espacements": [150, 200]})
    c("5. une puce d'espacement (25 cm) s'allume comme dans la note ; relu, le gabarit rend les mêmes écarts (stable)",
      lambda: d2["ecarts"]["espacements"] == [150, 200, 250] and g["stable"]["ecarts"] == d2["ecarts"])
    c("5. une saisie HORS du gabarit (h = 30) est refusée, le gabarit intact",
      lambda: g["hors"]["ok"] and g["hors"]["message"] == "Cette valeur n'est pas une option du gabarit du bureau"
      and g["hors"]["ecarts"] == GAB_D)
    c("5. la poutre : sa rubrique Options (jeux, arrondi, γs, d_g, peau, « Taux d'armature ») ; éteindre l'option retire ses réglages "
      "des écarts — rien n'y reste que le jeu entre lits",
      lambda: g["p1"]["ok"] and all(f'data-chemin="{k}"' in g["p1"]["html"] for k in ("jeu_premier_lit", "jeu_entre_lits", "arrondi", "gamma_s", "dg", "taux/actif"))
      and g["p1"]["ecarts"] == {"jeu_entre_lits": 3.0, "taux/majoration": 15.0})
    c("5. un module sans gabarit (soudures) : refus clair, jamais une exception",
      lambda: g["sans"] == {"ok": False, "erreur": "Ce module n'a pas de gabarit du bureau."})
    return out


for nom, ok in controles():
    chk(nom, ok)


# ---------------------------------------------------------------- 6. les mutations
def muter(fichier: str, ancien: str, nouveau: str, mod_nom: str):
    """Rejoue tous les contrôles avec ``fichier`` muté (le module ``mod_nom`` remplacé le temps de l'essai) ; rend
    [(nom, vrai/faux)], ou None si la cible n'existe pas une fois."""
    global G, R  # noqa: PLW0603
    src = open(os.path.join(RACINE, fichier), encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    vrai = sys.modules[mod_nom]
    mutant = types.ModuleType(mod_nom)
    mutant.__file__ = vrai.__file__
    mutant.__package__ = vrai.__package__
    exec(compile(src.replace(ancien, nouveau), fichier, "exec"), mutant.__dict__)  # noqa: S102
    sys.modules[mod_nom] = mutant
    parents = {"api_web.gabarit": [(R, "G")], "api_web.dalle": [], "api_web.routeur": []}
    remis = []
    for objet, attr in parents.get(mod_nom, []):
        remis.append((objet, attr, getattr(objet, attr)))
        setattr(objet, attr, mutant)
    g0, r0 = G, R
    if mod_nom == "api_web.gabarit":
        G = mutant
    if mod_nom == "api_web.routeur":
        R = mutant
    try:
        return controles("mutation · ")
    finally:
        sys.modules[mod_nom] = vrai
        for objet, attr, v in remis:
            setattr(objet, attr, v)
        G, R = g0, r0


MUTATIONS = [
    ("la note neuve sans le gabarit (le routeur l'oublie)", "api_web/routeur.py", "api_web.routeur",
     'if action == "nouveau" and q.get("gabarit") is not None and G.a_un_gabarit(mod):', 'if False:', "3. dalles : une note NEUVE"),
    ("le gabarit appliqué aussi à une note existante", "api_web/routeur.py", "api_web.routeur",
     'if action == "nouveau" and q.get("gabarit") is not None and G.a_un_gabarit(mod):',
     'if q.get("gabarit") is not None and G.a_un_gabarit(mod):', "4. une note EXISTANTE"),
    ("les écarts gardent tout (même les valeurs d'usine)", "api_web/gabarit.py", "api_web.gabarit",
     "return {k: v for k, v in valeurs.items() if k in usine and v != usine[k]}", "return dict(valeurs)", "5. une saisie de la page"),
    ("un refus du moteur tu (la valeur d'usine gardée sans le dire)", "api_web/gabarit.py", "api_web.gabarit",
     'refus.append(f"{msg} — gabarit du bureau : la valeur d\'usine est gardée")', "pass", "4. une valeur refusée"),
    ("une saisie hors du gabarit acceptée", "api_web/gabarit.py", "api_web.gabarit",
     'if "/".join(ch) not in mod.GABARIT:', "if False:", "5. une saisie HORS du gabarit"),
]
for nom, fichier, mod_nom, ancien, nouveau, cible in MUTATIONS:
    res = muter(fichier, ancien, nouveau, mod_nom)
    if res is None:
        chk(f"mutation · {nom} : la cible existe une fois dans {fichier}", False, ancien)
        continue
    rouges = [n for n, ok in res if not ok]
    chk(f"mutation · {nom} : le contrôle « {cible}… » rougit", any(n.startswith(f"mutation · {cible}") for n in rouges), rouges[:4])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
