# -*- coding: utf-8 -*-
"""Barres en acier — LA COMPARAISON DES DEUX MÉTHODES (acier/barre/comparaison.py, lot C, C0f) : les sept sujets du brief
v12 § 9, chaque identifiant de vérification rangé dans UN sujet, la forme de chaque ligne.

  1. LE RANGEMENT — sur quinze barres d'essai (barre chargée et efforts imposés, laminés, PRS, tubes, âme élancée,
     raidisseurs, forces rapprochées, refus) : tout identifiant que le moteur rend (méthodes et âme) est rangé dans un
     sujet, ou est le refus d'une méthode ; et chaque identifiant (ou motif) des sujets est rendu par au moins une barre —
     un identifiant inconnu ROUGIT (avant le lot C, ``pire()`` rendait None et la ligne disparaissait en silence).
  2. LA FORME — {"sujet", "historique", "ec3", "commun", "ecart"} ; une case {"taux", "statut", "gouvernant"} : le plus fort
     taux JUGÉ, le pire statut ; l'écart en points de % ; plus de ligne « Le plus défavorable » ; ce que le moteur rend.
  3. LES CAS — la console sous N + M (E-661 non couvert : la case « Flexion composée » n'est jamais « ok ») ; E-SECTION
     (l'Eurocode 3 refuse la section : « non couvert » sur chaque ligne, pas de ligne à lui) ; une méthode bloquée (ses
     taux pour information ne sont pas jugés) ; le refus du moteur.
  4. MUTATIONS — chaque altération plausible du texte fait rougir un contrôle (comparaison.py, et un producteur
     d'identifiant : ec3_flambement.py).

    /usr/bin/python3.12 -B tests/test_barre_comparaison.py
"""
import importlib
import importlib.util
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(RACINE)
sys.path.insert(0, RACINE)

import acier.barre as PAQUET  # noqa: E402
from acier.barre import comparaison as CP  # noqa: E402
from acier.barre import entree as EN  # noqa: E402
from acier.barre import moteur as MO  # noqa: E402
from acier.barre import resultat as RE  # noqa: E402

DOSSIER = os.path.join(RACINE, "acier", "barre")
OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("  OK    " if cond else "  ECHEC ") + nom + (f"   [{str(info)[:400]}]" if info and not cond else ""))


def proche(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def q(v, a, b):
    return {"type": "q", "v": v, "a": a, "b": b}


def P_(v, a, **k):
    return dict({"type": "P", "v": v, "a": a}, **k)


def G(*charges, N=0.0):
    return {"nom": "G", "nature": "permanent", "N": N, "charges": list(charges)}


def Q(*charges):
    return {"nom": "Q", "nature": "exploitation_A", "charges": list(charges)}


def barre(**k):
    b = EN.nouvelle_barre("B")
    b["maintiens"] = [{"x": 3.0, "genre": "complet"}]
    b.update(k)
    return b


def efforts(profil, service, elu, L=5.0, **k):
    return barre(profil=profil, mode="efforts", flambement={"y": {"L": L}, "z": {"L": L}}, Lc=L, efforts={"service": service, "elu": elu}, **k)


PRS_MONO = {"b1": 300, "t1": 30, "hw": 540, "tw": 15, "b2": 200, "t2": 30, "a": 6, "bords": "oxycoupes"}
# les QUINZE barres d'essai : chacune rend des identifiants que les autres ne rendent pas (le contrôle 1 le demande)
BARRES = {
    "R2 (brief § 11.3)": barre(poids_propre=False, maintiens=[], appuis={"mode": "poses", "ss": 0.0, "c": 0.0},
                               cas=[G(q(2.42, 0.0, 6.0), P_(10.0, 3.0, ss=0.0)), Q(P_(15.0, 3.0, ss=0.0))]),
    "console 3 m, N 50 kN": barre(modele="console", travees=[3.0], maintiens=[], cas=[G(q(5.0, 0.0, 3.0), N=50.0), Q(q(5.0, 0.0, 3.0))]),
    "poteau HEB 200 (N, M_y, M_z, V_z, V_y)": efforts("HEB 200", {"N": 300.0, "My": 20.0, "Mz": 5.0, "Vz": 30.0, "Vy": 10.0},
                                                       {"N": 420.0, "My": 28.0, "Mz": 7.0, "Vz": 42.0, "Vy": 14.0}),
    "IPE 300, 400 kN près d'un appui (E-MV)": barre(travees=[2.0], maintiens=[], poids_propre=False, cas=[G(P_(400.0, 0.5))]),
    "PRS monosymétrique comprimé (E-Nb-T)": efforts(None, {"N": 100.0, "My": 300.0}, {"N": 140.0, "My": 400.0}, L=10.0, prs=PRS_MONO, nuance="S355"),
    "PRS aux bords non saisis (ELS bloqué)": efforts(None, {"My": 300.0, "Vz": 100.0}, {"My": 400.0, "Vz": 140.0}, L=8.0, nuance="S355",
                                                     prs=dict(PRS_MONO, b2=300, bords=None)),
    "continue HEA 200, trois travées": barre(profil="HEA 200", modele="continue", travees=[5.0, 4.0, 3.0], maintiens=[{"x": 2.5, "genre": "haut"}],
                                             cas=[G(q(6.0, 0.0, 12.0)), {"nom": "Q", "nature": "exploitation_B", "charges": [q(8.0, 0.0, 5.0), P_(12.0, 7.0)]}]),
    "raidisseur sous 150 kN": barre(appuis={"mode": "poses", "ss": 100.0, "c": 0.0}, cas=[G(q(5.0, 0.0, 6.0), P_(150.0, 3.0, ss=0.0)), Q(q(5.0, 0.0, 6.0))]),
    "âme élancée IPE 750x134 S355 sous N": barre(profil="IPE 750x134", nuance="S355", travees=[4.0], maintiens=[{"x": 1.0}, {"x": 2.0}, {"x": 3.0}],
                                                appuis={"mode": "poses", "ss": 100.0, "c": 50.0},
                                                cas=[G(P_(900.0, 1.0, ss=100.0), P_(900.0, 3.0, ss=100.0), N=50.0)]),
    "forces rapprochées (A-GR)": barre(appuis={"mode": "poses", "ss": 100.0, "c": 0.0}, cas=[G(q(2.0, 0.0, 6.0), P_(60.0, 3.0, ss=20.0), P_(60.0, 3.05, ss=20.0))]),
    "charge sur l'appui (transmise, -B)": barre(appuis={"mode": "poses", "ss": 100.0, "c": 100.0}, cas=[G(q(2.0, 0.0, 6.0), P_(60.0, 0.0, ss=100.0))]),
    "PRS à âme très élancée (A-S)": barre(prs={"b1": 400, "t1": 30, "hw": 1500, "tw": 6, "b2": 400, "t2": 30, "a": 5, "bords": "oxycoupes"}, nuance="S355",
                                          travees=[10.0], maintiens=[{"x": 2.5}, {"x": 5.0}, {"x": 7.5}], appuis={"mode": "ame", "ss": None, "c": None},
                                          cas=[G(q(20.0, 0.0, 10.0))]),
    "SHS 400x400x6 S460 sous V_y (A-Vb-y)": efforts("SHS 400x400x6 (froid)", {"Vy": 100.0}, {"Vy": 150.0}, L=3.0, nuance="S460"),
    "CHS 508x6.3 S355 (classe 4 : E-SECTION)": barre(profil="CHS 508x6.3", nuance="S355"),
    "HD 400x818 S355 (refus de l'âme, E-SECTION)": barre(profil="HD 400x818.0", nuance="S355"),
}
for b_ in BARRES.values():
    if b_.get("prs"):
        b_.pop("profil", None)


def resultats():
    """Les quinze barres passées au moteur (le moteur du paquet tel qu'il est chargé) : {nom: résultat}."""
    return {k: sys.modules["acier.barre.moteur"].verifier(b) for k, b in BARRES.items()}


def rendus(R):
    return {v["id"] for r in R.values() for x in list(r["methodes"].values()) + [r["ame"]] for v in x["verifications"]}


def cles_ligne(li):
    return set(li) == {"sujet", "historique", "ec3", "commun", "ecart"}


def case_ok(c):
    return c is None or (set(c) == {"taux", "statut", "gouvernant"} and c["statut"] in RE.STATUTS and isinstance(c["gouvernant"], str))


def controles(cp, R):
    """[(nom, vrai ?, info)] — rejoués sous chaque mutation, avec le module ``cp`` (comparaison.py, muté ou non)."""
    out = []
    ids = rendus(R)
    inconnus = sorted(i for i in ids if cp.sujet_de(i) is None and i not in cp.REFUS)
    out.append(("1. tout identifiant que le moteur rend sur les quinze barres (méthodes et âme) est rangé dans un sujet, ou est le refus "
                "d'une méthode — un identifiant inconnu rougit", ids and not inconnus, inconnus))
    non_vus = [m for m in cp.ranges() if not any(re.fullmatch(m, i) for i in ids)]
    out.append(("1. chaque identifiant (ou motif) des sujets est rendu par au moins une barre d'essai : aucun nom perdu ou renommé dans un "
                "bloc, aucun sujet vide en silence", not non_vus, non_vus))
    noms = [s[0] for s in cp.SUJETS]
    out.append(("1. les sept sujets du brief § 9, dans son ordre", noms == ["Flambement, axe faible", "Flambement, axe fort", "Flexion composée",
                                                                         "Déversement", "Cisaillement", "Âme sous charges", "Raidisseurs"], noms))
    faux = []
    for k, r in R.items():
        lignes = cp.comparer(r["methodes"], r["ame"])
        for li in lignes:
            th, te = ((li[c] or {}).get("taux") for c in ("historique", "ec3"))
            if not cles_ligne(li) or not all(case_ok(li[c]) for c in ("historique", "ec3", "commun")):
                faux.append((k, li["sujet"], "forme"))
            elif all(li[c] is None for c in ("historique", "ec3", "commun")):
                faux.append((k, li["sujet"], "ligne vide"))
            elif (li["ecart"] is None) != (th is None or te is None) or (li["ecart"] is not None and not proche(li["ecart"], 100.0 * (te - th))):
                faux.append((k, li["sujet"], "écart", li["ecart"], th, te))
        if [li["sujet"] for li in lignes] != [n for n in noms if n in {li["sujet"] for li in lignes}]:
            faux.append((k, "ordre"))
        # la case : le plus fort taux JUGÉ et le pire statut des vérifications du sujet, recalculés ici
        for li in lignes:
            for c, x in (("historique", r["methodes"]["historique"]), ("ec3", r["methodes"]["ec3"]), ("commun", r["ame"])):
                if li[c] is None or li[c]["gouvernant"] in cp.REFUS or x.get("refus"):
                    continue
                motifs = next(s for s in cp.SUJETS if s[0] == li["sujet"])[1 + ("historique", "ec3", "commun").index(c)]
                vs = [v for v in x["verifications"] if any(re.fullmatch(m, v["id"]) for m in motifs)]
                juges = [v["taux"] for v in vs if v["statut"] in RE.JUGES and v["taux"] is not None]
                if li[c]["statut"] != RE.pire_statut(v["statut"] for v in vs) or li[c]["taux"] != (max(juges) if juges else None):
                    faux.append((k, li["sujet"], c, li[c], [(v["id"], v["statut"], v["taux"]) for v in vs]))
    out.append(("2. chaque ligne : {sujet, historique, ec3, commun, ecart} ; chaque case : le plus fort taux JUGÉ (ok, nok, hors domaine), le "
                "pire statut, sa vérification ; l'écart = taux Eurocode 3 − taux ELS en points (None si l'un manque) ; jamais une ligne vide ; "
                "l'ordre du § 9", not faux, faux[:4]))
    lignes = {k: cp.comparer(r["methodes"], r["ame"]) for k, r in R.items()}
    out.append(("2. plus de ligne « Le plus défavorable » ni de ligne de l'âme à part (elles redisaient les verdicts et le bloc de l'âme)",
                all(li["sujet"] in noms for ls in lignes.values() for li in ls), [li["sujet"] for ls in lignes.values() for li in ls if li["sujet"] not in noms]))
    co = {li["sujet"]: li for li in lignes["console 3 m, N 50 kN"]}
    fc = co.get("Flexion composée") or {}
    e_ = {v["id"]: v for v in R["console 3 m, N 50 kN"]["methodes"]["ec3"]["verifications"]}
    out.append(("3. la console de 3 m sous N = 50 kN : la case Eurocode 3 « Flexion composée » est « non couvert » (E-661, DÉCISION BUREAU "
                "n° 9), son taux le plus fort JUGÉ (E-N, E-My, E-NM) — jamais « ok »",
                (fc.get("ec3") or {}).get("statut") == RE.NON_COUVERT and e_["E-661"]["statut"] == RE.NON_COUVERT
                and proche(fc["ec3"]["taux"], max(e_[i]["taux"] for i in ("E-N", "E-My", "E-NM"))) and fc["ec3"]["gouvernant"] in ("E-N", "E-My", "E-NM"),
                fc.get("ec3")))
    ch = lignes["CHS 508x6.3 S355 (classe 4 : E-SECTION)"]
    out.append(("3. E-SECTION (l'Eurocode 3 ne couvre pas la section : CHS de classe 4) : pas de ligne à lui ; sur chaque ligne, la case "
                "Eurocode 3 est « non couvert », sans taux, E-SECTION ; la méthode ELS garde ses taux",
                ch and all(li["ec3"] == {"taux": None, "statut": RE.NON_COUVERT, "gouvernant": "E-SECTION"} for li in ch if li["ec3"] is not None)
                and sum(1 for li in ch if li["ec3"] is not None) >= 3 and any((li["historique"] or {}).get("taux") for li in ch)
                and not any("section" in li["sujet"].lower() for li in ch), ch))
    pb = lignes["PRS aux bords non saisis (ELS bloqué)"]
    out.append(("3. une méthode BLOQUÉE (les bords des semelles d'un PRS à saisir) : ses cases sont « bloque », sans taux — ses taux, calculés "
                "pour information, ne sont pas jugés — ; l'écart manque",
                pb and all(li["historique"]["statut"] == RE.BLOQUE and li["historique"]["taux"] is None and li["ecart"] is None
                           for li in pb if li["historique"] is not None) and any(li["historique"] is not None for li in pb), pb))
    r0 = R["R2 (brief § 11.3)"]
    refus = cp.comparer({"historique": MO._refusee("historique", "NBN B 51-001", "un refus"), "ec3": r0["methodes"]["ec3"]}, r0["ame"])
    out.append(("3. le REFUS du moteur (une méthode « Vérification impossible ») : « non couvert » sur chaque ligne où elle a un sujet, sans taux",
                refus and all(li["historique"]["statut"] == RE.NON_COUVERT and li["historique"]["taux"] is None for li in refus if li["historique"] is not None)
                and sum(1 for li in refus if li["historique"] is not None) >= 4, refus))
    out.append(("2. le moteur rend CETTE comparaison (moteur.verifier → comparaison.comparer), et rien quand une méthode manque",
                all(r["comparaison"] == CP.comparer(r["methodes"], r["ame"]) for r in R.values())
                and cp.comparer({"ec3": r0["methodes"]["ec3"]}, r0["ame"]) == [], ""))
    return out


# ------------------------------------------------------------------ 4. les mutations
def charger(src: str, nom: str = "comparaison") -> types.ModuleType:
    chemin = os.path.join(DOSSIER, nom + ".py")
    mod = types.ModuleType(f"acier.barre.{nom}_mute")
    mod.__file__, mod.__package__ = chemin, "acier.barre"
    exec(compile(src, chemin, "exec"), mod.__dict__)
    return mod


def _dependants(cible):
    """Les modules du paquet qui emploient ``cible`` (directement ou non), chacun après ce qu'il importe."""
    dep = {}
    for f in sorted(os.listdir(DOSSIER)):
        if f.endswith(".py") and f != "__init__.py":
            s = open(os.path.join(DOSSIER, f), encoding="utf-8").read()
            dep[f[:-3]] = set(re.findall(r"^\s*from \. import (\w+)", s, re.M)) | set(re.findall(r"^\s*from \.(\w+) import", s, re.M))
    touches, change = {cible}, True
    while change:
        change = False
        for k, v in dep.items():
            if k not in touches and v & touches:
                touches.add(k)
                change = True
    touches.discard(cible)
    ordre, faits = [], {cible}
    while touches - set(ordre):
        for k in sorted(touches - set(ordre)):
            if dep[k] & touches <= set(ordre) | faits:
                ordre.append(k)
    return ordre


def _installer(nom, mod):
    sys.modules[f"acier.barre.{nom}"] = mod
    setattr(PAQUET, nom, mod)
    for k in _dependants(nom):
        if f"acier.barre.{k}" in sys.modules:
            importlib.reload(sys.modules[f"acier.barre.{k}"])


MUTATIONS = [
    # (fichier, nom, ancien, nouveau) — comparaison.py se rejoue sur les résultats gardés ; un producteur, sur les barres recalculées
    ("comparaison", "un sujet disparaît (« Raidisseurs »)", '    ("Raidisseurs", (), (), (r"A-L\\d+-R", r"A-L\\d+-Rb")),\n', ""),
    ("comparaison", "un identifiant retiré de son sujet (E-NM)", '"E-My", "E-Mz", "E-MV", "E-NM", "E-661"', '"E-My", "E-Mz", "E-MV", "E-661"'),
    ("comparaison", "un identifiant de SUJETS renommé (« H-DEV » → « H-DEVERSEMENT »)", '("H-DEV",)', '("H-DEVERSEMENT",)'),
    ("comparaison", "un motif de lieu oublié (-FM : force concentrée et flexion)", 'r"A-L\\d+-FM", ', ""),
    ("comparaison", "E-SECTION rangé comme les autres (plus de « non couvert » partout)", 'REFUS = ("E-SECTION",)', "REFUS = ()"),
    ("comparaison", "le refus du moteur oublié", 'bool(r.get("refus")) or ', ""),
    ("comparaison", "le taux d'une case pris parmi les non jugés", 'v["statut"] in RE.JUGES and v["taux"] is not None', 'v["taux"] is not None'),
    ("comparaison", "le statut de la case : le premier, pas le pire", '"statut": RE.pire_statut(v["statut"] for v in vs)', '"statut": vs[0]["statut"]'),
    ("comparaison", "l'écart dans l'autre sens (ELS − Eurocode 3)", "100.0 * (te - th)", "100.0 * (th - te)"),
    ("comparaison", "une ligne vide gardée", "        if all(li[k] is None for k in COTES):\n            continue\n", ""),
    ("comparaison", "les sujets triés par nom (l'ordre du § 9 perdu)", "    for nom, h, e, a in SUJETS:", "    for nom, h, e, a in sorted(SUJETS):"),
    # (intégration du lot B, point 18 ; déplacé de tests/test_barre_moteur.py) l'identifiant renommé chez son PRODUCTEUR
    ("ec3_flambement", "« E-Nb-T » renommé par la vérification qui l'écrit (flexion-torsion)",
     '            J.verif("E-Nb-T", "Flambement par flexion-torsion"', '            J.verif("E-Nb-FT", "Flambement par flexion-torsion"'),
]


def muter(R, fichier, ancien, nouveau):
    """Les contrôles qui rougissent sous la mutation ; None si le texte n'y est pas une seule fois."""
    chemin = os.path.join(DOSSIER, fichier + ".py")
    src = open(chemin, encoding="utf-8").read()
    if src.count(ancien) != 1:
        return None
    if fichier == "comparaison":
        try:
            return [n for n, ok, _ in controles(charger(src.replace(ancien, nouveau)), R) if not ok]
        except Exception as e:  # noqa: BLE001 — une mutation qui fait planter un contrôle le fait rougir aussi
            return [f"exception {type(e).__name__}: {e}"]
    orig = sys.modules[f"acier.barre.{fichier}"]
    try:
        _installer(fichier, charger(src.replace(ancien, nouveau), fichier))
        return [n for n, ok, _ in controles(sys.modules["acier.barre.comparaison"], resultats()) if not ok]
    except Exception as e:  # noqa: BLE001
        return [f"exception {type(e).__name__}: {e}"]
    finally:
        _installer(fichier, orig)


def main():
    print("== 1 à 3. le rangement, la forme, les cas")
    R = resultats()
    for nom, ok, info in controles(CP, R):
        chk(nom, ok, info)
    print("== 4. mutations")
    for fichier, nom, ancien, nouveau in MUTATIONS:
        rouges = muter(R, fichier, ancien, nouveau)
        if rouges is None:
            chk(f"mutation « {nom} » : le texte à muter est dans {fichier}.py, une seule fois", False, "texte absent ou répété")
        else:
            chk(f"mutation « {nom} » ({fichier}.py) : {len(rouges)} contrôle(s) rougissent", bool(rouges), "AUCUN contrôle ne rougit")
    chk("après les mutations, le paquet est rendu : les contrôles repassent tous",
        all(ok for _, ok, _ in controles(sys.modules["acier.barre.comparaison"], resultats())))
    print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
    for nom, info in KO:
        print("   -", nom, "|", str(info)[:300])
    return 1 if KO else 0


if __name__ == "__main__":
    sys.exit(main())
