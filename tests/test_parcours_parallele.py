# -*- coding: utf-8 -*-
"""Le parcours du navigateur EN MORCEAUX (outils/parcours_parallele.py) : il joue le MÊME parcours que le passage
complet, sans un contrôle de moins.

- chaque section du parcours (une garde « if (!FILTRE || FILTRE === 'x' …) » de site/web/verifier_site.js) est jouée par
  au moins un morceau, et chaque morceau est un vrai filtre du parcours (ou « demarrage ») — une section ajoutée au
  parcours sans morceau fait rougir ce contrôle ;
- le journal se lit : les contrôles, OK ou KO, leur NOM sans le détail mesuré, et la clé de comparaison qui neutralise les
  mesures d'un nom (« 18 pages, 3.8 s ») ;
- la couverture : un contrôle du passage de référence absent de l'union des morceaux est nommé, et le passage rougit ;
- les morceaux SEULS (ceux qui mesurent des temps, « demarrage ») commencent quand tous les autres ont fini, et rien ne
  tourne avec eux (un faux « jouer » note les heures) ;
- mutations : un morceau retiré, une section nouvelle sans morceau, la clé sans neutralisation, aucun morceau seul —
  chacune fait rougir le contrôle qui la garde.

    /usr/bin/python3.12 tests/test_parcours_parallele.py
"""
import importlib.util
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

_spec = importlib.util.spec_from_file_location("parcours_parallele", os.path.join(RACINE, "outils", "parcours_parallele.py"))
PP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PP)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


# ---------------------------------------------------------------- 1. les sections du parcours et les morceaux
SOURCE = open(os.path.join(RACINE, "site", "web", "verifier_site.js"), encoding="utf-8").read()


def gardes(source: str):
    """Les gardes des sections : pour chaque « if (!FILTRE || … ) » (ou la liste des plaques), les filtres qui la font
    jouer — les noms exacts (« FILTRE === 'x' », « [...].includes(FILTRE) ») et les préfixes (« FILTRE.startsWith('x') »,
    « `module:${k}` »)."""
    out = []
    for m in re.finditer(r"!FILTRE \|\|([^{\n]*)", source):
        corps = m.group(1)
        noms = set(re.findall(r"FILTRE === '([^']+)'", corps))
        for liste in re.findall(r"\[([^\]]*)\]\.includes\(FILTRE\)", corps):
            noms.update(re.findall(r"'([^']+)'", liste))
        prefixes = set(re.findall(r"FILTRE\.startsWith\('([^']+)'\)", corps))
        if "`module:${k}`" in corps:
            prefixes.add("module:")
        out.append((m.start(), noms, prefixes))
    return out


def couverte(garde, morceaux) -> bool:
    """Une section est jouée par l'un des morceaux : son nom exact, ou — pour une section à préfixe (« module: »,
    « formats: ») — le morceau qui joue toute la famille (« modules », « formats ») et qui figure dans ses noms."""
    _pos, noms, _prefixes = garde
    return bool(noms & set(morceaux))


G = gardes(SOURCE)
chk(f"le parcours a ses gardes de section, lues dans verifier_site.js ({len(G)})", len(G) >= 20, len(G))
non_couvertes = [(SOURCE[:pos].count("\n") + 1, sorted(noms)) for pos, noms, _ in G if not couverte((pos, noms, _), PP.MORCEAUX)]
chk("chaque section du parcours est jouée par au moins un morceau (outils/parcours_parallele.MORCEAUX)", not non_couvertes, non_couvertes)
filtres = set().union(*(noms for _, noms, _ in G)) | {"demarrage"}
inconnus = [m for m in PP.MORCEAUX if m not in filtres]
chk("chaque morceau est un vrai filtre du parcours (ou « demarrage »)", not inconnus, inconnus)
chk("les morceaux ne se répètent pas", len(set(PP.MORCEAUX)) == len(PP.MORCEAUX), PP.MORCEAUX)
chk("« demarrage » est un morceau : le passage complet le joue à la fin (verifier_demarrage.js)",
    "demarrage" in PP.MORCEAUX and "if (!FILTRE) {" in SOURCE and "jouerDemarrage(sousFiltresDemarrage(FILTRE))" in SOURCE)
# les seuls contrôles qu'aucun filtre ne joue seul : aucun, hors de « demarrage » (la garde « if (!FILTRE) » n'en a pas d'autre)
seuls_complet = [SOURCE[:m.start()].count("\n") + 1 for m in re.finditer(r"if \(!FILTRE\) \{", SOURCE)]
chk("une seule partie n'est jouée QUE par le passage complet : demarrage, que le morceau du même nom joue",
    len(seuls_complet) == 1, seuls_complet)

# mutations : un morceau retiré, une section nouvelle sans morceau
reduits = [m for m in PP.MORCEAUX if m != "bureau"]
chk("mutation · le morceau « bureau » retiré : sa section n'est plus couverte, le contrôle rougit",
    any(not couverte(g, reduits) for g in G))
mutant = SOURCE + "\n  if (!FILTRE || FILTRE === 'nouvelle_section') {\n  }\n"
chk("mutation · une section nouvelle (« nouvelle_section ») sans morceau : le contrôle rougit",
    any(not couverte(g, PP.MORCEAUX) for g in gardes(mutant)))

# ---------------------------------------------------------------- 2. le journal
JOURNAL = """---- garde
OK  accueil vide : il le dit
OK  poutre : le moteur démarre dans le navigateur  1234 ms
KO  note PDF · dalle · pc : le clic télécharge la note (« n.pdf », 18 pages, 3.8 s)  {"x":1}
une ligne qui n'est pas un contrôle
OK  deux  espaces dans le détail  {"a": "b  c"}
"""
lu = PP.lire_journal(JOURNAL)
chk("le journal se lit : quatre contrôles, OK ou KO, le nom sans le détail mesuré",
    lu == [("OK", "accueil vide : il le dit"), ("OK", "poutre : le moteur démarre dans le navigateur"),
           ("KO", "note PDF · dalle · pc : le clic télécharge la note (« n.pdf », 18 pages, 3.8 s)"), ("OK", "deux")], lu)
chk("la clé de comparaison neutralise les mesures d'un nom (« 18 pages, 3.8 s » → « # pages, # s »)",
    PP.cle_controle("le clic télécharge la note (« n.pdf », 18 pages, 3.8 s)") == "le clic télécharge la note (« n.pdf », # pages, # s)"
    and PP.cle_controle("le clic (7 pages, 4,1 s)") == PP.cle_controle("le clic (9 pages, 12,25 s)"))
vrai = PP.cle_controle
PP.cle_controle = lambda n: n
try:
    chk("mutation · la clé sans neutralisation : deux passages qui ne diffèrent que par une durée ne se reconnaissent plus",
        PP.cle_controle("le clic (7 pages, 4,1 s)") != PP.cle_controle("le clic (9 pages, 12,25 s)"))
finally:
    PP.cle_controle = vrai

# ---------------------------------------------------------------- 3. la couverture, sur deux journaux factices
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    ref = os.path.join(tmp, "ref.log")
    with open(ref, "w", encoding="utf-8") as fh:
        fh.write("OK  a\nOK  b (3 s)\nOK  c\n")
    morceaux = [{"morceau": "m1", "code": 0, "duree": 1.0, "ok": 2, "ko": [], "noms": ["a", "b (5 s)"], "journal": ""}]
    refs = {PP.cle_controle(n) for _, n in PP.lire_journal(open(ref, encoding="utf-8").read())}
    vus = {PP.cle_controle(n) for r in morceaux for n in r["noms"]}
    chk("couverture : le contrôle « c » de la référence, absent des morceaux, est nommé ; « b (3 s) » et « b (5 s) » sont le même",
        sorted(refs - vus) == ["c"], sorted(refs - vus))

# ---------------------------------------------------------------- 4. les morceaux SEULS : après les autres, sans personne
import threading  # noqa: E402

chk("« demarrage » est joué SEUL (il mesure des temps) ; c'est un vrai morceau", "demarrage" in PP.SEULS and set(PP.SEULS) <= set(PP.MORCEAUX), PP.SEULS)


def ordre_joue(seuls) -> dict:
    """Les heures (début, fin) de chaque morceau, joués par PP.principal avec un faux « jouer » (aucun navigateur) ;
    ``seuls`` remplace PP.SEULS le temps de l'essai, le témoin des durées est écrit à part."""
    heures, verrou = {}, threading.Lock()
    vrai_jouer, vrais_seuls, vrai_temoin = PP.jouer, PP.SEULS, PP.TEMOIN

    def faux(base, dossier, morceau):
        t0 = time.time()
        time.sleep(0.15)
        with verrou:
            heures[morceau] = (t0, time.time())
        return {"morceau": morceau, "code": 0, "duree": 0.15, "ok": 1, "ko": [], "noms": [f"{morceau} : un contrôle"], "journal": ""}
    with tempfile.TemporaryDirectory() as tmp:
        PP.jouer, PP.SEULS, PP.TEMOIN = faux, seuls, os.path.join(tmp, "parcours.json")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = PP.principal(["http://x", tmp, "-j", "3", "demarrage", "garde", "poutre", "bureau"])
        finally:
            PP.jouer, PP.SEULS, PP.TEMOIN = vrai_jouer, vrais_seuls, vrai_temoin
    return {"code": code, "heures": heures}


import contextlib  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402

o = ordre_joue(PP.SEULS)
h = o["heures"]
autres = [v for k, v in h.items() if k != "demarrage"]
chk("« demarrage » commence quand les trois autres ont fini, et rien ne tourne avec lui (les trois autres, eux, tournent ensemble)",
    o["code"] == 0 and len(h) == 4 and h["demarrage"][0] >= max(f for _, f in autres) - 1e-6
    and max(d for d, _ in autres) < min(f for _, f in autres), {k: (round(a % 100, 2), round(b % 100, 2)) for k, (a, b) in h.items()})
m = ordre_joue(())
hm = m["heures"]
chk("mutation · aucun morceau seul : « demarrage » tourne avec les autres — le contrôle rougit",
    not (hm["demarrage"][0] >= max(f for k, (_, f) in hm.items() if k != "demarrage") - 1e-6))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:200])
sys.exit(1 if KO else 0)
