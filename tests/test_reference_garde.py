# -*- coding: utf-8 -*-
"""La GARDE du regel des photos (tests/test_reference_refonte.py, ``garde`` et ``geler``) : refaire la photo d'un module
est permis quand seul l'ÉCRAN change ; un écart de CALCUL fait refuser ``--geler``, sauf une raison, alors écrite dans
la photo (clé « regels »). Révision du 08/11/2026 de docs/site/REFONTE_ETABLI.md (règle B.2.1 revue).

1. ``garde`` sur des photos construites : la mise en page seule (un libellé changé, un bloc déplacé) ne compte pas ;
   un état changé, un message, un avis retiré, un verdict changé, une valeur qui ne se voit plus, un cas retiré comptent.
2. ``geler`` sur une copie des photos de la poutre, dans un dossier temporaire : une photo dont l'état a été altéré est
   REFUSÉE sans raison (le fichier reste intact à l'octet), acceptée avec une raison (écrite, avec ses écarts) ; une
   photo inchangée se réécrit à l'identique.
3. Les trois regels faits avant la garde (poutre, dalle, prédalle) sont écrits dans leurs photos, avec leurs écarts.

    /usr/bin/python3.12 tests/test_reference_garde.py
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

ICI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("reference_refonte", os.path.join(ICI, "test_reference_refonte.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append(nom)
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else "\n        | " + str(info)[:1500]))


with open(os.path.join(T.DOSSIER, "poutre.json"), encoding="utf-8") as f:
    POUTRE = json.load(f)
CAS = POUTRE["cas"][1]                                # « poutre 30 × 60 réaliste » : un résultat, des calculs, un état


def fichier(*photos):
    return {"cas": [{"nom": n, "photo": p} for n, p in photos]}


def variante(modif):
    p = copy.deepcopy(CAS["photo"])
    modif(p)
    return p


# ---------------------------------------------------------------------- 1. la garde, sur des photos construites
base = fichier((CAS["nom"], CAS["photo"]))
chk("garde : une photo identique, aucun écart", T.garde(base, copy.deepcopy(base)) == [])


def renomme(p):                                       # un libellé change, aucune valeur
    p["resultat"][0][0] = p["resultat"][0][0] + " (libellé revu)"


chk("garde : un libellé changé (mise en page) n'est pas un écart de calcul",
    T.garde(base, fichier((CAS["nom"], variante(renomme)))) == [], T.garde(base, fichier((CAS["nom"], variante(renomme)))))


def deplace(p):                                       # un groupe du calcul passe dans le résultat : ses valeurs se voient
    p["resultat"].append(p["calculs"].pop(0))


chk("garde : un bloc déplacé des fenêtres de calcul au résultat garde ses valeurs — pas d'écart",
    T.garde(base, fichier((CAS["nom"], variante(deplace)))) == [], T.garde(base, fichier((CAS["nom"], variante(deplace)))))


def retire(p):                                        # les fenêtres de calcul et le résultat (sauf sa tête) quittent l'écran
    p["calculs"] = []
    p["resultat"] = p["resultat"][:1]


e_ = T.garde(base, fichier((CAS["nom"], variante(retire))))
chk("garde : une valeur retirée de l'écran est nommée (« valeurs qui ne se voient plus »)",
    any("valeurs qui ne se voient plus" in x for x in e_), e_)
e_ = T.garde(base, fichier((CAS["nom"], variante(lambda p: p["etat"]["poutres"][0].update(b=99)))))
chk("garde : l'état de la note changé est un écart", any("etat changé" in x for x in e_), e_)
e_ = T.garde(base, fichier((CAS["nom"], variante(lambda p: p.update(messages=p["messages"][:-1] + ["refusé"])))))
chk("garde : un message changé est un écart", any("messages changé" in x for x in e_), e_)
e_ = T.garde(base, fichier((CAS["nom"], variante(lambda p: p.update(avis=(p.get("avis") or []) + ["⚠ nouvel avis"])))))
chk("garde : un avis apparu (ou retiré) est un écart", any("avis changés" in x for x in e_), e_)
e_ = T.garde(base, fichier((CAS["nom"], variante(lambda p: p["resume"].update(statut="Données incomplètes")))))
chk("garde : un verdict changé (statut du résumé) est un écart, nommé avant → après",
    any("verdict, statut" in x and "Données incomplètes" in x for x in e_), e_)
chk("garde : un cas retiré est un écart ; un cas ajouté n'en est pas un",
    T.garde(base, fichier()) == [f"{CAS['nom']} : cas retiré"]
    and T.garde(base, fichier((CAS["nom"], CAS["photo"]), ("cas nouveau", CAS["photo"]))) == [])

# ---------------------------------------------------------------------- 2. geler, gardé (dossier temporaire)
tmp = tempfile.mkdtemp(prefix="garde_")
try:
    chemin = os.path.join(tmp, "poutre.json")
    altere = copy.deepcopy(POUTRE)
    altere["cas"][1]["photo"]["etat"]["poutres"][0]["b"] = 99   # la photo gelée dit autre chose que le calcul du jour
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(T._dump(altere) + "\n")
    avant = open(chemin, "rb").read()
    with contextlib.redirect_stdout(io.StringIO()):
        refuse = T.geler("poutre", "", tmp)
    chk("geler : un écart de calcul SANS raison est refusé, la photo gelée reste intacte à l'octet",
        refuse is False and open(chemin, "rb").read() == avant)
    with contextlib.redirect_stdout(io.StringIO()):
        accepte = T.geler("poutre", "essai de la garde", tmp)
    with open(chemin, encoding="utf-8") as f:
        relu = json.load(f)
    dernier = (relu.get("regels") or [{}])[-1]
    chk("geler : avec une raison, la photo est écrite, et la raison et ses écarts avec elle (clé « regels »)",
        accepte is True and dernier.get("raison") == "essai de la garde"
        and any("etat changé" in x for x in dernier.get("ecarts", [])) and relu["cas"][1]["photo"]["etat"]["poutres"][0]["b"] != 99,
        dernier)
    shutil.copy(os.path.join(T.DOSSIER, "poutre.json"), chemin)
    avant = open(chemin, "rb").read()
    with contextlib.redirect_stdout(io.StringIO()):
        meme = T.geler("poutre", "", tmp)
    chk("geler : une photo inchangée se réécrit à l'identique (aucun regel ajouté)", meme is True and open(chemin, "rb").read() == avant)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---------------------------------------------------------------------- 3. les regels d'avant la garde, écrits
HISTOIRE = {"poutre": ["98085c7", "50c37dd"], "dalle": ["b3ec83a"], "predalle": ["b3ec83a"]}
for cle, commits in HISTOIRE.items():
    with open(os.path.join(T.DOSSIER, f"{cle}.json"), encoding="utf-8") as f:
        regels = json.load(f).get("regels") or []
    chk(f"{cle} : les regels faits avant la garde sont écrits dans la photo ({', '.join(commits)}), raison et écarts",
        [r.get("commit") for r in regels[:len(commits)]] == commits and all(r.get("raison") and r.get("ecarts") for r in regels),
        [(r.get("commit"), r.get("raison", "")[:60]) for r in regels])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
sys.exit(1 if KO else 0)
