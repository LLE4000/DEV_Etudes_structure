# -*- coding: utf-8 -*-
"""La BIBLIOTHÈQUE dans une note (docs/site/BIBLIOTHEQUES.md § 2.4 et § 2.5), vérifiée en Python ordinaire :

1. l'action « bibliotheque » du routeur rend la bibliothèque de la famille du module pour une NOTE : préremplie avec ce
   que la poutre sait (béton, Ø des barres et de l'étrier, largeur, « poutre »), l'hôte « note » (ses boutons), et les
   CIBLES que « Utiliser cette valeur » peut écrire (l'enrobage en cm, la classe de béton, la nuance) — jamais une
   classe que le module ne connaît pas (dite impossible, avec la raison) ; les paramètres de l'écran priment ;
2. « Utiliser cette valeur » passe par « modifier » : rien ne change avant ;
3. les FICHES (api_web/fiches.py) : « ajouter » copie la fiche figée dans l'état (identifiant, date du jour), le bloc
   « Justifications (n) » et la vue « fiche-<i> » la montrent ; « monter », « descendre », « retirer » ; la clé
   ``fiches`` est ABSENTE quand la note n'en a aucune ; la relecture est stricte (clé inconnue, sorte, extrait non
   rectangulaire, case hors du tableau, trop de fiches) et la note reste intacte sur un refus ; un titre piégé est
   échappé partout ; le calcul et le résumé ne changent pas quand des fiches sont là ;
4. la note PDF : après les planches, les pages « Justifications de la bibliothèque » (une ligne au sommaire), le titre
   et la valeur de chaque fiche s'y lisent (pdftotext) ; sans fiche, pas de page en plus ;
5. la brique ``ndc_pdf.kit.table`` : les notes garde-corps et soudures l'emploient (une source) ;
6. mutations : une relecture laxiste (case hors du tableau, clé inconnue) fait rougir ; une cible qui ignore les classes
   du module aussi.

    python3 tests/test_api_web_fiches.py
"""
import copy
import datetime
import json
import os
import subprocess
import sys
import tempfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import biblio_beton as B  # noqa: E402
from api_web import fiches as FI  # noqa: E402
from api_web import note_poutre as NP  # noqa: E402
from api_web import poutre as P  # noqa: E402
from api_web import routeur as R  # noqa: E402
from beton.poutre import entree as EN  # noqa: E402

OK, KO = [], []
PIEGE = '<img src=x onerror=alert(1)>'


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:300]}"))


def req(**k):
    return json.loads(R.traiter(json.dumps(k)))


def leve(f, *a, **k):
    try:
        f(*a, **k)
    except Exception as e:  # noqa: BLE001
        return e
    return None


class Abime:
    """Abîme un attribut d'un module le temps d'un bloc (mutation)."""

    def __init__(self, mod, **attrs):
        self.mod, self.attrs, self.avant = mod, attrs, {}

    def __enter__(self):
        for k, v in self.attrs.items():
            self.avant[k] = getattr(self.mod, k)
            setattr(self.mod, k, v)

    def __exit__(self, *_):
        for k, v in self.avant.items():
            setattr(self.mod, k, v)


def texte_pdf(octets: bytes) -> str:
    fd, chemin = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(chemin, "wb") as f:
            f.write(octets)
        return subprocess.run(["pdftotext", "-layout", chemin, "-"], capture_output=True, text=True, check=True).stdout
    finally:
        os.remove(chemin)


def pages_pdf(octets: bytes) -> int:
    return octets.count(b"/Type /Page\n") + octets.count(b"/Type /Page ")


# ============================================================================================ 1. l'action « bibliotheque »
print("=== 1 · La bibliothèque d'une note : préremplie, hôte « note », cibles")
e = P.nouvel_etat()
for chemin, v in ((["beton"], "C25/30"), (["b"], "30"), (["enrobage"], "4"), (["sections", "0", "inf", "0", "d"], "20"), (["sections", "0", "etriers", "0", "d"], "8"),
                  (["sections", "0", "M_inf"], "90"), (["sections", "0", "V"], "40")):
    assert P.modifier(e, chemin, v) is None, chemin
r = req(module="poutre", action="bibliotheque", etat=e)
chk("l'action « bibliotheque » aboutit, sur la bibliothèque béton, rubrique « Enrobages » d'abord (c'est elle que la poutre attend)",
    r["ok"] and r["bibliotheque"] == "beton" and r["rubrique"] == "enrobages", r.get("erreur") or (r.get("bibliotheque"), r.get("rubrique")))
chk("préremplie depuis la poutre : élément poutre, C25/30, Ø 20 (le plus gros lit), étrier Ø 8, b = 300 mm",
    r["params"]["element"] == "poutre" and r["params"]["classe"] == "C25/30" and r["params"]["phi"] == 20 and r["params"]["phi_etrier"] == 8 and r["params"]["b"] == 300,
    {k: r["params"].get(k) for k in ("element", "classe", "phi", "phi_etrier", "b")})
chk("l'hôte est la note : « Inclure dans la note », « Inclure l'extrait », « Utiliser cette valeur » dans la bande",
    all(x in r["bande"] for x in ('data-inclure="fiche"', 'data-inclure="extrait"', 'data-utiliser="1"')), r["bande"][-400:])
chk("la case allumée est celle de la poutre (S4 calculée, XC1)", 'class="cur"' in r["tableau"] and r["tableau"].count('class="cur"') == 1)
c0 = (r.get("cibles") or [{}])[0]
chk("la cible de « Utiliser cette valeur » : le champ « c enrobage » de la poutre, c_nom en cm, la valeur d'aujourd'hui dite",
    c0.get("chemin") == ["enrobage"] and abs(c0.get("valeur", 0) * 10 - r["fiche"]["valeur"]) < 1e-9 and c0.get("possible") is True
    and "cm" in c0.get("texte", "") and c0.get("actuel") == "4 cm", c0)
r2 = req(module="poutre", action="bibliotheque", etat=e, rubrique="enrobages", params={"classe": "C40/50", "exposition": "XD1"})
chk("les paramètres de l'écran priment sur le préremplissage (C40/50, XD1), le reste vient toujours de la poutre (Ø 20)",
    r2["params"]["classe"] == "C40/50" and r2["params"]["exposition"] == "XD1" and r2["params"]["phi"] == 20, r2["params"])
ra = req(module="poutre", action="bibliotheque", etat=e, rubrique="ancrages")
chk("rubrique « Ancrages » : Ø 20, C25/30, B500 de la poutre ; aucune cible (aucun champ de la poutre ne reçoit l_bd)",
    ra["ok"] and ra["params"]["phi"] == 20 and ra["params"]["classe"] == "C25/30" and ra["params"]["fyk"] == 500 and ra["cibles"] == [], (ra.get("params"), ra.get("cibles")))
rb = req(module="poutre", action="bibliotheque", etat=e, rubrique="betons")
cb = (rb.get("cibles") or [{}])[0]
chk("rubrique « Bétons » : la classe de la poutre allumée ; la cible écrit « beton »", rb["params"]["classe"] == "C25/30" and cb.get("chemin") == ["beton"]
    and cb.get("valeur") == "C25/30" and cb.get("possible") is True, (rb.get("params"), cb))
rb2 = req(module="poutre", action="bibliotheque", etat=e, rubrique="betons", params={"classe": "C60/75"})
cb2 = (rb2.get("cibles") or [{}])[0]
chk("une classe que le module des poutres ne connaît pas (C60/75) : la cible est dite IMPOSSIBLE, avec la raison — rien ne s'écrit",
    cb2.get("possible") is False and "C60/75" in cb2.get("raison", "") and cb2.get("chemin") == ["beton"], cb2)
rs = req(module="poutre", action="bibliotheque", etat=e, rubrique="aciers")
cs = (rs.get("cibles") or [{}])[0]
chk("rubrique « Aciers » : B500B de la poutre ; la cible écrit la nuance (fyk = 500)", rs["params"]["nuance"] == "B500B" and cs.get("chemin") == ["fyk"] and cs.get("valeur") == 500, (rs.get("params"), cs))
rx = req(module="garde_corps", action="bibliotheque", etat=None)
chk("un module sans bibliothèque est refusé doucement (pas d'exception)", rx["ok"] is False and "bibliothèque" in rx["erreur"], rx)
ri = req(module="poutre", action="bibliotheque", etat=e, rubrique="inconnue")
chk("une rubrique inconnue ouvre la première de la bibliothèque, et le dit", ri["ok"] and ri["rubrique"] == B.RUBRIQUE_DEFAUT and "inconnue" in (ri.get("message") or ""), ri.get("message"))

# ============================================================================================ 2. « Utiliser cette valeur »
print("\n=== 2 · Utiliser cette valeur : une saisie ordinaire, rien avant")
chk("consulter ne change pas la note", r["ok"] and "etat" not in r)
ru = req(module="poutre", action="modifier", etat=e, chemin=c0["chemin"], valeur=c0["valeur"])
chk("« Utiliser cette valeur » = modifier(enrobage) : la poutre prend c_nom en cm (30 mm → 3 cm, elle avait 4), le calcul suit",
    ru["ok"] and abs(ru["etat"]["poutres"][0]["enrobage"] - c0["valeur"]) < 1e-9 and ru["etat"]["poutres"][0]["enrobage"] == 3.0 and c0["valeur"] == 3.0,
    (ru["etat"]["poutres"][0]["enrobage"], c0["valeur"]))

# ============================================================================================ 3. les fiches
print("\n=== 3 · Les fiches de la note")
f0 = {**r["fiche"], "sorte": "fiche", "rubrique": r["rubrique"], "bibliotheque": r["bibliotheque"]}
avant = copy.deepcopy(e)
rf = req(module="poutre", action="fiches", etat=e, quoi="ajouter", fiche=f0)
chk("« ajouter » copie la fiche dans l'état : une fiche, identifiant j1, datée d'aujourd'hui, le moteur nommé",
    rf["ok"] and not rf.get("message") and len(rf["etat"]["fiches"]) == 1 and rf["etat"]["fiches"][0]["id"] == "j1"
    and rf["etat"]["fiches"][0]["date"] == datetime.date.today().isoformat() and "Bibliothèque" in rf["etat"]["fiches"][0]["moteur"], rf.get("message") or rf["etat"].get("fiches"))
fi = rf["etat"]["fiches"][0]
chk("la fiche garde tout pour être relue sans le moteur : titre, symbole, valeur, texte, unité, phrase, paramètres, lignes, sources, extrait (case)",
    fi["titre"].startswith("Enrobage poutre") and fi["symbole"] == "c_nom" and fi["unite"] == "mm" and fi["texte"] and fi["phrase"] and fi["params"]["classe"] == "C25/30"
    and len(fi["lignes"]) >= 6 and all(li["texte"] for li in fi["lignes"] if isinstance(li["valeur"], (int, float)) and not isinstance(li["valeur"], bool))
    and fi["sources"] and fi["extrait"] and fi["extrait"]["case"] is not None, {k: fi[k] for k in ("titre", "symbole", "unite", "texte", "phrase")})
chk("le bloc « Justifications (1) » à gauche nomme la fiche et sa valeur, avec son menu ⋯ (monter grisé, descendre grisé, retirer)",
    'class="et-fiches"' in rf["donnees"] and '<span>Justifications</span><span class="n">1</span>' in rf["donnees"] and fi["titre"] in rf["donnees"]
    and 'data-vue-fiche="fiche-0"' in rf["donnees"] and 'data-fiche-monter="0" disabled' in rf["donnees"] and 'data-fiche-descendre="0" disabled' in rf["donnees"]
    and 'data-fiche-retirer="0"' in rf["donnees"], rf["donnees"][-900:])
vf = [v for v in rf["vues"] if v["cle"] == "fiche-0"]
chk("une vue « fiche-0 » (onglet J1) montre la fiche : titre, valeur, lignes, extrait avec sa case, sources ; la clé n'est pas celle d'une section",
    len(vf) == 1 and vf[0]["court"] == "J1" and fi["titre"] in vf[0]["svg"] and 'class="et-fiche-t"' in vf[0]["svg"] and 'class="cur"' in vf[0]["svg"]
    and "EN 1992-1-1" in vf[0]["svg"] and [v["cle"] for v in rf["vues"]][:1] == ["s0"], [v["cle"] for v in rf["vues"]])
chk("le calcul et le résumé ne changent pas quand des fiches sont là",
    rf["resume"] == req(module="poutre", action="calculer", etat=avant)["resume"] and rf["resultat"] == req(module="poutre", action="calculer", etat=avant)["resultat"])
e1 = rf["etat"]
rx = req(module="poutre", action="bibliotheque", etat=e1, rubrique="ancrages")
f1 = {**rx["fiche"], "sorte": "extrait", "rubrique": "ancrages", "bibliotheque": "beton"}
rf2 = req(module="poutre", action="fiches", etat=e1, quoi="ajouter", fiche=f1)
chk("une deuxième fiche (l'extrait du tableau des ancrages) : j2, le bloc dit 2, la vue « fiche-1 » met l'extrait en premier",
    rf2["ok"] and [f["id"] for f in rf2["etat"]["fiches"]] == ["j1", "j2"] and '<span class="n">2</span>' in rf2["donnees"]
    and rf2["vues"][-1]["cle"] == "fiche-1" and rf2["vues"][-1]["svg"].index('class="et-fiche-x"') < rf2["vues"][-1]["svg"].index('class="et-fiche-t"'), [f["id"] for f in rf2["etat"]["fiches"]])
e2 = rf2["etat"]
rm = req(module="poutre", action="fiches", etat=e2, quoi="monter", index=1)
chk("« monter » la seconde la met en tête (les identifiants suivent)", [f["id"] for f in rm["etat"]["fiches"]] == ["j2", "j1"])
rd = req(module="poutre", action="fiches", etat=rm["etat"], quoi="descendre", index=0)
chk("« descendre » la remet", [f["id"] for f in rd["etat"]["fiches"]] == ["j1", "j2"])
rr = req(module="poutre", action="fiches", etat=rd["etat"], quoi="retirer", index=0)
chk("« retirer » la première garde la seconde, renumérotée dans le bloc (J1) mais avec son identifiant", [f["id"] for f in rr["etat"]["fiches"]] == ["j2"] and 'data-vue-fiche="fiche-0"' in rr["donnees"])
rr2 = req(module="poutre", action="fiches", etat=rr["etat"], quoi="retirer", index=0)
chk("retirer la dernière : la clé « fiches » DISPARAÎT de l'état (une note sans fiche est une note d'avant), plus de bloc ni de vue",
    "fiches" not in rr2["etat"] and 'class="et-fiches"' not in rr2["donnees"] and not [v for v in rr2["vues"] if v["cle"].startswith("fiche")], list(rr2["etat"]))
chk("… et son état est celui d'avant toute fiche, à la clé près", rr2["etat"] == avant)
for quoi, index, nom in (("retirer", 5, "un index hors de la liste"), ("retirer", "x", "un index illisible"), ("monter", None, "sans index"), ("danser", 0, "une action inconnue")):
    out = req(module="poutre", action="fiches", etat=e2, quoi=quoi, index=index)
    if not (out["ok"] and out["message"] and out["etat"] == e2):
        chk(f"{nom} est refusé avec un message, l'état intact", False, out.get("message"))
        break
else:
    chk("un index hors liste, illisible, absent, une action inconnue : un message, l'état intact", True)
# la relecture stricte
mauvaises = [
    ("pas un dictionnaire", "texte"), ("une clé inconnue", {**f0, "script": 1}), ("une sorte inconnue", {**f0, "sorte": "image"}),
    ("sans titre", {**f0, "titre": ""}), ("un titre qui n'est pas un texte", {**f0, "titre": ["a"]}),
    ("une ligne sans dictionnaire", {**f0, "lignes": ["x"]}), ("une ligne à clé inconnue", {**f0, "lignes": [{**f0["lignes"][0], "html": "<b>"}]}),
    ("un extrait non rectangulaire", {**f0, "extrait": {**f0["extrait"], "lignes": [f0["extrait"]["lignes"][0][:-1]]}}),
    ("une case hors du tableau", {**f0, "extrait": {**f0["extrait"], "case": [99, 0]}}),
    ("une case à demi", {**f0, "extrait": {**f0["extrait"], "case": [0]}}),
    ("des paramètres qui ne sont pas un dictionnaire", {**f0, "params": [1, 2]}),
    ("une valeur infinie", {**f0, "valeur": float("inf")}),
]
fautes = []
for nom, f in mauvaises:
    out = req(module="poutre", action="fiches", etat=e, quoi="ajouter", fiche=f)
    if not (out["ok"] and out.get("message", "").startswith("Fiche refusée") and out["etat"] == e):
        fautes.append((nom, out.get("message"), out.get("erreur")))
chk(f"la relecture est STRICTE : {len(mauvaises)} fiches abîmées refusées avec un message, l'état intact", not fautes, fautes[:3])
trop = copy.deepcopy(e)
for _ in range(FI.MAX_FICHES):
    assert FI.gerer(trop, "ajouter", f0) is None
msg = FI.gerer(trop, "ajouter", f0)
chk(f"au plus {FI.MAX_FICHES} fiches : la suivante est refusée", msg and str(FI.MAX_FICHES) in msg and len(trop["fiches"]) == FI.MAX_FICHES, msg)
# normaliser : la clé absente, la clé relue, la clé abîmée
n0 = P.normaliser(avant)
chk("normaliser une note sans fiche ne crée pas la clé « fiches » (une note ancienne reste identique à l'octet)", "fiches" not in n0 and n0 == avant)
n1 = P.normaliser(copy.deepcopy(e2))
chk("normaliser une note à fiches les relit telles quelles", n1.get("fiches") == e2["fiches"])
n2 = P.normaliser({**copy.deepcopy(e2), "fiches": []})
chk("une liste vide disparaît", "fiches" not in n2)
ex = leve(P.normaliser, {**copy.deepcopy(e2), "fiches": "non"})
chk("« fiches » qui n'est pas une liste : la note est refusée avec une raison (jamais réparée en silence)", isinstance(ex, ValueError) and "fiches" in str(ex), ex)
rk = req(module="poutre", action="calculer", etat={**copy.deepcopy(e2), "fiches": [{"titre": "x", "sorte": "film"}]})
chk("… et le routeur le dit sans exception", rk["ok"] is False and "sorte" in rk["erreur"], rk.get("erreur"))
# l'échappement
rp = req(module="poutre", action="fiches", etat=e, quoi="ajouter", fiche={**f0, "titre": PIEGE, "phrase": PIEGE, "sources": [PIEGE], "lignes": [{**f0["lignes"][0], "nom": PIEGE, "regle": PIEGE}],
                                                                       "extrait": {**f0["extrait"], "titre": PIEGE, "source": PIEGE, "entetes": [PIEGE] + f0["extrait"]["entetes"][1:]}})
chk("un titre, une phrase, une source, une ligne, un extrait piégés sont échappés dans le bloc et dans la vue",
    rp["ok"] and PIEGE not in rp["donnees"] and not any(PIEGE in v["svg"] for v in rp["vues"]) and "&lt;img" in rp["donnees"], rp.get("message"))

# ============================================================================================ 4. la note PDF
print("\n=== 4 · La note PDF : les pages « Justifications de la bibliothèque »")
from modules import export_pdf as EP  # noqa: E402
from ndc_pdf import data as ND  # noqa: E402
infos = {"nom_projet": "Essai", "indice": "A", "date": "09/11/2026"}
sans = NP.pdf(copy.deepcopy(avant), infos)
avec = NP.pdf(copy.deepcopy(e2), infos)
t_sans, t_avec = texte_pdf(sans), texte_pdf(avec)
chk("sans fiche : la garde et une planche (2 pages), rien de la bibliothèque", pages_pdf(sans) == 2 and "Justifications" not in t_sans, pages_pdf(sans))
n_just = EP._style_ndc(2).pages_justifications(ND.construire_doc(infos), e2["fiches"])
chk("deux fiches : les pages de justifications après la planche (autant que le tracé à blanc en compte), annoncées au sommaire de la garde",
    n_just >= 1 and pages_pdf(avec) == 2 + n_just and "Justifications de la bibliothèque" in t_avec and "2 fiches" in t_avec, (pages_pdf(avec), n_just))
chk("les pages portent J1 et J2, le titre et la valeur de chaque fiche, la source de la norme ; la fiche des ancrages, plus longue qu'une colonne, se poursuit (« J2 (suite) »)",
    all(x in t_avec for x in ("J1", "J2", e2["fiches"][0]["titre"], e2["fiches"][1]["titre"], e2["fiches"][0]["texte"], "EN 1992-1-1", "J2 (suite)")), t_avec[-1500:])
trois = copy.deepcopy(e2)
assert FI.gerer(trois, "ajouter", f0) is None
t3 = NP.pdf(trois, infos)
chk("trois fiches : au moins autant de pages que deux (J3 prend la colonne libre ou une page), J3 dessinée, et jamais un débordement tu (la note sort)",
    pages_pdf(t3) >= pages_pdf(avec) and "J3" in texte_pdf(t3) and "3 fiches" in texte_pdf(t3), (pages_pdf(t3), pages_pdf(avec)))
rb_ = req(module="poutre", action="bibliotheque", etat=e, rubrique="betons")
large = copy.deepcopy(avant)
assert FI.gerer(large, "ajouter", {**rb_["fiche"], "sorte": "extrait", "rubrique": "betons", "bibliotheque": "beton"}) is None
t_large = texte_pdf(NP.pdf(large, infos))
chk("un extrait de douze colonnes (le tableau 3.1 calculé) tient dans sa colonne : ses classes et sa source s'y lisent",
    "tableau 3.1" in t_large and "C90/105" in t_large and "C25/30" in t_large, t_large[-800:])
rt = json.loads(R.traiter(json.dumps({"module": "poutre", "action": "telecharger", "etat": e2, "quoi": "note_pdf", "infos": {"nom": "Essai"}})))
chk("le routeur rend la note à fiches (base 64)", rt["ok"] and rt["fichier"]["type"] == "application/pdf")

# ============================================================================================ 5. la brique table du kit
print("\n=== 5 · ndc_pdf.kit.table, une source")
from ndc_pdf import kit as K  # noqa: E402
src_gc = open(os.path.join(RACINE, "acier", "garde_corps", "note.py"), encoding="utf-8").read()
src_sd = open(os.path.join(RACINE, "acier", "soudures", "note.py"), encoding="utf-8").read()
import re  # noqa: E402
chk("les notes garde-corps et soudures appellent kit.table (plus de copie locale du corps du tableau)",
    callable(getattr(K, "table", None)) and re.search(r"\bKIT\.table\(", src_gc) and re.search(r"\bKIT\.table\(", src_sd)
    and "for cells in lignes:" not in src_gc and "for cells in lignes:" not in src_sd)

# ============================================================================================ 6. mutations
print("\n=== 6 · Mutations")
with Abime(FI, _extrait=lambda x, ou: x):
    out = req(module="poutre", action="fiches", etat=e, quoi="ajouter", fiche={**f0, "extrait": {**f0["extrait"], "case": [99, 0]}})
    chk("mutation : une relecture laxiste de l'extrait laisse passer la case hors du tableau (le contrôle 3 rougirait)", out["ok"] and not out.get("message"))
with Abime(FI, CLES=frozenset(FI.CLES | {"script"})):
    out = req(module="poutre", action="fiches", etat=e, quoi="ajouter", fiche={**f0, "script": 1})
    chk("mutation : une clé admise de plus laisse passer la fiche à clé inconnue", out["ok"] and not out.get("message"))
with Abime(EN, classes_beton=lambda: {c: {} for c in B.CLASSES_BETON}):
    out = req(module="poutre", action="bibliotheque", etat=e, rubrique="betons", params={"classe": "C60/75"})
    chk("mutation : une cible qui croit le module capable de C60/75 la dit possible (le contrôle 1 rougirait)", (out.get("cibles") or [{}])[0].get("possible") is True)

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("  KO", nom, "|", str(info)[:300])
sys.exit(1 if KO else 0)
