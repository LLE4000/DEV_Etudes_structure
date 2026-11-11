# -*- coding: utf-8 -*-
"""Fondations de grues — le CONTRAT du fichier d'échange avec RBAR : le document et le fichier disent la même chose.

``docs/beton/FONDATIONS_GRUE_RBAR.md`` est le brief remis à la conversation qui développe RBAR : il nomme chaque
paramètre que le site exporte, donne son sens, et porte un fichier d'exemple complet (Greenwood, les armatures du plan
de référence). Deux copies d'une même information — le document, le fichier — finissent par diverger si rien ne les
lie. Ce fichier les lie :

 1. le fichier d'exemple du document EST celui que le site produit (régénéré : ``--regenerer``) ;
 2. chaque nom de paramètre que le site exporte — les quatre classeurs, la paroi, les seconds groupes de barres — est
    nommé dans le document ;
 3. les ajouts de la version 2 sont dans le fichier, sous le nom et avec la valeur que le document annonce ;
 4. le tableau « d'où vient chaque barre » dit les cellules et les armatures du fichier ;
 5. le fichier ne change que par ajouts : les paramètres de la version 1 sont toujours là ;
 6. des mutations : un nom retiré du document, un exemple retouché, un paramètre renommé dans l'export — chacune vue.

    /usr/bin/python3.12 tests/test_fondation_grue_contrat_rbar.py [--regenerer]
"""
import copy
import json
import os
import re
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import fondation_grue as FG  # noqa: E402
from beton.fondation_grue import export as X  # noqa: E402
from beton.fondation_grue import moteur as MO  # noqa: E402

DOC = os.path.join(RACINE, "docs", "beton", "FONDATIONS_GRUE_RBAR.md")
OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:700]}"))


# ------------------------------------------------------------------ les fichiers du site
ARMATURES_DU_PLAN = (("support/inf/0/n", "9"), ("support/sup/0/n", "6"), ("semelle/inf/0/n", "6"), ("semelle/inf/0/d", "25"))
ETATS = (("BASE", ()), ("GWW", ()), ("VAR4", ()), ("VAR8", ()),
         ("GWW", (("cote_b", "paroi"), ("support/inf/1/n", "2"), ("support/sup/1/n", "2"), ("semelle/sup/1/n", "3"))))


def etat(exemple, saisies=()):
    e = FG.nouvel_etat()
    assert FG.modifier(e, ["exemple"], exemple) is None
    for chemin, v in saisies:
        assert FG.modifier(e, chemin.split("/"), v) is None, (chemin, v)
    return e


def fichier(e, x_mod=X) -> dict:
    """Le fichier RBAR d'une note, tel que le site le fait télécharger (identification vide)."""
    return x_mod.rbar(MO.verifier(FG.normaliser(copy.deepcopy(e))["fondations"][0]), {})


def exemple_du_document(x_mod=X) -> dict:
    return fichier(etat("GWW", ARMATURES_DU_PLAN), x_mod)


def serre(v, large=200, retrait=0, fond=3) -> str:
    """Un JSON valide et lisible : une valeur qui tient sur la ligne y reste ; sinon un niveau de plus est ouvert,
    jusqu'à ``fond`` niveaux — au-delà la valeur reste sur une ligne."""
    plat = json.dumps(v, ensure_ascii=False)
    if len(plat) + retrait <= large or not isinstance(v, (dict, list)) or not v or retrait >= fond:
        return plat
    pad = " " * (retrait + 1)
    if isinstance(v, dict):
        corps = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {serre(x, large, retrait + 1, fond)}" for k, x in v.items()]
        return "{\n" + ",\n".join(corps) + "\n" + " " * retrait + "}"
    return "[\n" + ",\n".join(f"{pad}{serre(x, large, retrait + 1, fond)}" for x in v) + "\n" + " " * retrait + "]"


def noms_de(v, chemin="", out=None) -> set:
    """Tous les noms de paramètres d'un fichier : {(chemin, nom)}."""
    out = set() if out is None else out
    if isinstance(v, dict):
        for k, x in v.items():
            out.add((chemin, k))
            noms_de(x, f"{chemin}/{k}", out)
    elif isinstance(v, list):
        for x in v:
            noms_de(x, chemin, out)
    return out


# ------------------------------------------------------------------ le document
BLOC = re.compile(r"```json\n(.*?)\n```", re.S)


def lire_document() -> str:
    with open(DOC, encoding="utf-8") as h:
        return h.read()


def prose(doc: str) -> str:
    """Le document sans son fichier d'exemple (un nom n'est « dit » que s'il est écrit hors de l'exemple)."""
    return BLOC.sub("", doc)


def nomme(texte: str, nom: str) -> bool:
    """Le document nomme-t-il ce paramètre ? Entre accents graves, dans une liste de champs « {a, b} », ou au bout
    d'un chemin (« note.fleche », « armatures[].origine »)."""
    return f"`{nom}`" in texte or re.search(r"[`{,\[ .]" + re.escape(nom) + r"[`},\] ]", texte) is not None


def lignes_du_tableau(doc: str, titre: str) -> list:
    """Les lignes (listes de cellules) du premier tableau qui suit le titre donné."""
    lignes = doc[doc.index(titre):].split("\n")
    debut = next(i for i, l in enumerate(lignes) if l.startswith("|"))
    out = []
    for l in lignes[debut + 2:]:
        if not l.startswith("|"):
            break
        out.append([c.strip() for c in l.strip().strip("|").split("|")])
    return out


V1 = {"": "format version note limites modelesPoutre pieux fondations materiaux pieces implantation armatures controle",
      "/note": "nom projet grue schema libelle classeur statut date indice chantier", "/materiaux": "beton acier",
      "/implantation": "unite repere pieds pieux pieces paroi conventions", "/controle": "lignes manques bloquants pret a_regarder ecarts",
      "/modelesPoutre": "nom bas haut lat etrier domaine stab", "/pieux": "code nom diametre_mm charge_t etat",
      "/armatures": "element nom_element poste designation genre texte etat saisie"}
# la traduction d'une ligne du tableau « d'où vient chaque barre » : repère → (élément, poste) du fichier
REPERES = {"1": ("semelle", "etriers"), "2": ("semelle", "sup"), "3": ("semelle", "inf"), "4": ("semelle", "laterales"),
           "5": ("semelle", "renfort_pieu"), "6": ("support", "etriers_extremites"), "7": ("support", "etriers_extremites"),
           "8": ("support", "etriers_centre"), "9": ("support", "sup"), "10": ("support", "inf"), "11": ("support", "laterales"),
           "12": ("perfos", "perfos")}


def fautes(doc: str, x_mod=X) -> list:
    """Ce que le document et le fichier ne disent pas de la même façon ; liste vide = ils concordent."""
    r = []
    ex = exemple_du_document(x_mod)
    # 1. le fichier d'exemple
    blocs = BLOC.findall(doc)
    if len(blocs) != 1:
        return [f"le document doit porter UN fichier d'exemple ({len(blocs)} blocs json)"]
    try:
        ecrit = json.loads(blocs[0])
    except ValueError as err:
        return [f"le fichier d'exemple du document n'est pas du JSON ({err})"]
    if ecrit != json.loads(json.dumps(ex)):
        diff = sorted(k for k in set(ecrit) | set(ex) if ecrit.get(k) != json.loads(json.dumps(ex)).get(k))
        r.append(f"le fichier d'exemple du document n'est plus celui que le site produit (blocs : {diff}) — --regenerer")
    # 2. chaque nom exporté est nommé
    texte = prose(doc)
    noms = set()
    for exemple, saisies in ETATS:
        noms |= noms_de(fichier(etat(exemple, saisies), x_mod))
    noms |= noms_de(ex)
    absents = sorted({n for _c, n in noms if not nomme(texte, n)})
    if absents:
        r.append(f"paramètres exportés que le document ne nomme pas : {absents}")
    if len(noms) < 120:
        r.append(f"seulement {len(noms)} noms relus")
    # 3. les ajouts de la version 2 : le nom, et la valeur annoncée pour Greenwood
    if f"`version` : {ex['version']}" not in texte:
        r.append(f"le document n'annonce pas la version {ex['version']} du fichier")
    for cellules in lignes_du_tableau(doc, "### 5.3"):
        chemin, valeur = cellules[0].strip("`"), cellules[2].strip("`")
        cible = ex
        try:
            for k in chemin.replace("[]", "").split("."):
                cible = cible[0][k] if isinstance(cible, list) else cible[k]
        except (KeyError, IndexError, TypeError):
            r.append(f"version 2 : « {chemin} » annoncé par le document, absent du fichier")
            continue
        if valeur:
            lu = f"{cible:g}" if isinstance(cible, float) else str(cible)
            if lu != valeur:
                r.append(f"version 2 : « {chemin} » vaut {lu!r} dans le fichier, {valeur!r} dans le document")
    # 4. d'où vient chaque barre : les cellules et l'armature du fichier
    arm = {(a["element"], a["poste"]): a for a in ex["armatures"]}
    table = lignes_du_tableau(doc, "## 6.")
    if [l[0] for l in table] != list(REPERES):
        r.append(f"le tableau des repères ne porte pas les 12 repères du plan ({[l[0] for l in table]})")
    for l in table:
        a = arm.get(REPERES.get(l[0], ("", "")))
        if a is None:
            r.append(f"repère {l[0]} : son poste n'est pas dans le fichier")
            continue
        if l[4] != (a["cellules"] or "—"):
            r.append(f"repère {l[0]} : cellules {l[4]!r} dans le document, {a['cellules']!r} dans le fichier")
        if l[0] != "7" and not (f"`{a['element']}` · `{a['poste']}`" in l[3]):
            r.append(f"repère {l[0]} : le paramètre nommé n'est pas « {a['element']} · {a['poste']} »")
        chiffres = {"barres": [("n", a.get("groupes", [{}])[0].get("n")), ("d", a.get("groupes", [{}])[0].get("d"))],
                    "etriers": [("brins", a.get("brins")), ("d", a.get("d")), ("pas", a.get("pas"))],
                    "renfort": [("brins", a.get("brins")), ("d", a.get("d")), ("brins_base", a.get("brins_base"))],
                    "perfos": [("n", a.get("n")), ("d", a.get("d"))], "prescription": [("d", a.get("d"))]}[a["genre"]]
        for nom, v in chiffres:
            if l[0] != "7" and f"`{nom}` {v:g}" not in l[3]:
                r.append(f"repère {l[0]} : « {nom} {v:g} » attendu dans « {l[3]} »")
    # 5. le fichier ne change que par ajouts
    presents = noms_de(ex)
    for chemin, cles in V1.items():
        manque = [k for k in cles.split() if (chemin, k) not in presents]
        if manque:
            r.append(f"paramètres de la version 1 disparus de « {chemin or '/'} » : {manque}")
    if ex["format"] != "stab.fondation_grue" or ex["version"] < 2:
        r.append(f"format ou version : {ex['format']} {ex['version']}")
    return r


if "--regenerer" in sys.argv:
    doc = lire_document()
    nouveau = BLOC.sub(lambda m: "```json\n" + serre(json.loads(json.dumps(exemple_du_document()))) + "\n```", doc, count=1)
    with open(DOC, "w", encoding="utf-8") as h:
        h.write(nouveau)
    print("fichier d'exemple régénéré dans", os.path.relpath(DOC, RACINE), "(" + ("changé" if nouveau != doc else "inchangé") + ")")
    sys.exit(0)

print("=== 1 à 5. Le document et le fichier disent la même chose ===")
DOCUMENT = lire_document()
reproches = fautes(DOCUMENT)
chk("le fichier d'exemple du document est celui que le site produit ; chaque paramètre exporté (4 classeurs, paroi, seconds groupes) y est "
    "nommé ; les ajouts de la version 2 y sont, à leur valeur ; le tableau des 12 repères dit les cellules et les armatures du fichier ; "
    "aucun paramètre de la version 1 n'a disparu", not reproches, reproches[:4])
EXEMPLE = exemple_du_document()
chk("l'exemple est Greenwood aux armatures du plan de référence : 9Ø20 et 6Ø16 dans la poutre support, 6Ø25 en bas de la semelle, prêt à partir",
    [a["texte"] for a in EXEMPLE["armatures"][:2]] == ["9Ø20", "6Ø16"] and EXEMPLE["armatures"][4]["texte"] == "6Ø25" and EXEMPLE["controle"]["pret"] is True,
    [a["texte"] for a in EXEMPLE["armatures"][:5]])
chk("l'origine d'une armature se lit telle qu'elle s'écrira sur le plan (« F1 · Poutre support châssis · Armatures inférieures · classeur G129:G132 ») ; "
    "une prescription le dit",
    EXEMPLE["armatures"][0]["origine"] == "F1 · Poutre support châssis · Armatures inférieures · classeur G129:G132"
    and all(a["origine"].endswith("prescription du classeur") and a["cellules"] == "" for a in EXEMPLE["armatures"] if a["genre"] == "prescription")
    and all(a["cellules"] and a["origine"].endswith("classeur " + a["cellules"]) for a in EXEMPLE["armatures"] if a["genre"] != "prescription"),
    [a["origine"] for a in EXEMPLE["armatures"]][:3])
chk("la charge maximale par pied est le plus grand A, B, C ou D des cas saisis (Greenwood : 1004 kN, le chiffre du plan) ; dans les variantes, la charge P",
    EXEMPLE["charges"] == {"max_par_pied_kN": 1004.0}
    and [fichier(etat(x))["charges"]["max_par_pied_kN"] for x in ("VAR4", "VAR8")] == [etat(x)["fondations"][0]["charges"]["P"] for x in ("VAR4", "VAR8")],
    EXEMPLE["charges"])
chk("une fiche sans enrobage ni flèche (les variantes) : enrobage_mm nul, textes vides — rien d'inventé",
    all(fichier(etat(x))["materiaux"]["enrobage_mm"] is None and fichier(etat(x))["note"]["fleche"] == "" for x in ("VAR4", "VAR8")))
chk("le document ne promet rien : ce qu'il nomme est dans le fichier (plus de « j'y travaille »)", "j'y travaille" not in DOCUMENT)

# ================================================================== 6. les mutations
print("\n=== 6. Les mutations : chacune vue ===")


class ExportMutant:
    """L'export, dont le fichier est retouché par ``retouche`` (le module chargé reste intact)."""

    def __init__(self, retouche):
        self.retouche = retouche

    def rbar(self, res, infos=None):
        d = json.loads(json.dumps(X.rbar(res, infos)))
        self.retouche(d)
        return d


def renommer(bloc, ancien, nouveau):
    def f(d):
        for a in (d[bloc] if isinstance(d[bloc], list) else [d[bloc]]):
            if ancien in a:
                a[nouveau] = a.pop(ancien)
    return f


def sans_ligne(doc, motif):
    return "\n".join(l for l in doc.split("\n") if motif not in l)


MUTATIONS_DOCUMENT = (
    ("le document ne nomme plus « hautSuivants » (le nom oublié le 11/11/2026)", lambda d: sans_ligne(d, "`hautSuivants`")),
    ("le document ne nomme plus « brins_base »", lambda d: d.replace("`brins_base`", "brins de base")),
    ("le fichier d'exemple est retouché à la main (8Ø20 au lieu de 9Ø20)", lambda d: d.replace('"bas": {"n": 9, "d": 20}', '"bas": {"n": 8, "d": 20}')),
    ("le document annonce une autre charge par pied", lambda d: d.replace("| `1004` |", "| `1040` |")),
    ("le tableau des repères donne d'autres cellules au repère 10", lambda d: d.replace("| G129:G132 | 9Ø20 |", "| G130:G133 | 9Ø20 |")),
    ("le tableau des repères annonce 8 barres au repère 10", lambda d: d.replace("`support` · `inf` : `n` 9, `d` 20", "`support` · `inf` : `n` 8, `d` 20")),
    ("le document oublie un repère du plan", lambda d: sans_ligne(d, "| 12 | Ø25 droit")),
    ("le document annonce la version 1", lambda d: d.replace("`version` : 2", "`version` : 1")),
)
MUTATIONS_EXPORT = (
    ("l'export renomme « origine » en « provenance »", renommer("armatures", "origine", "provenance")),
    ("l'export renomme « enrobage_mm »", renommer("materiaux", "enrobage_mm", "enrobage")),
    ("l'export ajoute un paramètre sans le dire au document", lambda d: d["note"].update(secret_de_fabrication="x")),
    ("l'export retire un paramètre de la version 1 (« pieces »)", lambda d: d.pop("pieces")),
    ("l'export écrit les cellules en liste et non en plage", lambda d: [a.update(cellules=a["cellules"].replace(":", ", ")) for a in d["armatures"]]),
    ("l'export retombe en version 1", lambda d: d.update(version=1)),
)
survivants = [nom for nom, mue in MUTATIONS_DOCUMENT if mue(DOCUMENT) == DOCUMENT or not fautes(mue(DOCUMENT))]
survivants += [nom for nom, retouche in MUTATIONS_EXPORT if not fautes(DOCUMENT, ExportMutant(retouche))]
chk(f"{len(MUTATIONS_DOCUMENT) + len(MUTATIONS_EXPORT)} mutations (le document : {len(MUTATIONS_DOCUMENT)} ; l'export : {len(MUTATIONS_EXPORT)}) : TOUTES vues",
    not survivants, survivants)
chk("le témoin (le document tel quel, l'export tel quel) n'est vu par aucun contrôle", not fautes(DOCUMENT, ExportMutant(lambda d: None)))

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:500])
sys.exit(1 if KO else 0)
