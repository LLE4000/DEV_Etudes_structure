# -*- coding: utf-8 -*-
"""Fondations de grues — le fichier « rbar.ouvrage » que le site écrit pour RBAR, et les OPTIONS DE CONCEPTION qui le
complètent (``beton/fondation_grue/ouvrage.py``, ``beton/fondation_grue/conception.py`` ; leur document :
``docs/beton/FONDATIONS_GRUE_OUVRAGE.md``).

 1. GREENWOOD : le fichier du site, pour le classeur Greenwood aux armatures du plan de référence, EST le fichier
    d'exemple de RBAR (sa copie nommée ci-dessous), à ses écarts VOULUS près — listés un à un, chacun avec sa raison ;
 2. LES RÈGLES DU FORMAT que le lecteur de RBAR fait respecter (sa spécification, recopiée en contrôles) : chaque
    fichier du site les tient — Greenwood, le classeur BASE, les options poussées à leurs bornes — et chaque
    contre-exemple est vu ;
 3. LES OPTIONS DE CONCEPTION : leurs défauts sont les valeurs de Greenwood ; la table « mise en page par châssis » est
    celle du message de RBAR du 12/11/2026 ; chaque option change le fichier là où elle le dit, et RIEN du calcul ;
 4. LA NORMALISATION : une clé inconnue, une valeur hors du domaine sont refusées avec leur message ; une note d'avant
    les options les reçoit toutes ; l'écran refuse sans rien écrire ;
 5. LE CONTRÔLE avant l'envoi : ce qui part et d'où, ce qui manque, ce qui est à regarder, ce qui bloque — un fichier
    bloqué n'est jamais écrit ; les schémas que le fichier ne décrit pas encore le disent ;
 6. des MUTATIONS du fichier et des options : chacune vue rouge ; le témoin, aucune.

Ce que ce test ne prouve pas : que RBAR DESSINE bien ce fichier. Le lecteur et le rendu de RBAR (C#, dans son dépôt)
ne tournent pas ici ; leur passage sur le fichier Greenwood du site est une mesure datée
(``docs/site/mesures/2026-11-12_fichier_rbar_lu_par_rbar.md``).

    /usr/bin/python3.12 tests/test_fondation_grue_ouvrage.py
    RBAR_EXEMPLE=<dépôt RBAR>/tests/Rbar.Core.Tests/Fixtures/ouvrage-exemple.json  /usr/bin/python3.12 tests/…
        compare aussi la copie à l'original : quand RBAR change son exemple, ce test le voit.
"""
import copy
import json
import math
import os
import re
import sys
import types

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RACINE)

from api_web import fondation_grue as FG  # noqa: E402
from beton.fondation_grue import conception as CO  # noqa: E402
from beton.fondation_grue import entree as EN  # noqa: E402
from beton.fondation_grue import export as X  # noqa: E402
from beton.fondation_grue import moteur as MO  # noqa: E402
from beton.fondation_grue import ouvrage as O  # noqa: E402

OK, KO = [], []


def chk(nom, cond, info=""):
    (OK if cond else KO).append((nom, info))
    print(("OK   " if cond else "KO   ") + nom + ("" if cond else f"  | {str(info)[:900]}"))


# ====================================================================== les copies NOMMÉES (R2)
# 1. Le fichier d'exemple de RBAR, sans son commentaire (« _doc ») : dépôt RBAR,
#    tests/Rbar.Core.Tests/Fixtures/ouvrage-exemple.json (écrit en 0.4.27, lu au commit 5252130, 0.4.35). C'est
#    Greenwood tel que la conversation RBAR l'a relevé sur le plan de référence (GWW VAL COF 001, indice A).
EXEMPLE_RBAR = json.loads(r"""
{
 "format": "rbar.ouvrage",
 "version": 1,
 "unite": "mm",
 "nom": "F1",
 "plan": {"titre": "Implantation"},
 "materiaux": {"beton": "C30/37", "acier": "BE500", "enrobage_mm": 40},
 "modelesPoutre": [
  {
   "nom": "F1 — Poutre transversale 140x70",
   "bas": {"n": 9, "d": 20, "longitudinal": {"extremites": "deux", "retour_mm": 500}},
   "haut": {"n": 6, "d": 16, "longitudinal": {"extremites": "deux", "retour_mm": 500}},
   "lat": {
    "nParFace": 2,
    "d": 12,
    "longitudinal": {"extremites": "deux", "retour_mm": 500, "recouvrements_mm": [3750]}
   },
   "etrier": {
    "d": 12,
    "espacement": 300,
    "brins": 4,
    "disposition": "cadres",
    "zones": [
     {"longueur_mm": 2000, "espacement": 100, "brins": 6},
     {"espacement": 300, "brins": 4},
     {"longueur_mm": 2000, "espacement": 100, "brins": 6}
    ]
   },
   "montage": {"d": -1},
   "scellements": [
    {
     "d": 25,
     "longueur_mm": 600,
     "scelle_mm": 400,
     "remarque": "scellement chimique, en seconde phase",
     "points_mm": [[330, 150], [330, -150], [640, 0], [6860, 0], [7170, 150], [7170, -150]]
    }
   ],
   "enrobage": 40
  },
  {
   "nom": "F1 — Poutre longitudinale 100x75",
   "bas": {"n": 6, "d": 25, "longitudinal": {"extremites": "deux", "retour_mm": 600}},
   "haut": {"n": 4, "d": 16, "longitudinal": {"extremites": "deux", "retour_mm": 600}},
   "lat": {
    "nParFace": 2,
    "d": 12,
    "longitudinal": {"extremites": "deux", "retour_mm": 800, "recouvrements_mm": [1350]}
   },
   "etrier": {"d": 12, "espacement": 125, "brins": 4, "disposition": "deuxCadres"},
   "montage": {"d": -1},
   "cages": [
    {
     "d": 12,
     "positions_mm": [500, 2200],
     "longueur_mm": 700,
     "largeur_mm": 700,
     "cadresLong": 5,
     "cadresTravers": 5,
     "jambe_mm": 650
    }
   ],
   "enrobage": 40
  }
 ],
 "typesPieux": [{"code": "PX53", "nom": "Pieu Ø53 — 75 T (ELS)", "diametre_mm": 530, "charge_t": 75, "etat": "ELS"}],
 "pieux": [
  {"nom": "P1", "type": "PX53", "centre": [-3250, 3850], "recepage_m": -1.2},
  {"nom": "P2", "type": "PX53", "centre": [-3250, 2150], "recepage_m": -1.2},
  {"nom": "P3", "type": "PX53", "centre": [-3250, -2150], "recepage_m": -1.2},
  {"nom": "P4", "type": "PX53", "centre": [-3250, -3850], "recepage_m": -1.2},
  {"nom": "P5", "type": "PX53", "centre": [3250, 3850], "recepage_m": -1.2},
  {"nom": "P6", "type": "PX53", "centre": [3250, 2150], "recepage_m": -1.2},
  {"nom": "P7", "type": "PX53", "centre": [3250, -2150], "recepage_m": -1.2},
  {"nom": "P8", "type": "PX53", "centre": [3250, -3850], "recepage_m": -1.2}
 ],
 "poutres": [
  {
   "titre": "Détail Poutres transversales",
   "designation": "Poutres transversales",
   "position": "retombee",
   "largeur_mm": 1400,
   "hauteur_mm": 700,
   "dessus_m": -0.5,
   "modele": "F1 — Poutre transversale 140x70",
   "echelle": 25,
   "coupes_mm": [1000, 3750],
   "elements": [
    {"nom": "PT1", "debut": [-3750, 3000], "fin": [3750, 3000]},
    {"nom": "PT2", "debut": [-3750, -3000], "fin": [3750, -3000]}
   ],
   "origines": [
    {"poste": "inf", "texte": "F1 · Poutre transversale · Armatures inférieures · classeur G129:G132"},
    {"poste": "sup", "texte": "F1 · Poutre transversale · Armatures supérieures · classeur G136:G139"},
    {
     "poste": "etriers",
     "entreAxes_mm": 100,
     "texte": "F1 · Poutre transversale · Étriers aux extrémités · classeur G145:G147"
    },
    {
     "poste": "etriers",
     "entreAxes_mm": 300,
     "texte": "F1 · Poutre transversale · Étriers en partie centrale · classeur G152:G154"
    },
    {"poste": "lat", "texte": "F1 · Poutre transversale · Armatures latérales · prescription du classeur"},
    {"poste": "scellements", "texte": "F1 · Poutre transversale · Barres scellées · classeur G160:G162"}
   ]
  },
  {
   "titre": "Détail Poutres longitudinales",
   "designation": "Poutres longitudinales",
   "position": "retombee",
   "largeur_mm": 1000,
   "hauteur_mm": 750,
   "dessus_m": -0.5,
   "modele": "F1 — Poutre longitudinale 100x75",
   "echelle": 25,
   "coupes_mm": [2200],
   "elements": [
    {"nom": "PL1", "debut": [-3250, 1650], "fin": [-3250, 4350]},
    {"nom": "PL2", "debut": [-3250, -4350], "fin": [-3250, -1650]},
    {"nom": "PL3", "debut": [3250, 1650], "fin": [3250, 4350]},
    {"nom": "PL4", "debut": [3250, -4350], "fin": [3250, -1650]}
   ],
   "origines": [
    {"poste": "inf", "texte": "F1 · Poutre longitudinale · Armatures inférieures · classeur G175:G178"},
    {"poste": "sup", "texte": "F1 · Poutre longitudinale · Armatures supérieures · classeur G181:G184"},
    {"poste": "etriers", "texte": "F1 · Poutre longitudinale · Étriers · classeur G189:G191"},
    {"poste": "lat", "texte": "F1 · Poutre longitudinale · Armatures latérales · prescription du classeur"},
    {
     "poste": "cages",
     "texte": "F1 · Poutre longitudinale · Cage de renfort au droit des pieux · classeur G195:G198"
    }
   ]
  }
 ],
 "encadres": [
  {
   "titre": "Matériaux",
   "lignes": [
    {"libelle": "Béton", "valeur": "C30/37"},
    {"libelle": "Acier", "valeur": "BE500"},
    {"libelle": "Enrobage", "valeur": "40 mm"},
    "distance entre bord béton et extérieur barres"
   ]
  },
  {
   "titre": "Remarques — Placement des barres",
   "lignes": [
    "1. Poser les barres inférieures dans les poutres longitudinales",
    "2. Poser la cage d'armatures des poutres transversales",
    "3. Poser les barres supérieures des poutres longitudinales"
   ]
  }
 ],
 "traces": [
  {
   "nom": "E1",
   "texte": "Équipement posé",
   "ferme": true,
   "points": [[-3000, -3000], [3000, -3000], [3000, 3000], [-3000, 3000]]
  },
  {"nom": "E1 diagonale 1", "points": [[-3000, -3000], [3000, 3000]]},
  {"nom": "E1 diagonale 2", "points": [[-3000, 3000], [3000, -3000]]}
 ]
}
""")

# 2. La table « mise en page par châssis » du message de RBAR du 12/11/2026 (§ 3.1 : « valeurs de départ, celles du
#    PDF d'essai du 12/11, sur A0 » ; pas de 1/30) : châssis (cm) → format et six échelles 1/N, dans l'ordre des noms.
NOMS_ECHELLES = ("plan", "details", "familles", "coupes", "croisements", "implantation")
TABLE_RBAR = {450: ("A0", 20, 10, 20, 20, 5, 50), 460: ("A0", 20, 10, 20, 20, 5, 50), 600: ("A0", 20, 10, 20, 10, 10, 50),
              800: ("A0", 25, 10, 25, 10, 10, 50), 1000: ("A0", 50, 20, 20, 10, 10, 100)}

# 3. Les clés du format (spécification de RBAR, docs/reference/ouvrages/regles/format-du-fichier.md, § 2 à § 7 bis ;
#    « feuilles.echelles », « feuilles.numeros », « cartouche », « revisions » : les ajouts du message du 12/11/2026,
#    que RBAR lit en note tant qu'il ne les a pas faits). Une clé hors de ces listes serait lue par RBAR comme une note.
CLES = {
    "": {"format", "version", "unite", "nom", "plan", "materiaux", "feuilles", "cartouche", "revisions", "modelesPoutre",
         "typesPieux", "pieux", "poutres", "encadres", "traces"},
    "plan": {"titre"}, "materiaux": {"beton", "acier", "enrobage_mm"}, "feuilles": {"format", "echelles", "numeros"},
    "cartouche": {"projet", "numeroProjet", "batiment", "titre", "phase", "discipline", "responsable", "dessine",
                  "verifie", "date", "intervenants"},
    "revisions": {"indice", "date", "description", "par", "verifie"},
    "modelesPoutre": {"nom", "bas", "haut", "basSuivants", "hautSuivants", "lat", "etrier", "enrobage", "montage",
                      "cages", "scellements"},
    "lit": {"n", "d", "longitudinal"}, "lat": {"nParFace", "d", "longitudinal"},
    "longitudinal": {"extremites", "retour_mm", "recouvrements_mm"},
    "etrier": {"d", "espacement", "brins", "disposition", "zones", "appuis", "premier_mm"},
    "zones": {"longueur_mm", "espacement", "brins"}, "montage": {"d"},
    "cages": {"d", "positions_mm", "longueur_mm", "largeur_mm", "cadresLong", "cadresTravers", "jambe_mm", "axe_mm"},
    "scellements": {"d", "longueur_mm", "scelle_mm", "points_mm", "remarque"},
    "typesPieux": {"code", "nom", "diametre_mm", "charge_t", "etat"}, "pieux": {"nom", "type", "centre", "recepage_m"},
    "poutres": {"titre", "designation", "largeur_mm", "hauteur_mm", "dessus_m", "position", "modele", "echelle",
                "coupes_mm", "elements", "origines"},
    "elements": {"nom", "debut", "fin"}, "origines": {"poste", "texte", "entreAxes_mm"}, "encadres": {"titre", "lignes"},
    "traces": {"points", "ferme", "nom", "texte"},
}
POSITIONS = {"longrine", "retombee", "remontee", "noyee"}
EXTREMITES = {"droites", "debut", "fin", "deux"}
DISPOSITIONS = {"cadreInterieur", "deuxCadres", "epingles", "cadres"}
POSTES = {"inf", "sup", "lat", "etriers", "cages", "scellements"}
DIAMETRES = {6, 8, 10, 12, 14, 16, 20, 25, 28, 32, 40}

# ====================================================================== les notes d'essai
ARMATURES_DU_PLAN = (("support/inf/0/n", "9"), ("support/sup/0/n", "6"), ("semelle/inf/0/n", "6"), ("semelle/inf/0/d", "25"))
INFOS = {"nom": "Note Greenwood", "chantier": "Greenwood", "indice": "B", "date": "12/11/2026"}
_CAS = {}


def etat(exemple="GWW", plus=(), saisies=ARMATURES_DU_PLAN):
    """La note d'un classeur, saisies faites PAR L'ÉCRAN (``FG.modifier`` : le chemin de l'utilisateur)."""
    e = FG.nouvel_etat()
    assert FG.modifier(e, ["exemple"], exemple) is None
    for chemin, v in tuple(saisies) + tuple(plus):
        msg = FG.modifier(e, chemin.split("/"), v)
        assert msg is None, (chemin, v, msg)
    return e


def cas(exemple="GWW", plus=(), saisies=ARMATURES_DU_PLAN):
    """(note, résultat du moteur), calculés une fois : les mutations ne rejouent que l'écriture du fichier."""
    cle = (exemple, tuple(plus), tuple(saisies))
    if cle not in _CAS:
        e = etat(exemple, plus, saisies)
        _CAS[cle] = (e, MO.verifier(FG.normaliser(copy.deepcopy(e))["fondations"][0]))
    return _CAS[cle]


def fichier(r, infos=None, o_mod=O):
    """Le fichier tel qu'il se télécharge (le texte JSON), relu comme RBAR le lira."""
    return json.loads(o_mod.fichier(copy.deepcopy(r), infos or {})["texte"])


def calcul(r):
    """Ce que dit le calcul : le résultat du moteur, sans la fondation relue (qui porte les options)."""
    return json.dumps({k: v for k, v in r.items() if k != "fondation"}, sort_keys=True, ensure_ascii=False, default=str)


def sans_commentaires(v):
    if isinstance(v, dict):
        return {k: sans_commentaires(x) for k, x in v.items() if not k.startswith("_")}
    if isinstance(v, list):
        return [sans_commentaires(x) for x in v]
    return v


def ecarts(a, b, chemin=""):
    """Les chemins (« /poutres[0]/coupes_mm[1] ») où ``b`` diffère de ``a`` : valeur différente, clé ou rang présent
    d'un seul côté. Un nombre se compare par sa valeur (40 et 40.0 : la même chose dans un JSON)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            out += ecarts(a[k], b[k], f"{chemin}/{k}") if k in a and k in b else [f"{chemin}/{k}"]
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        for i in range(max(len(a), len(b))):
            out += ecarts(a[i], b[i], f"{chemin}[{i}]") if i < len(a) and i < len(b) else [f"{chemin}[{i}]"]
        return out
    return [chemin] if a != b or isinstance(a, bool) != isinstance(b, bool) or isinstance(a, str) != isinstance(b, str) else []


def fr(x, nd=1):
    s = f"{x:.{nd}f}".replace(".", ",")
    return s[:-2] if nd == 1 and s.endswith(",0") else s


# ====================================================================== 1. Greenwood
ECARTS_VOULUS = (
    (r"/feuilles", "ajout du 12/11/2026 : le format et les échelles des deux feuilles, la ligne de la table « mise en page "
                   "par châssis »"),
    (r"/modelesPoutre\[[01]\]/nom|/poutres\[[01]\]/modele",
     "le nom du modèle porte la désignation des options de conception, au pluriel comme le plan (« Poutres transversales »)"),
    (r"/poutres\[[01]\]/echelle", "absente : la mise en page par châssis fixe les échelles (l'exemple de RBAR la précède)"),
    (r"/poutres\[[01]\]/origines\[\d\]/texte",
     "le poste et les cellules RÉELS du classeur, ceux de la note — l'exemple de RBAR les avait écrits à la main"),
    (r"/encadres\[[23]\]", "deux encadrés de plus : « Pieux » (la charge du pieu le plus chargé, en attendant sa clé) et "
                           "« Grue » (ses données et la plus grande charge par pied)"),
    (r"/traces\[[012]\]/nom|/traces\[0\]/texte",
     "le site sait que le tracé est le châssis d'une grue, RBAR non (exigence n° 1 du brief) : il porte son nom"),
)


def fautes_greenwood(o_mod=O):
    out = []
    _, r = cas()
    f = r["fondation"]
    o = fichier(r, o_mod=o_mod)
    vus = ecarts(EXEMPLE_RBAR, sans_commentaires(o))
    inattendus = [c for c in vus if not any(re.fullmatch(p, c) for p, _ in ECARTS_VOULUS)]
    if inattendus:
        out.append(f"écarts à l'exemple de RBAR qui ne sont pas voulus : {inattendus[:8]}")
    perdus = [raison[:50] for p, raison in ECARTS_VOULUS if not any(re.fullmatch(p, c) for c in vus)]
    if perdus:
        out.append(f"écarts voulus qui n'existent plus (la liste doit rester exacte) : {perdus}")
    # chaque écart voulu, à SA valeur
    if o.get("feuilles") != {"format": "A0", "echelles": dict(zip(NOMS_ECHELLES, TABLE_RBAR[600][1:]))}:
        out.append(f"feuilles : pas la ligne du châssis 6,00 m de la table de RBAR ({o.get('feuilles')})")
    noms = [m["nom"] for m in o["modelesPoutre"]]
    if noms != ["F1 — Poutres transversales 140x70", "F1 — Poutres longitudinales 100x75"] or [p["modele"] for p in o["poutres"]] != noms:
        out.append(f"noms des modèles : {noms}")
    arm = {(a["element"], a["poste"]): a for a in r["armatures"]}
    attendues = [[X.origine(f, arm[("support", p)]) for p in ("inf", "sup", "etriers_extremites", "etriers_centre", "laterales")]
                 + [X.origine(f, arm[("perfos", "perfos")])],
                 [X.origine(f, arm[("semelle", p)]) for p in ("inf", "sup", "etriers", "laterales", "renfort_pieu")]]
    if [[x["texte"] for x in p["origines"]] for p in o["poutres"]] != attendues:
        out.append("les origines ne sont pas celles de la note (le poste et les cellules du classeur)")
    v = r["valeurs"]
    pieux = {"titre": "Pieux", "lignes": [
        {"libelle": "Pieu le plus chargé", "valeur": f"{fr(v['pieux.sollicitation_T'])} T (service, sans coefficient)"},
        {"libelle": "Capacité portante", "valeur": f"{fr(v['pieux.capacite_T'])} T"}]}
    if o["encadres"][2:3] != [pieux] or pieux["lignes"][0]["valeur"] != "63,8 T (service, sans coefficient)":
        out.append(f"encadré « Pieux » : {o['encadres'][2:3]}")
    grue = o["encadres"][3] if len(o["encadres"]) > 3 else {}
    attendu = [{"libelle": lib, "valeur": str(f[cle]).strip()} for lib, cle in (("Grue", "grue"), ("Flèche", "fleche"),
                                                                   ("Hauteur sous crochet", "hauteur_crochet"),
                                                                   ("Châssis", "dimensions_chassis"), ("Niveau", "niveau_beton"))
                if str(f.get(cle) or "").strip()]
    if grue != {"titre": "Grue", "lignes": attendu + [{"libelle": "Charge maximale par pied", "valeur": "1004 kN"}]}:
        out.append(f"encadré « Grue » : {grue}")
    if [t.get("nom") for t in o["traces"]] != ["Châssis", "Châssis diagonale 1", "Châssis diagonale 2"] \
            or o["traces"][0].get("texte") != f"Châssis de la grue {f['grue']}":
        out.append(f"tracés du châssis : {[(t.get('nom'), t.get('texte')) for t in o['traces']]}")
    # les seconds groupes de barres : les lits suivants de leur face
    _, r2 = cas(plus=(("support/inf/1/n", "2"), ("semelle/sup/1/n", "3")))
    o2 = fichier(r2, o_mod=o_mod)
    g_inf = r2["fondation"]["support"]["inf"][1]
    g_sup = r2["fondation"]["semelle"]["sup"][1]
    if [(x["n"], x["d"]) for x in o2["modelesPoutre"][0].get("basSuivants", [])] != [(2, g_inf["d"])] \
            or [(x["n"], x["d"]) for x in o2["modelesPoutre"][1].get("hautSuivants", [])] != [(3, g_sup["d"])] \
            or "hautSuivants" in o2["modelesPoutre"][0] or "basSuivants" in o2["modelesPoutre"][1]:
        out.append("les seconds groupes de barres ne partent pas en lits suivants de leur face")
    # le cartouche et l'indice : ce que la note sait, rien d'autre
    o3 = fichier(r, INFOS, o_mod=o_mod)
    if o3.get("cartouche") != {"projet": "Greenwood", "titre": "Note Greenwood", "date": "2026-11-12"} \
            or o3.get("revisions") != [{"indice": "B", "date": "2026-11-12"}] or o3["_source"]["note"] != "Note Greenwood":
        out.append(f"cartouche et indice : {o3.get('cartouche')} {o3.get('revisions')}")
    o4 = fichier(r, {"nom": " ", "chantier": "", "indice": "", "date": "demain"}, o_mod=o_mod)
    if "cartouche" in o4 or "revisions" in o4:
        out.append("une note sans nom, chantier, indice ni date lisible écrit quand même un cartouche ou un indice")
    o5 = fichier(r, {"indice": "C", "date": "31/02/2026x"}, o_mod=o_mod)
    if o5.get("revisions") != [{"indice": "C"}] or "cartouche" in o5:
        out.append(f"une date illisible : {o5.get('cartouche')} {o5.get('revisions')}")
    return out


print("=== 1. Greenwood : le fichier du site est l'exemple de RBAR, à ses écarts voulus près ===")
reproches = fautes_greenwood()
chk(f"Greenwood (aux armatures du plan) : le fichier du site est l'exemple de RBAR, à {len(ECARTS_VOULUS)} écarts voulus près — "
    "la mise en page, les noms des modèles, les échelles par feuille, les origines réelles, deux encadrés, le nom du châssis — "
    "chacun à sa valeur ; les seconds groupes partent en lits suivants ; le cartouche ne dit que ce que la note sait",
    not reproches, reproches)
_, R_GWW = cas()
SITE = fichier(R_GWW)
chk("les armatures qui partent sont celles que la note vérifie : 9Ø20 / 6Ø16 et étriers 6 brins tous les 10 cm puis 4 tous les "
    "30 cm (poutre support) ; 6Ø25 / 4Ø16 et 4 brins tous les 12,5 cm (semelle) ; 8 pieux Ø53 de 75 T aux centres du site",
    [(SITE["modelesPoutre"][0][k]["n"], SITE["modelesPoutre"][0][k]["d"]) for k in ("bas", "haut")] == [(9, 20), (6, 16)]
    and [(SITE["modelesPoutre"][1][k]["n"], SITE["modelesPoutre"][1][k]["d"]) for k in ("bas", "haut")] == [(6, 25), (4, 16)]
    and SITE["modelesPoutre"][0]["etrier"]["zones"] == [{"longueur_mm": 2000, "espacement": 100, "brins": 6},
                                                        {"espacement": 300, "brins": 4},
                                                        {"longueur_mm": 2000, "espacement": 100, "brins": 6}]
    and SITE["typesPieux"] == [{"code": "PX53", "nom": "Pieu Ø53 — 75 T (ELS)", "diametre_mm": 530, "charge_t": 75, "etat": "ELS"}]
    and [p["centre"] for p in SITE["pieux"]] == [[round(p["x"] * 10), round(p["y"] * 10)] for p in O.G.plan(R_GWW["fondation"])["pieux"]],
    SITE["modelesPoutre"][0]["etrier"])
f_exp = FG.telecharger(cas()[0], "ouvrage", INFOS)
chk("« DWG pour AutoCAD » télécharge ce fichier : « F1 - ouvrage RBAR.json », JSON, le même octet pour octet à chaque export",
    f_exp["nom"] == "F1 - ouvrage RBAR.json" and f_exp["type"] == "application/json"
    and f_exp["texte"] == O.fichier(R_GWW, INFOS)["texte"] == FG.telecharger(cas()[0], "ouvrage", INFOS)["texte"]
    and f_exp["texte"].endswith("}\n") and "\\u00" not in f_exp["texte"], (f_exp["nom"], f_exp["type"]))
if os.environ.get("RBAR_EXEMPLE"):
    with open(os.environ["RBAR_EXEMPLE"], encoding="utf-8") as h:
        original = sans_commentaires(json.load(h))
    chk("la copie de l'exemple de RBAR est l'original (RBAR_EXEMPLE)", original == EXEMPLE_RBAR, ecarts(EXEMPLE_RBAR, original)[:8])


# ====================================================================== 2. les règles du format
def nombre(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def point(p):
    return isinstance(p, list) and len(p) == 2 and all(nombre(x) for x in p)


def inconnues(d, sorte, ou, out):
    if not isinstance(d, dict):
        out.append(f"{ou} : un objet attendu")
        return False
    out += [f"{ou}.{k} : clé que le format ne connaît pas" for k in d if not k.startswith("_") and k not in CLES[sorte]]
    return True


def fautes_longitudinal(lg, L, h, ou):
    r = []
    if not inconnues(lg, "longitudinal", ou + ".longitudinal", r):
        return r
    if lg.get("extremites", "droites") not in EXTREMITES:
        r.append(f"{ou} : extrémités « {lg.get('extremites')} » inconnues")
    if "retour_mm" in lg and not (nombre(lg["retour_mm"]) and 0 < lg["retour_mm"] < L / 2):
        r.append(f"{ou} : retour de {lg['retour_mm']} mm (hors de 0 à la demi-longueur)")
    for x in lg.get("recouvrements_mm", []):
        if not (nombre(x) and 0 < x < L):
            r.append(f"{ou} : recouvrement à {x} mm, hors de la poutre")
    return r


def fautes_modele(m, b, h, L, ou):
    """Un modèle de poutre, lu avec la section et la longueur de la famille qui le cite."""
    r = []
    enr = m.get("enrobage")
    if not (nombre(enr) and 0 < enr < min(b, h) / 2):
        r.append(f"{ou}.enrobage : {enr}")
        enr = 0
    lit_haut = 0
    for cle in ("bas", "haut"):
        for lit in ([m[cle]] if cle in m else []) + list(m.get(cle + "Suivants", [])):
            if not inconnues(lit, "lit", f"{ou}.{cle}", r):
                continue
            if not (isinstance(lit.get("n"), int) and not isinstance(lit.get("n"), bool) and lit["n"] >= 1 and lit.get("d") in DIAMETRES):
                r.append(f"{ou}.{cle} : lit illisible ({lit.get('n')}Ø{lit.get('d')})")
            lit_haut = max(lit_haut, lit.get("d", 0) if cle == "haut" else 0)
            if (lit.get("longitudinal") or {}).get("retour_mm", 0) > h - 2 * enr:
                r.append(f"{ou}.{cle} : retour plus haut que la poutre")
            r += fautes_longitudinal(lit.get("longitudinal", {}), L, h, f"{ou}.{cle}")
    if "lat" in m and inconnues(m["lat"], "lat", f"{ou}.lat", r):
        lat = m["lat"]
        if not (isinstance(lat.get("nParFace"), int) and lat["nParFace"] >= 1 and lat.get("d") in DIAMETRES):
            r.append(f"{ou}.lat : {lat.get('nParFace')} rangée(s) Ø{lat.get('d')} (le site n'écrit pas 0 : il retire la clé)")
        r += fautes_longitudinal(lat.get("longitudinal", {}), L, h, f"{ou}.lat")
    et = m.get("etrier")
    if et is not None and inconnues(et, "etrier", f"{ou}.etrier", r):
        if not (et.get("d") in DIAMETRES and nombre(et.get("espacement")) and et["espacement"] > 0
                and isinstance(et.get("brins"), int) and et["brins"] >= 2):
            r.append(f"{ou}.etrier : illisible ({et})")
        if "disposition" in et and et["disposition"] not in DISPOSITIONS:
            r.append(f"{ou}.etrier.disposition : « {et['disposition']} » inconnue")
        zones = et.get("zones", [])
        for z in zones:
            if inconnues(z, "zones", f"{ou}.etrier.zones", r) and not (
                    nombre(z.get("espacement")) and z["espacement"] > 0 and isinstance(z.get("brins", 2), int) and z.get("brins", 2) >= 2
                    and nombre(z.get("longueur_mm", 1)) and z.get("longueur_mm", 1) > 0):
                r.append(f"{ou}.etrier.zones : zone illisible ({z})")
        if zones:
            sans = [z for z in zones if "longueur_mm" not in z]
            total = sum(z.get("longueur_mm", 0) for z in zones)
            if len(sans) > 1 or (total >= L if sans else abs(total - L) > 1):
                r.append(f"{ou}.etrier.zones : {total:g} mm de zones pour une poutre de {L:g} mm")
    if (m.get("montage") or {}).get("d", 0) not in (0, -1) and m["montage"]["d"] not in DIAMETRES:
        r.append(f"{ou}.montage : {m['montage']}")
    for k, c in enumerate(m.get("cages", [])):
        if not inconnues(c, "cages", f"{ou}.cages[{k}]", r):
            continue
        lg, lr, j = c.get("longueur_mm"), c.get("largeur_mm", b - 2 * enr), c.get("jambe_mm", h - 2 * enr)
        if not (c.get("d") in DIAMETRES and nombre(lg) and lg > 0 and nombre(lr) and 0 < lr <= b - 2 * enr and nombre(j) and 0 < j <= h - 2 * enr):
            r.append(f"{ou}.cages[{k}] : cage à la donnée incohérente (RBAR ne la génère pas)")
            continue
        if not all(isinstance(c.get(n), int) and c[n] >= 0 for n in ("cadresLong", "cadresTravers")):
            r.append(f"{ou}.cages[{k}] : nombres de cadres illisibles")
        for x in c.get("positions_mm", []):
            if not (nombre(x) and lg / 2 <= x <= L - lg / 2):
                r.append(f"{ou}.cages[{k}] : cage qui dépasse d'un about (centre à {x} mm)")
    for k, s in enumerate(m.get("scellements", [])):
        if not inconnues(s, "scellements", f"{ou}.scellements[{k}]", r):
            continue
        if not (s.get("d") in DIAMETRES and nombre(s.get("longueur_mm")) and nombre(s.get("scelle_mm"))
                and 0 < s["scelle_mm"] <= s["longueur_mm"] and isinstance(s.get("remarque", ""), str)):
            r.append(f"{ou}.scellements[{k}] : donnée incohérente")
        for p in s.get("points_mm", []):
            if not (point(p) and 0 <= p[0] <= L and abs(p[1]) <= b / 2 - enr):
                r.append(f"{ou}.scellements[{k}] : point hors du dessus ({p})")
    return r


def fautes_format(o):
    """Ce que le lecteur de RBAR refuserait, noterait, ou ne générerait pas (les avertissements du générateur) ; [] :
    le fichier passe sans rien de tout cela (hors les ajouts du 12/11, lus en note tant que RBAR ne les a pas faits)."""
    r = []
    if not inconnues(o, "", "", r):
        return r
    if o.get("format") != "rbar.ouvrage" or o.get("unite") != "mm":
        r.append(f"format, unité : {o.get('format')} {o.get('unite')}")
    if not (isinstance(o.get("version"), int) and not isinstance(o.get("version"), bool) and o["version"] == 1):
        r.append(f"version : {o.get('version')!r}")
    for cle in ("plan", "materiaux"):
        if cle in o:
            inconnues(o[cle], cle, cle, r)
    if "feuilles" in o and inconnues(o["feuilles"], "feuilles", "feuilles", r):
        fe = o["feuilles"]
        if not re.fullmatch(r"A[0-4]", str(fe.get("format", ""))):
            r.append(f"feuilles.format : {fe.get('format')!r}")
        ech = fe.get("echelles", {})
        if not (isinstance(ech, dict) and list(ech) == list(NOMS_ECHELLES)
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 and v != 30 for v in ech.values())):
            r.append(f"feuilles.echelles : les six échelles, en 1/N entiers, jamais 1/30 ({ech})")
    if "cartouche" in o and inconnues(o["cartouche"], "cartouche", "cartouche", r):
        ca = o["cartouche"]
        if any(not isinstance(v, str) or not v.strip() for k, v in ca.items() if k != "intervenants"):
            r.append("cartouche : un champ vide (un champ inconnu reste ABSENT, jamais vide)")
        if "date" in ca and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(ca["date"])):
            r.append(f"cartouche.date : {ca['date']!r} (AAAA-MM-JJ)")
    for i, rev in enumerate(o.get("revisions", [])):
        if inconnues(rev, "revisions", f"revisions[{i}]", r):
            if not str(rev.get("indice", "")).strip():
                r.append(f"revisions[{i}] : sans indice")
            if "date" in rev and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(rev["date"])):
                r.append(f"revisions[{i}].date : {rev['date']!r}")
    modeles = {}
    for i, m in enumerate(o.get("modelesPoutre", [])):
        if inconnues(m, "modelesPoutre", f"modelesPoutre[{i}]", r):
            if not str(m.get("nom", "")).strip() or m["nom"] in modeles:
                r.append(f"modelesPoutre[{i}].nom : vide ou en double")
            modeles[m.get("nom")] = m
    codes = set()
    for i, t in enumerate(o.get("typesPieux", [])):
        if inconnues(t, "typesPieux", f"typesPieux[{i}]", r):
            if not str(t.get("code", "")).strip() or t["code"] in codes:
                r.append(f"typesPieux[{i}].code : vide ou en double")
            codes.add(t.get("code"))
            if not (nombre(t.get("diametre_mm")) and 50 <= t["diametre_mm"] <= 5000 and nombre(t.get("charge_t")) and t["charge_t"] > 0):
                r.append(f"typesPieux[{i}] : Ø hors de 50 à 5000 mm ou charge illisible")
    noms = set()
    for i, p in enumerate(o.get("pieux", [])):
        if not inconnues(p, "pieux", f"pieux[{i}]", r):
            continue
        n = p.get("nom")
        if not isinstance(n, str) or not n.strip() or len(n) > 16 or n in noms:
            r.append(f"pieux[{i}].nom : vide, de plus de 16 caractères ou en double ({n!r})")
        noms.add(n)
        if ("type" in p and p["type"] not in codes) or ("type" not in p and len(codes) != 1):
            r.append(f"pieux[{i}].type : type introuvable ({p.get('type')!r})")
        if not point(p.get("centre")) or ("recepage_m" in p and not nombre(p["recepage_m"])):
            r.append(f"pieux[{i}] : centre ou recépage illisible")
    if not o.get("pieux") and not o.get("poutres"):
        r.append("ni pieu ni poutre")
    noms_el = set()
    for i, fam in enumerate(o.get("poutres", [])):
        ou = f"poutres[{i}]"
        if not inconnues(fam, "poutres", ou, r):
            continue
        b, h = fam.get("largeur_mm"), fam.get("hauteur_mm")
        if not (nombre(b) and nombre(h) and 50 <= b <= 10000 and 50 <= h <= 10000):
            r.append(f"{ou} : section hors de 50 à 10 000 mm ({b} × {h})")
            continue
        if fam.get("position", "longrine") not in POSITIONS or ("dessus_m" in fam and not nombre(fam["dessus_m"])):
            r.append(f"{ou} : position ou niveau illisible")
        longueurs = []
        els = fam.get("elements", [])
        if not els or len(els) > 100:
            r.append(f"{ou} : {len(els)} poutres (1 à 100)")
        for j, el in enumerate(els):
            if not inconnues(el, "elements", f"{ou}.elements[{j}]", r):
                continue
            n = el.get("nom")
            if not isinstance(n, str) or not n.strip() or len(n) > 20 or n in noms_el:
                r.append(f"{ou}.elements[{j}].nom : vide, de plus de 20 caractères ou en double ({n!r})")
            noms_el.add(n)
            if point(el.get("debut")) and point(el.get("fin")):
                longueurs.append(math.dist(el["debut"], el["fin"]))
            else:
                r.append(f"{ou}.elements[{j}] : bouts illisibles")
        if longueurs and max(longueurs) - min(longueurs) > 1:
            r.append(f"{ou} : poutres de longueurs différentes dans une famille ({sorted(set(longueurs))})")
        L = longueurs[0] if longueurs else 0.0
        for x in fam.get("coupes_mm", []):
            if not (nombre(x) and 0 < x < L):
                r.append(f"{ou}.coupes_mm : {x} hors de la poutre (0 à {L:g})")
        m = modeles.get(fam.get("modele"))
        if "modele" in fam and m is None:
            r.append(f"{ou}.modele : modèle introuvable ({fam['modele']!r})")
        for j, og in enumerate(fam.get("origines", [])):
            if not inconnues(og, "origines", f"{ou}.origines[{j}]", r):
                continue
            if og.get("poste") not in POSTES or not str(og.get("texte", "")).strip():
                r.append(f"{ou}.origines[{j}] : poste inconnu ou texte vide ({og.get('poste')!r})")
            zones = ((m or {}).get("etrier") or {}).get("zones", [])
            if "entreAxes_mm" in og and og["entreAxes_mm"] not in {z.get("espacement") for z in zones}:
                r.append(f"{ou}.origines[{j}] : entre-axes {og['entreAxes_mm']} d'aucune zone d'étriers")
        if m is not None and L:
            r += fautes_modele(m, b, h, L, f"{ou} → modèle")
    for i, en in enumerate(o.get("encadres", [])):
        if not inconnues(en, "encadres", f"encadres[{i}]", r):
            continue
        if not str(en.get("titre", "")).strip() or not en.get("lignes"):
            r.append(f"encadres[{i}] : sans titre ou sans ligne")
        for li in en.get("lignes", []):
            textes = [li] if isinstance(li, str) else list(li.values()) if isinstance(li, dict) and set(li) == {"libelle", "valeur"} else []
            if not textes or not all(isinstance(t, str) and t.strip() and "\n" not in t for t in textes):
                r.append(f"encadres[{i}] : ligne illisible ou sur deux lignes ({li!r})")
    for i, t in enumerate(o.get("traces", [])):
        if inconnues(t, "traces", f"traces[{i}]", r):
            if len(t.get("points", [])) < 2 or not all(point(p) for p in t["points"]) or not isinstance(t.get("ferme", False), bool):
                r.append(f"traces[{i}] : points illisibles")
    return r


print("\n=== 2. Les règles du format : chaque fichier du site les tient, chaque contre-exemple est vu ===")
BORNES = (
    ("Greenwood", ()),
    ("retours arrêtés à 10 cm sous la face opposée", (("conception/support/retour", "arret"), ("conception/semelle/retour", "arret"))),
    ("barres droites (retours nuls)", (("conception/support/retour_mm", "0"), ("conception/semelle/retour_lat_mm", "0"))),
    ("retours fixes au plus long qui tienne", (("conception/support/retour_mm", "596"), ("conception/semelle/retour_mm", "646"))),
    ("aucune barre latérale", (("conception/support/laterales", "0"), ("conception/semelle/laterales", "0"))),
    ("dix rangées latérales, sans recouvrement", (("conception/support/laterales", "10"), ("conception/support/recouvrement_lat", False))),
    ("aucune coupe", (("conception/support/coupes", "aucune"), ("conception/semelle/coupes", "aucune"))),
    ("coupes au milieu", (("conception/support/coupes", "milieu"), ("conception/semelle/coupes", "milieu"))),
    ("sans cage de renfort", (("conception/semelle/cage_cadres_long", "0"), ("conception/semelle/cage_cadres_travers", "0"))),
    ("zones serrées de 10 cm", (("conception/support/zone_serree_mm", "100"),)),
    ("niveaux non saisis", (("conception/niveaux/dessus_m", ""), ("conception/niveaux/recepage_m", ""))),
    ("textes du plan vidés", tuple((f"conception/textes/{k}", "") for k in ("plan", "support", "semelle", "chassis", "pose"))),
    ("mise en page laissée à RBAR", (("conception/mise_en_page", "rbar"),)),
    ("6 perfos par plaque (le motif connu est à 3)", (("perfos/n", "2"),)),
    ("des seconds groupes de barres", (("support/inf/1/n", "2"), ("support/sup/1/n", "2"), ("semelle/sup/1/n", "3"))),
    ("6 brins dans la semelle", (("semelle/etriers/brins", "6"),)),
)
mauvais = []
for nom, plus in BORNES:
    _, r = cas(plus=plus)
    for infos in ({}, INFOS):
        fa = fautes_format(fichier(r, infos))
        if fa:
            mauvais.append((nom, fa[:3]))
for ex in ("BASE",):
    _, r = cas(ex, saisies=())
    fa = fautes_format(fichier(r, INFOS))
    if fa:
        mauvais.append((ex, fa[:3]))
chk(f"{len(BORNES)} fichiers de Greenwood (options poussées à leurs bornes, seconds groupes, 6 perfos par plaque) et celui du "
    "classeur BASE, avec ou sans cartouche : chacun tient les règles du format — rien que RBAR refuse, note ou ne génère pas",
    not mauvais, mauvais)


def corrompu(o, retouche):
    o = copy.deepcopy(o)
    retouche(o)
    return o


CONTRE_EXEMPLES = (
    ("la version écrite en texte", lambda o: o.update(version="1")),
    ("une unité en cm", lambda o: o.update(unite="cm")),
    ("une clé que le format ne connaît pas", lambda o: o["poutres"][0].update(hauteur=700)),
    ("un nom de pieu de 17 caractères", lambda o: o["pieux"][0].update(nom="P" * 17)),
    ("deux poutres du même nom", lambda o: o["poutres"][1]["elements"][0].update(nom="PT1")),
    ("une famille de poutres de longueurs différentes", lambda o: o["poutres"][1]["elements"][0].update(fin=[-3250, 4500])),
    ("une section en cm", lambda o: o["poutres"][0].update(largeur_mm=14)),
    ("un modèle introuvable", lambda o: o["poutres"][0].update(modele="?")),
    ("un type de pieu introuvable", lambda o: o["pieux"][0].update(type="PX99")),
    ("une coupe hors de la poutre", lambda o: o["poutres"][0].update(coupes_mm=[8000])),
    ("une origine au poste inconnu", lambda o: o["poutres"][0]["origines"][0].update(poste="bas")),
    ("un entre-axes d'origine qui ne désigne aucune zone", lambda o: o["poutres"][0]["origines"][2].update(entreAxes_mm=150)),
    ("des zones d'étriers plus longues que la poutre", lambda o: o["modelesPoutre"][0]["etrier"]["zones"][0].update(longueur_mm=6000)),
    ("une disposition d'étriers inconnue", lambda o: o["modelesPoutre"][0]["etrier"].update(disposition="U")),
    ("un retour plus haut que la poutre", lambda o: o["modelesPoutre"][0]["bas"]["longitudinal"].update(retour_mm=650)),
    ("des extrémités inconnues", lambda o: o["modelesPoutre"][0]["bas"]["longitudinal"].update(extremites="crochets")),
    ("une cage qui dépasse d'un about", lambda o: o["modelesPoutre"][1]["cages"][0].update(positions_mm=[100, 2200])),
    ("une cage plus large que la poutre", lambda o: o["modelesPoutre"][1]["cages"][0].update(largeur_mm=1000)),
    ("une barre scellée hors du dessus", lambda o: o["modelesPoutre"][0]["scellements"][0]["points_mm"].append([330, 700])),
    ("une longueur scellée plus longue que la barre", lambda o: o["modelesPoutre"][0]["scellements"][0].update(scelle_mm=700)),
    ("une ligne d'encadré sur deux lignes", lambda o: o["encadres"][1]["lignes"].append("a\nb")),
    ("une échelle au 1/30", lambda o: o["feuilles"]["echelles"].update(plan=30)),
    ("une échelle de moins", lambda o: o["feuilles"]["echelles"].pop("croisements")),
    ("une date du cartouche à la française", lambda o: o.update(cartouche={"date": "12/11/2026"})),
    ("un champ du cartouche laissé vide", lambda o: o.update(cartouche={"phase": ""})),
    ("une révision sans indice", lambda o: o.update(revisions=[{"date": "2026-11-12"}])),
    ("un tracé d'un seul point", lambda o: o["traces"][0].update(points=[[0, 0]])),
    ("des barres latérales à zéro rangée", lambda o: o["modelesPoutre"][0]["lat"].update(nParFace=0)),
)
muets = [nom for nom, retouche in CONTRE_EXEMPLES if not fautes_format(corrompu(SITE, retouche))]
chk(f"{len(CONTRE_EXEMPLES)} contre-exemples (version, unité, clé inconnue, noms, longueurs d'une famille, cages, perfos, "
    "encadrés, échelles, cartouche…) : chacun vu par ces règles — un contrôle qui ne rougit jamais ne contrôle rien",
    not muets, muets)


# ====================================================================== 3. les options de conception
def fautes_defauts(co=CO):
    """Les défauts sont les valeurs de Greenwood, lues dans l'exemple de RBAR ; la table est celle de son message."""
    out = []
    d = co.defauts()
    m0, m1 = EXEMPLE_RBAR["modelesPoutre"]
    p0, p1 = EXEMPLE_RBAR["poutres"]
    lu = {"support/retour_mm": m0["bas"]["longitudinal"]["retour_mm"], "support/retour_lat_mm": m0["lat"]["longitudinal"]["retour_mm"],
          "support/zone_serree_mm": m0["etrier"]["zones"][0]["longueur_mm"], "support/laterales": m0["lat"]["nParFace"],
          "semelle/retour_mm": m1["bas"]["longitudinal"]["retour_mm"], "semelle/retour_lat_mm": m1["lat"]["longitudinal"]["retour_mm"],
          "semelle/laterales": m1["lat"]["nParFace"], "semelle/cage_long_mm": m1["cages"][0]["longueur_mm"],
          "semelle/cage_larg_mm": m1["cages"][0]["largeur_mm"], "semelle/cage_cadres_long": m1["cages"][0]["cadresLong"],
          "semelle/cage_cadres_travers": m1["cages"][0]["cadresTravers"], "semelle/cage_jambe_mm": m1["cages"][0]["jambe_mm"],
          "perfos/longueur_mm": m0["scellements"][0]["longueur_mm"], "perfos/scelle_mm": m0["scellements"][0]["scelle_mm"],
          "perfos/remarque": m0["scellements"][0]["remarque"], "niveaux/dessus_m": p0["dessus_m"],
          "niveaux/recepage_m": EXEMPLE_RBAR["pieux"][0]["recepage_m"], "textes/plan": EXEMPLE_RBAR["plan"]["titre"],
          "textes/support": p0["designation"], "textes/semelle": p1["designation"],
          "textes/pose": "\n".join(EXEMPLE_RBAR["encadres"][1]["lignes"]),
          # ce que l'exemple montre sans le nommer : des retours de longueur fixe, un recouvrement des latérales au milieu
          # (3750 = L/2, 1350 = L/2), les coupes au milieu de la zone serrée et au milieu (1000, 3750), et au droit du
          # second pieu (2200 = L/2 + entraxe/2) ; et la règle du propriétaire (12/11) pour l'arrêt : 10 cm
          "support/retour": "fixe", "semelle/retour": "fixe", "support/recouvrement_lat": True, "semelle/recouvrement_lat": True,
          "support/coupes": "zone_milieu", "semelle/coupes": "pieu", "support/arret_mm": 100, "semelle/arret_mm": 100}
    for chemin, v in lu.items():
        a, b = chemin.split("/")
        if d[a][b] != v or isinstance(d[a][b], bool) != isinstance(v, bool):
            out.append(f"défaut « {chemin} » : {d[a][b]!r} — Greenwood : {v!r}")
    if d["mise_en_page"] != "auto" or d["textes"]["chassis"] != "Châssis de la grue":
        out.append(f"mise en page {d['mise_en_page']!r}, texte du châssis {d['textes']['chassis']!r}")
    # le motif des perfos, relu sur les points de l'exemple : le premier pied est à 750 mm du début de la poutre
    pts = m0["scellements"][0]["points_mm"]
    if [(750 - x, y) for x, y in pts[:3]] != list(co.MOTIF_PERFOS) or co.PERFOS_PAR_MOTIF != 3:
        out.append(f"motif des perfos : {co.MOTIF_PERFOS}")
    if {int(L[0]): tuple(L[1:]) for L in co.MISES_EN_PAGE} != TABLE_RBAR or tuple(co.ECHELLES) != NOMS_ECHELLES:
        out.append(f"la table « mise en page par châssis » n'est plus celle du message de RBAR : {co.MISES_EN_PAGE}")
    if [k for k, _ in co.CHOIX_MISE_EN_PAGE] != ["auto", "450", "460", "600", "800", "1000", "rbar"]:
        out.append(f"choix de mise en page : {co.CHOIX_MISE_EN_PAGE}")
    return out


def fautes_mise_en_page(co=CO):
    out = []

    def mp(chassis, choix="auto"):
        return co.mise_en_page({"chassis": chassis, "conception": {"mise_en_page": choix}})

    for ch, ligne in TABLE_RBAR.items():
        l, pourquoi = mp(float(ch))
        if l is None or tuple(l[1:]) != ligne or pourquoi != f"ligne du châssis {fr(ch / 100, 2)} m":
            out.append(f"châssis {ch} : {l} ({pourquoi})")
    for ch, attendu in ((500.0, 600), (470.0, 600), (300.0, 450), (900.0, 1000)):
        l, pourquoi = mp(ch)
        if l is None or l[0] != attendu or "absent de la table : ligne du châssis immédiatement plus grand" not in pourquoi:
            out.append(f"châssis {ch} absent de la table : {l} ({pourquoi}) — attendu la ligne {attendu}")
    l, pourquoi = mp(1100.0)
    if l is not None or "RBAR choisit" not in pourquoi:
        out.append(f"châssis 11,00 m (au-delà de la table) : {l} ({pourquoi})")
    l, pourquoi = mp(600.0, "800")
    if l is None or l[0] != 800 or "imposée" not in pourquoi:
        out.append(f"ligne imposée : {l} ({pourquoi})")
    l, pourquoi = mp(600.0, "rbar")
    if l is not None or "laissée à RBAR" not in pourquoi:
        out.append(f"« RBAR choisit » : {l} ({pourquoi})")
    if co.feuilles(co.MISES_EN_PAGE[2]) != {"format": "A0", "echelles": dict(zip(NOMS_ECHELLES, TABLE_RBAR[600][1:]))}:
        out.append("feuilles() : pas la ligne de la table")
    return out


print("\n=== 3. Les options de conception ===")
reproches = fautes_defauts() + fautes_mise_en_page()
chk("les défauts sont les valeurs de Greenwood relues dans l'exemple de RBAR (retours 50 et 60 cm, latérales 2 rangées et "
    "retours 50 et 80 cm, zones serrées 2 m, cages 70 × 70 de 5 + 5 cadres et jambe 65, perfos 60 dont 40 scellés et leur "
    "motif, niveaux −0,50 et −1,20, textes) ; l'arrêt des retours à 10 cm, la règle du propriétaire ; la table « mise en page "
    "par châssis » est celle du message de RBAR ; un châssis absent prend la ligne immédiatement plus grande, au-delà RBAR choisit",
    not reproches, reproches)

OPTIONS = (
    # (ce qui est saisi, les saisies, ce qui doit changer dans le fichier — None : rien —, la valeur écrite)
    ("retour des barres de la poutre support : 45 cm", (("conception/support/retour_mm", "450"),),
     r"/modelesPoutre\[0\]/(bas|haut)/longitudinal/retour_mm",
     lambda o: [o["modelesPoutre"][0][k]["longitudinal"]["retour_mm"] for k in ("bas", "haut")] == [450, 450]),
    ("retour des barres de la semelle : 55 cm", (("conception/semelle/retour_mm", "550"),),
     r"/modelesPoutre\[1\]/(bas|haut)/longitudinal/retour_mm",
     lambda o: [o["modelesPoutre"][1][k]["longitudinal"]["retour_mm"] for k in ("bas", "haut")] == [550, 550]),
    ("retour nul : des barres droites", (("conception/support/retour_mm", "0"),),
     r"/modelesPoutre\[0\]/(bas|haut)/longitudinal/(extremites|retour_mm)",
     lambda o: [o["modelesPoutre"][0][k]["longitudinal"] for k in ("bas", "haut")] == [{"extremites": "droites"}] * 2),
    ("poutre support : retour arrêté à 10 cm sous la face opposée (70 − 4 − 1,2 − 10 = 54,8 cm)",
     (("conception/support/retour", "arret"),), r"/modelesPoutre\[0\]/(bas|haut)/longitudinal/retour_mm",
     lambda o: [o["modelesPoutre"][0][k]["longitudinal"]["retour_mm"] for k in ("bas", "haut")] == [548, 548]),
    ("semelle : retour arrêté à 15 cm (75 − 4 − 1,2 − 15 = 54,8 cm)",
     (("conception/semelle/retour", "arret"), ("conception/semelle/arret_mm", "150")),
     r"/modelesPoutre\[1\]/(bas|haut)/longitudinal/retour_mm",
     lambda o: [o["modelesPoutre"][1][k]["longitudinal"]["retour_mm"] for k in ("bas", "haut")] == [548, 548]),
    ("l'arrêt ne compte qu'en « arrêté sous la face opposée »", (("conception/support/arret_mm", "200"),), None, lambda o: True),
    ("retour des latérales de la poutre support : 30 cm", (("conception/support/retour_lat_mm", "300"),),
     r"/modelesPoutre\[0\]/lat/longitudinal/retour_mm", lambda o: o["modelesPoutre"][0]["lat"]["longitudinal"]["retour_mm"] == 300),
    ("retour des latérales de la semelle : 70 cm", (("conception/semelle/retour_lat_mm", "700"),),
     r"/modelesPoutre\[1\]/lat/longitudinal/retour_mm", lambda o: o["modelesPoutre"][1]["lat"]["longitudinal"]["retour_mm"] == 700),
    ("zones d'étriers serrés de 1,50 m (et la coupe au milieu de la zone)", (("conception/support/zone_serree_mm", "1500"),),
     r"/modelesPoutre\[0\]/etrier/zones\[[02]\]/longueur_mm|/poutres\[0\]/coupes_mm\[0\]",
     lambda o: [z.get("longueur_mm") for z in o["modelesPoutre"][0]["etrier"]["zones"]] == [1500, None, 1500]
     and o["poutres"][0]["coupes_mm"] == [750, 3750]),
    ("aucune barre latérale dans la poutre support (ni son origine)", (("conception/support/laterales", "0"),),
     r"/modelesPoutre\[0\]/lat|/poutres\[0\]/origines\[[45]\](/.*)?",
     lambda o: "lat" not in o["modelesPoutre"][0] and "lat" not in [x["poste"] for x in o["poutres"][0]["origines"]]),
    ("trois rangées de latérales dans la semelle", (("conception/semelle/laterales", "3"),),
     r"/modelesPoutre\[1\]/lat/nParFace", lambda o: o["modelesPoutre"][1]["lat"]["nParFace"] == 3),
    ("latérales de la poutre support sans recouvrement au milieu", (("conception/support/recouvrement_lat", False),),
     r"/modelesPoutre\[0\]/lat/longitudinal/recouvrements_mm",
     lambda o: "recouvrements_mm" not in o["modelesPoutre"][0]["lat"]["longitudinal"]),
    ("poutre support : une coupe, au milieu", (("conception/support/coupes", "milieu"),), r"/poutres\[0\]/coupes_mm\[[01]\]",
     lambda o: o["poutres"][0]["coupes_mm"] == [3750]),
    ("poutre support : aucune coupe", (("conception/support/coupes", "aucune"),), r"/poutres\[0\]/coupes_mm",
     lambda o: "coupes_mm" not in o["poutres"][0]),
    ("semelle : une coupe, au milieu", (("conception/semelle/coupes", "milieu"),), r"/poutres\[1\]/coupes_mm\[0\]",
     lambda o: o["poutres"][1]["coupes_mm"] == [1350]),
    ("semelle : aucune coupe", (("conception/semelle/coupes", "aucune"),), r"/poutres\[1\]/coupes_mm",
     lambda o: "coupes_mm" not in o["poutres"][1]),
    ("cage de renfort : 80 cm de long", (("conception/semelle/cage_long_mm", "800"),), r"/modelesPoutre\[1\]/cages\[0\]/longueur_mm",
     lambda o: o["modelesPoutre"][1]["cages"][0]["longueur_mm"] == 800),
    ("cage de renfort : 60 cm de large", (("conception/semelle/cage_larg_mm", "600"),), r"/modelesPoutre\[1\]/cages\[0\]/largeur_mm",
     lambda o: o["modelesPoutre"][1]["cages"][0]["largeur_mm"] == 600),
    ("cage de renfort : jambe de 60 cm", (("conception/semelle/cage_jambe_mm", "600"),), r"/modelesPoutre\[1\]/cages\[0\]/jambe_mm",
     lambda o: o["modelesPoutre"][1]["cages"][0]["jambe_mm"] == 600),
    ("cage de renfort : 4 cadres dans le sens de la semelle", (("conception/semelle/cage_cadres_long", "4"),),
     r"/modelesPoutre\[1\]/cages\[0\]/cadresLong", lambda o: o["modelesPoutre"][1]["cages"][0]["cadresLong"] == 4),
    ("cage de renfort : 6 cadres en travers", (("conception/semelle/cage_cadres_travers", "6"),),
     r"/modelesPoutre\[1\]/cages\[0\]/cadresTravers", lambda o: o["modelesPoutre"][1]["cages"][0]["cadresTravers"] == 6),
    ("pas de cage de renfort (zéro cadre), ni son origine",
     (("conception/semelle/cage_cadres_long", "0"), ("conception/semelle/cage_cadres_travers", "0")),
     r"/modelesPoutre\[1\]/cages|/poutres\[1\]/origines\[4\]",
     lambda o: "cages" not in o["modelesPoutre"][1] and "cages" not in [x["poste"] for x in o["poutres"][1]["origines"]]),
    ("barres scellées de 70 cm", (("conception/perfos/longueur_mm", "700"),), r"/modelesPoutre\[0\]/scellements\[0\]/longueur_mm",
     lambda o: o["modelesPoutre"][0]["scellements"][0]["longueur_mm"] == 700),
    ("barres scellées sur 30 cm", (("conception/perfos/scelle_mm", "300"),), r"/modelesPoutre\[0\]/scellements\[0\]/scelle_mm",
     lambda o: o["modelesPoutre"][0]["scellements"][0]["scelle_mm"] == 300),
    ("la remarque des barres scellées", (("conception/perfos/remarque", "résine époxy"),),
     r"/modelesPoutre\[0\]/scellements\[0\]/remarque", lambda o: o["modelesPoutre"][0]["scellements"][0]["remarque"] == "résine époxy"),
    ("niveau du dessus à −0,60 m", (("conception/niveaux/dessus_m", "-0,6"),), r"/poutres\[[01]\]/dessus_m",
     lambda o: [p["dessus_m"] for p in o["poutres"]] == [-0.6, -0.6]),
    ("niveau du dessus vidé : absent", (("conception/niveaux/dessus_m", ""),), r"/poutres\[[01]\]/dessus_m",
     lambda o: all("dessus_m" not in p for p in o["poutres"])),
    ("recépage à −1,50 m", (("conception/niveaux/recepage_m", "-1,5"),), r"/pieux\[[0-7]\]/recepage_m",
     lambda o: {p["recepage_m"] for p in o["pieux"]} == {-1.5}),
    ("recépage vidé : absent", (("conception/niveaux/recepage_m", ""),), r"/pieux\[[0-7]\]/recepage_m",
     lambda o: all("recepage_m" not in p for p in o["pieux"])),
    ("titre du plan vidé : pas de titre", (("conception/textes/plan", ""),), r"/plan", lambda o: "plan" not in o),
    ("titre du plan", (("conception/textes/plan", "Plan d'implantation"),), r"/plan/titre",
     lambda o: o["plan"] == {"titre": "Plan d'implantation"}),
    ("désignation des poutres support", (("conception/textes/support", "Poutres support"),),
     r"/modelesPoutre\[0\]/nom|/poutres\[0\]/(titre|designation|modele)",
     lambda o: (o["poutres"][0]["designation"], o["poutres"][0]["titre"], o["poutres"][0]["modele"], o["modelesPoutre"][0]["nom"])
     == ("Poutres support", "Détail Poutres support", "F1 — Poutres support 140x70", "F1 — Poutres support 140x70")),
    ("désignation des semelles vidée : RBAR nomme la famille", (("conception/textes/semelle", ""),),
     r"/modelesPoutre\[1\]/nom|/poutres\[1\]/(titre|designation|modele)",
     lambda o: "designation" not in o["poutres"][1] and "titre" not in o["poutres"][1]
     and o["poutres"][1]["modele"] == o["modelesPoutre"][1]["nom"] == "F1 — Semelle 100x75"),
    ("texte du châssis", (("conception/textes/chassis", "Emprise du châssis"),), r"/traces\[0\]/texte",
     lambda o: o["traces"][0]["texte"] == "Emprise du châssis WOLF 7534"),
    ("remarques de pose vidées : pas d'encadré", (("conception/textes/pose", ""),), r"/encadres\[[123]\](/.*)?",
     lambda o: [x["titre"] for x in o["encadres"]] == ["Matériaux", "Pieux", "Grue"]),
    ("remarques de pose : deux lignes", (("conception/textes/pose", " 1. Une \n\n2. Deux"),), r"/encadres\[1\]/lignes\[[0-2]\]",
     lambda o: o["encadres"][1]["lignes"] == ["1. Une", "2. Deux"]),
    ("mise en page laissée à RBAR : pas de « feuilles »", (("conception/mise_en_page", "rbar"),), r"/feuilles",
     lambda o: "feuilles" not in o),
    ("mise en page du châssis 8,00 m imposée", (("conception/mise_en_page", "800"),), r"/feuilles/echelles/(plan|familles)",
     lambda o: o["feuilles"]["echelles"] == dict(zip(NOMS_ECHELLES, TABLE_RBAR[800][1:]))),
)


def fautes_options(o_mod=O):
    out = []
    _, r0 = cas()
    base, calc0 = fichier(r0, o_mod=o_mod), calcul(r0)
    for titre, saisies, motif, valeur in OPTIONS:
        _, r = cas(plus=saisies)
        o = fichier(r, o_mod=o_mod)
        vus = ecarts(base, o)
        if motif is None:
            if vus:
                out.append(f"{titre} : le fichier change ({vus[:4]})")
        elif not vus or any(not re.fullmatch(motif, c) for c in vus):
            out.append(f"{titre} : changements {vus[:6]} — attendus : {motif}")
        try:
            bonne = valeur(o)
        except (KeyError, IndexError, TypeError):
            bonne = False
        if not bonne:
            out.append(f"{titre} : la valeur écrite n'est pas celle de l'option")
        if calcul(r) != calc0:
            out.append(f"{titre} : le CALCUL a changé")
    return out


reproches = fautes_options()
chk(f"{len(OPTIONS)} options saisies par l'écran (retours fixes, nuls ou arrêtés sous la face opposée — Ø étrier compris —, "
    "latérales, zones serrées, coupes, cages, barres scellées, niveaux, textes, mise en page) : chacune change le fichier là où "
    "elle le dit, à sa valeur, et RIEN du calcul (valeurs, vérifications, armatures, statut identiques)",
    not reproches, reproches)


# ====================================================================== 4. la normalisation
REFUS = (
    ({"inconnue": 1}, "champ inconnu"), ({"support": {"zz": 1}}, "champ inconnu"),
    ({"support": {"retour_mm": -1}}, "hors du domaine"), ({"support": {"retour_mm": 6000}}, "hors du domaine"),
    ({"support": {"retour_mm": True}}, "nombre attendu"), ({"support": {"retour_mm": "500"}}, "nombre attendu"),
    ({"support": {"retour_mm": float("nan")}}, "hors du domaine"), ({"support": {"laterales": 2.5}}, "entier"),
    ({"support": {"retour": "crochet"}}, "inconnu"), ({"support": {"coupes": "pieu"}}, "inconnu"),
    ({"semelle": {"coupes": "zone_milieu"}}, "inconnu"), ({"support": {"recouvrement_lat": 1}}, "vrai ou faux"),
    ({"perfos": {"longueur_mm": 300, "scelle_mm": 400}}, "dépasse"), ({"textes": {"plan": "x" * 81}}, "caractères"),
    ({"textes": {"pose": "\n".join(f"{i}. ligne" for i in range(9))}}, "lignes"), ({"textes": {"pose": "y" * 81}}, "dépasse"),
    ({"niveaux": {"dessus_m": "bas"}}, "nombre attendu"), ({"mise_en_page": "A3"}, "inconnu"),
    ("illisible", "illisibles"), ({"support": 3}, "illisibles"), ({"semelle": {"cage_cadres_long": 51}}, "hors du domaine"),
)


def fautes_normalisation(co=CO):
    out = []
    d = co.defauts()
    if co.normaliser(None) != d or co.normaliser({}) != d:
        out.append("une note sans options ne reçoit pas les valeurs par défaut")
    d["support"]["retour_mm"] = 1.0
    if co.defauts()["support"]["retour_mm"] == 1.0:
        out.append("defauts() rend l'original, pas une copie")
    p = co.normaliser({"support": {"retour_mm": 450}})
    dd = co.defauts()
    if p["support"]["retour_mm"] != 450 or p["support"]["zone_serree_mm"] != dd["support"]["zone_serree_mm"] \
            or {k: v for k, v in p.items() if k != "support"} != {k: v for k, v in dd.items() if k != "support"}:
        out.append("un bloc partiel n'est pas complété par ses défauts")
    if co.normaliser(co.normaliser(p)) != co.normaliser(p):
        out.append("normaliser n'est pas idempotent")
    if co.normaliser({"textes": {"pose": "  a \n\n b "}})["textes"]["pose"] != "a\nb":
        out.append("les remarques de pose ne sont pas nettoyées (lignes vides, blancs)")
    if co.normaliser({"niveaux": {"dessus_m": None}})["niveaux"]["dessus_m"] is not None:
        out.append("un niveau vidé n'est pas gardé vide")
    for c, motif in REFUS:
        try:
            co.normaliser(c)
            out.append(f"accepté : {c!r}"[:120])
        except ValueError as e:
            if motif not in str(e):
                out.append(f"{c!r} : « {e} » (attendu « {motif} »)"[:160])
    # le retour : longueur fixe, nulle (barres droites), arrêtée sous la face opposée ; refusé s'il ne tient pas
    c = co.defauts()
    if co.retour(c, "support", 700, 40, 12) != 500:
        out.append("retour fixe")
    c["support"]["retour"] = "arret"
    if co.retour(c, "support", 700, 40, 12) != 548:
        out.append(f"retour arrêté : {co.retour(c, 'support', 700, 40, 12)} (attendu 700 − 40 − 12 − 100 = 548)")
    for arret, motif in ((700.0, "n'a pas de longueur"), (40.0, "entame l'enrobage")):
        c["support"]["arret_mm"] = arret
        try:
            co.retour(c, "support", 700, 40, 12)
            out.append(f"un arrêt de {arret} mm est accepté")
        except ValueError as e:
            if motif not in str(e):
                out.append(f"arrêt de {arret} mm : « {e} »")
    c = co.defauts()
    c["semelle"]["retour_mm"] = 647.0
    try:
        co.retour(c, "semelle", 750, 40, 12)
        out.append("un retour fixe plus haut que la cage (647 > 750 − 2 × 52) est accepté")
    except ValueError as e:
        if "ne tient pas dans la hauteur" not in str(e):
            out.append(f"retour trop haut : « {e} »")
    return out


print("\n=== 4. La normalisation ===")
reproches = fautes_normalisation()
f_avant = EN.exemple("GWW")
f_avant.pop("conception", None)
chk(f"une note sans options reçoit les valeurs de Greenwood ; un bloc partiel est complété ; {len(REFUS)} saisies hors du "
    "domaine (clé inconnue, nombre, entier, choix, case, longueur scellée, textes, remarques de pose) refusées avec leur "
    "message ; le retour arrêté vaut h − enrobage − Ø étrier − arrêt et doit tenir dans les étriers",
    not reproches and EN.normaliser(f_avant)["conception"] == CO.defauts(), reproches)
e = etat()
avant = copy.deepcopy(e["fondations"][0]["conception"])
messages = [FG.modifier(e, c.split("/"), v) for c, v in (("conception/support/retour_mm", "-5"), ("conception/support/inconnue", "5"),
                                                           ("conception/support/laterales", "2,5"),
                                                           ("conception/support/recouvrement_lat", "oui"),
                                                           ("conception/perfos/scelle_mm", "900"),
                                                           ("conception/mise_en_page", "A3"),
                                                           ("conception/textes/pose", "\n".join("x" for _ in range(9))))]
chk("l'écran refuse chaque saisie fautive avec un message, et la note reste comme elle était",
    all(isinstance(m, str) and m for m in messages) and e["fondations"][0]["conception"] == avant, messages)


# ====================================================================== 5. le contrôle avant l'envoi
BLOCAGES = (
    ("une vérification du classeur PAS OK (2 barres en bas de la poutre support)", (("support/inf/0/n", "2"),), "PAS OK"),
    ("des étriers de Ø différents aux extrémités et au centre", (("support/etriers_extremites/d", "14"),),
     "un modèle de RBAR n'a qu'un diamètre d'étrier"),
    ("des zones d'étriers serrés plus longues que la poutre", (("conception/support/zone_serree_mm", "4000"),),
     "ne tiennent pas dans la poutre"),
    ("un retour arrêté sans longueur", (("conception/support/retour", "arret"), ("conception/support/arret_mm", "700")),
     "n'a pas de longueur"),
    ("un retour arrêté dans l'enrobage", (("conception/semelle/retour", "arret"), ("conception/semelle/arret_mm", "30")),
     "entame l'enrobage"),
    ("un retour fixe plus haut que la poutre", (("conception/support/retour_mm", "620"),), "ne tient pas dans la hauteur"),
)
SCHEMAS = (("le schéma 4 pieux", "VAR4", (), "4 pieux · cadre"), ("le schéma 8 pieux primaires", "VAR8", (), "8 pieux · primaires"),
           ("la paroi de pieux sécants", "GWW", (("cote_b", "paroi"),), "paroi de pieux sécants"))


def fautes_controle(o_mod=O):
    out = []
    e, r = cas()
    f = r["fondation"]
    c = o_mod.controle(copy.deepcopy(r), {})
    if not c["pret"] or c["bloquants"]:
        out.append(f"Greenwood : le contrôle bloque ({c['bloquants']})")
    lignes = {li["quoi"]: li for li in c["lignes"]}
    for a in r["armatures"]:
        if a["element"] not in ("support", "semelle") or a["poste"] in ("laterales", "renfort_pieu"):
            continue
        nom = "Poutres support" if a["element"] == "support" else "Semelles"
        li = lignes.get(f"{nom} · {a['designation'].lower()}")
        cellules = X.plage(X._cellules(f, a["saisie"]))
        if li is None or li["valeur"] != a["texte"] or li["source"] != f"calcul · classeur {cellules}" \
                or not X.origine(f, a).endswith(f"classeur {cellules}"):
            out.append(f"contrôle : la ligne de « {a['designation']} » ({a['element']}) ne dit pas ce qui part et d'où")
    attendues = {"Pieux": "8 pieux Ø53 cm, capacité 75 T", "Barres scellées (perfos)": "3Ø25 par plaque, 60 cm dont 40 scellés",
                 "Cages de renfort au droit des pieux": "5 + 5 cadres Ø12, 70 × 70 cm",
                 "Mise en page des plans": "A0 · plan 1/20 · details 1/10 · familles 1/20 · coupes 1/10 · croisements 1/10 · implantation 1/50"}
    for quoi, valeur in attendues.items():
        if (lignes.get(quoi) or {}).get("valeur") != valeur:
            out.append(f"contrôle : « {quoi} » vaut {(lignes.get(quoi) or {}).get('valeur')!r}")
    if (lignes.get("Mise en page des plans") or {}).get("source") != "ligne du châssis 6,00 m":
        out.append("contrôle : la mise en page ne dit pas d'où elle vient")
    regard = " | ".join(c["a_regarder"])
    for motif in ("le classeur vérifie 16 brins Ø12", "la règle de passage des brins aux cadres est à confirmer",
                  "Niveau du dessus des poutres : -0,50 m, la valeur de Greenwood", "Niveau de recépage des pieux : -1,20 m, la valeur de Greenwood"):
        if motif not in regard:
            out.append(f"à regarder : « {motif} » absent")
    manque = " | ".join(c["manques"])
    for motif in ("Charge horizontale des pieux", "Charge verticale des pieux : RBAR n'a pas encore de clé",
                  "Cartouche : le chantier et le nom de la note ne sont pas saisis", "numéro de projet, bâtiment, phase, discipline"):
        if motif not in manque:
            out.append(f"manques : « {motif} » absent")
    if "Cartouche : chantier" in " | ".join(o_mod.controle(copy.deepcopy(r), INFOS)["manques"]):
        out.append("une note nommée, avec son chantier, se dit sans nom ni chantier")
    # les options qui sortent des valeurs de Greenwood, ou qui se taisent : le contrôle le suit
    _, r2 = cas(plus=(("conception/niveaux/dessus_m", "-0,6"), ("conception/niveaux/recepage_m", "")))
    c2 = o_mod.controle(copy.deepcopy(r2), {})
    if "Niveau du dessus" in " | ".join(c2["a_regarder"]) or "Niveau de recépage des pieux : non saisi" not in " | ".join(c2["manques"]):
        out.append("contrôle : un niveau saisi, un niveau vidé")
    _, r3 = cas(plus=(("perfos/n", "2"),))
    o3, c3 = fichier(r3, o_mod=o_mod), o_mod.controle(copy.deepcopy(r3), {})
    if "scellements" in o3["modelesPoutre"][0] or "scellements" in [x["poste"] for x in o3["poutres"][0]["origines"]] \
            or "6Ø25 par plaque — le motif de pose connu est celui de 3 barres" not in " | ".join(c3["manques"]):
        out.append("6 perfos par plaque : le motif de 3 est posé quand même, ou le contrôle se tait")
    _, r4 = cas(plus=(("support/etriers_centre/brins", "3"),))
    c4 = o_mod.controle(copy.deepcopy(r4), {})
    if not c4["pret"] or "Poutre support : 3 brins — aucune disposition de RBAR ne convient" not in " | ".join(c4["a_regarder"]):
        out.append("3 brins : le contrôle ne le dit pas")
    _, r5 = cas(plus=(("semelle/etriers/brins", "6"),))
    if fichier(r5, o_mod=o_mod)["modelesPoutre"][1]["etrier"].get("disposition") != "cadres" \
            or SITE["modelesPoutre"][1]["etrier"].get("disposition") != "deuxCadres" \
            or SITE["modelesPoutre"][0]["etrier"].get("disposition") != "cadres":
        out.append("dispositions des étriers : 4 brins de semelle en deux cadres, sinon des cadres qui se chevauchent")
    # ce qui bloque : dit, et le fichier n'est jamais écrit
    for titre, saisies, motif in BLOCAGES:
        e, r = cas(plus=saisies)
        c = o_mod.controle(copy.deepcopy(r), {})
        if c["pret"] or not any(motif in b for b in c["bloquants"]):
            out.append(f"{titre} : ne bloque pas ({c['bloquants']})")
            continue
        try:
            o_mod.fichier(copy.deepcopy(r), {})
            out.append(f"{titre} : le fichier part quand même")
        except ValueError as err:
            if not str(err).startswith("Fichier pour RBAR refusé : ") or motif not in str(err):
                out.append(f"{titre} : refus « {err} »")
    for titre, ex, plus, nom in SCHEMAS:
        e, r = cas(ex, plus=plus, saisies=())
        raison = o_mod.possible(r["fondation"])
        c = o_mod.controle(copy.deepcopy(r), {})
        if not raison or nom not in raison or c["pret"] or c["lignes"] or c["bloquants"][:1] != [raison[:1].upper() + raison[1:]]:
            out.append(f"{titre} : le contrôle ne dit pas que le fichier n'existe pas encore ({raison})")
        try:
            o_mod.fichier(copy.deepcopy(r), {})
            out.append(f"{titre} : un fichier part")
        except ValueError:
            pass
        except Exception as err:  # noqa: BLE001 — un refus doit être un message, jamais un plantage
            out.append(f"{titre} : l'écriture plante ({type(err).__name__}) au lieu de dire pourquoi")
    return out


print("\n=== 5. Le contrôle avant l'envoi ===")
reproches = fautes_controle()
chk("le contrôle dit ce qui part et d'où (chaque armature, sa valeur, ses cellules du classeur ; pieux, cages, perfos, mise en "
    "page), ce qui est à regarder (cages, niveaux de Greenwood), ce qui manque (charges des pieux, cartouche) ; "
    f"{len(BLOCAGES)} blocages (PAS OK, Ø d'étriers, zones serrées, retours impossibles) refusent le fichier avec leur raison ; "
    f"{len(SCHEMAS)} schémas pas encore décrits le disent",
    not reproches, reproches)
try:
    FG.telecharger(cas(plus=BLOCAGES[0][1])[0], "ouvrage", {})
    refus_ecran = None
except ValueError as err:
    refus_ecran = str(err)
chk("par l'écran aussi : « DWG pour AutoCAD » d'une note PAS OK est refusé avec sa raison",
    bool(refus_ecran) and refus_ecran.startswith("Fichier pour RBAR refusé : ") and "PAS OK" in refus_ecran, refus_ecran)


# ====================================================================== 6. les mutations
print("\n=== 6. Les mutations : chacune vue rouge ===")


def mutant(module, ancien, nouveau, **remplace):
    """Le module dont la source a ``ancien`` remplacé par ``nouveau`` (une seule occurrence) ; ``remplace`` : des noms
    du mutant posés après coup (CO=<options mutées>)."""
    src = open(module.__file__, encoding="utf-8").read()
    assert src.count(ancien) == 1, (module.__name__, ancien[:60], src.count(ancien))
    m = types.ModuleType(module.__name__ + "_mutant")
    m.__package__ = module.__package__
    m.__file__ = module.__file__
    exec(compile(src.replace(ancien, nouveau, 1), module.__file__, "exec"), m.__dict__)  # noqa: S102 — le module lui-même, muté
    m.__dict__.update(remplace)
    return m


def rouge(fn):
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 — un mutant qui plante est vu rouge aussi
        return True


def fautes_ouvrage(o_mod):
    return fautes_greenwood(o_mod) + fautes_options(o_mod) + fautes_controle(o_mod) + [
        f"{nom} : {x}" for nom, plus in BORNES for x in fautes_format(fichier(cas(plus=plus)[1], INFOS, o_mod))]


def fautes_options_de(co_mod):
    """Les options mutées : lues directement, et par le fichier (le module du fichier relit ``co_mod``)."""
    o_mod = mutant(O, *TEMOIN_O, CO=co_mod)
    return fautes_defauts(co_mod) + fautes_mise_en_page(co_mod) + fautes_normalisation(co_mod) + fautes_ouvrage(o_mod)


MUTATIONS_OUVRAGE = (
    ("les barres perdent leurs retours d'about", '"extremites": "deux", "retour_mm": _num(round(retour_mm, 1))',
     '"extremites": "droites", "retour_mm": _num(round(retour_mm, 1))'),
    ("un retour nul écrit un retour", "    if retour_mm <= 0:\n", "    if retour_mm < 0:\n"),
    ("les lits suivants sont perdus", "        if len(gs) > 1:", "        if False:"),
    ("la zone serrée prend le pas du centre", 'et["zones"] = [{"longueur_mm": z, "espacement": _num(ext["pas"])',
     'et["zones"] = [{"longueur_mm": z, "espacement": _num(cen["pas"])'),
    ("le motif des perfos n'est pas retourné au second pied", "pts += [[_num(a1 + dx)", "pts += [[_num(a1 - dx)"),
    ("le motif de 3 perfos posé quel que soit leur nombre", 'if pf and pf["n"] == CO.PERFOS_PAR_MOTIF:', "if pf:"),
    ("les cages centrées sur la semelle", '"positions_mm": [_num(round(L / 2 - s / 2, 3))', '"positions_mm": [_num(round(L / 2, 3))'),
    ("les pieux perdent leur recépage", '            q["recepage_m"] = _num(c["niveaux"]["recepage_m"])', "            pass"),
    ("le fichier part malgré un blocage", 'if not ctl["pret"]:', "if False:"),
    ("la charge du pieu le plus chargé devient sa capacité", "f\"{_fr(v['pieux.sollicitation_T'])} T (service, sans coefficient)\"",
     "f\"{_fr(v['pieux.capacite_T'])} T (service, sans coefficient)\""),
    ("les étriers de semelle à 4 brins deviennent des cadres", '    if piece == "semelle" and brins == 4:', "    if False:"),
    ("la coupe de la semelle au droit du premier pieu", 'return [_num(round(L / 2 + _mm(f["pieux"]["entraxe"]) / 2, 1))]',
     'return [_num(round(L / 2 - _mm(f["pieux"]["entraxe"]) / 2, 1))]'),
    ("la mise en page n'est pas écrite", '        out["feuilles"] = CO.feuilles(ligne)', "        pass"),
    ("les semelles prennent les noms des poutres support", '"nom": f"{PREFIXES[piece]}{i + 1}"', '"nom": f"PT{i + 1}"'),
    ("une section écrite en cm", '"largeur_mm": _num(_mm(sec["b"]))', '"largeur_mm": _num(sec["b"])'),
    ("la date du cartouche à la française", '        cartouche["date"] = date', '        cartouche["date"] = infos.get("date")'),
    ("le contrôle oublie les Ø d'étriers différents", '    if ext["d"] != cen["d"]:', "    if False:"),
    ("les zones serrées trop longues passent", "and not 0 < 2 * z < L_su:", "and False:"),
    ("le contrôle ne voit pas un retour impossible", 'CO.retour(c, piece, h_cm * 10, f["enrobage"], et["d"])', "None"),
    ("le retour oublie l'étrier", 'CO.retour(c, "support", su["h"] * 10, f["enrobage"], cen["d"])',
     'CO.retour(c, "support", su["h"] * 10, f["enrobage"], 0)'),
    ("le recouvrement des latérales hors du milieu", "[_num(round(longueur_mm / 2, 1))]", "[_num(round(longueur_mm / 3, 1))]"),
    ("la paroi de pieux sécants part quand même", '    if f.get("cote_b") == E.COTE_PAROI:', "    if False:"),
    ("les origines oublient leurs zones d'étriers", '{"poste": "etriers", "entreAxes_mm": _num(ext["pas"]), "texte": _origine(f, ext)}',
     '{"poste": "etriers", "texte": _origine(f, ext)}'),
    ("le contrôle tait un niveau vidé", 'manques.append(f"{nom} : non saisi (options de conception) — absent du plan")', "pass"),
    ("le châssis n'est plus fermé", 'trace = {"nom": "Châssis", "ferme": True,', 'trace = {"nom": "Châssis", "ferme": False,'),
    ("les poutres en longrine (tirets)", '"position": "retombee"', '"position": "longrine"'),
    ("l'unité change", 'FORMAT, VERSION, UNITE = "rbar.ouvrage", 1, "mm"', 'FORMAT, VERSION, UNITE = "rbar.ouvrage", 1, "cm"'),
)
MUTATIONS_CONCEPTION = (
    ("un défaut change (retour de la poutre support 45 cm)", '"support": {"retour": RETOUR_FIXE, "retour_mm": 500.0,',
     '"support": {"retour": RETOUR_FIXE, "retour_mm": 450.0,'),
    ("un défaut change (40 → 45 cm scellés)", '"perfos": {"longueur_mm": 600.0, "scelle_mm": 400.0,',
     '"perfos": {"longueur_mm": 600.0, "scelle_mm": 450.0,'),
    ("l'arrêt des retours passe à 15 cm", '"semelle": {"retour": RETOUR_FIXE, "retour_mm": 600.0, "arret_mm": 100.0,',
     '"semelle": {"retour": RETOUR_FIXE, "retour_mm": 600.0, "arret_mm": 150.0,'),
    ("une échelle de la table change", '(600.0, "A0", 20, 10, 20, 10, 10, 50),', '(600.0, "A0", 25, 10, 20, 10, 10, 50),'),
    ("un châssis absent prend la ligne inférieure", "ligne = next((L for L in MISES_EN_PAGE if L[0] >= Lc - 1e-9), None)",
     "ligne = next((L for L in reversed(MISES_EN_PAGE) if L[0] <= Lc + 1e-9), None)"),
    ("« RBAR choisit » n'est plus compris", "    if choix == RBAR_CHOISIT:\n", "    if False:\n"),
    ("une clé inconnue passe", '_exiger(not inconnues, f"options de conception, {nom} : champ inconnu',
     '_exiger(True, f"options de conception, {nom} : champ inconnu'),
    ("la longueur scellée peut dépasser la barre", '_exiger(out["perfos"]["scelle_mm"] <= out["perfos"]["longueur_mm"],',
     "_exiger(True,"),
    ("le retour arrêté oublie l'enrobage", 'r = h_mm - enrobage_mm - etrier_mm - a', "r = h_mm - etrier_mm - a"),
    ("un retour fixe trop haut passe", '_exiger(r <= cage + 1e-9, f"{nom} : un retour de', '_exiger(True, f"{nom} : un retour de'),
    ("le motif des perfos change", "MOTIF_PERFOS = ((420.0, 150.0), (420.0, -150.0), (110.0, 0.0))",
     "MOTIF_PERFOS = ((400.0, 150.0), (400.0, -150.0), (110.0, 0.0))"),
    ("les remarques de pose acceptent 80 lignes", "POSE_LIGNES_MAX = 8", "POSE_LIGNES_MAX = 80"),
    ("un nombre entier accepte une décimale", '_exiger(float(x).is_integer(), f"{nom} : nombre entier attendu")',
     '_exiger(True, f"{nom} : nombre entier attendu")'),
    ("une note sans options reçoit des options vides", "    if c is None:\n        return defauts()", "    if c is None:\n        return {}"),
)
TEMOIN_O = ("# ------------------------------------------------------------------ le fichier\n", "# (témoin)\n")
TEMOIN_CO = ("# ------------------------------------------------------------------ normalisation\n", "# (témoin)\n")
LISTES = ((O, MUTATIONS_OUVRAGE), (CO, MUTATIONS_CONCEPTION))
absents = [(mod.__name__.rsplit(".", 1)[-1], mu[0]) for mod, liste in LISTES for mu in liste
           if open(mod.__file__, encoding="utf-8").read().count(mu[1]) != 1]
chk("chaque mutation vise un texte présent UNE fois dans sa source (sinon elle ne mute rien)", not absents, absents)
def present(mod, ancien):
    return open(mod.__file__, encoding="utf-8").read().count(ancien) == 1


survivants = [t for t, a, n in MUTATIONS_OUVRAGE if present(O, a) and not rouge(lambda a=a, n=n: fautes_ouvrage(mutant(O, a, n)))]
survivants += [t for t, a, n in MUTATIONS_CONCEPTION
               if present(CO, a) and not rouge(lambda a=a, n=n: fautes_options_de(mutant(CO, a, n)))]
chk(f"{len(MUTATIONS_OUVRAGE) + len(MUTATIONS_CONCEPTION)} mutations (le fichier : {len(MUTATIONS_OUVRAGE)} ; les options : "
    f"{len(MUTATIONS_CONCEPTION)}) : TOUTES vues rouges", not survivants, survivants)
temoins = fautes_ouvrage(mutant(O, *TEMOIN_O)) + fautes_options_de(mutant(CO, *TEMOIN_CO))
chk("le témoin (un commentaire changé dans chaque module) n'est vu rouge par AUCUN de ces contrôles", not temoins, temoins[:4])

print(f"\nRÉSULTAT : {len(OK)} OK, {len(KO)} échec(s)")
for nom, info in KO:
    print("   -", nom, "|", str(info)[:500])
sys.exit(1 if KO else 0)
