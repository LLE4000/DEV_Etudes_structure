# -*- coding: utf-8 -*-
"""La NOTE DE CALCUL PDF du module « Choix de profilé » (les barres en acier) sur le site — le générateur et sa plomberie.

Elle a LA PLANCHE des notes Poutre et Dalle (le kit ``ndc_pdf.styles`` : la garde A4 portrait, puis une planche A4
paysage par barre — le panneau de gauche et son dessin, deux colonnes de calcul, des vérifications numérotées, une
bande de verdict à la couleur de l'état ; une planche qui déborde se poursuit sur une page suivante, jamais sur une
troisième colonne) et elle dit LES CHIFFRES DE L'ÉCRAN : elle ne calcule rien.

- le résultat est celui de ``api_web.profil.calculer`` (``acier.barre.moteur.verifier``), la fonction de l'écran ;
- chaque formule est écrite par ``acier.barre.ecriture`` — « symbole = formule = valeurs = résultat », la ligne même de
  la fenêtre de calcul de l'écran — et la liste des formules d'une vérification est celle de l'écran
  (``profil_resultat.formules_de``) ;
- le dessin du panneau est celui de l'écran (``dessin_barre.barre`` : les cas de charge, les appuis, les maintiens, les
  enveloppes de M et de V ; en efforts imposés, la section à l'échelle), peint tel quel (``ndc_pdf.svg``) ;
- les mots sont ceux de l'écran (``profil_mots``, ``profil_resultat``).

CE QUI S'IMPRIME (lot C, C5 ; brief v12 § 0, § 9) : le moteur calcule TOUJOURS les deux méthodes ; la NOTE imprime ce que
son option dit (``profil.IMPRESSION`` : « deux », le défaut ; « historique », la méthode ELS seule ; « ec3 », l'Eurocode 3
seul). Par barre, dans l'ordre du § 9 :

  1. la barre, ses charges et les hypothèses COMMUNES (combinaisons, type d'ouvrage, poids propre, torsion non couverte,
     l'origine des longueurs) — et, quand la méthode NON imprimée est plus défavorable, une ligne qui le dit ;
  2. (« deux ») la SYNTHÈSE : les deux méthodes sujet par sujet (acier/barre/comparaison.py), « — » quand une méthode
     n'a rien sur un sujet, le numéro de la vérification qui gouverne ;
  3. (« deux » et « historique ») les ÉCARTS AVEC LES CLASSEURS DU BUREAU : une ligne par correction qui joue (brief
     § 6.7 ; jamais un choix du bureau) ;
  4. le détail de chaque méthode imprimée (ses hypothèses PROPRES en tête), puis l'âme et ses raidisseurs (EN 1993-1-5)
     — « Contrôles complémentaires EN 1993-1-5 » dans une note « méthode ELS seule » ;
  5. les décisions du bureau appliquées, les paramètres à confirmer (Eurocode 3 imprimé), ce que la note ne vérifie pas.

Le STATUT imprimé (la pastille, la garde) est le plus défavorable des méthodes IMPRIMÉES et de l'âme : jamais la couleur
d'une méthode que la note ne montre pas. La note dit chaque chose une fois : un message déjà imprimé sur la planche
n'est pas répété dans une vérification suivante ; chaque ligne de texte tient en 120 caractères (la garde des textes,
tests/test_note_profil.py).

    sections(etat) -> [planche]          une planche par barre (le dictionnaire que ``Style.planche`` dessine)
    pdf(etat, infos) -> bytes            infos = {nom, chantier, indice, date} de la note du site
    fichier(etat, infos) -> {"nom", "type": "application/pdf", "base64"}      (le contrat de api_web/note_poutre.fichier)

L'IDENTIFICATION de la garde et des en-têtes est TOUJOURS celle de la note du site : projet = le chantier, à défaut le nom
de la note ; partie = le nom de la note quand il y a un chantier, sinon le titre du module ; indice ; date de la page.

Une barre que le moteur refuse arrête la note avec son message ; une planche qui ne tiendrait pas (un bloc plus haut
qu'une colonne entière) aussi : jamais une note tronquée, jamais un chiffre faux imprimé.
"""
import base64
import copy
import os
import tempfile
import xml.etree.ElementTree as ET_
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from acier.barre import combinaisons as CO
from acier.barre import ame as AMB
from acier.barre import ec3_deversement as DV
from acier.barre import ecriture as W
from acier.barre import moteur as MO
from acier.barre import parametres as PA
from acier.barre import resultat as RE
from acier.barre import statique as ST
from ndc_pdf import data as ndc_data
from ndc_pdf import riche as RI
from ndc_pdf.fonts import register_all, string_width
from ndc_pdf.styles import Encre
from ndc_pdf.svg import peindre_svg

from . import commun as C
from . import dessin_barre as DB
from . import dessin_profil as DP
from . import profil as P
from . import profil_mots as M
from . import profil_prs as PP
from . import profil_resultat as R

TITRE = "Barres en acier"
# λ̄ le temps d'une conversion : UNE lettre, que les motifs de ndc_pdf.riche reconnaissent (une barre combinante n'en est
# pas une) ; elle s'écrit ``\bar{λ}`` au bout
BARRE = "ƛ"
LARGEUR_DESSIN = 400.0          # la largeur de la scène de la barre : ses textes font 8 pt dans le panneau de gauche
MARGE_DESSIN = 2.0              # la marge du peintre autour d'un dessin (ndc_pdf.svg.peindre_svg)
# la hauteur des données du panneau, sous le dessin (``Style._data`` : 8 pt d'écart, un titre de 12 pt, puis 10,8 pt
# par ligne — corps 7,6, interligne 1,42) et 6 pt de marge au-dessus du bord du panneau
TETE_DONNEES, LIGNE_DONNEES, PIED_DONNEES = 20.0, 7.6 * 1.42, 6.0
COMBINAISONS_MAX = 8            # au-delà, le tableau des combinaisons ne montre que les plus chargées
# la couleur du kit pour un niveau de l'écran (commun.niveau, profil._niveau) ; ses mots sont ceux de l'écran
# (profil_resultat.mot_niveau)
COULEURS = {"p": "ok", "f": "att", "r": "ko", "v": "att", "n": "att", "c": "att"}
COURT = {"historique": M.METHODES_COURT["historique"], "ec3": "Eurocode 3"}        # une méthode au fil d'une phrase (« 74 % ELS »)
LEGENDE = "Vert : conforme   ·   Ocre : attention (95 à 100 %), une vérification non couverte ou hors domaine   ·   Rouge : non conforme"
# le nom d'une dimension et d'une propriété du catalogue, à gauche de son symbole
LIBELLES = {"h": "Hauteur", "b": "Largeur", "t_w": "Âme", "t_f": "Semelle", "r": "Congé", "r_1": "Congé", "r_2": "Arrondi", "t": "Épaisseur",
            "d": "Diamètre", "masse": "Masse", "A": "Aire", "I_y": "Inertie, axe fort", "I_z": "Inertie, axe faible", "W_el,y": "Module élastique",
            "W_pl,y": "Module plastique", "W_pl,z": "Module plastique, faible", "A_v,z": "Aire de cisaillement", "I_t": "Inertie de torsion",
            "I_w": "Gauchissement"}
# le nom d'un effort imposé, à gauche de son symbole (profil_mots.EFFORTS en donne l'aide, plus longue)
EFFORTS = {"N": "Effort normal", "My": "Moment, axe fort", "Vz": "Effort tranchant, âme", "Mz": "Moment, axe faible", "Vy": "Effort tranchant, semelles"}
# une vérification ne réécrit pas les formules que la précédente vient d'écrire, mot pour mot, à partir de ce nombre (les
# facteurs de l'annexe A, communs à (6.61) et (6.62)) ; en dessous, elle les répète : un bloc se lit seul
COMMUNES_MIN = 4
# (lot C, C5) ce que la note ne vérifie pas : profil_mots.limites (une source, lue aussi par « Tout le calcul » de l'écran)
# l'origine d'une longueur de flambement rendue par le moteur (moteur._longueurs : « portee », « troncon », « console »,
# « debord », « saisie ») — une console : U_CONSOLE × sa longueur (doc p. 16, u recommandé « encastrée – libre »)
ORIGINES = {"portee": "la portée", "troncon": "le plus long tronçon entre sections tenues", "saisie": "saisie",
            "debord": "la barre entière et son débord libre"}
OUVRAGES = {"batiment": "bâtiment", "autre": "autre qu'un bâtiment"}
# les titres des blocs neufs (lot C, C5)
TITRE_SYNTHESE = "Synthèse des deux méthodes"
TITRE_ECARTS = "Écarts avec les classeurs du bureau"
TITRE_DECISIONS = "Décisions du bureau appliquées"
TITRE_LIMITES = "Ce que la note ne vérifie pas"
AUCUN_ECART = "Aucune correction des classeurs ne joue sur cette barre."
LIGNE_MAX = 120                 # une ligne de texte de la note : 120 caractères au plus (la garde des textes)
# un message plus court ne se cherche pas dans un texte déjà imprimé (« ok » serait partout)
REDITE_MIN = 20
# le titre d'un bloc : le plus petit corps du kit (styles._titre) et ce que le carré du numéro prend à la colonne
TITRE_MIN, TITRE_RETRAIT = 7.0, Encre.s_tit * 1.25 + 8


# ------------------------------------------------------------------ les textes : ceux du moteur, composés
def tex(s) -> str:
    """Un symbole ou une formule du moteur (« N_b,z,Rd », « χ_LT·W_pl,y·f_y/γ_M1 », « λ̄_LT² ») en mini-LaTeX."""
    return RI.tex(str(s).replace("λ̄", BARRE)).replace(BARRE, r"\bar{λ}")


def sym(s) -> str:
    """Un symbole du moteur, composé dans un texte riche."""
    return "$" + tex(s) + "$"


def riche(s) -> str:
    """Un texte du moteur (une hypothèse, un message, un critère) en texte riche : ses symboles indicés composés,
    λ̄ avec sa barre."""
    morceaux = RI.auto(str(s).replace("λ̄", BARRE)).split("$")
    return "$".join(m.replace(BARRE, r"\bar{λ}") if i % 2 else m.replace(BARRE, r"$\bar{λ}$") for i, m in enumerate(morceaux))


def pr(t: str) -> tuple:
    """Une ligne de texte de la note : (« pr », le texte riche, le texte lui-même) — la garde des textes lit le troisième
    (tests/test_note_profil.py : 120 caractères au plus, aucune redite dans un bloc) ; le kit ne lit que les deux premiers."""
    return ("pr", riche(t), t)


def brut(t: str) -> tuple:
    """Une ligne de texte déjà composée (sans symbole à composer) : (« pr », le texte, le texte)."""
    return ("pr", t, t)


def _cle(t: str) -> str:
    """Un texte comparé à un autre : en minuscules, sans point final ni blancs autour."""
    return str(t).strip().rstrip(".").lower()


def taux_ecrit(t: Optional[float]) -> str:
    """« 51,9 % » — la décimale de la ligne « η = … » de l'écran ; « > 999 % » au-delà ; « — » sans taux."""
    if t is None:
        return "—"
    return f"> {C.PLAFOND_TAUX} %" if t * 100.0 > C.PLAFOND_TAUX else W.nombre(t * 100.0, 1) + " %"


def comparaison(v: Dict) -> str:
    """Ce qu'une vérification compare, en texte riche : « M_y,Ed = 66,6 kN·m ≤ M_b,Rd = 128,3 kN·m » ; une interaction,
    « η = 0,549 ≤ 1 » ; une vérification sans taux dit son critère, ou son premier message."""
    crit = v.get("critere") or ""
    if v.get("taux") is None or v.get("valeur") is None:
        return riche(crit) if crit else riche((v.get("messages") or [""])[0])
    signe = "≤" if v["statut"] != RE.NOK and v["taux"] <= 1.0 else ">"      # un taux > 1 pour information (non couvert) : « > »
    g, _, d = crit.partition(" ≤ ")
    if R.interaction(v):
        return f"η = {C.fr(v['valeur'], 3)} {signe} 1"
    u = v["unite"]
    if not u:                   # (lot C, C4) une somme sans unité à sa limite, comme l'écran : « η₂ + 0,8·η₁ = 0,513 ≤ 1,4 »
        return f"{sym(g)} = {C.fr(v['valeur'], 3)} {signe} {M.n(v['limite'], 3)}"
    return f"{sym(g)} = {M.r1(v['valeur'])} {u} {signe} {sym(d)} = {M.r1(v['limite'])} {u}"


def etat_verification(v: Dict) -> str:
    """La couleur du verdict d'une vérification : celle de son niveau à l'écran (``profil_mots.niveau``)."""
    return COULEURS[M.niveau(v)]


def lieu(v: Dict) -> str:
    """« Poids propre + G + Q · x = 3,00 m » : la combinaison et l'endroit où une vérification a été trouvée."""
    return " · ".join(t for t in (v.get("combinaison") or "", M.lieu(v)) if t)


def _minuscule(s: str) -> str:
    return s[:1].lower() + s[1:]


# ------------------------------------------------------------------ les blocs d'une planche
def _charge(ch: Dict) -> str:
    """Une charge, en clair : « 5 kN/m de 0 à 6,00 m », « 10 kN à 3,00 m »."""
    v = M.n(abs(ch["v"]), 2)
    haut = " (vers le haut)" if ch["v"] < 0 else ""
    t = ch["type"]
    if t == "q":
        return f"{v} kN/m de {M.m(ch['a'])} à {M.m(ch['b'])} m{haut}"
    if t == "tri":
        sens = "croissante" if ch.get("sens", "croissante") == "croissante" else "décroissante"
        return f"triangulaire {sens}, {v} kN/m, de {M.m(ch['a'])} à {M.m(ch['b'])} m{haut}"
    if t == "P":
        appui = f", sur {M.n(ch['ss'], 0)} mm" if ch.get("ss") else ""             # sa longueur d'appui rigide, quand elle est saisie
        return f"{v} kN à {M.m(ch['a'])} m{appui}{haut}"
    return f"couple de {v} kN·m à {M.m(ch['a'])} m"


def _lignes(tete: str, morceaux: List[str], sep: str = " ; ", fin: str = ".") -> List[str]:
    """« tête : a ; b ; c. » en lignes de 120 caractères au plus (la garde des textes) : la suite d'une ligne trop longue
    passe à la ligne suivante, sous « tête (suite) : »."""
    out, cur = [], ""
    for m in morceaux:
        t = f"{cur}{sep}{m}" if cur else f"{tete if not out else tete + ' (suite)'} : {m}"
        if cur and len(t) + len(fin) > LIGNE_MAX:
            out.append(cur + fin)
            cur = f"{tete} (suite) : {m}"
        else:
            cur = t
    return out + [cur + fin] if cur else out


def _longueurs(el: Dict, Ls: Dict) -> str:
    """L'origine des longueurs de flambement que le moteur a prises (``r["longueurs"][a]["origine"]``), en une ligne :
    « L_y : la portée ; L_z : le plus long tronçon entre sections tenues » ; une console, « 2,1 × 3,00 m (console, doc
    p. 16) »."""
    def mot(a):
        o = Ls[a].get("origine")
        if o == "console":
            return f"{C.fr(MO.U_CONSOLE, 1)} × {M.m(sum(el['travees']))} m (console, doc p. 16)"
        return ORIGINES.get(o, "saisie")
    y, z = mot("y"), mot("z")
    return f"L_y et L_z : {y}." if y == z else f"L_y : {y} ; L_z : {z}."


def _poids_propre(el: Dict, r: Dict) -> str:
    """La ligne du poids propre d'une barre chargée : sa valeur (celle du moteur : 78,6 kN/m³ × A), où il agit (au centre de
    gravité) et selon la POSITION de la barre (horizontale : en flexion ; verticale : dans N ; inclinée : les deux)."""
    pp = f" ({M.n(ST.POIDS_VOLUMIQUE, 1)} kN/m³ × A = {M.n(r['poids_propre'], 3)} kN/m)" if r.get("poids_propre") else ""
    pos = el.get("position") or "horizontale"
    ou = {"horizontale": "en flexion", "verticale": "dans N (barre verticale)"}.get(
        pos, f"barre inclinée de {M.n(el.get('angle'), 1)}° : en flexion et dans N")
    return f"Poids propre du profilé{pp} : compté, cas permanent, au centre de gravité, {ou}."


def autre_methode(r: Dict, rv: Dict) -> Optional[str]:
    """La méthode que la note N'IMPRIME PAS, en une ligne, quand elle est plus défavorable que celles imprimées (son statut
    est pire, ou le même avec un taux plus fort) : « Eurocode 3, non imprimé : non conforme, 110 % » — l'ingénieur le sait."""
    caches = [k for k in r["methodes"] if k not in rv["methodes"]]
    if not caches or not rv["methodes"]:
        return None
    k = caches[0]
    x = r["methodes"][k]
    vus = RE.pire_statut([y["statut"] for y in rv["methodes"].values()])
    t_vus = max((y["taux_max"] for y in rv["methodes"].values() if y.get("taux_max") is not None), default=None)
    pire = RE.pire_statut([vus, x["statut"]]) == x["statut"] and x["statut"] != vus
    plus_fort = x["statut"] == vus and x.get("taux_max") is not None and (t_vus is None or x["taux_max"] > t_vus)
    if not (pire or plus_fort):
        return None
    t = x.get("taux_max")
    taux = "" if t is None else ", " + (f"> {C.PLAFOND_TAUX} %" if C.pct(t) > C.PLAFOND_TAUX else f"{C.pc(C.pct(t))} %")
    nom = f"Méthode {M.ELS}, non imprimée" if k == "historique" else f"{M.METHODES[k]}, non imprimé"
    return f"{nom} : {M.MOT_STATUT[x['statut']].lower()}{taux}."


def communes(el: Dict, r: Dict) -> List[str]:
    """Les hypothèses COMMUNES aux deux méthodes (brief § 9), une par ligne : les combinaisons de calcul (barre chargée ;
    acier/barre/combinaisons.hypotheses, sur les cas du moteur, poids propre compris), le type d'ouvrage, le poids propre
    ajouté aux efforts imposés, la torsion non couverte (brief § 17.5)."""
    out: List[str] = []
    if el["mode"] == "chargee":
        out += CO.hypotheses(el["combinaisons"], ST.cas_complets(el, MO.section_de(el)), PA.jeu(el["parametres"]))
    elif any((el.get("poids_propre_efforts") or {}).values()):
        out.append(MO.MOT_POIDS_PROPRE_IMPOSES)
    out.append(f"Type d'ouvrage : {OUVRAGES.get(el.get('ouvrage'), el.get('ouvrage'))}.")
    out.append(M.HYPOTHESE_TORSION)
    return out


def donnees(el: Dict, r: Dict, autre: Optional[str] = None) -> Dict:
    """Le premier bloc : la barre, ses charges (ou ses efforts imposés), ses maintiens, ses longueurs de flambement et leur
    origine, puis les hypothèses communes (``communes``). ``r`` : le résultat tel que la note l'imprime (``imprime``) ;
    ``autre`` : la ligne de la méthode non imprimée (``autre_methode``)."""
    items: List[tuple] = [("v", f"Profilé {r['profil']}", "", f"{M.n(r['masse'], 1)}", "kg/m")]
    # la f_y de l'Eurocode 3 (la table, selon l'épaisseur) d'abord ; la méthode ELS prend la NOMINALE, que ses hypothèses
    # disent (« f_y nominale, comme les classeurs du bureau ») : seule, la ligne dit la sienne
    g = next((r["methodes"][k]["grandeurs"] for k in ("ec3", "historique") if (r["methodes"].get(k) or {}).get("grandeurs", {}).get("fy")), None)
    if g:
        src = g.get("fy_ref")                # (la source entière — bande d'épaisseur comprise — est dans les hypothèses des méthodes)
        items.append(("v", f"Acier {el['nuance']}" + (f" · {src}" if src else ""), "f_y", M.n(g["fy"], 0), "N/mm²"))
    if autre:
        items.append(brut(autre))
    Ls = r["longueurs"]
    if el["mode"] == "chargee":
        items.append(("v", "Modèle", "", M.NOM_MODELE[el["modele"]], ""))
        for i, L in enumerate(el["travees"]):
            items.append(("v", "Portée" if len(el["travees"]) == 1 else f"Travée {i + 1}", "L" if len(el["travees"]) == 1 else f"L_{i + 1}", M.m(L), "m"))
        items.append(("s", "Charges (valeurs caractéristiques)"))
        if el["poids_propre"]:
            items.append(brut(_poids_propre(el, r)))
        for c in el["cas"]:
            quoi = [_charge(ch) for ch in c["charges"]] + ([f"effort normal {M.n(c['N'], 1)} kN"] if c.get("N") else [])
            items += [brut(t) for t in _lignes(f"{c['nom']} — {_minuscule(M.NATURE_COURTE[c['nature']])}", quoi or ["aucune charge"])]
        items.append(("s", "Maintiens et flambement"))
        # les maintiens, par genre : « Maintiens latéraux, semelle haute : 3,11 · 6,22 · 9,33 m. » (une ligne de 120 au plus)
        genres: Dict[str, List[str]] = {}
        for mt in el["maintiens"]:
            genres.setdefault(mt["genre"], []).append(M.m(mt["x"]))
        for g_, xs in genres.items():
            items += [brut(t) for t in _lignes(f"Maintiens latéraux, {M.NOM_GENRE[g_].lower()}", xs, " · ", " m.")]
        if not genres:
            items.append(brut("Aucun maintien latéral entre les appuis."))
    else:
        items.append(("s", "Efforts imposés"))
        jeux = [(k, {"historique": "service", "ec3": "elu"}[k]) for k in r["methodes"]]
        items.append(brut("Valeurs : " + " · ".join({"service": f"de service (méthode {M.ELS})", "elu": "de calcul (Eurocode 3)"}[j]
                                                    for _k, j in jeux) + "."))
        for cle, s_, unite, _aide in M.EFFORTS:
            vals = [el["efforts"][j][cle] for _k, j in jeux]
            if any(vals):
                items.append(("v", EFFORTS[cle], tex(s_), " · ".join(M.r1(x) for x in vals), unite))
        formes = dict(M.DIAGRAMMES)
        items.append(brut(f"Diagramme du moment : {formes[el['diagramme']['y']['forme']].lower()} (axe fort), "
                          f"{formes[el['diagramme']['z']['forme']].lower()} (axe faible)."))
        items.append(("s", "Flambement et déversement"))
    for a, nom in (("y", "axe fort"), ("z", "axe faible")):
        items.append(("v", f"Flambement, {nom}", f"u_{a}·L_{a}", f"{M.n(Ls[a]['u'], 3)} × {M.m(Ls[a]['L'])}", "m"))
    items.append(pr(_longueurs(el, Ls)))
    items.append(("s", "Hypothèses communes"))
    items += [pr(h) for h in communes(el, r)]
    return {"titre": "La barre et ses charges" if el["mode"] == "chargee" else "La barre et ses efforts", "items": items, "verdicts": []}


def case_taux(statut: Optional[str], taux: Optional[float], gras: bool = False) -> tuple:
    """La case « TAUX » d'un tableau de la note — les lieux de l'âme, le bilan des combinaisons, la comparaison : la couleur
    d'un taux sous son statut, celle de l'écran (``profil_resultat.niveau_taux``, qui lit ``profil_mots.couleur_taux``). Rouge
    pour un refus ; À L'ENCRE, sans la couleur « conforme », pour un taux montré pour information (bloqué, non couvert, hors
    domaine) ou absent ; vert sinon. (Lot C, reprise de C0 : les lieux et les combinaisons peignaient en vert le taux d'une
    décision impossible — 156,7 % à la borne, aux appuis de la barre neuve.)"""
    niv = R.niveau_taux({"statut": statut, "taux": taux})
    if niv in ("v", "c"):
        return ("num", taux_ecrit(taux), gras)
    return ("eta", taux_ecrit(taux), niv != "r", gras)


def _tableau_combinaisons(x: Dict, largeur: float, noms: Optional[Dict[str, str]] = None) -> Optional[tuple]:
    """Le bilan des combinaisons d'une méthode (deux au moins) : chacune, la vérification qui y gouverne, son taux ;
    les plus chargées d'abord, COMBINAISONS_MAX au plus. ``noms`` : un autre nom pour certaines vérifications."""
    combs = x.get("combinaisons") or []
    if len(combs) < 2:
        return None
    titres = dict({v["id"]: v["titre"] for v in x["verifications"]}, **(noms or {}))
    tri = sorted(combs, key=lambda c: -(c["taux"] if c["taux"] is not None else -1.0))
    lignes = []
    for c in tri[:COMBINAISONS_MAX]:
        gov = c["nom"] == x.get("combinaison")
        quoi = titres.get(c["gouvernant"], c["gouvernant"] or "—")
        lignes.append([("lab", c["nom"], gov), ("ref", _minuscule(quoi)), case_taux(c["statut"], c["taux"], gov)])
    cols = [("COMBINAISON", largeur - 78 - 40, "left"), ("CE QUI Y GOUVERNE", 78, "left"), ("TAUX", 40, "right")]
    note = None
    if len(combs) > COMBINAISONS_MAX:
        note = f"{len(combs)} combinaisons calculées ; les {COMBINAISONS_MAX} plus chargées sont montrées."
    return ("tab", cols, lignes, note)


# les hypothèses que le moteur écrit dans une méthode et que le premier bloc dit déjà (une fois) : le poids propre
DITES_EN_TETE = (MO.MOT_POIDS_PROPRE, MO.MOT_POIDS_PROPRE_IMPOSES)


def propres(el: Dict, k: str, x: Dict) -> List[str]:
    """Les hypothèses PROPRES d'une méthode, en tête de son détail (brief § 9) : celles que le moteur écrit (sans celles du
    premier bloc), puis ce que la note compose de ce que le moteur NOMME — méthode ELS : la tolérance de 5 % retirée d'une
    ancienne barre ; Eurocode 3 : l'origine de f_u, k_c et sa justification, la source de M_cr et de C₁, celle de I_t et
    I_w."""
    out = [h for h in x["hypotheses"] if h not in DITES_EN_TETE]
    if k == "historique":
        if (el.get("historique") or {}).get("tolerance_retiree"):
            out.append("Tolérance de 5 % retirée (13.2 point 5).")
        return out
    G = x.get("grandeurs") or {}
    if G.get("fu"):
        out.append(f"f_u = {M.n(G['fu'], 0)} N/mm² ({G.get('fu_source') or G.get('fu_ref') or ''}).".replace(" ().", "."))
    D = G.get("deversement") or {}
    if D.get("cas") == "calcule":
        out.append("M_cr : solveur du module (moments réels, hauteur des charges, maintiens).")
        if D.get("methode") == DV.METHODE_LAMINES:
            out.append(f"k_c = {C.fr(D['kc'], 2)} : tableau 6.6." if D.get("kc_tabule") else "k_c = 1 : diagramme hors tableau 6.6.")
    ids = {v["id"] for v in x["verifications"]}
    if ids & {"E-661", "E-662"} and D.get("cas") == "calcule":
        out.append("C₁ = M_cr,c/M_cr,0 du solveur (annexe E ANB : méthode avancée admise).")
    if D.get("cas") == "calcule" or (G.get("flambement") or {}).get("T"):
        out.append("I_t et I_w : formules des trois plats (I_t côté sûr), comme au panneau." if el.get("prs") else "I_t et I_w : catalogue du site.")
    return out


def methode(el: Dict, k: str, x: Dict, largeur: float) -> Dict:
    """Le bloc de tête d'une méthode : sa norme, ses hypothèses propres (``propres``), le bilan de ses combinaisons, ses
    avertissements, puis son verdict — le taux le plus fort et ce qu'il compare."""
    items: List[tuple] = [pr(h) for h in propres(el, k, x)]
    tab = _tableau_combinaisons(x, largeur)
    if tab:
        items.append(tab)
    items += [pr(a["message"]) for a in x["alertes"]]
    g = R.gouvernante(x)
    niv = R.niveau_methode(x)
    if g is None:
        verdict = {"texte": "Aucun effort à vérifier.", "etat": "att"}
    else:
        ou = lieu(g)
        verdict = {"texte": f"{R.mot_methode(x)} — {_minuscule(g['titre'])} : {comparaison(g)}" + (f" ({ou})" if ou else ""), "etat": COULEURS[niv],
                   "riche": True, "cale": taux_ecrit(x["taux_max"]) if x.get("taux_max") is not None else ""}
    items.append(("k", 0))
    # (« Méthode ELS (NBN B 51-001) » porte déjà sa norme ; « Eurocode 3 — NBN EN 1993-1-1 »)
    titre = M.METHODES[k] if x["norme"] in M.METHODES[k] else f"{M.METHODES[k]} — {x['norme']}"
    return {"titre": titre, "items": items, "verdicts": [verdict]}


def titre_ame(r: Dict) -> str:
    """Le titre du bloc de l'âme : « Contrôles complémentaires EN 1993-1-5 » dans une note « méthode ELS seule »
    (la méthode ELS ne les traite pas) ; « Âme et raidisseurs — NBN EN 1993-1-5 » sinon."""
    return M.TITRE_AME_SEUL if list(r["methodes"]) == ["historique"] else f"{M.TITRE_AME} — {r['ame']['norme']}"


def ame(el: Dict, r: Dict, largeur: float) -> Dict:
    """Le bloc de tête de l'âme : ses hypothèses, LA DÉCISION de chaque lieu (un appui, une charge ponctuelle : sans
    raidisseur, ou la paire de plats — et son taux), le bilan de ses combinaisons, ses avertissements, puis son verdict."""
    a = r["ame"]
    # (lot C, C5, reprise) les soudures des raidisseurs : dites une fois, dans « Ce que la note ne vérifie pas » (``limites``)
    items: List[tuple] = [pr(h) for h in a["hypotheses"] if h != AMB.HYP_SOUDURES]
    if a["lieux"]:
        lignes = []
        for L in a["lieux"]:
            plats = M.decision_lieu(L)
            gov = a.get("gouvernant") in L["verifications"]
            lignes.append([("lab", L["nom"], gov), ("ref", f"{M.r1(L['F_Ed'])} kN"), ("lab", plats, L["decision"] == "raidisseur"),
                           case_taux(L["statut"], L["taux"], gov)])
        cols = [("LIEU", 58, "left"), ("FORCE", 44, "left"), ("DÉCISION", largeur - 58 - 44 - 40, "left"), ("TAUX", 40, "right")]
        items.append(("tab", cols, lignes))
    # (dans le bilan des combinaisons, une vérification d'un lieu se dit par son lieu : « appui A »)
    tab = _tableau_combinaisons(a, largeur, {k: L["nom"] for L in a["lieux"] for k in L["verifications"]})
    if tab:
        items.append(tab)
    items += [pr(x["message"]) for x in a["alertes"]]
    verdicts = []
    if a["verifications"]:
        niv = R.niveau_methode(a)
        g = R.gouvernante(a)
        quoi = M.resume_ame(a) if a["statut"] == RE.OK else (_minuscule(g["titre"]) if g else "")
        mot = R.mot_methode(a)                     # (lot C, C0c : « Bloqué » quand une donnée manque, le mot de l'écran)
        verdicts = [{"texte": f"{mot} — {_minuscule(quoi)}" if quoi else mot, "etat": COULEURS[niv], "riche": True,
                     "cale": taux_ecrit(a["taux_max"]) if a.get("taux_max") is not None else ""}]
        items.append(("k", 0))
    return {"titre": titre_ame(r), "items": items, "verdicts": verdicts}


def lignes_ecrites(x: Dict, v: Dict) -> List[Tuple[Tuple[str, ...], str]]:
    """Les formules d'une vérification telles qu'elles s'écrivent : [((symbole, formule, valeurs), résultat)] — celles
    de la fenêtre de calcul de l'écran, sans la ligne du taux (le verdict le dit)."""
    out = []
    for f, syms in R.formules_de(x, v, taux=False):
        parts, res = W.morceaux(f, syms)
        out.append((tuple(parts), res))
    return out


def redit(t: str, dits) -> bool:
    """Le message ``t`` est-il déjà imprimé sur la planche (``dits`` : les textes imprimés, ``_cle``) — le même, ou contenu
    dans un texte plus long (« Décision impossible : s_s à saisir. » dans « Appui A, appui B : décision impossible : s_s à
    saisir. ») ?"""
    c = _cle(t)
    return c in dits or (len(c) >= REDITE_MIN and any(c in d for d in dits))


def avant_apres(L: Dict, largeur: float) -> List[tuple]:
    """Le tableau AVANT / APRÈS d'un lieu qui reçoit un raidisseur (brief § 15.4) : sans raidisseur, la compression de l'âme
    à la jonction (σ_z,Ed), son taux η₂ et l'interaction (7.2) ou le critère 6.2.1(5) ; avec le raidisseur proposé, la
    contrainte de sa section efficace, son flambement hors plan, l'écrasement au contact. Les chiffres du moteur
    (``L["sans"]``, ``L["avec"]`` : raidisseurs.vue)."""
    sa, av = L.get("sans") or {}, L.get("avec")
    if not av:
        return []

    def t(x):
        return ("num", "—") if x is None else case_taux(RE.OK if x <= 1.0 else RE.NOK, x)

    inter = sa.get("interaction") if sa.get("interaction") is not None else sa.get("critere")
    lignes = [[("lab", "Contrainte : $σ_{z,Ed}$ sans, σ avec (N/mm²)"), ("num", M.r1(sa.get("sigma_z"))), ("num", M.r1(av.get("sigma")))],
              [("lab", "Âme sous la force : $η_{2}$"), t(sa.get("eta2")), ("num", "—")],
              [("lab", "Interaction (7.2) ou critère 6.2.1(5)"), t(inter), ("num", "—")],
              [("lab", "Raidisseur : flambement hors plan"), ("num", "—"), t(av.get("taux_flambement"))],
              [("lab", "Raidisseur : écrasement au contact"), ("num", "—"), t(av.get("taux_contact"))]]
    cols = [("", largeur - 2 * 58, "left"), ("SANS RAIDISSEUR", 58, "right"), ("AVEC RAIDISSEUR", 58, "right")]
    return [("s", "Sans et avec le raidisseur proposé"), ("tab", cols, lignes)]


def verification(k: str, x: Dict, v: Dict, precedente: Optional[Tuple[int, List]] = None, dits=None, lieu_=None,
                 largeur: float = 0.0) -> Dict:
    """Une vérification : sa référence et l'endroit où elle a été trouvée, ses formules, ses messages, son verdict.
    ``precedente`` : (le numéro, les formules écrites) de la vérification d'avant — celles qu'elle vient d'écrire mot
    pour mot ne sont pas répétées quand il y en a COMMUNES_MIN au moins : une ligne le dit, à leur place. ``dits`` : les
    textes déjà imprimés sur la planche (``_cle``) — un message déjà dit n'est pas répété (``redit``), ceux de ce bloc
    s'y ajoutent. ``lieu_`` : le lieu de l'âme dont c'est la première vérification — son tableau avant/après."""
    dits = set() if dits is None else dits
    items: List[tuple] = [("pr", " · ".join(t for t in (v["reference"], lieu(v)) if t))]
    lignes = lignes_ecrites(x, v)
    communes_ = set(precedente[1]) & set(lignes) if precedente else set()
    if len(communes_) < COMMUNES_MIN:
        communes_ = set()
    dit = False
    for parts, res in lignes:
        if (parts, res) in communes_:
            if not dit:
                n = sum(1 for l in lignes if l in communes_)
                items.append(brut(f"Les {n} facteurs écrits à la vérification {precedente[0]} sont les mêmes : ils ne sont pas répétés."))
                dit = True
            continue
        items.append(("e", None, [tex(p) for p in parts], tex(res)))
    if lieu_ is not None:
        items += avant_apres(lieu_, largeur)
    neufs = []
    for t in v["messages"]:
        if not redit(t, dits) and _cle(t) not in {_cle(u) for u in neufs}:
            neufs.append(t)
    items += [pr(t) for t in neufs]
    dits.update(_cle(t) for t in neufs)
    items.append(("k", 0))
    verdict = {"texte": comparaison(v), "etat": etat_verification(v), "riche": True,
               "cale": taux_ecrit(v["taux"]) if v["taux"] is not None else M.MOT_STATUT[v["statut"]].lower()}
    # (une vérification de l'âme porte déjà son lieu : « Appui A : âme sous la réaction » ; un titre qui ne tiendrait pas dans la
    # colonne au plus petit corps du kit perd le nom de sa méthode, que la tête du bloc dit déjà : « Flexion composée et
    # flambement (6.61), (6.62) » sortait de la page de 2 pt, en deuxième colonne)
    titre = v["titre"] if k == M.AME else f"{M.METHODES_COURT[k]} · {_minuscule(v['titre'])}"
    if largeur and k != M.AME and string_width(titre, Planche.f_h, TITRE_MIN) > largeur - TITRE_RETRAIT:
        titre = v["titre"]
    return {"titre": titre, "items": items, "verdicts": [verdict]}


def _case_taux(c: Optional[Dict]) -> str:
    """Le taux d'une case de la comparaison, écrit ; son état quand elle n'en a pas (« non couvert »)."""
    if c is None:
        return "—"
    return taux_ecrit(c["taux"]) if c["taux"] is not None else M.MOT_STATUT[c["statut"]].lower()


def _ecart(e: Optional[float]) -> str:
    """L'écart d'une ligne de la synthèse (points de pourcentage, Eurocode 3 − ELS) : « +3,5 », « −30,8 », « — »."""
    return "—" if e is None else ("+" if e > 0 else "") + W.nombre(e, 1)


def _colonne_ec3(li: Dict) -> Tuple[Optional[Dict], bool]:
    """(la case de la colonne Eurocode 3, est-ce celle de l'Eurocode 3 ?) d'une ligne de la synthèse : celle de l'Eurocode 3,
    ou l'âme (EN 1993-1-5, commune aux deux méthodes) quand l'Eurocode 3 n'a rien sur le sujet ou qu'elle y est plus chargée
    — la plus grande des deux cases que le moteur NOMME (acier/barre/comparaison.py), rien n'est recalculé."""
    e, a = li.get("ec3"), li.get("commun")
    if a is None or (e is not None and (a.get("taux") is None or (e.get("taux") is not None and e["taux"] >= a["taux"]))):
        return e, True
    return a, False


def comparaison_methodes(r: Dict, largeur: float, num: Optional[Dict[str, int]] = None) -> Dict:
    """LA SYNTHÈSE (bloc 2 d'une note « les deux », brief § 9) : les deux méthodes côte à côte, sujet par sujet
    (acier/barre/comparaison.py) — le plus fort taux de chacune, « — » quand une méthode n'a rien sur le sujet (l'âme :
    « Contrôles complémentaires » de la méthode ELS), l'écart en points, le numéro de la vérification qui gouverne
    chaque méthode (``num`` : identifiant → numéro du bloc). Un taux sous un statut bloqué, non couvert ou hors domaine
    s'écrit sans la couleur « conforme » (l'écran le pose en « v », ``profil_resultat.niveau_taux``, la même fonction)."""
    num = num or {}

    def case(c):
        if c is None or c["taux"] is None:
            return ("num", _case_taux(c))
        return case_taux(c["statut"], c["taux"])

    lignes = []
    for li in r["comparaison"]:
        ce, est_ec3 = _colonne_ec3(li)
        if li["historique"] is None and ce is None:
            continue
        nums = " · ".join(str(num[c["gouvernant"]]) if c is not None and c.get("gouvernant") in num else "—" for c in (li["historique"], ce))
        lignes.append([("lab", li["sujet"], False), case(li["historique"]), case(ce), ("num", _ecart(li.get("ecart")) if est_ec3 else "—"),
                       ("num", nums)])
    w = (44, 44, 30, 34)
    cols = [("SUJET", largeur - sum(w), "left"), (M.METHODES_COURT["historique"].upper(), w[0], "right"),
            (M.METHODES_COURT["ec3"].upper(), w[1], "right"), ("ÉCART", w[2], "right"), ("N°", w[3], "right")]
    return {"titre": TITRE_SYNTHESE, "items": [("tab", cols, lignes)], "verdicts": []}


def ecarts(r: Dict, num: Optional[Dict[str, int]] = None) -> Dict:
    """« Écarts avec les classeurs du bureau » (bloc 3, brief § 6.7) : une ligne par correction qui JOUE sur la barre (le
    moteur les rend : ``methodes.historique.corrections``, leur texte, ``historique_deversement.CORRECTIONS``), avec le
    numéro de la vérification qui la porte. Jamais un choix du bureau (f_y nominale, courbes et classes, s, C, C_d)."""
    num = num or {}
    items = []
    for c in (r["methodes"].get("historique") or {}).get("corrections") or []:
        ou = ", ".join(f"n° {num[i]}" for i in c.get("verifications") or () if i in num)
        t = c["texte"][:1].upper() + c["texte"][1:]
        items.append(pr(t + (f" ({ou})" if ou else "")))
    return {"titre": TITRE_ECARTS, "items": items or [brut(AUCUN_ECART)], "verdicts": []}


def _nums(ids, num: Dict[str, int]) -> str:
    """« vérification 14 », « vérifications 14 et 17 »."""
    ns = sorted({num[i] for i in ids if i in num})
    if not ns:
        return ""
    return ("vérification " if len(ns) == 1 else "vérifications ") + (" et ".join([", ".join(map(str, ns[:-1])), str(ns[-1])]) if len(ns) > 1 else str(ns[0]))


# (lot C, C5, reprise) le message de la variante du contact inconnu dans une vérification de l'âme (acier/barre/ec3_ame.py,
# ``mots`` : « … ; avec contact, F_Rd = … kN, pour information » ou « … ; sans contact, F_Rd = … ») — la décision n° 2 cite
# les vérifications qui l'IMPRIMENT (un message déjà dit n'est pas répété : ``redit``)
MOT_CONTACT_VARIANTE = "contact, F_Rd = "


def _variante_plastique(sa: Dict) -> bool:
    """La variante plastique de η₁ existe-t-elle sur ce lieu (classes 1 et 2) ? Les deux seuils de résistance locale que le
    moteur NOMME (``sans.seuil`` : η₁ élastique, η₁ plastique) diffèrent — égaux en classes 3 et 4, où il n'y a qu'η₁
    élastique (acier/barre/ame.py : « η₁ élastique seul (classe … : pas de variante plastique) »)."""
    s = sa.get("seuil") or {}
    e, p = s.get("elastique"), s.get("plastique")
    return e is not None and p is not None and abs(e - p) > 1e-9 * max(1.0, abs(e))


def _decision_eta1(a: Dict, jeu: Dict, num: Dict[str, int]) -> Optional[str]:
    """La ligne de la décision n° 1 (η₁ de (7.2), EN 1993-1-5) : le défaut appliqué et le RÉSULTAT de l'autre variante — la
    plus forte interaction plastique des lieux (``sans.interaction_plastique``), avec la vérification qui l'imprime ; « sans
    objet » quand aucun lieu n'a de variante plastique (classes 3 et 4 : la vérification le dit). Une seule vérification
    citée (le lieu le plus chargé) : la ligne tient en 120 caractères. None : aucun lieu sous force et flexion."""
    lieux = [L for L in a.get("lieux") or [] if (L.get("sans") or {}).get("interaction") is not None
             and f"{L['base']}-FM" in (L.get("verifications") or ())]
    if not lieux:
        return None
    tete = "n° 1 · η₁ de (7.2), EN 1993-1-5 : "
    avec = [L for L in lieux if _variante_plastique(L["sans"])]
    if not avec:
        L = max(lieux, key=lambda L: L["sans"]["interaction"])
        return tete + f"élastique (appliqué) ; plastique : sans objet, pas de variante ({_nums([L['base'] + '-FM'], num)})."
    if jeu["eta1_ame"] == "plastique":
        L = max(avec, key=lambda L: L["sans"]["interaction"])
        return tete + f"plastique (appliqué) ; élastique pour information ({_nums([L['base'] + '-FM'], num)})."
    L = max(avec, key=lambda L: L["sans"]["interaction_plastique"])
    return tete + (f"élastique (appliqué) ; plastique : {taux_ecrit(L['sans']['interaction_plastique'])}, pour information "
                   f"({_nums([L['base'] + '-FM'], num)}).")


def decisions(el: Dict, r: Dict, num: Dict[str, int], imprimes: Optional[Dict[str, List[str]]] = None) -> Dict:
    """« Décisions du bureau appliquées » (brief § 13.2, règle 8) : une ligne par décision qui joue sur la barre — le défaut
    appliqué (acier/barre/parametres.py) et l'autre variante, son résultat ou la vérification qui l'IMPRIME ; n° 7, le seuil
    « à la limite », cité avec sa source (celui de toutes les notes du site). Les décisions d'une méthode non imprimée
    ne sont pas dites. ``imprimes`` : identifiant → les messages que sa vérification imprime sur la planche (``planche``)."""
    jeu = PA.jeu(el.get("parametres"))
    T = PA.PARAMETRES
    items: List[tuple] = []
    a = r["ame"]
    n1 = _decision_eta1(a, jeu, num)
    if n1:
        items.append(pr(n1))
    inconnus = [f"{L['base']}-F" for L in a.get("lieux") or [] if L.get("type") == "c" and (L.get("sans") or {}).get("contact") == "inconnu"
                and L.get("ss")]
    if inconnus:
        autre = T["contact_inconnu"]["variante"] if jeu["contact_inconnu"] == T["contact_inconnu"]["valeur"] else T["contact_inconnu"]["valeur"]
        # (lot C, C5, reprise) seulement les vérifications qui IMPRIMENT la variante : la même, dite à l'appui A, n'est pas
        # redite à l'appui B
        cites = [i for i in inconnus if any(MOT_CONTACT_VARIANTE in t for t in (imprimes or {}).get(i, ()))]
        ou = _nums(cites, num)
        items.append(pr(f"n° 2 · contact d'extrémité inconnu (type (c)) : « {jeu['contact_inconnu']} » (appliqué) ; "
                        f"« {autre} » pour information" + (f", {ou}." if ou else ".")))
    e = r["methodes"].get("ec3")
    if e:
        G = e.get("grandeurs") or {}
        av = G.get("A_v_variante")
        if av:
            items.append(pr(f"n° 3 · A_v,y d'un laminé : 2·b·t_f (appliqué) ; A − h_w·t_w : {taux_ecrit(av['taux'])}, pour information "
                            f"({_nums(['E-Vy'], num)})."))
        D = G.get("deversement") or {}
        info = D.get("information")
        if D.get("cas") == "calcule" and info:
            autre = f"{info['methode']} : M_b,Rd = {M.r1(info['M_b_Rd'] / 1e6)} kN·m, {taux_ecrit(info.get('taux'))}"
            if el.get("prs"):
                quoi = "PRS bisymétrique « soudée équivalente »" if D.get("soudee_equivalente") else "PRS bisymétrique"
                items.append(pr(f"n° 10 · {quoi} : {D['methode']} (appliqué) ; {autre}."))
            else:
                items.append(pr(f"n° 4 · déversement des laminés : {D['methode']} (appliqué) ; {autre}."))
        if any(x.get("code") == "console_NM" for x in e.get("alertes") or ()):
            items.append(pr("n° 9 · console sous N + M : non couvert (défaut v1) ; aucune variante tant que le bureau n'a pas fixé de méthode."))
    items.append(pr(f"n° 7 · « à la limite » : taux de {C.pc(C.pct(C.ME.SEUIL_ATTENTION))} à 100 %, le seuil de toutes les notes du site."))
    return {"titre": TITRE_DECISIONS, "items": items, "verdicts": []}


def a_confirmer(r: Dict, largeur: float) -> Optional[Dict]:
    """Les paramètres de l'annexe nationale belge dont la source reste à confirmer : la valeur prise, d'où elle vient
    (Eurocode 3 imprimé seulement)."""
    ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
    if not ac:
        return None
    cols = [("PARAMÈTRE", 62, "left"), ("VALEUR PRISE", 56, "left"), ("D'OÙ VIENT LA VALEUR", largeur - 62 - 56, "left")]
    lignes = [[("lab", riche(p["nom"]), True), ("lab", R.valeur_param(p["valeur"])), ("ref", riche(p["source"]))] for p in ac]
    return {"titre": "Paramètres à confirmer — annexe belge", "items": [("tab", cols, lignes)], "verdicts": []}


def limites(el: Dict, r: Dict) -> Dict:
    """Ce que la note ne vérifie pas : la flèche (donnée, non vérifiée), puis le périmètre du module (``profil_mots.limites`` :
    une ligne chacune ; la torsion est dite dans les hypothèses communes)."""
    items: List[tuple] = []
    fl = (r.get("statique") or {}).get("fleche")
    if fl and fl.get("valeur"):
        rapport = f" (L/{C.fr(fl['rapport'], 0)})" if fl.get("rapport") else ""
        items.append(brut(f"La flèche de service — {C.fr(abs(fl['valeur']), 1)} mm{rapport} sous {fl['combinaison']}, à {M.m(fl['x'])} m : "
                          "une information, elle n'est pas vérifiée."))
    items.append(("s", "Ne sont pas vérifiés"))
    items += [pr(t[:1].upper() + t[1:] + ".") for t in M.limites(bool(el.get("prs"))) if M.TORSION not in t]
    return {"titre": TITRE_LIMITES, "items": items, "verdicts": []}


# ------------------------------------------------------------------ le panneau de gauche
# (lot C, C5) le nom d'une propriété d'un PRS au panneau (profil_prs.proprietes : son symbole) ; son ORIGINE est dite une fois,
# dans les hypothèses de l'Eurocode 3 (« I_t et I_w : formules des trois plats ») et sur le croquis (la méthode)
LIBELLES_PRS = {"A": "Aire", "z_g": "Centre de gravité", "I_y": "Inertie, axe fort", "I_z": "Inertie, axe faible", "W_el,y": "Module élastique",
                "W_el,y,haut": "Module él., fibre haute", "W_el,y,bas": "Module él., fibre basse", "W_pl,y": "Module plastique",
                "I_t": "Inertie de torsion", "I_w": "Gauchissement", "z_s": "Centre de cisaillement", "z_j": "Wagner", "ψ_f": "Monosymétrie",
                "f_y,f1": "Semelle haute", "f_y,w": "Âme", "f_y,f2": "Semelle basse", "f_y": "De calcul"}


def blocs_panneau(el: Dict) -> List[Tuple[str, List[tuple]]]:
    """Sous le dessin : les dimensions du profilé, puis ses propriétés — celles de la vue « Section » de l'écran. Un PRS
    (lot C, C5) : ses trois plats et sa gorge, puis le tableau de ``profil_prs.proprietes`` (A, z_g, I, W, I_t, I_w, z_s,
    z_j et ψ_f d'un monosymétrique, f_y de chaque plat et f_y de calcul) — la même source que l'écran."""
    def lignes(triplets, noms=LIBELLES):
        return [(noms.get(s_, ""), tex(s_) if s_ != "masse" else "", val, unite) for s_, val, unite in triplets]

    if el.get("prs"):
        f = P.fiche_profil(el["profil"], el)
        d = {s_: v for s_, v, _u in f["dimensions"]}
        dims = [("Semelle haute", tex("b_1 × t_1"), f"{d['b_1']} × {d['t_1']}", "mm"), ("Âme", tex("h_w × t_w"), f"{d['h_w']} × {d['t_w']}", "mm"),
                ("Semelle basse", tex("b_2 × t_2"), f"{d['b_2']} × {d['t_2']}", "mm"), ("Hauteur", "h", d["h"], "mm"),
                ("Gorge", "a", d["a"], "mm" if d["a"] != "non saisie" else "")]
        props = [(LIBELLES_PRS.get(s_, ""), tex(s_), v, u) for s_, v, u, _o in f["tableau"]]
        return [("DIMENSIONS", dims), ("PROPRIÉTÉS", props)]
    f = P.fiche_profil(el["profil"])
    return [("DIMENSIONS", lignes(f["dimensions"])), ("PROPRIÉTÉS", lignes(f["proprietes"]))]


def _interieur(svg: str) -> Tuple[List[float], str]:
    """(la viewBox, le contenu) d'un dessin SVG : ce qui est entre sa balise ouvrante et sa balise fermante."""
    vb = [float(t) for t in ET_.fromstring(svg).get("viewBox").split()]
    return vb, svg[svg.index(">", svg.index("<svg")) + 1:svg.rindex("</svg>")]


def empiler(haut: str, bas: str, part: float = 0.62, ecart: float = 10.0) -> str:
    """UN dessin SVG : ``haut`` (la barre), puis ``bas`` (le croquis du PRS) dessous, centré, à ``part`` de sa largeur —
    un seul SVG, que le panneau peint tel quel (ndc_pdf.svg : ``g`` et ``transform``)."""
    (x1, y1, w1, h1), c1 = _interieur(haut)
    (x2, y2, w2, h2), c2 = _interieur(bas)
    k = part * w1 / w2
    tx, ty = x1 + (w1 - k * w2) / 2.0, y1 + h1 + ecart
    H = h1 + ecart + k * h2
    def c(x):                                           # une coordonnée du dessin (pas un nombre écrit pour le lecteur)
        return repr(round(x, 4))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{c(x1)} {c(y1)} {c(w1)} {c(H)}">'
            f'<g>{c1}</g><g transform="translate({c(tx)},{c(ty)}) scale({repr(round(k, 6))}) translate({c(-x2)},{c(-y2)})">{c2}</g></svg>')


def dessin_panneau(el: Dict, r: Dict) -> Tuple[str, str]:
    """(le titre du panneau, le dessin de l'écran en SVG) : la barre chargée et ses enveloppes ; en efforts imposés, la
    section à l'échelle. Un PRS (lot C, C5 ; brief § 18.7, charte E1 : un seul endroit) : son croquis coté sous la barre,
    dans le même dessin — ou seul, en efforts imposés."""
    if el["mode"] == "chargee":
        env = DB.enveloppes(r)
        quoi = " — M ET V À L'ELU" if env and env[0] == "ec3" else (" — M ET V EN SERVICE" if env else "")
        svg = DB.barre("note_barre", el, r, largeur=LARGEUR_DESSIN, legende=False)
        if el.get("prs"):
            return "LA BARRE" + quoi + " · LE PRS", empiler(svg, DP.section_prs(PP.section(el), None, "note_prs"))
        return "LA BARRE" + quoi, svg
    return "LA SECTION", P.vue_section(el, "note_section", fiche=False)


def reserve_donnees(blocs) -> float:
    """La hauteur à garder sous le dessin pour les données du panneau : ses deux blocs sont côte à côte, le plus long
    décide (la planche du kit, sans cette clé, réserverait la SOMME de leurs lignes — le dessin en serait réduit)."""
    return TETE_DONNEES + LIGNE_DONNEES * max((len(rows) for _t, rows in blocs), default=0) + PIED_DONNEES


def hauteur_naturelle(svg: str, largeur: float) -> float:
    """La hauteur que le dessin prend quand il occupe toute la largeur du panneau."""
    vb = [float(t) for t in ET_.fromstring(svg).get("viewBox").split()]
    return vb[3] * (largeur - 2 * MARGE_DESSIN) / vb[2] + 2 * MARGE_DESSIN


# ------------------------------------------------------------------ une planche par barre
class Planche(Encre):
    """La planche des notes Poutre et Dalle, pour une barre : elle se poursuit sur une page suivante quand elle déborde ;
    la couleur de sa pastille est celle de l'état de la barre à l'écran."""
    planche_suite = True
    LEGENDE_GARDE = LEGENDE
    MARGE_GARDE, RETRAIT_GARDE = 52, 26            # la marge de la garde, le retrait du titre d'une ligne (``garde_lignes``)

    def coul_etat(self, sec):
        return {"ok": self.ok, "att": self.att, "ko": self.ko}[sec["couleur"]]

    def garde(self, d, doc, sections, fiches=None):
        """La garde nomme chaque barre EN ENTIER, avec son résumé entier — elle en a la place — ; un nom qui passerait
        sous la pastille d'état cède quand même (« … »)."""
        lignes = []
        for s in sections:
            place = d.W - 2 * self.MARGE_GARDE - self.RETRAIT_GARDE - (string_width(s["etat"].upper(), self.f_v, 6.4) + 14) - 10
            nom = _ceder(s["nom_entier"], lambda n: string_width(f"{n} — {s['section']}", self.f_h, 11), place)
            lignes.append(dict(s, poutre=nom, acier=s["resume"]))
        return super().garde(d, doc, lignes, fiches)

    def largeur_colonne(self) -> float:
        """La largeur d'une colonne de calcul (la grille de ``_page_planche``)."""
        zx = self.margin + self.coupe_w + self.gap
        return (self.page[0] - self.margin - zx - self.col_gap * (self.n_cols - 1)) / self.n_cols

    # (mise à jour 1, étape 2) UN genre d'élément de plus, ("d", svg) : un dessin peint à la largeur de sa colonne, dans le fil du
    # bloc ; quand il n'entre plus dans la colonne, il passe à la suivante avec le rappel « (suite) » du kit, comme une formule.
    # Le kit ndc_pdf ne change pas ; un bloc sans dessin passe tel quel au kit.
    AVANT_DESSIN, APRES_DESSIN = 5.0, 3.0

    def _items(self, d, fr, v, ms, cols, suivante=None):
        if not any(it[0] == "d" for it in v["items"]):
            return super()._items(d, fr, v, ms, cols, suivante)
        for it in v["items"]:
            if it[0] != "d":
                fr = super()._items(d, fr, dict(v, items=[it]), ms, cols, suivante)
                continue
            h = hauteur_dessin(it[1], fr.w)
            if not fr.fits(self.AVANT_DESSIN + h + self.APRES_DESSIN) and (cols.index(fr) + 1 < len(cols) or suivante is not None):
                fr = cols[cols.index(fr) + 1] if cols.index(fr) + 1 < len(cols) else suivante()
                self._suite(d, fr, v)
            fr.down(self.AVANT_DESSIN + h)
            peindre_svg(d.c, it[1], fr.x, fr.y, fr.w, h, marge=0.0, riche=True)
            fr.down(self.APRES_DESSIN)
        return fr


def hauteur_dessin(svg: str, largeur: float) -> float:
    """La hauteur d'un dessin posé à la largeur de sa colonne."""
    vb = [float(t) for t in ET_.fromstring(svg).get("viewBox").split()]
    return vb[3] * largeur / vb[2]


def sous_les_formules(bloc: Dict, svg: str) -> None:
    """Pose un dessin dans un bloc de la note, SOUS SA DERNIÈRE FORMULE : avant ses remarques et son verdict."""
    k = max((i for i, it in enumerate(bloc["items"]) if it[0] == "e"), default=0)
    bloc["items"].insert(k + 1, ("d", svg))


def resume(el: Dict, r: Dict, court: bool = False) -> str:
    """« S235 · 6,00 m · 74 % ELS · 58 % Eurocode 3 » : sous le nom de la barre, dans la garde et dans l'en-tête ;
    ``court`` : les seuls taux (« 74 % ELS · 58 % Eurocode 3 »). Le taux AVANT le nom de la méthode, comme le
    verdict de l'écran (« 74 % Méthode ELS (NBN B 51-001) ») : « Eurocode 3 9 % » se lisait « 39 % ». Une barre qui demande des
    raidisseurs le dit au bout (« · 2 raidisseurs »)."""
    morceaux = [] if court else [el["nuance"]]
    if el["mode"] == "chargee" and not court:
        morceaux.append(f"{' + '.join(M.m(x) for x in el['travees'])} m")
    for k, x in r["methodes"].items():
        if x.get("taux_max") is not None:
            p = C.pct(x["taux_max"])
            morceaux.append((f"> {C.PLAFOND_TAUX} %" if p > C.PLAFOND_TAUX else f"{C.pc(p)} %") + f" {COURT[k]}")
    k = len(M.raidis(r["ame"]))
    if k:                                              # ce que la barre demande en plus de son profilé
        morceaux.append(f"{k} raidisseur{'s' if k > 1 else ''}")
    return " · ".join(morceaux)


def _ceder(texte: str, large, place: float) -> str:
    """Le texte, coupé par « … » pour que ``large(texte)`` tienne dans ``place`` (ses dernières lettres cèdent)."""
    if large(texte) <= place:
        return texte
    while len(texte) > 1 and large(texte.rstrip() + "…") > place:
        texte = texte[:-1]
    return texte.rstrip() + "…"


def entete(el: Dict, r: Dict, mot: str, style: "Planche") -> Tuple[str, str]:
    """(le nom, le résumé) de l'en-tête d'une planche — le titre « nom — profilé (suite) » au corps 14, à gauche ; le
    résumé au corps 8 puis la pastille d'état, à droite. Le résumé entier s'il tient entre les deux ; sinon ses seuls
    taux ; sinon rien ; et si le titre seul ne tient pas à gauche de la pastille (un nom de 60 caractères), le nom cède."""
    register_all()                                 # les polices de la note : la planche se mesure avant qu'un document existe
    nom = el.get("nom") or P.NOM_DEFAUT

    def titre(n: str) -> float:
        return string_width(f"{n} — {r['profil']} (suite)", style.f_h, 14)

    place = style.page[0] - 2 * style.margin - (string_width(mot.upper(), style.f_v, 6.6) + 14) - 8
    for res in (resume(el, r), resume(el, r, court=True)):
        if titre(nom) + 12 + string_width(res, style.f_v, 8) <= place:
            return nom, res
    return _ceder(nom, titre, place - 6), ""


def resume_entete(el: Dict, r: Dict, mot: str, style: "Planche") -> str:
    """Le résumé de l'en-tête d'une planche (``entete``)."""
    return entete(el, r, mot, style)[1]


def planche(el: Dict, r: Dict, style: Planche, impression: str = "deux") -> Dict:
    """La planche d'une barre, dans l'ordre du § 9 (la docstring du fichier) : ``r`` est le résultat ENTIER du moteur (les
    deux méthodes) ; la planche imprime ce que l'option ``impression`` dit (``imprime``) — son en-tête, sa couleur et son
    mot (le statut des méthodes imprimées et de l'âme), son panneau, ses vérifications numérotées."""
    register_all()                                 # (les titres se mesurent : verification)
    largeur = style.largeur_colonne()
    rv = imprime(r, impression)
    synthese = impression == "deux" and R.comparable(rv)
    avec_ecarts = "historique" in rv["methodes"]
    tete = [donnees(el, rv, autre_methode(r, rv))]
    n0 = len(tete) + synthese + avec_ecarts             # les blocs avant le détail : la barre, la synthèse, les écarts
    details: List[Dict] = []
    num: Dict[str, int] = {}
    dits = {_cle(t) for b in tete for it in b["items"] if it[0] == "pr" for t in it[2:3]}

    def poser(b: Dict) -> None:
        details.append(b)
        dits.update(_cle(it[2]) for it in b["items"] if it[0] == "pr" and len(it) > 2)

    imprimes: Dict[str, List[str]] = {}            # ce que chaque vérification imprime (la décision n° 2 les cite)
    premieres = {L["verifications"][0]: L for L in rv["ame"]["lieux"] if L.get("avec") and L.get("verifications")}
    # (mise à jour 1, étape 2) les CONTRAINTES dans la section, sous les formules du bloc qui les calcule
    s_ = MO.section_de(rv["barre"]) if rv.get("barre") else None
    vs_ = {k: DP.volets(k, x, s_) for k, x in rv["methodes"].items()}
    for k, x in list(rv["methodes"].items()) + [(M.AME, rv["ame"])]:
        poser(methode(el, k, x, largeur) if k != M.AME else ame(el, rv, largeur))
        precedente = None
        for v in x["verifications"]:
            details.append(verification(k, x, v, precedente, dits, premieres.get(v["id"]), largeur))
            dessin = DP.volets_du_bloc(k, v, vs_.get(k) or {})
            if dessin:
                sous_les_formules(details[-1], DP.contraintes(f"note_{k}_{v['id']}", s_, dessin, largeur, hs=84.0))
            imprimes[v["id"]] = [it[2] for it in details[-1]["items"] if it[0] == "pr" and len(it) > 2]
            num[v["id"]] = n0 + len(details)
            precedente = (n0 + len(details), lignes_ecrites(x, v))
    verifs = list(tete)
    if synthese:
        verifs.append(comparaison_methodes(rv, largeur, num))
    if avec_ecarts:
        verifs.append(ecarts(rv, num))
    verifs += details
    verifs.append(decisions(el, rv, num, imprimes=imprimes))
    ac = a_confirmer(rv, largeur)
    if ac:
        verifs.append(ac)
    verifs.append(limites(el, rv))
    for i, v in enumerate(verifs, 1):
        v["num"] = i
    titre, svg = dessin_panneau(el, rv)
    niv = P._niveau(rv)
    couleur, mot = COULEURS[niv], R.mot_niveau(niv)

    def dessin(d, x, y, w, h, S):
        peindre_svg(d.c, svg, x, y, w, h, marge=MARGE_DESSIN, riche=True)

    blocs = blocs_panneau(el)
    nom, court = entete(el, rv, mot, style)
    return {"poutre": nom, "nom_entier": el.get("nom") or P.NOM_DEFAUT, "section": rv["profil"], "beton": "", "acier": court, "resume": resume(el, rv),
            "etat": mot, "couleur": couleur,
            "titre_coupe": titre, "dessin": dessin, "coupe_h": hauteur_naturelle(svg, style.coupe_w), "reserve": reserve_donnees(blocs),
            "blocs": blocs, "verifs": verifs}


def imprime(r: Dict, impression: str) -> Dict:
    """Le résultat d'une barre tel que la note l'IMPRIME : la vue de la méthode affichée (``profil.vue`` : l'écran et la
    note, une seule règle — mise à jour 1, § 1.3)."""
    return P.vue(r, impression)


def sections(etat: Dict, style: Optional[Planche] = None) -> List[Dict]:
    """Les planches de la note, dans l'ordre des barres, avec ce que l'option de la note imprime (``planche``). Une barre que
    le moteur refuse arrête la note (ValueError), avec son nom et le message du moteur."""
    style = style or Planche()
    impression = etat.get(P.IMPRESSION, P.IMPRESSIONS[0])
    out = []
    for i, el in enumerate(etat["elements"]):
        try:
            out.append(planche(el, P.calculer(el), style, impression))
        except (ValueError, KeyError, TypeError) as err:
            raise ValueError(f"La note n'est pas préparée : barre « {el.get('nom') or f'n° {i + 1}'} », {err}") from err
    return out


# ------------------------------------------------------------------ le document
def document(etat: Dict, infos: Optional[Dict], chemin: str):
    """Le document construit (pas encore écrit) : la garde et les planches. ``d.warnings`` : ce qui déborde encore."""
    etat = P.normaliser(copy.deepcopy(etat))
    style = Planche()
    doc_meta = ndc_data.construire_doc(infos or {}, date_defaut=datetime.today().strftime("%d/%m/%Y"))
    return style.build(chemin, sections=sections(etat, style), doc=doc_meta)


def nom_note(etat: Dict, infos: Optional[Dict]) -> str:
    """Le nom de la note : celui que la note du site porte ; à défaut, le nom de la barre quand elle est seule, le titre
    du module sinon."""
    nom = str((infos or {}).get("nom") or "").strip()
    if nom:
        return nom
    els = etat.get("elements") or []
    if len(els) == 1 and isinstance(els[0], dict):
        return str(els[0].get("nom") or "").strip() or TITRE
    return TITRE


def infos_doc(etat: Dict, infos: Optional[Dict]) -> Dict:
    """Les « infos » de la garde ({"nom_projet", "partie", "indice", "date"}) depuis celles de la note du site : la
    correspondance de la note de la poutre (api_web/note_poutre.py)."""
    infos = infos or {}
    nom, chantier = nom_note(etat, infos), str(infos.get("chantier") or "").strip()
    out = {"nom_projet": chantier or nom, "partie": nom if chantier else TITRE, "indice": str(infos.get("indice") or "").strip() or "A"}
    if str(infos.get("date") or "").strip():
        out["date"] = str(infos.get("date")).strip()
    return out


def pdf(etat: Dict, infos: Optional[Dict] = None) -> bytes:
    """La note PDF (octets) : la garde A4 portrait, puis la planche A4 paysage de chaque barre (une page, ou plusieurs
    quand elle se poursuit). Ce qui déborderait encore est REFUSÉ avec un message : jamais une note tronquée."""
    fd, chemin = tempfile.mkstemp(suffix=".pdf", prefix="note_profil_")
    os.close(fd)
    try:
        d = document(etat, infos_doc(etat, infos), chemin)
        if d.warnings:
            raise ValueError("La note ne tient pas sur sa planche (" + " ; ".join(d.warnings) + ") : elle ne sort pas tronquée. À signaler "
                             "au bureau ; l'écran garde tout le calcul.")
        d.save()
        with open(chemin, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(chemin)
        except OSError:
            pass


def fichier(etat: Dict, infos: Optional[Dict] = None) -> Dict:
    """Le fichier à télécharger : {"nom", "type": "application/pdf", "base64"} (le routeur le contrôle)."""
    infos = dict(infos or {})
    octets = pdf(etat, infos)
    return {"nom": C.nom_fichier(f"Note de calcul - {nom_note(etat, infos)}", "pdf"), "type": "application/pdf",
            "base64": base64.b64encode(octets).decode("ascii")}
