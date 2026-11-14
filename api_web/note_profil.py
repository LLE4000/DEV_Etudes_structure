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

CE QUI S'IMPRIME, par barre : les MÉTHODES que la barre a de cochées à l'écran (rubrique « Méthodes » : la méthode
historique, l'Eurocode 3, ou les deux) — la note imprime ce que l'écran montre. La méthode historique vient la
première : ses contraintes admissibles, son flambement et son déversement, tels que les classeurs du bureau les
écrivent. Avec les deux méthodes, un tableau les compare sujet par sujet. Les paramètres de l'annexe belge dont la source
reste à confirmer sont nommés, avec la valeur prise ; ce que la note ne vérifie pas est dit.

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

from acier.barre import ecriture as W
from acier.barre import resultat as RE
from ndc_pdf import data as ndc_data
from ndc_pdf import riche as RI
from ndc_pdf.fonts import register_all, string_width
from ndc_pdf.styles import Encre
from ndc_pdf.svg import peindre_svg

from . import commun as C
from . import dessin_barre as DB
from . import profil as P
from . import profil_mots as M
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
COURT = {"historique": "historique", "ec3": "Eurocode 3"}        # une méthode au fil d'une phrase
LEGENDE = "Vert : conforme   ·   Ocre : attention (95 à 100 %), ou une vérification non couverte   ·   Rouge : non conforme"
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
# ce que la note ne vérifie pas (docs/acier/BARRES_ACIER.md : le périmètre de la version 1)
LIMITES = ("le voilement de l'âme, les charges concentrées et les raidisseurs (EN 1993-1-5)", "les assemblages et les appuis",
           "le feu et la fatigue")


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
    signe = "≤" if v["statut"] != RE.NOK else ">"
    g, _, d = crit.partition(" ≤ ")
    if R.interaction(v):
        return f"η = {C.fr(v['valeur'], 3)} {signe} 1"
    u = v["unite"]
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
        return f"{v} kN à {M.m(ch['a'])} m{haut}"
    return f"couple de {v} kN·m à {M.m(ch['a'])} m"


def donnees(el: Dict, r: Dict) -> Dict:
    """Le premier bloc : la barre, ses charges (ou ses efforts imposés), ses maintiens, ses longueurs de flambement."""
    items: List[tuple] = [("v", f"Profilé {r['profil']}", "", f"{M.n(r['masse'], 1)}", "kg/m")]
    fy = next((x["grandeurs"].get("fy") for x in r["methodes"].values() if x["grandeurs"].get("fy")), None)
    if fy:
        items.append(("v", f"Acier {el['nuance']}", "f_y", M.n(fy, 0), "N/mm²"))
    Ls = r["longueurs"]
    if el["mode"] == "chargee":
        items.append(("v", "Modèle", "", M.NOM_MODELE[el["modele"]], ""))
        for i, L in enumerate(el["travees"]):
            items.append(("v", "Portée" if len(el["travees"]) == 1 else f"Travée {i + 1}", "L" if len(el["travees"]) == 1 else f"L_{i + 1}", M.m(L), "m"))
        items.append(("s", "Charges (valeurs caractéristiques)"))
        if el["poids_propre"]:
            items.append(("pr", "Poids propre du profilé : compté, cas permanent."))
        for c in el["cas"]:
            quoi = [_charge(ch) for ch in c["charges"]] + ([f"effort normal {M.n(c['N'], 1)} kN"] if c.get("N") else [])
            items.append(("pr", f"{c['nom']} — {_minuscule(M.NATURE_COURTE[c['nature']])} : " + (" ; ".join(quoi) if quoi else "aucune charge") + "."))
        items.append(("s", "Maintiens et flambement"))
        mts = " ; ".join(f"{M.NOM_GENRE[mt['genre']].lower()} à {M.m(mt['x'])} m" for mt in el["maintiens"])
        items.append(("pr", f"Maintiens latéraux : {mts}." if mts else "Aucun maintien latéral entre les appuis."))
    else:
        items.append(("s", "Efforts imposés"))
        jeux = [(k, {"historique": "service", "ec3": "elu"}[k]) for k in r["methodes"]]
        items.append(("pr", "Valeurs : " + " · ".join({"service": "de service (méthode historique)", "elu": "de calcul (Eurocode 3)"}[j]
                                                      for _k, j in jeux) + "."))
        for cle, s_, unite, _aide in M.EFFORTS:
            vals = [el["efforts"][j][cle] for _k, j in jeux]
            if any(vals):
                items.append(("v", EFFORTS[cle], tex(s_), " · ".join(M.r1(x) for x in vals), unite))
        formes = dict(M.DIAGRAMMES)
        items.append(("pr", f"Diagramme du moment : {formes[el['diagramme']['y']['forme']].lower()} (axe fort), "
                            f"{formes[el['diagramme']['z']['forme']].lower()} (axe faible)."))
        items.append(("s", "Flambement et déversement"))
    for a, nom in (("y", "axe fort"), ("z", "axe faible")):
        items.append(("v", f"Flambement, {nom}", f"u_{a}·L_{a}", f"{M.n(Ls[a]['u'], 3)} × {M.m(Ls[a]['L'])}", "m"))
    return {"titre": "La barre et ses charges" if el["mode"] == "chargee" else "La barre et ses efforts", "items": items, "verdicts": []}


def _tableau_combinaisons(x: Dict, largeur: float) -> Optional[tuple]:
    """Le bilan des combinaisons d'une méthode (deux au moins) : chacune, la vérification qui y gouverne, son taux ;
    les plus chargées d'abord, COMBINAISONS_MAX au plus."""
    combs = x.get("combinaisons") or []
    if len(combs) < 2:
        return None
    titres = {v["id"]: v["titre"] for v in x["verifications"]}
    tri = sorted(combs, key=lambda c: -(c["taux"] if c["taux"] is not None else -1.0))
    lignes = []
    for c in tri[:COMBINAISONS_MAX]:
        gov = c["nom"] == x.get("combinaison")
        quoi = titres.get(c["gouvernant"], c["gouvernant"] or "—")
        lignes.append([("lab", c["nom"], gov), ("ref", _minuscule(quoi)), ("eta", taux_ecrit(c["taux"]), c["statut"] != RE.NOK, gov)])
    cols = [("COMBINAISON", largeur - 78 - 40, "left"), ("CE QUI Y GOUVERNE", 78, "left"), ("TAUX", 40, "right")]
    note = None
    if len(combs) > COMBINAISONS_MAX:
        note = f"{len(combs)} combinaisons calculées ; les {COMBINAISONS_MAX} plus chargées sont montrées."
    return ("tab", cols, lignes, note)


def methode(el: Dict, k: str, x: Dict, largeur: float) -> Dict:
    """Le bloc de tête d'une méthode : sa norme, ses hypothèses (celles que le moteur écrit), le bilan de ses
    combinaisons, ses avertissements, puis son verdict — le taux le plus fort et ce qu'il compare."""
    items: List[tuple] = [("pr", riche(h)) for h in x["hypotheses"]]
    tab = _tableau_combinaisons(x, largeur)
    if tab:
        items.append(tab)
    items += [("pr", riche(a["message"])) for a in x["alertes"]]
    g = R.gouvernante(x)
    niv = R.niveau_methode(x)
    if g is None:
        verdict = {"texte": "Aucun effort à vérifier.", "etat": "att"}
    else:
        ou = lieu(g)
        verdict = {"texte": f"{R.mot_niveau(niv)} — {_minuscule(g['titre'])} : {comparaison(g)}" + (f" ({ou})" if ou else ""), "etat": COULEURS[niv],
                   "riche": True, "cale": taux_ecrit(x["taux_max"]) if x.get("taux_max") is not None else ""}
    items.append(("k", 0))
    return {"titre": f"{M.METHODES[k]} — {x['norme']}", "items": items, "verdicts": [verdict]}


def lignes_ecrites(x: Dict, v: Dict) -> List[Tuple[Tuple[str, ...], str]]:
    """Les formules d'une vérification telles qu'elles s'écrivent : [((symbole, formule, valeurs), résultat)] — celles
    de la fenêtre de calcul de l'écran, sans la ligne du taux (le verdict le dit)."""
    out = []
    for f, syms in R.formules_de(x, v, taux=False):
        parts, res = W.morceaux(f, syms)
        out.append((tuple(parts), res))
    return out


def verification(k: str, x: Dict, v: Dict, precedente: Optional[Tuple[int, List]] = None) -> Dict:
    """Une vérification : sa référence et l'endroit où elle a été trouvée, ses formules, ses messages, son verdict.
    ``precedente`` : (le numéro, les formules écrites) de la vérification d'avant — celles qu'elle vient d'écrire mot
    pour mot ne sont pas répétées quand il y en a COMMUNES_MIN au moins : une ligne le dit, à leur place."""
    items: List[tuple] = [("pr", " · ".join(t for t in (v["reference"], lieu(v)) if t))]
    lignes = lignes_ecrites(x, v)
    communes = set(precedente[1]) & set(lignes) if precedente else set()
    if len(communes) < COMMUNES_MIN:
        communes = set()
    dit = False
    for parts, res in lignes:
        if (parts, res) in communes:
            if not dit:
                n = sum(1 for l in lignes if l in communes)
                items.append(("pr", f"Les {n} facteurs écrits à la vérification {precedente[0]} sont les mêmes : ils ne sont pas répétés."))
                dit = True
            continue
        items.append(("e", None, [tex(p) for p in parts], tex(res)))
    items += [("pr", riche(t)) for t in v["messages"]]
    items.append(("k", 0))
    verdict = {"texte": comparaison(v), "etat": etat_verification(v), "riche": True,
               "cale": taux_ecrit(v["taux"]) if v["taux"] is not None else M.MOT_STATUT[v["statut"]].lower()}
    return {"titre": f"{M.METHODES_COURT[k]} · {_minuscule(v['titre'])}", "items": items, "verdicts": [verdict]}


def comparaison_methodes(r: Dict, largeur: float) -> Dict:
    """Les deux méthodes côte à côte, sujet par sujet : le taux le plus fort de chacune."""
    def case(t):
        return ("num", "—") if t is None else ("eta", taux_ecrit(t), t <= 1.0, False)

    lignes = [[("lab", li["sujet"], li["sujet"] == "Le plus défavorable"), case(li["historique"]), case(li["ec3"])] for li in r["comparaison"]]
    cols = [("SUJET", largeur - 2 * 52, "left"), ("HISTORIQUE", 52, "right"), ("EUROCODE 3", 52, "right")]
    return {"titre": "Comparaison des deux méthodes", "items": [("tab", cols, lignes), ("pr", riche(R.NOTE_COMPARAISON))], "verdicts": []}


def a_confirmer(r: Dict, largeur: float) -> Optional[Dict]:
    """Les paramètres de l'annexe nationale belge dont la source reste à confirmer : la valeur prise, d'où elle vient."""
    ac = (r["methodes"].get("ec3") or {}).get("a_confirmer") or []
    if not ac:
        return None
    cols = [("PARAMÈTRE", 62, "left"), ("VALEUR PRISE", 56, "left"), ("D'OÙ VIENT LA VALEUR", largeur - 62 - 56, "left")]
    lignes = [[("lab", riche(p["nom"]), True), ("lab", R.valeur_param(p["valeur"])), ("ref", riche(p["source"]))] for p in ac]
    return {"titre": "Paramètres à confirmer — annexe belge", "items": [("tab", cols, lignes), ("pr", R.NOTE_A_CONFIRMER)], "verdicts": []}


def limites(r: Dict) -> Dict:
    """Ce que la note ne vérifie pas : la flèche (donnée, non vérifiée), puis le périmètre du module."""
    items: List[tuple] = []
    fl = (r.get("statique") or {}).get("fleche")
    if fl and fl.get("valeur"):
        rapport = f" (L/{C.fr(fl['rapport'], 0)})" if fl.get("rapport") else ""
        items.append(("pr", f"La flèche de service — {C.fr(abs(fl['valeur']), 1)} mm{rapport} sous {fl['combinaison']}, à {M.m(fl['x'])} m : "
                            "une information, elle n'est pas vérifiée."))
    items.append(("pr", "Ne sont pas vérifiés : " + " ; ".join(LIMITES) + "."))
    return {"titre": "Ce que la note ne vérifie pas", "items": items, "verdicts": []}


# ------------------------------------------------------------------ le panneau de gauche
def blocs_panneau(el: Dict) -> List[Tuple[str, List[tuple]]]:
    """Sous le dessin : les dimensions du profilé, puis ses propriétés — celles de la vue « Section » de l'écran."""
    f = P.fiche_profil(el["profil"])

    def lignes(triplets):
        return [(LIBELLES.get(s, ""), tex(s) if s != "masse" else "", val, unite) for s, val, unite in triplets]

    return [("DIMENSIONS", lignes(f["dimensions"])), ("PROPRIÉTÉS", lignes(f["proprietes"]))]


def dessin_panneau(el: Dict, r: Dict) -> Tuple[str, str]:
    """(le titre du panneau, le dessin de l'écran en SVG) : la barre chargée et ses enveloppes ; en efforts imposés, la
    section à l'échelle."""
    if el["mode"] == "chargee":
        env = DB.enveloppes(r)
        quoi = " — M ET V À L'ELU" if env and env[0] == "ec3" else (" — M ET V EN SERVICE" if env else "")
        return "LA BARRE" + quoi, DB.barre("note_barre", el, r, largeur=LARGEUR_DESSIN, legende=False)
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


def resume(el: Dict, r: Dict, court: bool = False) -> str:
    """« S235 · 6,00 m · 74 % historique · 52 % Eurocode 3 » : sous le nom de la barre, dans la garde et dans l'en-tête ;
    ``court`` : les seuls taux (« 74 % historique · 52 % Eurocode 3 »). Le taux AVANT le nom de la méthode, comme le
    verdict de l'écran (« 74 % Méthode historique ») : « Eurocode 3 9 % » se lisait « 39 % »."""
    morceaux = [] if court else [el["nuance"]]
    if el["mode"] == "chargee" and not court:
        morceaux.append(f"{' + '.join(M.m(x) for x in el['travees'])} m")
    for k, x in r["methodes"].items():
        if x.get("taux_max") is not None:
            p = C.pct(x["taux_max"])
            morceaux.append((f"> {C.PLAFOND_TAUX} %" if p > C.PLAFOND_TAUX else f"{C.pc(p)} %") + f" {COURT[k]}")
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


def planche(el: Dict, r: Dict, style: Planche) -> Dict:
    """La planche d'une barre : son en-tête, son panneau, ses vérifications numérotées — les données ; par méthode, sa
    tête puis chacune de ses vérifications ; la comparaison ; les paramètres à confirmer ; ce qui n'est pas vérifié."""
    largeur = style.largeur_colonne()
    verifs = [donnees(el, r)]
    for k, x in r["methodes"].items():
        verifs.append(methode(el, k, x, largeur))
        precedente = None
        for v in x["verifications"]:
            verifs.append(verification(k, x, v, precedente))
            precedente = (len(verifs), lignes_ecrites(x, v))
    if R.comparable(r):
        verifs.append(comparaison_methodes(r, largeur))
    ac = a_confirmer(r, largeur)
    if ac:
        verifs.append(ac)
    verifs.append(limites(r))
    for i, v in enumerate(verifs, 1):
        v["num"] = i
    titre, svg = dessin_panneau(el, r)
    niv = P._niveau(r)
    couleur, mot = COULEURS[niv], R.mot_niveau(niv)

    def dessin(d, x, y, w, h, S):
        peindre_svg(d.c, svg, x, y, w, h, marge=MARGE_DESSIN, riche=True)

    blocs = blocs_panneau(el)
    nom, court = entete(el, r, mot, style)
    return {"poutre": nom, "nom_entier": el.get("nom") or P.NOM_DEFAUT, "section": r["profil"], "beton": "", "acier": court, "resume": resume(el, r),
            "etat": mot, "couleur": couleur,
            "titre_coupe": titre, "dessin": dessin, "coupe_h": hauteur_naturelle(svg, style.coupe_w), "reserve": reserve_donnees(blocs),
            "blocs": blocs, "verifs": verifs}


def sections(etat: Dict, style: Optional[Planche] = None) -> List[Dict]:
    """Les planches de la note, dans l'ordre des barres. Une barre que le moteur refuse arrête la note (ValueError), avec
    son nom et le message du moteur."""
    style = style or Planche()
    out = []
    for i, el in enumerate(etat["elements"]):
        try:
            out.append(planche(el, P.calculer(el), style))
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
