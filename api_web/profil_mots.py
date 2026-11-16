# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — les MOTS et les NOMBRES de l'écran : ce que la saisie, les résultats,
les fenêtres de calcul et les dessins écrivent de la même façon (api_web/profil.py et ses fichiers).

Rien n'est calculé ici ; les titres des vérifications, leurs critères et leurs formules viennent du moteur
(acier/barre) — ce fichier les met en forme.
"""
import re
from typing import Dict, Optional

from acier.barre import combinaisons as CO
from acier.barre import ec3_deversement as DV
from acier.barre import historique as HI
from acier.barre import resultat as RE

from . import commun as C

NB = C.NBSP
FINE = " "                  # l'espace fine des milliers (celle de commun.fr)
MOINS = "−"

# la méthode du bureau (clé interne « historique », inchangée : état, notes enregistrées, identifiants) se NOMME
# « Méthode ELS (NBN B 51-001) » — titres, en-têtes de rubrique, aides, phrases — et « ELS » seul là où la place manque
# (en-têtes de colonnes, segment du choix des méthodes, titre court d'un bloc, taux du résumé)
ELS = "ELS"
METHODES = {"historique": f"Méthode {ELS} ({HI.NORME})", "ec3": "Eurocode 3"}
METHODES_COURT = {"historique": ELS, "ec3": "Eurocode 3"}


def minuscule(s: str) -> str:
    """Un nom au fil d'une phrase : sa première lettre en minuscule (« Eurocode 3 » → « eurocode 3 », « Méthode ELS (NBN B
    51-001) » → « méthode ELS (NBN B 51-001) ») — un sigle reste en capitales (« ELS »)."""
    return s if s[1:2].isupper() else s[:1].lower() + s[1:]


AME = "ame"                                        # le bloc de l'âme (EN 1993-1-5) : ni l'une ni l'autre méthode, toujours rendu
TITRE_AME = "Âme et raidisseurs"
TITRE_AME_SEUL = "Contrôles complémentaires EN 1993-1-5"      # son titre dans une note « méthode ELS seule »
PREFIXE = {"historique": "h", "ec3": "e", AME: "a"}          # le préfixe des fenêtres de calcul d'une méthode, de l'âme
MODES_APPUI = (("poses", "Posée", "La barre repose sur ses appuis : la réaction appuie sur l'âme."),
               ("ame", "Par l'âme", "Platine d'about, cornières, gousset : aucune force concentrée sur l'âme aux appuis."))
NOM_MODE_APPUI = {"poses": "posée", "ame": "par l'âme"}
AIDE_SS = ("Longueur d'appui rigide : la longueur sur laquelle la force entre dans la semelle (largeur de l'appui, platine, semelle du "
           "poteau qui charge). 0 : le cas le plus défavorable.")
AIDE_DEBORD = "Débord : la distance du bout de la barre au bord de l'appui de rive. 0 : la barre s'arrête au bord de l'appui."
MODES = (("chargee", "Barre chargée", "Appuis, charges, maintiens : le module fait la statique et les combinaisons."),
         ("efforts", "Efforts imposés", f"Les efforts sont donnés : ceux de service pour la {minuscule(METHODES['historique'])}, ceux de calcul pour l'Eurocode."))
MODELES = (("appuis", "Deux appuis"), ("console", "Console"), ("enc_app", "Encastrée – appuyée"), ("bi_enc", "Bi-encastrée"),
           ("continue", "Continue"))
NOM_MODELE = dict(MODELES)
TYPES_CHARGE = (("q", "Répartie"), ("tri", "Triangulaire"), ("P", "Ponctuelle"), ("C", "Couple"))
UNITE_CHARGE = {"q": "kN/m", "tri": "kN/m", "P": "kN", "C": "kN·m"}
LETTRE_CHARGE = {"q": "q", "tri": "q", "P": "P", "C": "C"}
SENS = (("croissante", "0 → q"), ("decroissante", "q → 0"))
NATURES = tuple((k, v[0]) for k, v in CO.NATURES.items())
NATURE_COURTE = {k: v[0].split(" — ")[0].replace("Permanente (G)", "Permanente") for k, v in CO.NATURES.items()}
GENRES = (("complet", "Complet", "Déplacement latéral ET torsion empêchés (une poutre secondaire, un contreventement)."),
          ("haut", "Semelle haute", "Seule la semelle haute est tenue (un platelage, une panne)."),
          ("bas", "Semelle basse", "Seule la semelle basse est tenue (un bracon)."))
NOM_GENRE = {k: t for k, t, _ in GENRES}
NIVEAUX = (("haut", "Semelle haute", "La charge est posée sur la semelle haute : elle aggrave le déversement."),
           ("centre", "Centre", "La charge agit au centre de cisaillement."),
           ("bas", "Semelle basse", "La charge est suspendue à la semelle basse : elle stabilise."))
DIAGRAMMES = ((DV.CONSTANT, "Constant"), (DV.LINEAIRE, "Linéaire (ψ)"), (DV.REPARTI, "Charge répartie"), (DV.PONCTUEL, "Charge à mi-portée"))
CHOIX_METHODES = (("deux", "Les deux"), ("historique", METHODES_COURT["historique"]), ("ec3", METHODES_COURT["ec3"]))
C_MODES = (("unite", "C = 1", "Les classeurs du bureau."), ("norme", "Selon la norme", "0,85 sous des charges ; selon ψ sous des moments d'extrémité."))
EFFORTS = (("N", "N", "kN", "Effort normal ; positif en compression."), ("My", "M_y", "kN·m", "Moment autour de l'axe fort."),
           ("Vz", "V_z", "kN", "Effort tranchant parallèle à l'âme."), ("Mz", "M_z", "kN·m", "Moment autour de l'axe faible."),
           ("Vy", "V_y", "kN", "Effort tranchant parallèle aux semelles."))
# le niveau (couleur) d'un statut sans taux ; « hors domaine » (brief § 9) : le niveau « v » existant (à vérifier), avec
# son mot — un taux calculé hors du domaine de la méthode n'est jamais peint « conforme »
NIVEAU_STATUT = {RE.OK: "p", RE.NOK: "r", RE.NON_COUVERT: "c", RE.HORS_DOMAINE: "v"}
MOT_STATUT = {RE.OK: "Conforme", RE.NOK: "Non conforme", RE.NON_COUVERT: "Non couvert", RE.HORS_DOMAINE: "Hors domaine"}
CLASSE_BLOC = {"p": "ok", "f": "warn", "r": "nok", "c": "inc", "v": "inc"}

_PRIME = re.compile(r"([A-Za-zΑ-Ωα-ω])([′'])_([A-Za-z0-9]+(?:,[A-Za-z0-9]+)*)")      # A′_ed : l'indice derrière la prime


# ------------------------------------------------------------------ nombres
def n(x: Optional[float], dec: int = 3) -> str:
    """Une saisie ou une donnée du catalogue, sans zéro inutile, à la française, les milliers séparés ; « — » sans valeur.
    L'arrondi est celui du site (``commun.nb`` : demi vers l'extérieur) — le catalogue et la barre écrivent le même
    chiffre d'un profilé (HEB 180 : I_y = 3 831,25 cm⁴ → « 3 831,3 », que l'arrondi de Python écrivait « 3 831,2 »)."""
    if x is None:
        return "—"
    s = C.nb(float(x), dec)
    if s == "—":
        return s
    ent, _, d = s.partition(",")
    signe = MOINS if ent.startswith("-") else ""
    ent = f"{int(ent.lstrip('-')):,}".replace(",", FINE)
    return signe + ent + ("," + d if d else "")


def r1(x: Optional[float]) -> str:
    """Un effort ou une résistance calculés : au dixième."""
    return C.fr(x, 1).replace("-", MOINS) if x is not None else "—"


def m(x: float) -> str:
    """Une abscisse ou une longueur en mètres : deux décimales."""
    return C.fr(x, 2)


def taux_txt(t: Optional[float]) -> str:
    """« 92 % » ; « > 999 % » au-delà ; « — » sans taux."""
    if t is None:
        return "—"
    p = C.pct(t)
    return f"&gt;{NB}{C.PLAFOND_TAUX}{NB}%" if p > C.PLAFOND_TAUX else f"{C.pc(p)}{NB}%"


# ------------------------------------------------------------------ symboles
def sub(s: str) -> str:
    """Un symbole ou une formule du moteur en HTML : les indices en indice (``commun.symb_math``), ceux qui suivent une
    prime (A′_ed) aussi ; λ̄ surligné par la feuille (la barre combinante se dessine mal dans certaines polices)."""
    h = C.symb_math(_PRIME.sub("\\1\\2⁣\\3⁣", s))
    h = re.sub("([′'])⁣([^⁣]*)⁣", r"\1<sub>\2</sub>", h)
    return h.replace("λ̄", '<span class="pf-barre">λ</span>')


# ------------------------------------------------------------------ l'âme et ses raidisseurs
def raidis(a: Dict) -> list:
    """Les lieux du bloc de l'âme qui reçoivent un raidisseur."""
    return [L for L in a.get("lieux") or [] if L["decision"] == "raidisseur"]


def decision_lieu(L: Dict) -> str:
    """Ce qu'un lieu décide, en mots (du texte) : la paire de plats, « sans raidisseur », ou la décision impossible et ce
    qui manque (« décision impossible : s_s à fournir » — brief § 8.3, point 5)."""
    if L["decision"] == "raidisseur":
        return L["raidisseur"]["designation"]
    if L["decision"] == "impossible":
        r = L.get("raison") or "Décision impossible."
        return r[:1].lower() + r[1:].rstrip(".")
    return f"sans raidisseur ({L['borne']})" if L.get("borne") else "sans raidisseur"


def resume_ame(a: Dict) -> str:
    """Ce que le bloc de l'âme décide, en une phrase (du texte) : « Pas de raidisseur nécessaire » ; « Raidisseurs :
    2 plats 70 × 12 mm (appui A, appui B) » ; « Décision impossible : s_s à fournir (appui A, appui B) » ; « Âme élancée
    vérifiée au cisaillement » ; vide quand rien n'appuie sur l'âme. La même à l'écran et dans la note."""
    r = raidis(a)
    imp: Dict[str, list] = {}
    for L in a.get("lieux") or []:
        if L["decision"] == "impossible":
            imp.setdefault((L.get("raison") or "Décision impossible.").rstrip("."), []).append(L["nom"][:1].lower() + L["nom"][1:])
    sans = " ; ".join(f"{t} ({', '.join(noms)})" for t, noms in imp.items())
    if r:
        par = {}
        for L in r:
            par.setdefault(L["raidisseur"]["designation"], []).append(L["nom"][:1].lower() + L["nom"][1:])
        return (("Raidisseur : " if len(r) == 1 else "Raidisseurs : ") + " ; ".join(f"{d} ({', '.join(noms)})" for d, noms in par.items())
                + (" ; " + sans[:1].lower() + sans[1:] if sans else ""))
    if sans:
        return sans
    if a.get("lieux"):
        return "Pas de raidisseur nécessaire"
    return "Âme élancée vérifiée au cisaillement" if a.get("verifications") else ""


def niveau(v: Dict) -> str:
    """La couleur d'une vérification : celle de son taux ; sans taux, celle de son statut ; hors domaine : « v »."""
    if v.get("statut") == RE.HORS_DOMAINE:
        return NIVEAU_STATUT[RE.HORS_DOMAINE]
    if v.get("taux") is not None:
        return "r" if v["statut"] == RE.NOK else C.niveau(v["taux"])
    return NIVEAU_STATUT.get(v["statut"], "c")


def lieu(v: Dict) -> str:
    """Où une vérification d'une barre chargée a été trouvée : « x = 3,00 m », « de 0 à 6,00 m » ; vide sinon (du texte)."""
    if v.get("x") is not None:
        return f"x = {m(v['x'])} m"
    if v.get("troncon"):
        return f"de {m(v['troncon'][0])} à {m(v['troncon'][1])} m"
    return ""


def comparaison(v: Dict, court: bool = True) -> str:
    """Ce qu'une vérification compare, en clair (HTML) : « M_y,Ed 66,6 ≤ M_b,Rd 72,5 kN·m » ; une interaction, « η =
    0,549 ≤ 1 » (``court``) ou son critère entier ; une vérification sans taux dit son critère, ou son premier message."""
    crit = v.get("critere") or ""
    if v.get("taux") is None or v.get("valeur") is None:
        return sub(crit) if crit else C.esc((v.get("messages") or [""])[0])
    signe = "≤" if v["statut"] != RE.NOK and v["taux"] <= 1.0 else "&gt;"      # un taux > 1 pour information (non couvert) : « > »
    g, _, d = crit.partition(" ≤ ")
    if not v["unite"] and abs((v["limite"] or 0.0) - 1.0) < 1e-12:
        return f"{'η' if court else sub(g)} = {C.fr(v['valeur'], 3)} {signe} 1"
    return f"{sub(g)} {r1(v['valeur'])} {signe} {sub(d)} {r1(v['limite'])}{NB}{C.esc(v['unite'])}"
