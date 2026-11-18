# -*- coding: utf-8 -*-
"""Barres en acier (module « Choix de profilé ») — les MOTS et les NOMBRES de l'écran : ce que la saisie, les résultats,
les fenêtres de calcul et les dessins écrivent de la même façon (api_web/profil.py et ses fichiers).

Rien n'est calculé ici ; les titres des vérifications, leurs critères et leurs formules viennent du moteur
(acier/barre) — ce fichier les met en forme.
"""
import re
from typing import Dict, Optional

from acier.barre import ame as AMEB
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
           "poteau qui charge). Vide : calcul à la borne 0 (prudent).")
AIDE_DEBORD = "Débord : la distance du bout de la barre au bord de l'appui de rive. 0 : la barre s'arrête au bord de l'appui."
# le champ s_s ou c vide (non saisi) : il se vide comme il se remplit (api_web/commun.num, auto), et dit le sens du vide — le calcul à
# la borne 0, comme l'aide (« à saisir » n'est dit que par une décision impossible : la borne suffit souvent)
SS_VIDE = "borne 0"
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
# ---- (lot C, C3) les libellés et les aides des champs neufs de la SAISIE : un mot par valeur, le sens au survol (charte E6)
NIVEAUX_CHARGE = (("haut", "▲", "Semelle haute"), ("centre", "●", "Centre de cisaillement"), ("bas", "▼", "Semelle basse"))
POSITIONS = (("horizontale", "Horizontale"), ("verticale", "Verticale"), ("inclinee", "Inclinée"))
ANGLE_DEFAUT = 45.0             # ° : l'angle qu'écrit le clic « Inclinée » quand aucun n'est saisi — affiché, à corriger (À VÉRIFIER)
OUVRAGES = (("batiment", "Bâtiment"), ("autre", "Autre", "Autre : hors ponts, tours, silos, chemins de roulement."))
# (mise à jour 1, § 1.3) les deux DÉCISIONS BUREAU du bloc EUROCODE 3 : n° 11 et n° 12 (acier/barre/parametres.py)
AUCUN_EFFORT = "Aucun effort à vérifier"          # (une méthode sans vérification : sa carte le dit)
AME_CLASSE_3 = (("classe_2_efficace", "Classe 2 efficace", "Âme de classe 3 d'un I soudé ou d'un PRS, flexion d'axe fort seule : W_pl,eff (§ 5.5.2(11))."),
                ("elastique", "Élastique", "Âme de classe 3 : W_el, toute la section en élastique."))
MCR_RETENU = (("annexe_E", "Formule", "M_cr par la formule de l'annexe E ANB, tronçon par tronçon ; le solveur est affiché."),
              ("solveur", "Solveur", "M_cr par le solveur du module ; la formule est affichée."))
EXPRESSIONS = (("6.10", "6.10"), ("6.10ab", "6.10a/b", "(6.10a) et (6.10b) : l'enveloppe des deux."))
VARIANTES_DEV = (("classeurs", "Classeurs", "Les classeurs « Déversement » du bureau (NBN B 51-002, tableaux 11 et 12)."),
                 ("fascicule", "Fascicule p. 19", "La semelle comprimée du fascicule (doc p. 19) : variante."))
INTRODUCTIONS = (("auto", "auto", "Le type que le module déduit de la géométrie."),
                 ("a", "a", "(a) Sur une semelle, reprise par le cisaillement de l'âme."),
                 ("b", "b", "(b) Transmise d'une semelle à l'autre."),
                 ("c", "c", "(c) Près d'une extrémité non raidie."))
CONTACTS = (("oui", "oui"), ("non", "non"), ("inconnu", "inconnu"))
AIDE_CONTACT = "Type (c) : le dispositif de chargement garde-t-il s_s en suivant la rotation d'extrémité ? (8.2)"
TORSIONS = (("fourche", "Fourche"), ("encastre", "Encastré"))
DETAIL_APPUIS = (("identiques", "Identiques"), ("un_par_un", "Un par un", "s_s, c et la torsion appui par appui."))
N_MODES = (("constant", "constant"), ("troncons", "par tronçon"))
COURBES_AXE = (("", "norme"), ("a", "a"), ("b", "b"), ("c", "c"))
AIDE_COURBE = "Courbe du fascicule (doc p. 4) : le tableau n'en donne pas pour ce profil."
AIDE_U = "13.3 : 1 articulé ; 0,80 encastré–articulé ; 0,65 bi-encastré ; console : déjà 2,1·L proposé (u = 1)."
# l'origine d'une longueur de flambement proposée (moteur._longueurs_rendues, « origine ») : la fin de l'aide du champ vide
ORIGINES = {"portee": "la portée", "troncon": "le plus long tronçon entre sections tenues", "console": "2,1 × la console (13.3)",
            "debord": "la barre entière (débord)"}
AIDE_REGULIERS = ("« 8 » : 8 maintiens complets, réguliers entre les bouts de la barre ; « 8 2 20 » : entre 2 et 20 m. "
                  "Ceux qui y étaient sont remplacés.")
AIDE_NEGLIGE = "Négligé dans M_cr (variante belge du déversement, 7.7) ; il reste compté au flambement."
AIDE_DEBORDS = "Porte-à-faux au-delà du premier et du dernier appui (m). Les abscisses partent du bout gauche de la barre."
AIDE_COURTE_DUREE = "De courte durée (moins d'un mois) : règle belge de courte durée avec la neige et le vent (tableau A1.1 ANB, note 3)."
AIDE_REGLE_COURTE = "ψ₀ = 0,3 entre la neige et le vent (tableau A1.1 ANB, note 3, mise en œuvre restreinte du module)."
AIDE_CD = "Coefficient de moment du déversement des classeurs (1 à 3)."
AIDE_CLASSE_D = "Le tableau 12 ne donne pas de classe pour ce profil : la classe est à fournir."
C_MODES = (("unite", "C = 1", "Les classeurs du bureau."), ("norme", "Selon la norme", "0,85 sous des charges ; selon ψ sous des moments d'extrémité."))
EFFORTS = (("N", "N", "kN", "Effort normal ; positif en compression."), ("My", "M_y", "kN·m", "Moment autour de l'axe fort."),
           ("Vz", "V_z", "kN", "Effort tranchant parallèle à l'âme."), ("Mz", "M_z", "kN·m", "Moment autour de l'axe faible."),
           ("Vy", "V_y", "kN", "Effort tranchant parallèle aux semelles."))
# le niveau (couleur) d'un statut sans taux ; « hors domaine » (brief § 9) : le niveau « v » existant (à vérifier), avec
# son mot — un taux calculé hors du domaine de la méthode n'est jamais peint « conforme » ; « bloqué » (lot C) : une
# donnée à saisir, « v » aussi (il y a quelque chose à faire)
NIVEAU_STATUT = {RE.OK: "p", RE.NOK: "r", RE.BLOQUE: "v", RE.NON_COUVERT: "c", RE.HORS_DOMAINE: "v"}
MOT_STATUT = {RE.OK: "Conforme", RE.NOK: "Non conforme", RE.BLOQUE: "Bloqué", RE.NON_COUVERT: "Non couvert", RE.HORS_DOMAINE: "Hors domaine"}
CLASSE_BLOC = {"p": "ok", "f": "warn", "r": "nok", "c": "inc", "v": "inc"}
# les statuts sous lesquels un TAUX n'est jamais peint « conforme » : il est montré pour information (lot C, C0b)
STATUTS_A_VERIFIER = (RE.BLOQUE, RE.NON_COUVERT, RE.HORS_DOMAINE)
# (lot C, C4) le contrôle HÉRITÉ du classeur (H-EQ, brief § 6.1 : H11 et H12) ne décide jamais seul : il n'est jamais en
# tête du verdict de la méthode ELS quand une autre vérification a un taux jugé (profil_resultat.tete)
HERITES = ("H-EQ",)
# (lot C, C4) le niveau d'une charge sur le dessin : un SYMBOLE lisible en noir et blanc, et sa légende en mots
SYMBOLE_NIVEAU = {"haut": "▲", "centre": "●", "bas": "▼"}
NOM_NIVEAU = {"haut": "semelle haute", "centre": "centre de cisaillement", "bas": "semelle basse"}
# (lot C, C4) la page « Écarts avec les classeurs du bureau » (brief § 6.7) : la ligne du bloc ELS qui l'ouvre, sa fenêtre
FENETRE_ECARTS = "ecarts"
TITRE_ECARTS = "écarts avec les classeurs du bureau"


def mot_corrections(n_: int) -> str:
    """« 1 correction des classeurs », « 3 corrections des classeurs » (du texte)."""
    return f"{n_} correction{'s' if n_ > 1 else ''} des classeurs"

# ---- (lot C, C5) CE QUE LE MODULE NE VÉRIFIE PAS (docs/acier/BARRES_ACIER.md : le périmètre de la version 1) — une source,
# lue par la note PDF (« Ce que la note ne vérifie pas ») et par « Tout le calcul » de l'écran. La torsion (brief § 17.5,
# lot 8) : la note la dit dans les hypothèses communes (« Torsion non couverte »), les autres limites à la fin.
TORSION = "charges supposées dans le plan de l'âme, par le centre de cisaillement"
HYPOTHESE_TORSION = f"Torsion non couverte (lot 8) : {TORSION}."
LIMITES = (f"la torsion ({TORSION})",
           "les assemblages, les appuis et les soudures des raidisseurs (EN 1993-1-8)",
           # (B4 : le voilement par cisaillement se calcule sans raidisseur intermédiaire, montant d'extrémité non rigide,
           # sans la contribution des semelles — le côté sûr)
           "les raidisseurs intermédiaires d'une âme élancée (calcul sans eux, côté sûr)",
           "la flexion d'axe faible d'une barre chargée (M_z : en efforts imposés seulement)",
           "un appui intermédiaire encastré, une continue à rive encastrée (modèles fixes seulement)",
           "les tassements d'appui", "le feu et la fatigue")
# un PRS ajoute les siennes (brief § 18.7)
LIMITES_PRS = ("les soudures âme-semelles du PRS", "la qualité d'acier des plats épais (EN 1993-1-10)")


def limites(prs: bool = False) -> tuple:
    """Ce que le module ne vérifie pas pour cette barre : ``LIMITES``, et ``LIMITES_PRS`` pour un PRS."""
    return LIMITES + (LIMITES_PRS if prs else ())

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
    qui manque (« décision impossible : s_s à saisir » — brief § 8.3, point 5)."""
    if L["decision"] == "raidisseur":
        return L["raidisseur"]["designation"]
    if L["decision"] == "impossible":
        # seul le maintien manque : la raison sans son taux (« ligne »), que la pastille de la ligne donne déjà
        r = L.get("ligne") or L.get("raison") or "Décision impossible."
        return r[:1].lower() + r[1:].rstrip(".")
    return f"sans raidisseur ({L['borne']})" if L.get("borne") else "sans raidisseur"


def _sans_prefixe(t: str) -> str:
    """Une raison de décision impossible placée APRÈS son lieu : sans « Décision impossible : » (ame.DECISION_IMPOSSIBLE), en minuscule."""
    t = t[len(AMEB.DECISION_IMPOSSIBLE):] if t.startswith(AMEB.DECISION_IMPOSSIBLE) else t
    return t[:1].lower() + t[1:]


def resume_ame(a: Dict) -> str:
    """Ce que le bloc de l'âme décide, en une phrase (du texte) : « Pas de raidisseur nécessaire » ; « Raidisseurs :
    2 plats 70 × 12 mm (appui A, appui B) » ; « Décision impossible : s_s à saisir (appui A, appui B) » ; « Âme élancée
    vérifiée au cisaillement » ; vide quand rien n'appuie sur l'âme. La même à l'écran et dans la note."""
    r = raidis(a)
    imp: Dict[str, list] = {}
    for L in a.get("lieux") or []:
        if L["decision"] == "impossible":
            imp.setdefault((L.get("raison") or "Décision impossible.").rstrip("."), []).append(L["nom"][:1].lower() + L["nom"][1:])
    # une raison qui finit par une parenthèse (« … 6.1(1)) », « … (À VÉRIFIER) ») : le lieu en tête, jamais « …) (charge à 3,00 m) » —
    # et sans « décision impossible : » après lui (deux « : » de suite ; intégration du lot B) : « Charge à 0,50 m : s_s à saisir ; … »
    sans = " ; ".join(f"{', '.join(noms)} : {_sans_prefixe(t)}" if t.endswith(")") else f"{t} ({', '.join(noms)})"
                      for t, noms in imp.items())
    if r:
        par = {}
        for L in r:
            par.setdefault(L["raidisseur"]["designation"], []).append(L["nom"][:1].lower() + L["nom"][1:])
        return (("Raidisseur : " if len(r) == 1 else "Raidisseurs : ") + " ; ".join(f"{d} ({', '.join(noms)})" for d, noms in par.items())
                + (" ; " + sans[:1].lower() + sans[1:] if sans else ""))
    if sans:
        return sans[:1].upper() + sans[1:]
    if a.get("lieux"):
        return "Pas de raidisseur nécessaire"
    return "Âme élancée vérifiée au cisaillement" if a.get("verifications") else ""


def couleur_taux(statut: Optional[str], taux: Optional[float]) -> str:
    """LA couleur d'un TAUX sous un STATUT (lot C, C0b : une seule fonction, que lisent la pastille du verdict, la ligne
    « Profilé », la comparaison de l'écran et celle de la note, et chaque ligne de vérification) : « r » pour un statut
    « nok » ; « v » (à vérifier) pour « bloqué », « non couvert » et « hors domaine » — le taux est pour information,
    jamais vert ; sinon la couleur du taux (« c » sans taux). Mesuré avant (766659c, toujours à l'intégration du lot B) :
    une console de 3 m sous N = 50 kN, Eurocode 3 « non couvert » (E-661), portait une pastille VERTE « 57 % »."""
    if statut == RE.NOK:
        return "r"
    if statut in STATUTS_A_VERIFIER:
        return "v"
    return C.niveau(taux)


def niveau(v: Dict) -> str:
    """La couleur d'une vérification : celle de son taux sous son statut (``couleur_taux``) ; sans taux, celle de son
    statut."""
    if v.get("taux") is not None:
        return couleur_taux(v.get("statut"), v["taux"])
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
    if not v["unite"]:                  # (lot C, C4) une somme sans unité, écrite comme la norme : « η₂ + 0,8·η₁ = 0,513 ≤ 1,4 »
        return f"{sub(g)} = {C.fr(v['valeur'], 3)} {'≤' if v['valeur'] <= v['limite'] else '&gt;'} {n(v['limite'], 3)}"
    return f"{sub(g)} {r1(v['valeur'])} {signe} {sub(d)} {r1(v['limite'])}{NB}{C.esc(v['unite'])}"
