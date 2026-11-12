# -*- coding: utf-8 -*-
"""Les CAS de ``tests/test_reference_refonte.py``, module par module : des suites de requêtes du routeur
(``api_web/routeur.traiter``), rejouées sur l'API JSON. Une requête sans « etat » reprend l'état rendu par la
précédente ; la première est en général « nouveau » (la note neuve) ou « calculer » sur une note d'exemple.

Par module : au moins trois cas de RÉFÉRENCE (note neuve, exemple publié ou réaliste, cas chargé près de 100 %) et au
moins trois cas LIMITES (valeur minimale ou nulle admise, valeur hors domaine refusée avec message, élément ajouté ou
supprimé, géométrie non conforme, champ vide « à compléter », note abîmée…). ``attendu`` dit ce que le cas doit
montrer ; le test le contrôle, au gel comme à chaque comparaison.

D'où viennent les cas (inventaire du 07/11/2026) : les exemples publiés des benchmarks (assemblages : BM1… de chaque
configuration, VAL-A/B et VAL-M1… du pied de poteau, chargés par « exemple » comme l'onglet Benchmark de Streamlit),
les notes d'exemple des modules (``api_web/age._etat_releve``/``_etat_meteo``, ``api_web/plancher.exemple``,
``api_web/taux._RICHE`` et ``_RECUE``), les requêtes propres du parcours (``PARCOURS`` des modules : refus, relevés, fichiers)
et les cas limites des suites ``tests/test_api_web_<clé>.py``. Les charges « près de 100 % » ont été réglées par
dichotomie sur le moteur du jour (07/11/2026) puis arrondies.
"""
import copy
import os
import sys

RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RACINE not in sys.path:
    sys.path.insert(0, RACINE)

from api_web import age as AGE  # noqa: E402
from api_web import plancher as PL  # noqa: E402
from api_web import taux as TX  # noqa: E402

CAS = {}
REF, LIM = "référence", "limite"


def cas(cle, nom, sorte, pas, attendu=None, selecteurs=False, fichiers=False):
    """Un cas : ``sorte`` REF ou LIM ; ``pas`` : les requêtes ; ``attendu`` : {"message": True} (la dernière requête
    est refusée avec un message), {"erreur": True} (le routeur rend une erreur), {"taux": [min, max]},
    {"niveau": "pf"}, {"elements": n}, {"avis": "texte"}, {"texte": "texte"}, {"sans_message": True} ;
    ``selecteurs`` / ``fichiers`` : photographier aussi chaque sélecteur annoncé / chaque fichier à télécharger."""
    if any(c["nom"] == nom for c in CAS.get(cle, [])):
        raise ValueError(f"{cle} : deux cas nommés « {nom} »")
    CAS.setdefault(cle, []).append({"nom": nom, "sorte": sorte, "pas": copy.deepcopy(pas), "attendu": attendu or {},
                                    "selecteurs": selecteurs, "fichiers": fichiers})


def neuf():
    return {"action": "nouveau"}


def mod(chemin, valeur, **k):
    return dict({"action": "modifier", "chemin": chemin.split("/") if isinstance(chemin, str) else list(chemin), "valeur": valeur}, **k)


def act(action, **k):
    return dict({"action": action}, **k)


def calc(etat):
    return {"action": "calculer", "etat": copy.deepcopy(etat)}


ABIMEE = {"action": "calculer", "etat": {"version": 99, "elements": "abîmé", "cas": None, "poutres": None}}


# ====================================================================== poutre (béton armé)
cas("poutre", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
_P_REAL = [neuf(), mod("b", "30"), mod("h", "60"), mod("beton", "C25/30"), mod("sections/0/inf/0/n", "3"),
           mod("sections/0/inf/0/d", "20"), mod("sections/0/M_inf", "120"), mod("sections/0/V", "40"), mod("sections/0/pas", "20")]
cas("poutre", "poutre 30 × 60 réaliste (3Ø20, M = 120 kN·m, V = 40 kN, étriers à 20 cm)", REF, _P_REAL, {"niveau": "p"}, selecteurs=True)
cas("poutre", "chargée près de 100 % (M = 142 kN·m, V = 60 kN)", REF, _P_REAL[:6] + [mod("sections/0/V", "60"), mod("sections/0/M_inf", "142")],
    {"taux": [95, 100]})
cas("poutre", "deux sections (sur appui M_sup = 90 kN·m)", REF,
    _P_REAL + [mod("sections/+", None), mod("sections/1/M_sup", "90"), mod("sections/1/sup/0/n", "3")])
cas("poutre", "moments nuls (valeur minimale admise)", LIM, [neuf(), mod("sections/0/M_inf", "0"), mod("sections/0/V", "0")],
    {"sans_message": True})
cas("poutre", "largeur négative refusée", LIM, [neuf(), mod("b", "-20")], {"message": True})
cas("poutre", "nombre illisible refusé", LIM, [neuf(), mod("h", "quarante")], {"message": True})
cas("poutre", "dernier lit non retiré", LIM, [neuf(), mod("sections/0/inf/0/-", None)], {"message": True})
cas("poutre", "surchargée (M = 400 kN·m)", LIM, _P_REAL[:6] + [mod("sections/0/M_inf", "400")], {"niveau": "r"})
cas("poutre", "poutres ajoutée, copiée, la première supprimée", LIM,
    [neuf(), mod("sections/0/M_inf", "40"), act("ajouter"), mod("b", "25"), act("dupliquer"), act("supprimer", index=0)],
    {"elements": 2})
cas("poutre", "8Ø32 dans 20 cm : barres qui ne tiennent pas (rouge sous 100 %)", LIM,
    [neuf(), mod("sections/0/inf/0/n", "8"), mod("sections/0/inf/0/d", "32")], {"niveau": "r"})
cas("poutre", "note abîmée", LIM, [ABIMEE])
# l'option « Taux d'armature » (10/11/2026, docs/site/LOT_Q_QUATRE_MODULES.md § 6.1) : éteinte, les cas d'avant ne
# gagnent que sa case ; allumée, la ligne de chaque section, le taux de la poutre et leurs fenêtres de calcul
cas("poutre", "taux d'armature : deux sections de 4 m et 2 m (option allumée, 30 × 60)", REF,
    _P_REAL + [mod("sections/+", None), mod("sections/1/M_sup", "90"), mod("sections/1/sup/0/n", "3"), mod("taux/actif", True),
               mod("sections/0/longueur", "4"), mod("sections/1/longueur", "2")], {"sans_message": True})
cas("poutre", "taux d'armature : majoration de 80 % refusée, puis une section sans longueur", LIM,
    [neuf(), mod("taux/actif", True), mod("taux/retour", "12"), mod("taux/majoration", "80")], {"message": True})

# ====================================================================== dalle
cas("dalle", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
_D_REAL = [neuf(), mod("h", "22"), mod("sections/0/My_inf", "30"), mod("sections/0/Mx_inf", "15"), mod("sections/0/V", "60")]
cas("dalle", "dalle 22 cm réaliste (M_y = 30, M_x = 15 kN·m/m)", REF, _D_REAL, {"niveau": "p"})
cas("dalle", "chargée près de 100 % (M_y = 38,5 kN·m/m)", REF, [neuf(), mod("sections/0/My_inf", "38,5")], {"taux": [95, 100]})
cas("dalle", "barres Ø12/150 en nappe inférieure Y", REF, _D_REAL + [act("choisir", role="inf_y-0", nom="Ø12/150")])
cas("dalle", "moment négatif refusé (PARCOURS)", LIM, [neuf(), mod("sections/0/Mx_inf", "-20")], {"message": True})
cas("dalle", "espacement de 600 mm refusé", LIM, [neuf(), mod("sections/0/faces/inf_x/couches/0/type", "barres"),
                                                        mod("sections/0/faces/inf_x/couches/0/esp", "600")], {"message": True})
cas("dalle", "distance vidée : automatique", LIM, [neuf(), mod("sections/0/faces/sup_x/couches/0/dist", "")], {"sans_message": True})
cas("dalle", "section ajoutée, dalle ajoutée", LIM, [neuf(), mod("sections/+", None), mod("sections/1/My_inf", "30"), act("ajouter")],
    {"elements": 2})
cas("dalle", "fichier Streamlit illisible refusé", LIM, [neuf(), mod("fichier_streamlit", {"nom": "abime.json", "texte": "{pas du json"})],
    {"message": True})

# ====================================================================== prédalle
cas("predalle", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
_PR_REAL = [neuf(), mod("h", "24"), mod("sections/0/My_inf", "40"), mod("sections/0/Mx_inf", "15"), mod("sections/0/V", "50")]
cas("predalle", "prédalle 24 cm réaliste", REF, _PR_REAL, {"niveau": "p"})
cas("predalle", "chargée près de 100 % (M_y = 43,5 kN·m/m)", REF, [neuf(), mod("sections/0/My_inf", "43,5")], {"taux": [95, 100]})
cas("predalle", "barres Ø12/150 en nappe inférieure Y", REF, _PR_REAL + [act("choisir", role="inf_y-0", nom="Ø12/150")])
cas("predalle", "prédalle plus épaisse que la dalle refusée", LIM, [neuf(), mod("h_pre", "21")], {"message": True})
cas("predalle", "dalle de 6 cm refusée", LIM, [neuf(), mod("h", "6")], {"message": True})
cas("predalle", "distance vidée : automatique", LIM, [neuf(), mod("sections/0/faces/sup_x/couches/0/dist", "")], {"sans_message": True})
cas("predalle", "nappe de la prédalle déplacée (dist = 8 cm)", LIM, [neuf(), mod("sections/0/faces/inf_x/couches/0/dist", "8")])
cas("predalle", "prédalle ajoutée", LIM, [neuf(), act("ajouter"), mod("sections/0/My_inf", "30")], {"elements": 2})

# ====================================================================== garde-corps
cas("garde_corps", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("garde_corps", "lisses, en L, catégorie C1", REF, [neuf(), mod("type", "lisses"), mod("geometrie/forme", "L"), mod("charges/categorie", "C1")])
cas("garde_corps", "chargé près de 100 % (catégorie libre, q_h = 0,78 kN/m)", REF, [neuf(), mod("charges/categorie", "perso"), mod("charges/qh", "0,78")], {"taux": [95, 100]})
cas("garde_corps", "Eurocode, fixation latérale", REF, [neuf(), mod("methode/mode", "eurocode"), mod("fixation/type", "laterale")])
cas("garde_corps", "hauteur négative : non calculable", LIM, [neuf(), mod("geometrie/H", "-1100")], {"niveau": "r"})
cas("garde_corps", "tronçon de longueur nulle : calcul impossible", LIM, [neuf(), mod("geometrie/troncons/0/L", "0")], {"niveau": "x"})
cas("garde_corps", "hauteur illisible refusée", LIM, [neuf(), mod("geometrie/H", "abc")], {"message": True})
cas("garde_corps", "charge horizontale nulle (catégorie libre)", LIM, [neuf(), mod("charges/categorie", "perso"), mod("charges/qh", "0")])
cas("garde_corps", "surchargé (q_h = 3 kN/m)", LIM, [neuf(), mod("charges/categorie", "perso"), mod("charges/qh", "3")], {"niveau": "r"})
cas("garde_corps", "garde-corps ajouté", LIM, [neuf(), act("ajouter"), mod("geometrie/H", "1000")], {"elements": 2})

# ====================================================================== cornière (refonte du 10/11/2026 : la méthode du classeur)
# (les cas du modèle lame — V, e, charge… — n'existent plus : les photos sont regelées, CORNIERE_DECISIONS § 7)
from api_web import corniere as COR  # noqa: E402

cas("corniere", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("corniere", "cas de contrôle du classeur (R = 42,76 kN, R_ELU = 68,9, sans jeu : 88,86 % en section 3)", REF,
    [neuf(), mod("R", "42,76"), mod("R_ELU", "68,9"), mod("dx", "0"), mod("dy", "0")], {"taux": [88, 89]})
cas("corniere", "mode équilibre", REF, [neuf(), mod("mode", "equilibre")])
cas("corniere", "configuration B (aile horizontale en haut : section nette, pression sur le support)", REF, [neuf(), mod("configuration", "B")],
    {"niveau": "r"})
cas("corniere", "ailes inégales L 150×100×12, 150 à l'horizontale (Y = 65)", REF, [neuf(), mod("Y", "65"), mod("profil", "L 150×100×12")])
# (R = 36 kN avant la révision § 9 : les jeux par défaut 20/10 donnaient 98,5 % ; avec 10/5 il faut 40 kN pour 96,2 %)
cas("corniere", "chargée près de 100 % (R = 40 kN)", REF, [neuf(), mod("R", "40")], {"taux": [95, 100]})
cas("corniere", "charge dans l'angle refusée (X = 10)", LIM, [neuf(), mod("X", "10")], {"message": True})
cas("corniere", "cornière trop petite pour la charge refusée (L 45×45×4,5)", LIM, [neuf(), mod("profil", "L 45×45×4,5")], {"message": True})
cas("corniere", "largeur d'appui a = 100 mm (L_eff)", LIM, [neuf(), mod("a", "100")])
cas("corniere", "saisie libre (les dimensions de la cornière affichée)", LIM, [neuf(), mod("standard", "saisie"), mod("r", "0")])
cas("corniere", "ancrage bas (Y = 40, sans jeu) : la garde rougit", LIM, [neuf(), mod("dx", "0"), mod("dy", "0"), mod("d", "10"), mod("Y", "40")],
    {"texte": "Équilibre P1"})
cas("corniere", "note v1 (modèle lame) rouverte : l'avis d'ouverture", LIM, [calc(COR.NOTE_V1)], {"avis": "modèle lame"})
cas("corniere", "cornière ajoutée, la première supprimée", LIM, [neuf(), act("ajouter"), mod("R", "20"), act("ajouter"), act("supprimer", index=0)],
    {"elements": 2})
# la révision § 9 (10/11 soir) : l'état limite ELU (R saisie à l'ELU, f_y/γ_M0, coef et R_ELU sans effet), le support poutre
# (ses deux étendues, les distances aux bords en information), l'entraxe et le type d'ancrage, un entraxe refusé
cas("corniere", "ELU : R = 60 kN saisie à l'ELU (f_y/γ_M0 = 355, coef et R_ELU sans effet)", REF, [neuf(), mod("etat_limite", "ELU"), mod("R", "60")],
    {"taux": [95, 100]})
cas("corniere", "A sur poutre, 160 mm au-dessus et 120 au-dessous (bords de la poutre en information)", REF,
    [neuf(), mod("support", "poutre"), mod("support_haut", "160"), mod("support_bas", "120")])
cas("corniere", "B sur poutre (150/150 : le contact du bout libre sur la poutre)", LIM, [neuf(), mod("configuration", "B"), mod("support", "poutre")], {"niveau": "r"})
cas("corniere", "entraxe 80 mm, boulons traversants, deux rondelles, profondeur 200", LIM,
    [neuf(), mod("entraxe", "80"), mod("type_ancrage", "traversant"), mod("rondelles", "2"), mod("profondeur", "200")])
cas("corniere", "entraxe 30 mm refusé (s < 2,2·d0, EN 1993-1-8 Tableau 3.3)", LIM, [neuf(), mod("entraxe", "30")], {"message": True})

# ====================================================================== bois (refait le 10/11/2026 sur l'Eurocode 5)
# (les cas de la copie de l'écran Streamlit — flèches 1 000 fois trop grandes, une seule combinaison, « duree » choisie à
# la main — n'existent plus : les photos sont regelées, docs/site/LOT_Q_QUATRE_MODULES.md § 1 ; les taux annoncés sont
# ceux du moteur du 10/11/2026, recalculés à la main dans tests/test_poutre_bois_moteur.py)
from api_web import bois as BO  # noqa: E402

cas("bois", "note neuve", REF, [neuf()], {"taux": [56, 56], "niveau": "p"}, selecteurs=True, fichiers=True)
cas("bois", "gîte de plancher 63 × 175 C24 à 40 cm d'entraxe, 3,60 m, 1,2 + 2,0 kN/m² (habitation)", REF,
    [neuf(), act("choisir", role="section", nom="63 × 175"), mod("L", "3,6"), mod("charges/mode", "surfacique"), mod("charges/g", "1,2"), mod("charges/q", "2")],
    {"taux": [90, 92], "niveau": "p", "sans_message": True})
cas("bois", "chargée près de 100 % (Q = 2,06 kN/m : flèche finale nette)", REF, [neuf(), mod("charges/Q", "2,06")], {"taux": [95, 100], "niveau": "f"})
cas("bois", "G seule gouverne toute la note (2,5 m, G = 7, Q = 1 kN/m : flexion, k_mod permanent)", REF,
    [neuf(), mod("L", "2,5"), mod("charges/G", "7"), mod("charges/Q", "1")], {"taux": [106, 107], "niveau": "r"})
cas("bois", "G seule gouverne l'ELU, la flèche finale la note (G = 3, Q = 0,5 kN/m)", REF, [neuf(), mod("charges/G", "3"), mod("charges/Q", "0,5")],
    {"taux": [153, 154], "niveau": "r"})
cas("bois", "déversement : 45 × 245 sur 5 m, maintenue aux appuis seulement (k_crit = 0,398)", REF,
    [neuf(), act("choisir", role="section", nom="45 × 245"), mod("L", "5"), mod("maintien/type", "appuis")], {"taux": [244, 245], "niveau": "r"})
cas("bois", "entretoises à 1,25 m (45 × 245 sur 5 m)", REF,
    [neuf(), act("choisir", role="section", nom="45 × 245"), mod("L", "5"), mod("maintien/type", "entretoises"), mod("maintien/entraxe", "1,25")],
    {"taux": [137, 138]})
cas("bois", "lamellé-collé GL24h 140 × 560 sur 8 m (G = 3, Q = 4 kN/m, appui de 160 mm)", REF,
    [neuf(), mod("classe", "GL24h"), act("choisir", role="section", nom="140 × 560"), mod("L", "8"), mod("charges/G", "3"), mod("charges/Q", "4"),
     mod("appui/a", "160")], {"niveau": "p", "sans_message": True}, selecteurs=True)
cas("bois", "l'appui gouverne (100 × 300 sur 3 m, 8 + 12 kN/m, a = 40 mm)", REF,
    [neuf(), act("choisir", role="section", nom="100 × 300"), mod("L", "3"), mod("charges/G", "8"), mod("charges/Q", "12"), mod("appui/a", "40")],
    {"taux": [269, 270], "niveau": "r"})
cas("bois", "chevron de toiture 63 × 175 à 60 cm, 3,50 m, classe de service 2, neige (court terme)", REF,
    [neuf(), act("choisir", role="section", nom="63 × 175"), mod("L", "3,5"), mod("service", "2"), mod("charges/mode", "surfacique"), mod("charges/g", "0,9"),
     mod("charges/q", "0,5"), mod("charges/entraxe", "0,6"), mod("charges/nature", "neige")], {"niveau": "p"})
cas("bois", "classe de service 3 (k_mod 0,5 et 0,65, k_def 2 : la flèche finale passe de 56 à 92 %)", REF, [neuf(), mod("service", "3")],
    {"taux": [91, 92], "niveau": "p"})
cas("bois", "effet système (45 × 220 à 60 cm d'entraxe)", REF,
    [neuf(), act("choisir", role="section", nom="45 × 220"), mod("charges/mode", "surfacique"), mod("charges/entraxe", "0,6"), mod("systeme", True)])
cas("bois", "nature « autre » : instantané, ψ₂ = 0,5", REF, [neuf(), mod("charges/nature", "autre"), mod("charges/duree", "instantane"), mod("charges/psi2", "0,5")],
    {"sans_message": True})
cas("bois", "lamellé cambré GL28c 160 × 640 sur 10 m (G = 4, Q = 5 kN/m, appui 200 mm, contre-flèche 20 mm)", REF,
    [neuf(), mod("classe", "GL28c"), act("choisir", role="section", nom="160 × 640"), mod("L", "10"), mod("charges/G", "4"), mod("charges/Q", "5"),
     mod("appui/a", "200"), mod("contre_fleche", "20")], {"sans_message": True})
cas("bois", "section sur mesure 62,5 × 187,5", REF, [neuf(), mod("standard", "mesure"), mod("b", "62,5"), mod("h", "187,5")], {"sans_message": True})
cas("bois", "portée de 0,2 m refusée (PARCOURS)", LIM, [neuf(), mod("L", "0,2")], {"message": True})
cas("bois", "charge variable négative refusée", LIM, [neuf(), mod("charges/Q", "-3")], {"message": True})
cas("bois", "appui de 20 mm (valeur minimale admise)", LIM, [neuf(), mod("appui/a", "20")], {"sans_message": True})
cas("bois", "appui de 19 mm refusé", LIM, [neuf(), mod("appui/a", "19")], {"message": True})
cas("bois", "ψ₂ = 0,7 refusé en habitation (fixé par la nature de Q)", LIM, [neuf(), mod("charges/psi2", "0,7")], {"message": True})
cas("bois", "entretoises à 9 m sur une portée de 4 m refusées", LIM, [neuf(), mod("maintien/type", "entretoises"), mod("maintien/entraxe", "9")],
    {"message": True})
cas("bois", "section d'un autre type refusée par le sélecteur (120 × 400 en bois massif)", LIM, [neuf(), act("choisir", role="section", nom="120 × 400")],
    {"erreur": True})
cas("bois", "sans charge variable (Q = 0 : la combinaison G + Q disparaît)", LIM, [neuf(), mod("charges/Q", "0")], {"sans_message": True, "niveau": "p"})
cas("bois", "aucune charge (G = Q = 0, poids propre non compté) : 0 %, et c'est dit", LIM,
    [neuf(), mod("charges/G", "0"), mod("charges/Q", "0"), mod("charges/poids_propre", False)], {"taux": [0, 0], "avis": "Aucune charge"})
cas("bois", "appuis rapprochés (100 × 300 sur 0,65 m) : k_c,90 ramené à 1, et c'est dit", LIM,
    [neuf(), act("choisir", role="section", nom="100 × 300"), mod("L", "0,65")], {"avis": "Appuis rapprochés"})
cas("bois", "compression d'appui retirée de la note : la ligne reste, « non vérifié »", LIM, [neuf(), mod("appui/verifier", False)], {"avis": "retirée de la"})
cas("bois", "très au-delà (12 m, 20 + 30 kN/m : plus de 999 %)", LIM, [neuf(), mod("L", "12"), mod("charges/G", "20"), mod("charges/Q", "30")],
    {"niveau": "r"})
cas("bois", "du massif au lamellé-collé : la section 75 × 225 gardée sur mesure, l'avis", LIM, [neuf(), mod("classe", "GL24h")], {"avis": "sur mesure"})
cas("bois", "note v1 (copie de l'écran Streamlit) rouverte : les noms gardés, l'avis d'ouverture", LIM, [calc(BO.NOTE_V1)],
    {"avis": "les données sont à ressaisir", "elements": 2})
cas("bois", "poutres ajoutée, copiée, la première supprimée", LIM,
    [neuf(), mod("charges/Q", "1,5"), act("ajouter"), mod("L", "3"), act("dupliquer"), act("supprimer", index=0)], {"elements": 2})
cas("bois", "note sans poutre : une note neuve", LIM, [ABIMEE], {"taux": [56, 56]})
cas("bois", "note abîmée (« poutres » n'est pas une liste)", LIM, [{"action": "calculer", "etat": {"version": 2, "poutres": "abîmé"}}], {"erreur": True})
cas("bois", "note d'une version plus récente refusée", LIM, [{"action": "calculer", "etat": {"version": 3, "poutres": [{"nom": "Demain"}]}}], {"erreur": True})

# ====================================================================== flambement
cas("flambement", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("flambement", "IPE 400, S355, maintiens z", REF, [neuf(), mod("profil", "IPE 400"), mod("nuance", "S355"), mod("axes/z/maintiens", "2")])
cas("flambement", "chargé près de 100 % (N = 780 kN)", REF, [neuf(), mod("N", "780")], {"taux": [95, 100]})
cas("flambement", "encastrée-libre, L 6 m", REF, [neuf(), mod("axes/y/appuis", "EL"), mod("axes/z/appuis", "EL"), mod("L", "6")])
cas("flambement", "effort négatif refusé (PARCOURS)", LIM, [neuf(), mod("N", "-5")], {"message": True})
cas("flambement", "effort nul (valeur minimale admise)", LIM, [neuf(), mod("N", "0")], {"sans_message": True})
cas("flambement", "longueur nulle refusée (PARCOURS)", LIM, [neuf(), mod("L", "0")], {"message": True})
cas("flambement", "γ_M1 = 0,9 refusé (PARCOURS)", LIM, [neuf(), mod("gamma_M1", "0,9")], {"message": True})
cas("flambement", "famille des cornières (PARCOURS)", LIM, [neuf(), mod("famille", "L")])
cas("flambement", "barre ajoutée", LIM, [neuf(), act("ajouter"), mod("N", "300")], {"elements": 2})

# ====================================================================== choix de profilé
cas("profil", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("profil", "IPE, M_y = 150, V_z = 100", REF, [neuf(), mod("familles/IPE", True), mod("familles/HEA", False), mod("My", "150"), mod("Vz", "100")])
cas("profil", "chargé près de 100 % (IPE 300 imposé, M_y = 130 kN·m)", REF,
    [neuf(), mod("auto", False), act("choisir", role="profil", nom="IPE 300"), mod("My", "130")], {"taux": [95, 100]})
cas("profil", "flexion, effort normal et déversement", REF,
    [neuf(), mod("My", "80"), mod("avec_N", True), mod("N", "200"), mod("flambement", True), mod("Lcr_y", "4"), mod("Lcr_z", "4"),
     mod("deversement", True), mod("L_LT", "4")])
cas("profil", "moment énorme 1e308 refusé (PARCOURS)", LIM, [neuf(), mod("My", "1e308")], {"message": True})
cas("profil", "longueur de flambement négative refusée", LIM, [neuf(), mod("Lcr_y", "-1")], {"message": True})
cas("profil", "aucun profilé ne tient (M_y = 5000)", LIM, [neuf(), mod("My", "5000")], {"niveau": "r", "avis": "Aucun profilé"})
cas("profil", "effort tranchant négatif (signe admis)", LIM, [neuf(), mod("Vz", "-50")], {"sans_message": True})
cas("profil", "élément ajouté", LIM, [neuf(), act("ajouter"), mod("My", "60")], {"elements": 2})

# ====================================================================== soudures
cas("soudures", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("soudures", "C1 en ELS", REF, [neuf(), mod("config", "C1"), mod("methode/mode", "els")])
cas("soudures", "chargé près de 100 % (V = 97 kN)", REF, [neuf(), mod("efforts/V", "97")], {"taux": [95, 100]})
cas("soudures", "C3 sur poteau HEA, pénétration partielle", REF, [neuf(), mod("params/poteau_famille", "HEA"), mod("params/jonction", "partielle")])
cas("soudures", "gorge nulle refusée", LIM, [neuf(), mod("params/a", "0")], {"message": True})
cas("soudures", "console de 10 mm : calcul impossible", LIM, [neuf(), mod("params/h", "10")], {"niveau": "x"})
cas("soudures", "effort nul", LIM, [neuf(), mod("efforts/V", "0")])
cas("soudures", "soudure discontinue (tronçons)", LIM, [neuf(), mod("params/troncons/on", True)])
cas("soudures", "cas ajouté, configuration C5", LIM, [neuf(), act("ajouter"), mod("config", "C5")], {"elements": 2})

# ====================================================================== charges sur plancher
# Décision de l'ingénieur du 11/11/2026 : un cas neuf part du SP 200-4X du catalogue Ergon (le hourdis « le plus banal ») ;
# la fiche générique « à renseigner » reste au menu, et c'est elle qui garde l'état « à compléter ».
cas("plancher", "note neuve (SP 200-4X du catalogue, poids propre seul)", REF, [neuf()], {"niveau": "p"}, selecteurs=True, fichiers=True)
cas("plancher", "fiche générique à renseigner (à compléter)", LIM,
    [neuf(), mod("plancher/hourdis/type_cat", "Hourdis générique 200 · à renseigner")], {"niveau": "c"})
cas("plancher", "exemple hourdis (engin, plateaux, filière)", REF, [calc(PL.exemple("hourdis"))], selecteurs=True)
cas("plancher", "exemple dalle (télescopique à roues)", REF, [calc(PL.exemple("dalle"))])
cas("plancher", "exemple prédalle sans chape durcie (refus P-7)", REF, [calc(PL.exemple("predalle"))])
cas("plancher", "hourdis : filière retirée (non conforme)", LIM,
    [calc(PL.exemple("hourdis")), mod("appuis/4/repartiteur", "aucun"), mod("filieres/0/-", "1")])
cas("plancher", "portée négative refusée", LIM, [neuf(), mod("plancher/L", "-7")], {"message": True})
cas("plancher", "charge ponctuelle ajoutée puis dupliquée", LIM, [neuf(), mod("charges/+", "ponctuel"), mod("charges/0/P", "30"),
                                                                    mod("charges/0/dupliquer", "1")])
cas("plancher", "hourdis en service, sens 2", LIM, [calc(PL.exemple("hourdis")), mod("plancher/hourdis/nature", "service"), mod("plancher/sens", "2")])
cas("plancher", "cas ajouté", LIM, [neuf(), act("ajouter")], {"elements": 2})

# ====================================================================== maturité du béton (âge) — refait le 10/11/2026 sur la norme
# (les cas de l'écran d'avant — âge choisi ①, seuil ②, classe pour un délai ③, comparateur ④ — n'existent plus : les photos
# sont regelées, docs/site/LOT_Q_QUATRE_MODULES.md § 2 ; les échéances annoncées sont celles du moteur du 10/11/2026,
# recalculées à la main dans tests/test_maturite_moteur.py : 70 % de f_ck du C30/37 « moyen » à 20 °C = J + 10,7)
cas("age", "note neuve (C30/37, béton moyen, 20 °C, décoffrage à 70 % de f_ck : J + 10,7)", REF, [neuf()], {"sans_message": True},
    selecteurs=True, fichiers=True)
cas("age", "ciment N au lieu du béton moyen (J + 6,5)", REF, [neuf(), mod("durcissement/choix", "ciment:N")], {"sans_message": True})
cas("age", "décoffrage prévu à J + 11 : près de 100 % (la variante prudente ne tient pas)", REF, [neuf(), mod("objectifs/0/age", "11")],
    {"taux": [98, 99], "niveau": "f"})
cas("age", "relevé collé (Namur), coulé le 02/11/2026 à 16 h", REF, [calc(AGE._etat_releve())])
cas("age", "météo importée (Bruxelles), décoffrage à J + 3", REF, [calc(AGE._etat_meteo(True))], selecteurs=True)
cas("age", "trois objectifs datés et modèles (hors gel, reprise), coulé le 05/11/2026 à 7 h", REF,
    [neuf(), mod("date", "05/11/2026"), mod("heure", "7"), mod("temperature/T", "8"), mod("objectifs/+", "hors_gel"), mod("objectifs/+", "reprise"),
     mod("objectifs/0/age", "14"), mod("objectifs/2/age", "4")], selecteurs=True)
cas("age", "essai sur cube (21,6 MPa à J + 7) et courbe calée, loi EC2", REF,
    [neuf(), mod("loi/mode", "ec2"), mod("essai/actif", True), mod("essai/mpa", "21,6"), mod("essai/caler", True)])
cas("age", "relevé illisible refusé (PARCOURS)", LIM, [calc(AGE._etat_releve()), mod("temperature/releve", "12\nabc\n13")], {"message": True})
cas("age", "fichier illisible refusé (PARCOURS)", LIM, [neuf(), mod("temperature/fichier", {"nom": "faux.csv", "texte": "douze"})], {"message": True})
cas("age", "température de −20 °C (valeur minimale admise : la loi belge compte encore)", LIM, [neuf(), mod("temperature/T", "-20")])
cas("age", "température de −21 °C refusée", LIM, [neuf(), mod("temperature/T", "-21")], {"message": True})
cas("age", "cible au-dessus de la classe (40 MPa sur un C30/37) : non atteignable", LIM,
    [neuf(), mod("objectifs/0/mode", "mpa"), mod("objectifs/0/mpa", "40")])
cas("age", "le dernier objectif ne se retire pas", LIM, [neuf(), mod("objectifs/0/-", "1")], {"message": True})
cas("age", "météo : lieu introuvable (PARCOURS)", LIM, [calc(AGE._etat_meteo()), mod("temperature/meteo", AGE._meteo_exemple(trouve=False))],
    {"message": True})
cas("age", "note du module d'avant relue (noms gardés)", LIM, [calc(AGE.NOTE_V1)], {"elements": 2})
cas("age", "coulage ajouté", LIM, [neuf(), act("ajouter"), mod("classe", "C40/50")], {"elements": 2})

# ====================================================================== tableau des armatures
cas("armatures", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("armatures", "nappe, A_s,req = 754 mm²/m (Ø12/15 = 753,98 : juste insuffisant)", REF, [neuf(), mod("mode", "nappe"), mod("As_req", "754")])
cas("armatures", "barres, A_s,req = 440 mm² (4Ø12 tient à 97 %)", REF, [neuf(), mod("As_req", "440")])
cas("armatures", "barres, Ø20 imposé, 2 400 mm²", REF, [neuf(), mod("d", "20"), mod("As_req", "2400"), mod("n_max", "8")])
cas("armatures", "A_s,req nul", LIM, [neuf(), mod("As_req", "0")])
cas("armatures", "A_s,req négatif refusé", LIM, [neuf(), mod("As_req", "-5")], {"message": True})
cas("armatures", "zéro barre refusé", LIM, [neuf(), mod("n", "0")], {"message": True})
cas("armatures", "Ø min plus grand que Ø max", LIM, [neuf(), mod("d_min", "32"), mod("d_max", "10")])
cas("armatures", "repère ajouté", LIM, [neuf(), act("ajouter"), mod("As_req", "900")], {"elements": 2})

# ====================================================================== catalogue des profilés
cas("catalogue", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("catalogue", "recherche « IPE 3 », fiche IPE 300", REF, [neuf(), mod("recherche", "IPE 3"), act("choisir", role="fiche", nom="IPE 300")])
cas("catalogue", "filtre I_y ≥ 20 000 cm⁴, tri par masse", REF, [neuf(), mod("filtres/0/cle", "Iy"), mod("filtres/0/min", "20000"), mod("tri", "masse")])
cas("catalogue", "profilé reconstitué PRS 600 × 300", REF, [neuf(), mod("perso/dims/PRS/h", "600"), mod("perso/dims/PRS/b", "300"), mod("perso/+", "1")])
cas("catalogue", "filtre illisible refusé", LIM, [neuf(), mod("filtres/0/cle", "h"), mod("filtres/0/min", "abc")], {"message": True})
cas("catalogue", "PRS d'âme nulle refusé", LIM, [neuf(), mod("perso/dims/PRS/tw", "0")], {"message": True})
cas("catalogue", "recherche sans résultat", LIM, [neuf(), mod("recherche", "ZZZ 999")])
cas("catalogue", "toutes les familles décochées sauf CHS", LIM,
    [neuf()] + [mod(f"familles/{f}", False) for f in ("IPE", "HEA", "HEB", "HEM", "UPE", "UPN", "L", "LI", "SHS", "RHS")])
cas("catalogue", "recherche ajoutée", LIM, [neuf(), act("ajouter"), mod("recherche", "HEB")], {"elements": 2})

# ====================================================================== données béton
cas("donnees_beton", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("donnees_beton", "aciers B400", REF, [neuf(), mod("rubrique", "aciers"), act("choisir", role="ligne", nom="B400")])
cas("donnees_beton", "enrobages : XD3/XS3, R120", REF, [neuf(), mod("rubrique", "enrobages"), act("choisir", role="ligne", nom="XD3/XS3"),
                                                       mod("enrobages/r", "R120")])
cas("donnees_beton", "armatures Ø40", REF, [neuf(), mod("rubrique", "armatures"), act("choisir", role="ligne", nom="Ø40")])
cas("donnees_beton", "classes du bureau", REF, [neuf(), mod("classes/groupe", "bureau")])
cas("donnees_beton", "ancrages : tableaux à venir", LIM, [neuf(), mod("rubrique", "ancrages")])
cas("donnees_beton", "résistance au feu R240 (PARCOURS)", LIM, [neuf(), mod("rubrique", "enrobages"), mod("enrobages/r", "R240")])
cas("donnees_beton", "rubrique inconnue refusée", LIM, [neuf(), mod("rubrique", "pas-une-rubrique")], {"message": True})
cas("donnees_beton", "consultation ajoutée", LIM, [neuf(), act("ajouter"), mod("rubrique", "aciers")], {"elements": 2})

# ====================================================================== enrobage
cas("enrobage", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("enrobage", "poteau extérieur XC4 / S4", REF, [neuf(), mod("type", "Poteau"), mod("position", "Extérieur"), mod("classe_expo", "XC4"),
                                                  mod("classe_struct", "S4")])
cas("enrobage", "poutre XD3/XS3, feu R120", REF, [neuf(), mod("classe_expo", "XD3/XS3"), mod("feu", True), act("choisir", role="feu", nom="b ≥ 300 mm")])
cas("enrobage", "voile, Δc_dev = 0", REF, [neuf(), mod("type", "Voile"), mod("delta_dev", "0")])
cas("enrobage", "barre Ø4 refusée (PARCOURS)", LIM, [neuf(), mod("diam_barre", "4")], {"message": True})
cas("enrobage", "barre Ø20,5 refusée (PARCOURS)", LIM, [neuf(), mod("diam_barre", "20,5")], {"message": True})
cas("enrobage", "étrier Ø0 (valeur minimale admise)", LIM, [neuf(), mod("diam_etrier", "0")], {"sans_message": True})
cas("enrobage", "étrier négatif refusé (PARCOURS)", LIM, [neuf(), mod("diam_etrier", "-2")], {"message": True})
cas("enrobage", "élément ajouté", LIM, [neuf(), act("ajouter"), mod("type", "Dalle")], {"elements": 2})

# ====================================================================== rigidité du sol
cas("sol", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("sol", "essai collé (PARCOURS)", REF, [neuf(), mod("sondages/0/colle", "0,5;2,1;30\n1,0;2,4;35\n1,5;3,0;40\n2,0;5,2;41\n2,5;6,0;44\n3,0;8,0;50\n")])
cas("sol", "méthodes rapide, théories, abaque, plat", REF, [neuf(), mod("mode", "theories")])
cas("sol", "plaque (mode plat)", REF, [neuf(), mod("mode", "plat")])
cas("sol", "nappe active à 1,5 m, angle", REF, [neuf(), mod("nappe_active", True), mod("nappe", "1,5"), mod("position", "angle")])
cas("sol", "semelle de 0,2 m (borne basse admise)", LIM, [neuf(), mod("B", "0,2")], {"sans_message": True})
cas("sol", "semelle de 0,1 m refusée (PARCOURS)", LIM, [neuf(), mod("B", "0,1")], {"message": True})
cas("sol", "essai illisible collé (PARCOURS)", LIM, [neuf(), mod("sondages/0/colle", "rien de lisible")])
cas("sol", "largeur négative refusée", LIM, [neuf(), mod("B", "-2")], {"message": True})
cas("sol", "fichier GEF illisible (PARCOURS)", LIM, [neuf(), mod("sondages/0/fichier", {"nom": "faux.gef", "texte": "pas un essai"})], {"message": True})
cas("sol", "couche retirée, sondage ajouté", LIM, [neuf(), mod("sondages/0/couches/2/-", 1), mod("sondages/+", 1)])
cas("sol", "fondation ajoutée", LIM, [neuf(), act("ajouter"), mod("B", "3")], {"elements": 2})

# ====================================================================== taux d'armature
# refait le 10/11/2026 (Lot Q, docs/site/LOT_Q_QUATRE_MODULES.md § 3) : un QUANTITATIF, sans taux d'utilisation — pas de
# cas « près de 100 % » ; la note neuve porte la poutre type (85,8 kg/m³, recalculée à la main dans tests/test_api_web_taux.py)
cas("taux", "note neuve : la poutre type (85,8 kg/m³)", REF, [neuf()], {"texte": "85,8"}, fichiers=True)
cas("taux", "riche : cadre, cadre intérieur et épingle, retours, peau, portion sans étriers", REF, [calc(TX._note(**TX._RICHE))],
    {"texte": "Sans étriers de 2,00 à 3,00"}, fichiers=True)
cas("taux", "poutre reçue de « Poutres béton armé » (provenance)", REF, [calc(TX._note(**TX._RECUE))], {"texte": "reçue de la poutre V1, le 10/11/2026"})
cas("taux", "deux talons et leur cadre", REF, [neuf(), mod("forme", "deux_talons")], {"sans_message": True})
cas("taux", "14 m : recouvrements de 50 Ø, les zones d'about suivent", REF, [neuf(), mod("L", "14")], {"texte": "Ont suivi"})
cas("taux", "largeur de 5 cm refusée", LIM, [neuf(), mod("b", "5")], {"message": True})
cas("taux", "deux zones qui se recouvrent refusées", LIM, [calc(TX._note(**TX._RICHE)), mod("zones/1/de", "1,5")], {"message": True})
cas("taux", "type employé par une zone : retrait refusé (PARCOURS)", LIM, [neuf(), mod("types/0/-", "1")], {"message": True})
cas("taux", "toutes les zones retirées : sans étriers", LIM, [neuf(), mod("zones/0/-", "1"), mod("zones/0/-", "1"), mod("zones/0/-", "1")],
    {"texte": "Sans étriers sur toute la longueur"})
cas("taux", "ancienne note (v1) : noms gardés, avis", LIM, [calc(TX.NOTE_V1)], {"texte": "ancien module"})
cas("taux", "poutre ajoutée (PARCOURS)", LIM, [neuf(), act("ajouter")], {"elements": 2})

# ====================================================================== assemblages : sept configurations, deux modes du pied
_TYPES = ("doubles_cornieres", "plat_ame", "platine_about", "plat_ame_poteau", "doubles_cornieres_poteau", "platine_about_poteau",
          "platine_beton")
MANUEL = "Calcul manuel de la platine (boulons et éléments libres)"


def _relue_sans_V():
    """Une note relue (fichier .stab) dont V_Ed est vide : le site la garde à corriger et suspend le verdict."""
    from api_web import assemblages as AS
    e = AS.nouvel_etat()
    e["assemblages"][0]["values"]["V_Ed"] = None
    return e


COMPLET = "Vérification complète EN 1993-1-8 + EN 1992-4 (grille régulière)"
cas("assemblages", "note neuve (doubles cornières)", REF, [neuf()], selecteurs=True, fichiers=True)
for _t in _TYPES[1:]:
    cas("assemblages", f"{_t} : neuf", REF, [neuf(), mod("type", _t)], selecteurs=True)
for _t, _ex in (("doubles_cornieres", "BM1"), ("doubles_cornieres", "BM4"), ("plat_ame", "BM1"), ("platine_about", "BM1"),
                ("plat_ame_poteau", "BM2"), ("plat_ame_poteau", "BM4"), ("doubles_cornieres_poteau", "BM3"),
                ("platine_about_poteau", "BM4"), ("platine_beton", "VAL-A"), ("platine_beton", "VAL-M1"), ("platine_beton", "VAL-M3")):
    cas("assemblages", f"{_t} : exemple publié {_ex}", REF, [neuf(), mod("type", _t), mod("exemple", _ex)])
cas("assemblages", "pied de poteau : vérification complète (grille régulière)", REF, [neuf(), mod("type", "platine_beton"), mod("v/mode", COMPLET)],
    selecteurs=True)
for _t, _v in (("doubles_cornieres", "165"), ("plat_ame", "165"), ("platine_about", "290"), ("plat_ame_poteau", "285"),
               ("doubles_cornieres_poteau", "370"), ("platine_about_poteau", "510")):
    cas("assemblages", f"{_t} : chargé près de 100 % (V_Ed = {_v} kN)", REF, [neuf(), mod("type", _t), mod("v/V_Ed", _v)], {"taux": [95, 100]})
cas("assemblages", "platine_beton : chargé près de 100 % (N_Ed = 1 770 kN, calcul manuel)", REF, [neuf(), mod("type", "platine_beton"), mod("v/N_Ed", "1770")],
    {"taux": [95, 100]})
cas("assemblages", "platine_beton : chargé près de 100 % (N_Ed = 1 610 kN, vérification complète)", REF,
    [neuf(), mod("type", "platine_beton"), mod("v/mode", COMPLET), mod("v/N_Ed", "1610")], {"taux": [95, 100]})
cas("assemblages", "doubles cornières : prédimensionnement", LIM, [neuf(), mod("v/mode_calc", "PRÉDIMENSIONNEMENT")])
cas("assemblages", "effort tranchant nul", LIM, [neuf(), mod("v/V_Ed", "0")])
cas("assemblages", "effort illisible refusé", LIM, [neuf(), mod("v/V_Ed", "beaucoup")], {"message": True})
cas("assemblages", "longueur de cornière vidée refusée", LIM, [neuf(), mod("v/LC_u", "")], {"message": True})
cas("assemblages", "profilé personnalisé, hauteur vidée (à compléter)", LIM, [neuf(), mod("v/prof_S", "Personnalisé"), mod("v/hS_u", "")],
    {"niveau": "c"})
cas("assemblages", "note relue avec V_Ed vide (verdict suspendu)", LIM, [calc(_relue_sans_V())], {"niveau": "c"})
cas("assemblages", "pinces trop courtes (e1 = 10 mm)", LIM, [neuf(), mod("v/e1S_u", "10")])
cas("assemblages", "plat d'âme : configuration changée, valeurs gardées", LIM, [neuf(), mod("v/V_Ed", "90"), mod("type", "plat_ame")])
cas("assemblages", "pied : tige retirée (calcul manuel)", LIM, [neuf(), mod("type", "platine_beton"), mod("v/tiges/3/-", 1)])
cas("assemblages", "pied : traction N = −100 kN (vérification complète)", LIM,
    [neuf(), mod("type", "platine_beton"), mod("v/mode", COMPLET), mod("v/N_Ed", "-100")])
cas("assemblages", "assemblage ajouté (platine d'about)", LIM, [neuf(), act("ajouter"), mod("type", "platine_about")], {"elements": 2})
cas("assemblages", "note abîmée", LIM, [ABIMEE])


# ====================================================================== fondations de grues (module né le 11/11/2026)
# (les quatre classeurs validés par le bureau, repris par « exemple » ; les taux annoncés sont ceux des classeurs, que
# tests/test_fondation_grue_moteur.py tient cellule par cellule ; « v » : le classeur dit LIMITE, ou le cisaillement
# est en zone 3 — docs/beton/FONDATIONS_GRUE.md)
_FG = [neuf(), mod("exemple", "GWW")]
cas("fondation_grue", "note neuve (8 pieux sur semelles, valeurs du classeur BASE)", REF, [neuf()], {"taux": [100, 100], "niveau": "v"}, fichiers=True)
cas("fondation_grue", "classeur BASE", REF, [neuf(), mod("exemple", "BASE")], {"taux": [100, 100], "niveau": "v", "sans_message": True})
cas("fondation_grue", "classeur Greenwood (GWW)", REF, _FG, {"taux": [98, 98], "niveau": "v", "sans_message": True}, fichiers=True)
cas("fondation_grue", "classeur de la variante 4 pieux (cadre)", REF, [neuf(), mod("exemple", "VAR4")], {"taux": [90, 90], "niveau": "p", "sans_message": True},
    fichiers=True)
cas("fondation_grue", "classeur de la variante 8 pieux (poutres primaires à 45°)", REF, [neuf(), mod("exemple", "VAR8")],
    {"taux": [96, 97], "niveau": "f", "sans_message": True}, fichiers=True)
cas("fondation_grue", "Greenwood contre une paroi de pieux sécants", REF, _FG + [mod("cote_b", "paroi")], {"taux": [98, 98], "niveau": "v"})
cas("fondation_grue", "Greenwood, un second groupe de barres (2Ø16) et un 7e cas de la grue (900 + 900 kN)", REF,
    _FG + [mod("support/inf/1/n", "2"), mod("charges/cas/+", "1"), mod("charges/cas/6/A", "900"), mod("charges/cas/6/B", "900")], {"sans_message": True},
    fichiers=True)
cas("fondation_grue", "pieu PAS OK (capacité 40 T) : l'export vers RBAR est bloqué", LIM, _FG + [mod("pieux/capacite", "40")],
    {"niveau": "r", "texte": "Fichier pour RBAR bloqué"}, fichiers=True)
cas("fondation_grue", "cisaillement au-delà de la zone 3 (poutre support de 60 cm de large)", LIM, _FG + [mod("support/b", "60")], {"niveau": "r"})
cas("fondation_grue", "géométrie non respectée (entredistance des pieux 150 cm < 3 Ø)", LIM, _FG + [mod("pieux/entraxe", "150")], {"texte": "Non respecté"})
cas("fondation_grue", "un seul cas de la grue", LIM, _FG + [mod("charges/cas/1/-", "1")] * 5, {"sans_message": True})
cas("fondation_grue", "excentricité hors domaine refusée (e = 80 cm : le pied sortirait de la portée)", LIM, [neuf(), mod("e_support", "80")], {"message": True})
cas("fondation_grue", "diamètre hors de la liste refusé (Ø17)", LIM, [neuf(), mod("support/inf/0/d", "17")], {"message": True})
cas("fondation_grue", "le dernier cas de la grue ne se retire pas", LIM, _FG + [mod("charges/cas/1/-", "1")] * 5 + [mod("charges/cas/0/-", "1")], {"message": True})
# les options de conception du plan (12/11/2026, beton/fondation_grue/conception.py) : elles ne changent aucun calcul ;
# elles changent le fichier « rbar.ouvrage » (photographié : fichiers=True) — docs/beton/FONDATIONS_GRUE_OUVRAGE.md
cas("fondation_grue", "options de conception : retours arrêtés à 10 cm sous la face opposée, mise en page du châssis 8,00 m", REF,
    _FG + [mod("conception/support/retour", "arret"), mod("conception/semelle/retour", "arret"), mod("conception/mise_en_page", "800")],
    {"taux": [98, 98], "niveau": "v", "sans_message": True}, fichiers=True)
cas("fondation_grue", "retour plus haut que la poutre : le fichier pour RBAR est bloqué", LIM, _FG + [mod("conception/support/retour_mm", "650")],
    {"taux": [98, 98], "avis": "ne tient pas dans la hauteur"}, fichiers=True)
cas("fondation_grue", "option de conception hors domaine refusée (longueur scellée plus longue que la barre)", LIM,
    [neuf(), mod("conception/perfos/scelle_mm", "900")], {"message": True})
cas("fondation_grue", "fondation ajoutée, passée en cadre de 4 pieux", LIM, [neuf(), act("ajouter"), mod("type", "cadre4")], {"elements": 2})
cas("fondation_grue", "note abîmée (châssis négatif)", LIM, [calc({"version": 1, "fondations": [{"type": "semelles8", "chassis": -5}], "actif": 0})],
    {"erreur": True})
