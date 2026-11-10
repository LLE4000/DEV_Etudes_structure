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
``api_web/taux._ETAT_EXEMPLE``), les requêtes propres du parcours (``PARCOURS`` des modules : refus, relevés, fichiers)
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

# ====================================================================== bois
cas("bois", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("bois", "poutre 75×225 C24 (choix du sélecteur)", REF, [neuf(), act("choisir", role="section", nom="75x225")])
# la flèche de l'écran Streamlit est 1 000 fois trop grande (défaut reproduit et affiché) : 100 % ne s'atteint qu'en petite portée
cas("bois", "chargée près de 100 % (100×250, L = 0,5 m, G = 1,5, Q = 1,4 kN/m)", REF,
    [neuf(), act("choisir", role="section", nom="100x250"), mod("L", "0,5"), mod("G", "1,5"), mod("Q", "1,40")], {"taux": [95, 100]})
cas("bois", "section libre 80 × 240, classe 2, charge courte", REF,
    [neuf(), mod("standard", "non"), mod("service", "2"), mod("duree", "court"), mod("lim_inst", "L/400")])
cas("bois", "portée de 0,2 m refusée (PARCOURS)", LIM, [neuf(), mod("L", "0,2")], {"message": True})
cas("bois", "charge négative refusée", LIM, [neuf(), mod("Q", "-3")], {"message": True})
cas("bois", "appui de 30 mm (valeur minimale admise)", LIM, [neuf(), mod("appui", True), mod("a", "30")], {"sans_message": True})
cas("bois", "appui de 20 mm refusé", LIM, [neuf(), mod("appui", True), mod("a", "20")], {"message": True})
cas("bois", "poutre ajoutée", LIM, [neuf(), act("ajouter"), act("choisir", role="section", nom="100x250")], {"elements": 2})

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
cas("plancher", "note neuve (à compléter)", REF, [neuf()], {"niveau": "c"}, selecteurs=True, fichiers=True)
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

# ====================================================================== maturité du béton (âge)
cas("age", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("age", "relevé collé (Namur, 6 jours)", REF, [calc(AGE._etat_releve())])
cas("age", "tout affiché (seuil, optimisation, comparateur, essai)", REF,
    [calc(AGE._etat_releve(seuil__afficher=True, optim__afficher=True, comparateur__afficher=True, essai__mpa=12.0, essai__afficher=True))])
cas("age", "météo importée (Bruxelles)", REF, [calc(AGE._etat_meteo(True))])
cas("age", "seuil à 95 % de f_ck", REF, [neuf(), mod("seuil/afficher", True), mod("seuil/pct", "95")])
cas("age", "relevé illisible refusé (PARCOURS)", LIM, [calc(AGE._etat_releve()), mod("temperature/releve", "12\nabc\n13")], {"message": True})
cas("age", "fichier illisible refusé (PARCOURS)", LIM, [neuf(), mod("temperature/fichier", {"nom": "faux.csv", "texte": "douze"})], {"message": True})
cas("age", "température de −5 °C (PARCOURS)", LIM, [neuf(), mod("temperature/T", "-5")])
cas("age", "météo : lieu introuvable (PARCOURS)", LIM, [calc(AGE._etat_meteo()), mod("temperature/meteo", AGE._meteo_exemple(trouve=False))],
    {"message": True})
cas("age", "âge de 3 jours (le plus court)", LIM, [neuf(), mod("age", "3")], {"sans_message": True})
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
# P2 : la poutre de tests/test_api_web_taux.py (deux types d'étriers, Ø7,5, zones qui se recouvrent, B500C, 7,5 %)
_P2 = {"nom": "P2", "forme": "rectangulaire", "L": 7.35, "b": 35.0, "h": 70.0, "enrobage": 2.5, "majoration": 7.5, "acier": "B500C",
       "familles": [{"role": "principal", "position": "ame_inf", "n": 4, "d": 25}, {"role": "principal", "position": "ame_inf", "n": 3, "d": 20},
                    {"role": "renfort", "position": "ame_sup", "n": 3, "d": 16, "de": 0, "a": 25},
                    {"role": "lateral", "position": "lat_d", "n": 1, "d": 12, "retours": 1, "h_retour": 12.5}],
       "types_etriers": [{"nom": "E1", "d": 8.0}, {"nom": "E2", "d": 7.5}],
       "zones": [{"nom": "Z1", "de": 0, "a": 20, "s": 10, "type": "E2"}, {"nom": "Z2", "de": 20, "a": 80, "s": 20, "type": "E1"},
                 {"nom": "Z3", "de": 60, "a": 100, "s": 12.5, "type": "E1"}]}
cas("taux", "note neuve", REF, [neuf()], selecteurs=True, fichiers=True)
cas("taux", "exemple (talon gauche, latérales à retours, deux zones)", REF, [calc(TX._ETAT_EXEMPLE)], selecteurs=True)
cas("taux", "exemple en mètres, 3Ø25 choisis", REF, [calc(TX._ETAT_EXEMPLE), mod("longueurs", "metre"), act("choisir", role="d-0", nom="3Ø25")])
cas("taux", "P2 (deux types, Ø7,5, zones qui se recouvrent, B500C, 7,5 %)", REF, [calc({"version": 1, "poutres": [_P2], "actif": 0})])
cas("taux", "forme à deux talons refusée (renfort posé sur un talon)", LIM, [calc(TX._ETAT_EXEMPLE), mod("forme", "deux_talons")], {"message": True})
cas("taux", "portée nulle refusée", LIM, [calc(TX._ETAT_EXEMPLE), mod("L", "0")], {"message": True})
cas("taux", "majoration nulle", LIM, [calc(TX._ETAT_EXEMPLE), mod("majoration", "0")])
cas("taux", "dernier type d'étrier retiré (PARCOURS)", LIM, [calc(TX._ETAT_EXEMPLE), mod("types_etriers/0/-", "1")])
cas("taux", "poutre ajoutée (PARCOURS)", LIM, [calc(TX._ETAT_EXEMPLE), act("ajouter")], {"elements": 2})

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
