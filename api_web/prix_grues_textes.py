# -*- coding: utf-8 -*-
"""Les TEXTES du module « Prix des grues » qui se lisent à l'impression (cahier § 11.1, § 11.3) : l'encadré « Ce qui est
calculé » de chaque logique et la présentation de l'outil (bouton « ? »). Une seule source pour le rapport PDF et l'écran.

Aucun prix ici : le taux de la contribution environnementale vient de la base (``contribution``)."""
from typing import Dict, List, Optional

COMMUN = "Mois = mois complets + fraction du mois entamé. Jours facturés = mois × 20."

LOGIQUES: Dict[str, List[str]] = {
    "T": ["Location : prix par jour selon le châssis × jours facturés. Châssis hors standard : prix par semaine × semaines. "
          "Châssis roulant : + le châssis, les rails au mètre, leur pose et leur transport.",
          "Électricité : consommation kWh/mois × prix du kWh × mois ; sinon forfait par jour.",
          "Montage et démontage : forfaits du loueur, transport compris.",
          "Accessoires : prix par jour × jours facturés ; montage et démontage au forfait.",
          "Grutier : heures par jour × jours × taux ; au-delà de 8 h par jour, taux des heures sup.",
          "Fondation : prix réels de nos chantiers pour la même grue et le même type de fondation ; sinon le tableau.",
          "Total arrondi à l'euro, séparé en matériel et main-d'œuvre."],
    "R": ["Location : prix par jour × jours facturés.",
          "Électricité : kWh/mois × prix du kWh × mois si la consommation est connue ; sinon forfait par jour.",
          "Montage, démontage, transport des poids de test : forfaits du loueur.",
          "Conduite : télécommande par défaut.",
          "Accessoires, grutier, fondation et total : comme la grue tour."],
    "MS": ["Location : prix par jour × jours facturés (sans opérateur ; avec opérateur si on le choisit, et alors sans ligne grutier).",
           "Transport et montage, démontage : forfaits du loueur.",
           "Carburant : selon le tarif du loueur.",
           "Grutier : heures par jour × jours × taux ; au-delà de 8 h, taux des heures sup.",
           "Total arrondi à l'euro, séparé en matériel et main-d'œuvre."],
    "MA": ["Location : prix horaire × heures facturées par jour × jours. Minimum 5 h jusqu'à 80 t, 8 h au-delà, 8 h par jour sur plusieurs jours.",
           "Heures au-delà de 8 h : supplément heures sup.",
           "Déplacement de la grue : prix horaire × heures de trajet et de montage sur place.",
           "Déplacement du personnel : prix par trajet, si la location dure plus d'un jour.",
           "Ballast : transport au forfait, majoré le samedi, le dimanche et la nuit.",
           "Nuit, samedi, dimanche, férié : supplément horaire. Rigger, chef de levage et services si on les choisit.",
           "Contribution environnementale : {contribution} des lignes du loueur.",
           "Total arrondi à l'euro."],
}
NOMS = {"T": "Grue tour (T)", "R": "Grue à montage rapide (R)", "MS": "Grue mobile, grutier à part (MS)", "MA": "Grue mobile, grutier compris (MA)"}

PRESENTATION = [
    ("Ce qu'il fait.", "Il calcule le prix d'une grue pour un chantier : grue tour, grue à montage rapide ou grue mobile. On choisit la grue, "
                       "les dates et les heures du grutier. Il sort le prix détaillé, poste par poste, avec la source de chaque prix."),
    ("D'où viennent les prix.", "Des loueurs : une fois par an, on leur envoie un fichier Excel avec nos grues et nos prix, ils le corrigent "
                                "et on le réimporte. Des fondations réellement payées sur nos chantiers : on leur envoie un fichier Excel "
                                "pré-rempli, ils le renvoient avec les montants. De nos réglages : prix de l'électricité, jours par mois, taux."),
    ("Comment on sait si c'est fiable.", "Chaque prix porte un écusson : À jour, À vérifier ou Pas à jour."),
    ("Ce qu'on en fait.", "On chiffre l'installation de chantier, on imprime le rapport pour le faire valider, et on garde le prix de chaque "
                          "grue, chantier par chantier."),
]
CYCLES = [("Loueurs", ["fichier envoyé au loueur", "rempli par le loueur", "importé et contrôlé", "prix à jour"]),
          ("Fondations", ["fichier envoyé au chantier", "renvoyé", "importé et validé", "prix de la fondation à jour"])]


def ce_qui_est_calcule(logique: str, contribution: Optional[str] = None) -> List[str]:
    """Le texte de l'encadré d'une logique : la ligne commune, puis le bloc de la logique (le taux de la contribution
    environnementale lu dans la base ; inconnu : « un pourcentage »)."""
    bloc = [t.format(contribution=contribution or "un pourcentage") for t in LOGIQUES.get(logique, [])]
    return [COMMUN] + bloc
