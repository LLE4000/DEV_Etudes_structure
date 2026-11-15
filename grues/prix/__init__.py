# -*- coding: utf-8 -*-
"""Le moteur du module « Prix des grues » : le prix d'une grue (tour, montage rapide, mobile) pour un chantier, à partir
d'une BASE DES PRIX (les loueurs, leurs catalogues, leurs grues, les postes communs, nos réglages, le prix réel des
fondations de nos chantiers).

AUCUN PRIX ICI : la base vit dans le stockage du site, importée une fois depuis le fichier de départ (confidentiel, jamais
dans le dépôt). Ce paquet ne sait que la LIRE et CALCULER :

- ``valeurs``    : ce qu'une case de prix peut contenir (un nombre, « compris », « non disponible », « sur demande », vide) ;
- ``base``       : la lecture de la base — catalogues, grues, postes, réglages —, chaque prix avec son ORIGINE (valeur,
                   année, source, validité, notre correction) ;
- ``periode``    : les durées (mois, jours facturés, semaines, jours du grutier, calendrier réel belge) ;
- ``ecussons``   : l'écusson d'un prix, d'une grue, d'un calcul, d'un dossier de loueur (À jour, Pas à jour, À vérifier…) ;
- ``fondations`` : le prix d'une fondation (notre correction, la moyenne des retours validés de nos chantiers, l'ancien
                   tableau) ;
- ``calcul``     : le prix d'une grue, poste par poste, selon sa logique (T grue tour, R montage rapide, MS grue mobile
                   grutier à part, MA grue mobile grutier compris), ses prix manquants et ses points d'attention.

Tout est en bibliothèque standard : le paquet tourne tel quel dans le navigateur (Pyodide).
"""
