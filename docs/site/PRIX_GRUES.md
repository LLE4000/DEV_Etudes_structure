# Module « Prix des grues »

Le prix d'une grue (tour, montage rapide, mobile) pour un chantier, poste par poste, avec la source et l'écusson de chaque
prix ; la base des prix des loueurs ; le prix réel des fondations de grue de nos chantiers. Catégorie « Autres » de
l'accueil. Le cahier des charges (15/11/2026) est hors du dépôt, avec ses fichiers (confidentiels).

## 1. Où est quoi

| Partie | Fichier |
|---|---|
| Moteur (bibliothèque standard seule) | `grues/prix/` : `valeurs`, `base`, `periode`, `ecussons`, `fondations`, `calcul`, `edition`, `dossiers` |
| Échange avec les loueurs (§ 7) | `grues/prix/xlsx.py` (écrire et lire un .xlsx sans openpyxl), `fichier_loueur.py` (préparer), `lecture_loueur.py` (relire), `import_loueur.py` (appliquer, annuler) ; écran `api_web/prix_grues_echange.py` |
| Écran (trois pages en trois colonnes) | `api_web/prix_grues.py` (état, saisies, routage), `prix_grues_calcul.py`, `prix_grues_base.py`, `prix_grues_fondations.py`, `prix_grues_commun.py` (écussons, montants) |
| Rapport PDF (une grue, le projet) | `api_web/prix_grues_rapport.py`, textes `api_web/prix_grues_textes.py` |
| Feuille | `site/web/css/modules/prix_grues.css` |
| Base de TEST | `tests/donnees/prix_grues_base_test.json` |
| Tests | `tests/test_prix_grues_moteur.py`, `tests/test_prix_grues_echange.py`, `tests/test_api_web_prix_grues.py`, harnais `verifier_site.js` (filtre `prix_grues`) ; fichier rempli de test `tests/donnees/prix_grues_loueur_a_rempli.xlsx` ; carnet d'essai des tests communs `tests/carnets_essai.py` |

## 2. Confidentialité

- **Aucun prix dans le code ni dans le dépôt.** La vraie base (le fichier JSON de départ) n'est jamais versionnée ; elle
  s'importe une fois depuis la page, sur l'appareil.
- La **base de test** du dépôt est la base de départ rendue anonyme : tous les prix multipliés par un même facteur (gardé
  hors du dépôt, avec le script qui la fabrique), loueurs renommés « Loueur A, B, C », textes libres neutres, chantiers
  numérotés, aucune personne. `test_prix_grues_moteur.py` § 7 le contrôle (par empreintes : les noms interdits ne sont
  écrits nulle part). Les calculs sont linéaires dans les prix : les tests T1 à T4 s'y retrouvent au facteur près.
- Le site entier est servi avec `X-Robots-Tag: noindex, nofollow` (`site/web/_headers`) : les pages du module aussi.
- Le relevé de la note PDF (`construire.NOTES_RELEVE`) joue le rapport sur une mini-base FICTIVE écrite dans la recette :
  aucun fichier de prix n'entre dans le paquet du site.

## 3. Stockage (décision du propriétaire, 15/11/2026 : « si plus tard on doit ajouter Cloudflare, on le fera »)

- **La base des prix est le CARNET du module** (`CARNET = True`) : l'écran la garde dans ce navigateur
  (`stab.carnets.v1`), la joint à chaque requête et l'enregistre à chaque réponse ; le moteur la reçoit dans
  `etat["carnet"]`, jamais dans la note. Le téléchargement (export JSON, rapport) la reçoit aussi (paramètre `carnet` de
  `telecharger`, `api_web/routeur.py`).
- **Une note = un projet** : ses grues (GT1, GT2…), chacune avec son calcul. Le résumé de la note (accueil) dit le total.
- Limite : la base est **par appareil**. Le stockage partagé entre utilisateurs (Cloudflare) viendra plus tard ; d'ici
  là, « Exporter → Base des prix (JSON) » et « Remplacer la base (JSON) » font passer la base d'un appareil à l'autre.

## 4. Règles de calcul (cahier § 6)

- Mois = mois complets + fraction du mois entamé (fin = le jour où la grue n'est plus louée) ; jours facturés = mois ×
  20 ; semaines = mois × 4 ; jours du grutier = mois × 20, ou le calendrier réel (lundi à vendredi, fériés belges, congés
  du bâtiment des réglages). Logique MA : des jours.
- Grutier (T, R, MS) : min(h ; 8) × jours au taux normal, max(0 ; h − 8) × jours au taux des heures sup.
- MA : heures facturées = max(h ; minimum) — le minimum de la grue pour un jour, 8 h par jour sur plusieurs jours ;
  déplacement × heures ; ballast le samedi × coefficient ; personnel max(h ; 8) × jours × nombre ; contribution
  environnementale = taux × toutes les lignes du loueur, sauf la fondation et les lignes libres.
- Fondation : prix forcé → notre correction → moyenne des retours VALIDÉS → ancien tableau (pieux = dalle × 1,3 arrondi au
  multiple de 500 €, réglages) → manquant.
- Un prix inconnu n'est JAMAIS 0 € : la ligne retenue est « manquante » (rouge), le calcul « INCOMPLET ».
- Codes des lignes : ceux des tests (LOC, ELEC, FOND, MONT, DEMONT, POIDS, TEL, TELV, CAB, AI, CHROUL, TLS, DEPL, BAL,
  CARB) et les codes des postes de la base. Les montages d'accessoires suivent leur location (`calcul.SUIT`).

## 5. Ce qui est fait (lot du 15/11/2026)

- Moteur complet pour les quatre logiques ; T1 à T4 retrouvés à l'euro près (sur la vraie base hors du dépôt : 150 994 /
  50 244 / 137 080 / 13 618 €) ; § 13.5.4 (l'ancien retour validé : T1 au prix attendu) ; écussons des dossiers au
  14/11/2026 (§ 13.2.4 : les trois loueurs et l'électricité « Pas à jour »).
- Page Calcul : entrées, détail groupé (choix Auto/Oui/Non, quantité, prix forcé tapé dans la case avec sa justification,
  lignes libres, postes non retenus repliés), résultat (total, INCOMPLET, écusson et raison, matériel/main-d'œuvre,
  manquants, points d'attention, « Prendre la variante »), fenêtre de chaque prix (source, année, « Corriger dans la
  base », « Voir les chantiers »).
- Page Base des prix : loueurs (écusson, % à jour, manquants), catalogues et accessoires corrigeables dans la case,
  questions, réglages, historique ; la case choisie (revenir au prix du loueur, marquer à vérifier, motif).
- Page Fondations : grille (sans / avec démolition), détail d'une case (bande min–max, chantiers, valider, écarter,
  corriger, revenir au calcul automatique).
- Rapport PDF d'une grue et du projet ; présentation « ? » ; liste « À faire ».

## 6. L'échange annuel avec les loueurs (§ 7, lot du 15/11/2026)

- **Préparer** (dossier du loueur, page « Base des prix ») : l'année (l'année suivante à partir de novembre), « Demandé
  par », « Retour souhaité avant le » → `DEMANDE_PRIX_{ANNÉE}_{LOUEUR}.xlsx`. Le CONTENU est celui des modèles du cahier :
  sur la vraie base (hors du dépôt), les trois fichiers 2027 sont identiques aux modèles, feuille par feuille, case par
  case (valeurs, identifiants, clés, cases cachées, fusions, validations, verrous, volets) ; la mise en forme est refaite
  aux couleurs du site (§ 11.5). La valeur envoyée est celle du LOUEUR (jamais notre correction) ; nos remarques ne partent
  jamais ; les grues retirées non plus ; les questions « traitées » non plus.
- **Écrire le mail** : un lien `mailto:` (objet et texte du cahier ; le fichier se joint à la main). **Le fichier est
  parti** : le dossier passe en « Demande envoyée » jusqu'à l'import de la réponse.
- **Importer le fichier reçu** : relu par identifiant et clé (jamais par position) ; sur la vraie base, le fichier rempli
  du cahier donne exactement `lecture_attendue.json` ; l'aller-retour à vide 0 changement (186 / 109 / 0 prix confirmés).
  La page de **contrôle** (gauche : le fichier et les filtres ; centre : objet, champ, avant, après, écart, « appliquer » ;
  droite : compteurs, avertissements, « Appliquer », « Abandonner ») ; écart > seuil ou baisse surligné ; cases vidées
  décochées ; valeurs non lues à saisir ; autres prestations décochées (« créer un poste »).
- **Appliquer** : l'année, la période et la source « Fichier {loueur} {année} reçu le … » ; les prix confirmés datés si
  le loueur dit qu'ils restent valables ; une correction de notre part reste, la valeur du loueur passe dessous (« à
  revoir ») ; une nouvelle grue au numéro suivant ; réponses aux questions, remarques au dossier (« Traité ») ; tout dans
  l'historique ; un même fichier ne s'importe pas deux fois. **Annuler le dernier import** remet chaque objet touché.
- **Limites** : le fichier reçu n'est pas gardé (seuls son nom et son empreinte le sont) — il le sera avec le stockage
  partagé ; la « version courte » (§ 12 : seulement les prix manquants et les questions) attend la décision du
  propriétaire ; deux écrivains .xlsx coexistent (`saf/classeur.py`, valeurs et tableaux du SAF ; `grues/prix/xlsx.py`,
  mise en forme et lecture) — à fondre (docs/site/AMELIORATIONS_PAR_MODULE.md).

## 7. Ce qui reste (à faire dans l'ordre)

1. Demande au chantier (fichier Excel réalisation et démolition ; import ; saisie à la main) ; prix figés d'un calcul
   enregistré.
2. Convertisseurs (liste Excel du loueur A ; PDF du loueur B : pas de lecteur de PDF dans le navigateur aujourd'hui →
   saisie guidée d'abord).
3. Stockage partagé (Cloudflare), quand le propriétaire le décidera.
