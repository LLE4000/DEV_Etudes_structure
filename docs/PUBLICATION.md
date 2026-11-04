# Publication : du dépôt de développement au dépôt partagé

*Écrit le 04/11/2026, à la question du bureau : « peut-on transférer tout le dépôt
de développement vers le dépôt partagé avec les collègues, en un clic ? »*

## Ce que sont les deux dépôts

| | `LLE4000/DEV_Etudes_structure` (développement) | `LLE4000/Etudes-structure` (partagé) |
|---|---|---|
| Contenu | l'application complète : 313 fichiers de plus, 96 000 lignes ajoutées | la copie initiale (streamlit_app, modules, assets, requirements) |
| Dernière poussée | novembre 2026 | 2 février 2026 |
| Historique | 110 commits propres | ses propres commits (« Update poutre.py », …) |
| Ancêtre commun | **aucun** : les deux historiques sont indépendants (mesuré : `git merge-base` vide) | |

Conséquence : une fusion Git ordinaire est impossible (« unrelated histories ») et
n'aurait de toute façon aucun sens, puisque le dépôt partagé n'a reçu aucune
modification à conserver depuis la copie. La bonne opération est un **remplacement** :
le contenu du dépôt partagé devient celui du dépôt de développement.

## La réponse : oui, en un clic, sans jeton

Le dépôt de développement est public : un workflow GitHub Actions placé **dans le
dépôt partagé** peut le lire sans aucune clé, et pousser sur lui-même avec le
`GITHUB_TOKEN` que GitHub lui fournit. Le fichier est prêt :

```
outils/publication/synchroniser_depuis_dev.yml
```

Installation (une fois) : copier ce fichier dans le dépôt partagé sous
`.github/workflows/synchroniser_depuis_dev.yml` et le committer. Ensuite, à chaque
publication : onglet **Actions** du dépôt partagé → « Synchroniser depuis le dépôt
de développement » → **Run workflow** (la branche source est `main` par défaut).
Trente secondes plus tard, le dépôt partagé est identique au dépôt de développement
et Streamlit Community Cloud le redéploie.

L'installation elle-même peut être faite depuis cette session, sur demande explicite
(le dépôt partagé n'est jamais modifié sans qu'on le demande).

## Ce qu'il faut savoir avant de cliquer

- **L'historique du dépôt partagé est remplacé** (`push --force`) : un collègue qui
  avait cloné le dépôt doit le re-cloner ; celui qui n'utilise que l'application
  déployée ne voit qu'une mise à jour. Rien n'est perdu côté développement.
- **Les fichiers propres au dépôt partagé disparaissent** s'ils n'existent pas dans le
  dépôt de développement. Aujourd'hui, un seul : `.devcontainer/` (configuration
  Codespaces). S'il doit survivre, on le recopie d'abord dans le dépôt de développement.
- **Les secrets Streamlit** (fichier `secrets.toml` du déploiement) ne sont pas dans
  Git : ils restent attachés à l'application déployée.
- **Le sens est unique** : développement → partagé. Une modification faite directement
  dans le dépôt partagé serait écrasée à la synchronisation suivante ; toute correction
  se fait dans le dépôt de développement, puis on republie.

## Et si le dépôt de développement devenait le dépôt principal ?

C'est l'autre réponse possible, et elle est encore plus simple : pointer l'application
déployée des collègues sur `LLE4000/DEV_Etudes_structure`, branche `main`, et laisser le
dépôt partagé en lecture seule (archivé). Le dépôt de développement travaille déjà
ainsi : les lots se développent sur une branche, et `main` ne reçoit que des fusions
demandées. `main` est donc toujours un état livrable — exactement le rôle du dépôt
partagé. La règle serait alors : **`main` = ce que les collègues voient**, la branche de
travail = le développement. Le workflow ci-dessus reste utile si le bureau tient à deux
dépôts distincts (droits différents, nom plus parlant pour les collègues).
