Modèle du jour : la **forêt aléatoire** (`RandomForestClassifier`), précédée d'une
vraie préparation des données :

- **imputation** des valeurs manquantes (médiane, ou indicateur « valeur
  manquante ») ;
- **encodage** des variables catégorielles (sexe, origine, groupe de maladie,
  revenu) ;
- repérage des **valeurs aberrantes** (valeurs physiologiquement impossibles) ;
- réglage des **hyperparamètres** (`n_estimators`, `max_depth`,
  `min_samples_leaf`) par validation croisée.

Comparez votre score à celui de la régression logistique de la séance 1.
