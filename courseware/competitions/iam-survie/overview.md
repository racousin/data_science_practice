# Survie à 2 mois — __SESSION_TITLE__

Prédire si un **patient adulte gravement malade**, hospitalisé, sera **vivant** ou
**décédé** deux mois après son inclusion, à partir des seules données cliniques
recueillies dans les premiers jours d'hospitalisation.

## Contexte clinique

Les patients proviennent de l'étude SUPPORT (cinq centres hospitaliers américains) :
des adultes atteints d'une affection engageant le pronostic vital — défaillance
respiratoire ou multiviscérale, insuffisance cardiaque, BPCO ou cirrhose avancées,
coma, cancer métastatique. Un pronostic fiable à 2 mois aide à discuter de
l'intensité des soins avec le patient et sa famille. Le modèle n'utilise que les
informations disponibles à l'admission, avant que l'issue ne soit connue.

## Tâche

- **Type :** classification binaire, une prédiction par patient.
- **Cible `outcome` :** `alive` (vivant) ou `dead` (décédé) à 2 mois.
- **Métrique :** **exactitude** (*accuracy*), la proportion de patients bien classés.
  Environ **59 %** des patients sont vivants : prédire `alive` pour tout le monde
  donne déjà ≈ 0,59. C'est le seuil à dépasser.

## Les données

Chaque patient est identifié par `patient_id` et décrit par une trentaine de
variables cliniques :

- **démographie** : âge, sexe, origine, années d'études, revenu ;
- **diagnostic** : groupe et classe de maladie, nombre de comorbidités, cancer ;
- **constantes vitales** : pression artérielle moyenne, fréquence cardiaque et
  respiratoire, température ;
- **biologie** : leucocytes, sodium, créatinine, albumine, bilirubine, urée,
  glycémie, pH artériel, rapport PaO₂/FiO₂, diurèse ;
- **scores de gravité** : score physiologique aigu, score de coma, intensité des
  interventions ; autonomie (activités de la vie quotidienne).

Beaucoup de valeurs manquent, souvent parce qu'un examen n'a pas été prescrit :
le fait qu'une valeur manque peut lui-même être informatif.

| Fichier | Lignes | Contenu |
| --- | --- | --- |
| [`X_train.csv`](https://raw.githubusercontent.com/racousin/ai_for_medicine/main/ia-medecine/donnees/support2/X_train.csv) | 6 164 | `patient_id` + variables cliniques |
| [`y_train_survie.csv`](https://raw.githubusercontent.com/racousin/ai_for_medicine/main/ia-medecine/donnees/support2/y_train_survie.csv) | 6 164 | `patient_id,outcome` |
| [`X_test.csv`](https://raw.githubusercontent.com/racousin/ai_for_medicine/main/ia-medecine/donnees/support2/X_test.csv) | 1 541 | `patient_id` + variables, sans la cible |

## Cette séance

__SESSION_BLOCK__

## Soumission

Un fichier CSV nommé `submission.csv`, avec une ligne par `patient_id` de
`X_test.csv` (1 541 lignes) :

```
patient_id,outcome
a1b2c3d4e5,alive
f6a7b8c9d0,dead
```

Déposez-le sur cette page, ou envoyez-le depuis le notebook de la séance avec
votre clé d'API (page Profil). Chaque séance a son propre défi et son propre
classement.
