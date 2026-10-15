# Le projet : aller plus loin sur la survie à 2 mois

En binôme — un étudiant en médecine et un étudiant en apprentissage
automatique — vous reprenez la tâche des défis : prédire si un patient de
réanimation de l'étude SUPPORT2 est **vivant ou décédé 2 mois** après
l'admission. Les séances l'ont abordée avec une régression logistique, une
forêt aléatoire et un réseau de neurones ; le projet consiste à **aller plus
loin**, puis à présenter votre méthode à l'oral.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_medicine/blob/main/ia-medecine/notebooks/iam-projet.ipynb)

Le notebook décrit les données (28 variables mesurées à l'admission), entraîne
une régression logistique, mesure son exactitude sur un jeu de validation
(environ 0,76) et écrit `submission.csv`.

<!-- notes: 15 minutes. Former les binômes (un profil médecine, un profil
apprentissage automatique) avant la fin de la séance. Le notebook est une
baseline, pas un exercice : il n'a pas de correction. Insister sur la
validation : le score qui compte dans la présentation est celui d'une
validation honnête, pas un score mesuré sur les données d'entraînement. -->

---

## La métrique

Le classement utilise l'**exactitude**, la part des patients bien classés :

$$
\text{exactitude} = \frac{VP + VN}{VP + VN + FP + FN}
$$

Prédire `alive` pour tout le monde donne déjà environ **0,59**.

Cliniquement, toutes les erreurs ne se valent pas : manquer un décès n'a pas
le même coût qu'une fausse alerte. Présentez aussi la **matrice de
confusion** et le **rappel** des décès, la part des décès détectés :

$$
\text{rappel} = \frac{VP}{VP + FN}
$$

---

## Ce qui est attendu

Dépasser les modèles des séances, en mesurant chaque essai sur des patients
que le modèle n'a pas vus :

- **nouvelles variables** cliniquement pertinentes ;
- **la donnée manquante comme information** (indicateurs de valeur manquante) ;
- **gradient boosting** et réglage des hyperparamètres en validation croisée ;
- **calibration** des probabilités prédites ;
- **fuite d'information** : chaque variable est-elle connue au moment de la
  prédiction ?

Soumettez `submission.csv` (colonnes `patient_id,outcome`, une ligne par
patient du test) sur le challenge du projet,
[ml-arena.com/viewchallenge/172](https://ml-arena.com/viewchallenge/172).
Le classement est pour la gloire : il n'est pas noté.

---

## La présentation

**10 minutes** de présentation par binôme, puis **5 minutes** de questions
individuelles à chaque membre.

1. **Le problème** : contexte médical, données, cible.
2. **La méthodologie** : objectif, métrique, protocole de validation.
3. **Le prétraitement** : valeurs manquantes, variables catégorielles,
   nouvelles variables.
4. **Le modèle** : modèles essayés, résultats comparés à la baseline et aux
   modèles des séances.
5. **Les problèmes rencontrés et la suite** : limites, pistes.

**À retenir**

- Le score qui compte est mesuré sur des patients jamais vus.
- L'exactitude classe ; la matrice de confusion et le rappel des décès
  disent si le modèle est utile cliniquement.
- Chaque choix doit pouvoir être justifié, cliniquement et
  méthodologiquement, par chacun des deux membres du binôme.
