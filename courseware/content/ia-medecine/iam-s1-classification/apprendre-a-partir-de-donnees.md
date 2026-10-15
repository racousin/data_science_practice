# Apprendre à partir de données

L'**intelligence artificielle** regroupe les algorithmes qui automatisent des
capacités cognitives : raisonner, percevoir, décider. L'**apprentissage
automatique** (*machine learning*) en est la branche qui apprend ces
capacités à partir d'exemples, et l'**apprentissage profond** (*deep
learning*) la famille de modèles, les réseaux de neurones à nombreuses
couches, qui domine depuis les années 2010 grâce à trois facteurs réunis :
de grandes masses de données, la puissance de calcul des cartes graphiques
et des innovations algorithmiques diffusées en code ouvert.

Ce cours porte sur la méthode commune à tous ces modèles, sur des données
médicales **tabulaires** : un tableau de patients décrits par des variables.

<!-- notes: 25 minutes. Histoire de l'IA en 3 minutes au plus : retenir
IA ⊃ apprentissage automatique ⊃ apprentissage profond, et les trois facteurs
(données, calcul, algorithmes). Le cœur de la leçon est le vocabulaire
observations × variables, cible, supervisé, classification vs régression :
il sert dans toutes les séances. -->

---

## Programmer ou apprendre

> « La science qui permet aux ordinateurs d'apprendre sans être explicitement
> programmés. » — Arthur Samuel, 1959

| Programmation classique | Apprentissage automatique |
|---|---|
| Un développeur écrit des **règles** explicites | On fournit des **données** et les **réponses** attendues |
| L'ordinateur les applique aux entrées | La machine **découvre les règles** |
| Exemple : « si IMC ≥ 30 alors obésité » | Exemple : prédire la malignité d'une tumeur à partir de milliers de biopsies déjà diagnostiquées |

L'objectif est de **prédire** pour un nouveau cas, à partir de cas passés
dont on connaît l'issue.

---

## Les données : observations × variables

Un jeu de données est un **tableau** :

- chaque **ligne** est une **observation** (un patient, une biopsie, un
  séjour) ; leur nombre $n$ est la taille de l'échantillon ;
- chaque **colonne** est une **variable** (*feature*) mesurée sur chaque
  observation ; leur nombre $p$ est la dimension.

| Patient | Âge | Créatinine (mg/dL) | Pression artérielle moyenne (mmHg) | Décédé à 2 mois |
|---|---|---|---|---|
| 1 | 80 | 0,7 | 132 | oui |
| 2 | 69 | 1,8 | 71 | non |
| 3 | 41 | 0,7 | 77 | non |

Ici $n = 3$ observations et $p = 3$ variables, plus la dernière colonne :
la **cible**, ce qu'on veut prédire. Ce sont les trois premiers patients du
défi de la séance (étude SUPPORT2), qui en compte 6 164 décrits par 28
variables.

---

## Trois façons d'apprendre

![Non supervisé, supervisé, par renforcement](assets/iam-s1-classification/types-apprentissage.png)

- **Non supervisé** (à gauche) : des données sans étiquette, on cherche une
  structure, par exemple des groupes de patients semblables (*clustering*).
- **Supervisé** (au centre) : des paires (variables, cible) ; on apprend à
  prédire la cible. C'est l'objet de ce cours.
- **Par renforcement** (à droite) : un agent agit dans un environnement et
  apprend des récompenses qu'il reçoit.

---

## L'apprentissage supervisé

On note $X \in \mathbb{R}^{n \times p}$ le tableau des variables
explicatives et $y$ la colonne cible. Une observation est un couple
$(x_i, y_i)$ : les $p$ mesures du patient $i$ et son issue.

![Tumeurs du sein sur deux variables](assets/iam-s1-classification/cancer-sein-deux-variables.png)

Les 569 biopsies du jeu *Breast Cancer Wisconsin* (TP 1), sur 2 de leurs
30 variables : chaque point est une tumeur, sa couleur est la cible. Apprendre,
c'est trouver une frontière qui sépare au mieux les deux couleurs.

---

## Classification ou régression

La nature de la cible fixe la tâche :

| | Classification | Régression |
|---|---|---|
| **Cible** | Une **catégorie** : $y \in \{1, \dots, K\}$ | Un **nombre** : $y \in \mathbb{R}$ |
| **Question** | Quelle classe ? | Combien ? |
| **Exemples médicaux** | Tumeur maligne ou bénigne ; vivant ou décédé à 2 mois ; stade d'une maladie | Durée de séjour ; dose ; tension artérielle à 6 mois |

Avec $K = 2$ classes, on parle de **classification binaire**. Les
séances 1 à 3 traitent des classifications binaires ; le projet, une
classification à 4 classes.

<!-- notes: Faire classer à la salle quelques exemples : diagnostic
(classification), prix d'un médicament (régression), reconnaissance d'une
lésion sur image (classification), concentration d'un marqueur
(régression). -->

---

## Les données d'abord

Un modèle reproduit ce que contiennent ses données d'entraînement :

- **volume** : assez d'exemples pour chaque situation à prédire ;
- **qualité** : les erreurs de saisie et de mesure dégradent fortement les
  performances ;
- **biais** : une distorsion systématique (un hôpital, une population) est
  apprise et reproduite ;
- **diversité** : les données doivent couvrir les cas réels rencontrés.

Préparer les données (valeurs manquantes, variables catégorielles, valeurs
aberrantes, mise à l'échelle) est l'objet de la séance 2.

---

## À retenir

- Apprentissage automatique : on fournit des exemples, la machine trouve la
  règle.
- Un jeu de données = $n$ observations × $p$ variables, plus une cible en
  apprentissage supervisé.
- Cible catégorielle → **classification** ; cible numérique → **régression**.
- Un modèle ne vaut que par ses données : volume, qualité, biais, diversité.

<!-- notes: Quiz oral : quelles sont les deux dimensions d'un jeu de données ?
Prédire la survie à 2 mois, est-ce une classification ou une régression ? -->
