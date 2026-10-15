# Exercices — Calculs cliniques

Dans ce notebook, vous programmez les calculs que l'on fait chaque jour à la
main dans un service : IMC, conversions d'unités de biologie, clairance de la
créatinine, surface corporelle et dose, lecture d'une tension saisie en texte,
et un court compte rendu mis en forme. Comptez environ 1 h 15. Le livrable est
le notebook complété, dont toutes les cellules de vérification affichent
« ✅ ».

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s1-exercices.ipynb)

La [correction](https://colab.research.google.com/github/racousin/data_science_practice/blob/main/website/public/modules/python-medecine/notebooks/pym-s1-exercices-correction.ipynb)
est à ouvrir après avoir cherché.

1. Indice de masse corporelle, arrondi à une décimale.
2. Booléens : surpoids et obésité (classes OMS), hypertension (ESC/ESH 2018).
3. Conversions glycémie g/L ↔ mmol/L et créatininémie µmol/L ↔ mg/dL.
4. Lire du code : division entière, reste, flottants approchés.
5. Clairance de la créatinine de Cockcroft-Gault.
6. Surface corporelle de Mosteller et dose de chimiothérapie en mg/m².
7. Lire du code : découper un code CIM-10.
8. Extraire des valeurs d'une saisie texte (« TA: 142/91 mmHg », « 1,26 »).
9. Trouvez l'erreur : quatre bogues dans un calcul de clairance.
10. Un compte rendu en trois lignes avec des f-strings.
11. Pour aller plus loin (facultatif) : le DFG estimé par CKD-EPI 2021.

Chaque exercice donne la ligne `def` d'une fonction ; vous complétez la ligne
`return`. Les fonctions elles-mêmes sont l'objet de la séance 2.

Hors de Colab : ouvrez le `.ipynb` téléchargé avec `jupyter lab` ou VS Code
(bibliothèque standard seule, rien à installer).

<!-- notes: Faire les exercices 1 à 4 ensemble au tableau si la salle est
débutante (20 min), puis laisser travailler. Points de blocage attendus :
ex. 3, le sens de la conversion ; ex. 5, l'astuce `1 - 0.15 * femme` (rappeler
que True vaut 1) ; ex. 8, enchaîner deux split ; ex. 9, la priorité des
opérateurs (le bogue silencieux) — c'est le message principal de la séance :
un résultat sans erreur n'est pas forcément juste. Exercice 10 est la synthèse ;
11 est pour les rapides. -->
