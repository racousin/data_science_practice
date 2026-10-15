# Exercices — Dossiers patients

Vous allez manipuler des données cliniques avec les structures de la séance :
séries de glycémies et de tensions, dossier patient imbriqué, codes CIM-10,
interactions médicamenteuses, puis un fichier de constantes vitales de
réanimation que vous lirez et résumerez sans pandas. Comptez environ 75 minutes.
Livrable : le notebook complété, toutes les cellules de vérification affichant ✅
(l'exercice 10 est facultatif).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s3-exercices.ipynb)

Correction : [pym-s3-exercices-correction.ipynb](https://colab.research.google.com/github/racousin/ai_for_modecine/blob/main/python-medecine/notebooks/pym-s3-exercices-correction.ipynb) — à ouvrir après avoir cherché.

1. Moyenne, minimum et maximum d'une série de glycémies, sans bibliothèque.
2. Compréhensions de liste et slicing : filtrer les hyperglycémies, convertir en mmol/L.
3. Tensions artérielles en tuples, unpacking et `enumerate`.
4. Un dossier patient en dictionnaire imbriqué : modifier, `get`, résumer.
5. Compter des codes CIM-10 avec un dictionnaire, puis avec `Counter`.
6. Ensembles : symptômes communs, interactions médicamenteuses, patients distincts.
7. Trouvez l'erreur : supprimer dans une liste pendant qu'on la parcourt.
8. Lire `constantes_vitales.csv` avec `csv.DictReader` : température maximale par patient.
9. Écrire un fichier CSV de résumé avec `csv.writer`.
10. Pour aller plus loin : séries par patient et évolution de la température sur 48 h.

Trois items « Lire du code » (alias de listes, unpacking étoilé, parcours d'un
dictionnaire) se font sans clavier : prédire l'affichage, puis exécuter.

Hors de Colab : téléchargez le `.ipynb` (menu *File → Download* dans Colab, ou
depuis GitHub) et ouvrez-le dans Jupyter ou VS Code. Aucune installation n'est
nécessaire : le notebook n'utilise que la bibliothèque standard de Python ; le
fichier de données est téléchargé par le notebook lui-même.

<!-- notes: 75 minutes. Les exercices 1 à 3 prennent ~25 min ; faire une mise
en commun après « Lire du code 1 » (alias) : c'est le point de la séance qui
reste. Points de blocage attendus : exercice 4 (les doubles crochets
dossier["traitements"]["ramipril"]), exercice 7 (faire expliquer le décalage
d'indice au tableau avec la liste dessinée), exercice 8 (oubli de float(),
la comparaison de chaînes "38.8" > "100" donne un résultat faux sans erreur).
Si la cellule de téléchargement échoue (pare-feu), le fichier est aussi dans
python-medecine/donnees/ du dépôt racousin/ai_for_modecine. -->
