# Colab, Jupyter et scripts

Ce cours se fait dans **Google Colab** : un notebook Jupyter hébergé par
Google, qui s'ouvre dans le navigateur, sans rien installer. Cette leçon
explique comment on y exécute du code, en quoi un notebook diffère d'un
script, puis comment installer Python sur sa propre machine pour la suite.

<!-- notes: 25 minutes. Faire ouvrir le notebook d'exercices de la séance dès
maintenant (badge sur la page Exercices) : tout le monde doit avoir une cellule
qui affiche quelque chose avant la fin de la leçon. Blocage fréquent : compte
Google universitaire qui refuse Colab → utiliser un compte personnel. -->

---

## Google Colab : un notebook en ligne

Un **notebook** est une suite de **cellules** : des cellules de texte
(Markdown) qui expliquent, et des cellules de code que l'on exécute une par
une avec **Maj + Entrée**. Le résultat s'affiche sous la cellule.

```python
frequence_cardiaque = 88
frequence_cardiaque * 60 * 24      # battements par jour
```

La dernière expression d'une cellule est affichée automatiquement (`126720`) ;
pour afficher autre chose, on utilise `print(...)`.

Pour ouvrir un notebook du cours : cliquer sur le badge *Open in Colab*, puis
**Fichier → Enregistrer une copie dans Drive**, sinon vos modifications sont
perdues à la fermeture.

---

## L'état du notebook

Derrière le notebook tourne un interpréteur Python (le *noyau*, ou
*runtime*). Les variables créées dans une cellule restent en mémoire pour les
cellules suivantes, **dans l'ordre où on les exécute**, pas dans l'ordre où
elles sont écrites.

```python
dose_mg = 500
```
```python
dose_mg = dose_mg * 2     # exécutée deux fois : 2000, pas 1000
```

Le numéro `[3]` à gauche d'une cellule indique l'ordre d'exécution. En cas de
doute : **Exécution → Redémarrer la session et tout exécuter**. Un notebook
correct s'exécute de haut en bas sans erreur.

<!-- notes: Faire l'expérience en direct : exécuter la seconde cellule trois
fois. C'est la source principale de "ça marchait tout à l'heure". -->

---

## Trois façons d'exécuter du code

![Shell interactif, script et notebook](assets/pym-s1-premiers-pas/trois-facons-executer.png)

Le même code peut être tapé dans un **shell** interactif (`python` ou
`ipython` dans un terminal), enregistré dans un **script** `imc.py` puis lancé
par `python imc.py`, ou découpé en cellules de **notebook**.

Le notebook sert à explorer et à expliquer une analyse ; le script sert à
produire un programme réutilisable (séance 7). Dans un script, rien ne
s'affiche sans `print`.

---

## Installer Python sur sa machine

Colab suffit pour ce cours ; une installation locale devient utile pour
travailler hors ligne ou sur des données qui ne doivent pas quitter
l'hôpital. Trois options :

- **Anaconda** : Python et des centaines de bibliothèques déjà installées
  (environ 4 Go), avec une interface graphique (*Anaconda Navigator*).
- **Miniconda** : la même chose sans les bibliothèques (environ 500 Mo).
- **python.org** (ou `brew install python`, `sudo apt install python3 python3-pip`) :
  Python seul, on installe ensuite ce dont on a besoin.

On vérifie ensuite dans un terminal :

```bash
python --version      # ou python3 --version
```

<!-- notes: Ne pas faire installer en séance. Sous Windows, conseiller Anaconda
(ou WSL pour ceux qui veulent un vrai terminal Linux). -->

---

## Environnements et paquets

Un **environnement virtuel** isole les bibliothèques d'un projet de celles
d'un autre. Avec l'outil standard `venv` et l'installeur `pip` :

```bash
python -m venv mon_env
source mon_env/bin/activate      # Windows : mon_env\Scripts\activate
pip install pandas jupyterlab
deactivate
```

Avec conda, l'équivalent est
`conda create --name mon_env python=3.12`, puis
`conda activate mon_env` et `conda install pandas` (ou `pip install`).

Dans l'environnement activé, `jupyter lab` (ou `jupyter notebook`) ouvre
Jupyter dans le navigateur, à l'adresse `localhost:8888` : c'est l'équivalent
local de Colab. Dans Colab, on installe une bibliothèque avec `!pip install …`
dans une cellule.

---

## Le terminal et les éditeurs

Quelques commandes de terminal (Linux, macOS, WSL sous Windows) suffisent :

| Commande | Rôle | Exemple |
|---|---|---|
| `pwd`, `ls`, `cd` | où suis-je, lister, se déplacer | `cd ~/analyses` |
| `mkdir`, `cp`, `mv`, `rm` | créer, copier, déplacer, supprimer | `cp brut.csv copie.csv` |
| `head`, `tail`, `less` | voir le début, la fin, tout un fichier | `head -n 5 sejours.csv` |
| `grep` | chercher un texte dans un fichier | `grep "I21" diagnostics.csv` |
| `python`, `pip` | lancer Python, installer un paquet | `python imc.py` |

Pour écrire des scripts, on utilise un **IDE** (environnement de
développement intégré) : **VS Code** ou **PyCharm** colorent le code,
complètent les noms, signalent les erreurs et ouvrent aussi les notebooks.
D'autres outils viendront plus tard : `pytest` pour tester, `pylint` pour
relire (séance 7).

---

## À retenir

- Dans Colab, une cellule s'exécute avec Maj + Entrée ; enregistrer une copie
  dans Drive.
- Les variables vivent dans le noyau, dans l'ordre d'exécution : en cas de
  doute, redémarrer et tout exécuter.
- Shell pour essayer, notebook pour explorer, script pour réutiliser.
- En local : Anaconda ou python.org, un environnement par projet (`venv` ou
  conda), `pip install` pour les bibliothèques.
