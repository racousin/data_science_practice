# Lire et écrire des fichiers

Les données cliniques arrivent sous forme de fichiers : exports CSV du
logiciel de soins, comptes rendus en texte. Avant pandas (séance 5), voyons
comment Python lit et écrit un fichier avec ses seuls outils de base, et
comment ranger son contenu dans des listes et des dictionnaires.

<!-- notes: 25 minutes. Dans Colab, montrer le panneau Fichiers (à gauche) :
c'est là qu'apparaît le fichier téléchargé ou écrit. Erreur classique :
FileNotFoundError parce que la cellule de téléchargement n'a pas été exécutée. -->

---

## Ouvrir un fichier avec `with`

`open(chemin, mode)` ouvre un fichier ; `with` le referme automatiquement à la fin du bloc, même en cas d'erreur.

```python
with open("compte_rendu.txt", encoding="utf-8") as f:
    texte = f.read()          # tout le fichier dans une chaîne
```

Le mode par défaut est `"r"` (lecture). `"w"` écrit en écrasant le fichier, `"a"` ajoute à la fin. On précise toujours `encoding="utf-8"` pour les accents.

---

## Lire ligne à ligne

Un fichier ouvert se parcourt comme une liste de lignes. Chaque ligne se termine par `"\n"`, que `strip()` retire.

```python
with open("constantes_vitales.csv", encoding="utf-8") as f:
    entete = f.readline().strip()
    for ligne in f:
        champs = ligne.strip().split(",")
        print(champs[0], champs[5])   # id_patient, temperature
```

`split(",")` découpe la ligne en une liste de **chaînes** : `"38.2"` doit encore être converti avec `float(...)`.

---

## Le module `csv`

Découper à la main échoue dès qu'un champ contient une virgule entre guillemets. Le module `csv` de la bibliothèque standard gère ces cas.

```python
import csv

with open("constantes_vitales.csv", encoding="utf-8", newline="") as f:
    lecteur = csv.reader(f)
    entete = next(lecteur)      # la première ligne
    for ligne in lecteur:       # chaque ligne est une liste de str
        print(ligne)
```

---

## `csv.DictReader` : une ligne = un dictionnaire

`DictReader` lit l'en-tête et renvoie chaque ligne sous forme de dictionnaire. On accède aux colonnes par leur nom.

```python
with open("constantes_vitales.csv", encoding="utf-8", newline="") as f:
    for ligne in csv.DictReader(f):
        if float(ligne["temperature"]) >= 38.0:
            print(ligne["id_patient"], ligne["heure"], ligne["temperature"])
```

On combine ensuite avec les structures de la séance : un dictionnaire `{patient: liste de mesures}`, un ensemble de patients fébriles…

---

## Écrire un fichier

En mode `"w"`, `write` écrit une chaîne ; on ajoute soi-même les retours à la ligne.

```python
with open("resume.txt", "w", encoding="utf-8") as f:
    f.write("Patients fébriles\n")
    for p in ["R03", "R07"]:
        f.write(f"- {p}\n")
```

Pour un CSV, `csv.writer` (ou `csv.DictWriter`) s'occupe des séparateurs.

```python
with open("maxima.csv", "w", encoding="utf-8", newline="") as f:
    ecrivain = csv.writer(f)
    ecrivain.writerow(["id_patient", "temp_max"])
    ecrivain.writerow(["R03", 38.8])
```

---

## Récupérer un fichier en ligne

Dans Colab, le fichier n'est pas sur la machine : on le télécharge d'abord avec la bibliothèque standard.

```python
from urllib.request import urlretrieve

BASE = ("https://raw.githubusercontent.com/racousin/"
        "ai_for_medicine/main/python-medecine/donnees/")
urlretrieve(BASE + "constantes_vitales.csv", "constantes_vitales.csv")
```

---

## À retenir

- `with open(chemin, mode, encoding="utf-8") as f:` ouvre et referme le fichier.
- Un fichier se parcourt ligne par ligne ; tout ce qu'on lit est du texte à convertir.
- `csv.reader` donne des listes, `csv.DictReader` des dictionnaires indexés par l'en-tête.
- À la séance 5, `pandas.read_csv` fera tout cela en une ligne — mais c'est ce mécanisme qui tourne dessous.
