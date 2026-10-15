# Tests unitaires avec pytest

Une erreur de facteur dans une conversion d'unités ne provoque aucun message d'erreur :
le programme affiche un nombre faux, avec assurance. Les tests unitaires vérifient
automatiquement, à chaque modification, que chaque fonction rend le résultat attendu.

<!-- notes: 35 minutes. Montrer en direct un test qui passe, puis introduire un bug
et relancer pytest : la lecture du rapport d'échec est le moment clé. Les étudiants
bloquent sur les flottants (0.1 + 0.2 != 0.3) : ne pas éviter le piège, le provoquer,
puis introduire pytest.approx. -->

---

## Qu'est-ce qu'un test unitaire ?

Un test unitaire est une petite fonction qui vérifie le bon fonctionnement d'**une**
fonction du package, sur des cas dont on connaît la réponse.

```python
def simuler_glycemies(n):
    """n glycémies à jeun simulées, en g/L."""
    return np.random.normal(0.95, 0.10, n)
```

Comment la tester ? On vérifie ce qui doit être vrai quel que soit le tirage :
le résultat est-il un `np.ndarray` ? contient-il `n` valeurs ? sa moyenne est-elle
proche de 0,95 sur un grand échantillon ?

---

## Écrire un test

Trois conventions suffisent :

1. les tests sont rangés dans un dossier `tests/` ;
2. un **fichier de test par module** : `conversions.py` → `tests/test_conversions.py` ;
3. une **fonction de test par fonction testée** : `imc` → `test_imc`.

Un test appelle la fonction et vérifie le résultat avec `assert` :

```python
from medcalc.scores import imc

def test_imc():
    assert imc(80, 2.0) == 20
```

`assert condition` ne fait rien si la condition est vraie et lève `AssertionError`
sinon.

---

## Lancer pytest

`pytest` cherche les fichiers `test_*.py`, y exécute les fonctions `test_*`, et résume :

```bash
python -m pytest -q           # à la racine du projet
python -m pytest -v tests/test_scores.py   # un seul fichier, détaillé
```

```
..........                                   [100%]
10 passed in 0.03s
```

Un point par test réussi, `F` pour un échec. Le module `unittest` de la bibliothèque
standard fait la même chose avec une syntaxe à base de classes ; `pytest` est plus court
et lit aussi les tests `unittest`.

---

## Lire un échec

Une erreur a été introduite dans `hello` : la f-string a perdu ses accolades.

```python
def hello(name):
    return f" Hello name "

def test_hello():
    assert hello("Etienne") == " Hello Etienne "
```

![Rapport d'échec de pytest](assets/pym-s7-code-de-qualite/pytest_error.png)

Le rapport donne le test en échec (`test_hello`), la ligne (`>`), les deux valeurs
comparées (`E`) et leur différence (`-` attendu, `+` obtenu). On lit d'abord la
dernière ligne, puis on remonte.

---

## Plusieurs cas : `parametrize`

Plutôt que de copier un test pour chaque cas, on lui donne une table de cas. Exemple
avec le score qSOFA (Sepsis-3 : FR ≥ 22/min, PAS ≤ 100 mmHg, Glasgow < 15) :

```python
import pytest
from medcalc.scores import qsofa

@pytest.mark.parametrize("fr, pas, glasgow, attendu", [
    (16, 120, 15, 0),
    (22, 120, 15, 1),     # seuil exact : compte
    (24, 95, 13, 3),
])
def test_qsofa(fr, pas, glasgow, attendu):
    assert qsofa(fr, pas, glasgow) == attendu
```

Chaque ligne devient un test séparé dans le rapport. Les **valeurs seuils** sont les
cas les plus utiles : c'est là que se cachent les `>` écrits à la place de `>=`.

---

## Flottants et erreurs attendues

Les calculs en virgule flottante sont approchés : `0.1 + 0.2 == 0.3` vaut `False`.
On compare avec une tolérance grâce à `pytest.approx` :

```python
def test_glycemie_g_vers_mmol():
    assert glycemie_g_vers_mmol(1.0) == pytest.approx(5.551)
```

Une fonction doit aussi **refuser** les entrées absurdes. `pytest.raises` vérifie
qu'une exception est bien levée :

```python
def test_imc_taille_invalide():
    with pytest.raises(ValueError):
        imc(70, 0)
```

---

## Quelques conseils

- Des modules **par thème** (`conversions`, `scores`), une fonction par fonctionnalité.
- Pas de **code dupliqué** : une formule écrite deux fois sera corrigée une seule fois.
- Des tests **courts**, un comportement par test, avec des valeurs calculées à la main.
- La **couverture** mesure la part du code exécutée par les tests (plugin `pytest-cov`) :

```bash
python -m pytest --cov=medcalc
```

---

## À vous de jouer

- Écrivez des tests unitaires pour **chacune** des fonctions de votre package.
- Lancez-les avec `python -m pytest -q`.
- Introduisez volontairement un bug : vos tests le détectent-ils ?
