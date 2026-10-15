# Fonctions

Une fonction donne un nom à un calcul pour le réutiliser : on l'écrit une fois,
on la teste une fois, on l'appelle partout. Un score clinique, une conversion
d'unité ou une posologie sont naturellement des fonctions.

<!-- notes: 35 min. Points où ils calent : confondre print et return (la
fonction « affiche le bon résultat » mais renvoie None), l'ordre des
arguments positionnels, la portée (« pourquoi ma variable n'a pas changé ? »).
Faire prédire la sortie de l'exemple de portée avant de l'exécuter. -->

---

## Vous en utilisez déjà

`print`, `len`, `round` sont des fonctions : un nom, des arguments entre
parenthèses, un résultat.

```python
print("FC", 72, "bpm", sep=" | ")     # FC | 72 | bpm
round(37.8567, 1)                     # 37.9
```

La signature de `print` est `print(*objects, sep=' ', end='\n')` : des
arguments en nombre libre, puis des arguments **nommés** avec une valeur par
défaut.

---

## def et return

```python
def calculer_imc(poids_kg, taille_m):
    imc = poids_kg / taille_m ** 2
    return imc

calculer_imc(70, 1.75)    # 22.857...
```

`def` définit la fonction, ses **paramètres** sont entre parenthèses, et
`return` renvoie le résultat à l'appelant et termine la fonction. Sans `return`,
une fonction renvoie `None` : afficher un résultat avec `print` n'est pas le
renvoyer.

---

## Paramètres nommés et valeurs par défaut

La clairance de la créatinine selon Cockcroft et Gault (1976), créatinine en
mg/dL :

$$
Cl_{cr} = \frac{(140 - \mathrm{age}) \times \mathrm{poids}}{72 \times \mathrm{creatinine}}
$$

multipliée par 0,85 chez la femme.

```python
def clairance_cg(age, poids_kg, creatinine_mg_dl, femme=False):
    cl = (140 - age) * poids_kg / (72 * creatinine_mg_dl)
    if femme:
        cl = cl * 0.85
    return cl
```

```python
clairance_cg(70, 60, 1.2)                    # 48.6 mL/min (homme)
clairance_cg(70, 60, 1.2, femme=True)        # 41.3 mL/min
clairance_cg(age=70, creatinine_mg_dl=1.2, poids_kg=60)
```

Les arguments **positionnels** sont associés dans l'ordre ; les arguments
**nommés** peuvent être donnés dans n'importe quel ordre et rendent l'appel
lisible. Un paramètre avec valeur par défaut devient facultatif.

---

## Renvoyer plusieurs valeurs

`return a, b` renvoie un tuple, que l'appelant décompose en plusieurs variables.

```python
def dose_paracetamol(poids_kg):
    dose_mg = min(15 * poids_kg, 1000)   # 15 mg/kg/prise, 1 g max par prise
    dose_jour_mg = 4 * dose_mg           # 4 prises par 24 h
    return dose_mg, dose_jour_mg

dose, dose_jour = dose_paracetamol(12)   # 180, 720
```

---

## Docstring et annotations de type

```python
def convertir_glycemie(g_par_l: float) -> float:
    """Convertit une glycémie de g/L en mmol/L."""
    return g_par_l * 5.551
```

La **docstring** documente la fonction (pour le glucose, 1 g/L = 5,551 mmol/L) :
`help(convertir_glycemie)` l'affiche.
Les **annotations** (`: float`, `-> float`) indiquent les types attendus.
Python ne les vérifie pas : il est typé **dynamiquement** (le type d'une
variable est fixé à l'exécution et peut changer) mais **fortement** (il refuse
`"38" + 1` au lieu de convertir lui-même : `TypeError`).

---

## Portée des variables

Une variable créée dans une fonction est **locale** : elle n'existe que pendant
l'appel et ne modifie pas une variable **globale** de même nom.

```python
dose = 0
def calculer(poids):
    dose = 15 * poids     # variable locale
    return dose

calculer(20)    # 300
print(dose)     # 0 : la variable globale n'a pas changé
```

Écrire `dose += 1` dans la fonction provoque une `UnboundLocalError` : Python
voit une affectation, donc une variable locale, lue avant d'exister. Le mot-clé
`global dose` l'autoriserait, mais une fonction qui reçoit ses données en
paramètres et les renvoie avec `return` est plus sûre et plus facile à tester.

---

## Fonctions lambda

`lambda` définit une petite fonction anonyme d'une seule expression.

```python
mg_dl_vers_umol_l = lambda creat: creat * 88.4    # créatinine
mg_dl_vers_umol_l(1.2)                            # 106.08
```

On la rencontre surtout comme argument d'une autre fonction, par exemple pour
trier selon un critère : `sorted(patients, key=lambda p: p["age"])`
(séance 3). Pour tout ce qui a un nom et plus d'une ligne, on préfère `def`.

---

## À retenir

- `def nom(parametres):` puis `return` ; sans `return`, la fonction renvoie `None`.
- Arguments positionnels dans l'ordre, nommés dans n'importe quel ordre ; les valeurs par défaut rendent un paramètre facultatif.
- Une docstring et des annotations documentent la fonction, sans contrôle à l'exécution.
- Les variables d'une fonction sont locales : on passe les données en paramètre et on les récupère avec `return`.
