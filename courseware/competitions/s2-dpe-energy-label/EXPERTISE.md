# EXPERTISE — what the regulation says about these columns

The scorer's model is fixed: a default `LogisticRegression()`. It draws one
straight line through whatever numbers you give it. It cannot decide that a
code is a category, that a construction year of 1300 is a typing error, or that an empty
cell means "this dwelling has no second heating system". You decide that, and
the reasons are written down: the DPE is a regulated calculation, and the
documents that define it say how each input is used.

This brief collects the eleven facts that matter most for preprocessing, with
the choice each one justifies. It does not tell you how to write the code, and
it does not rank the choices: measuring them is your job. Column meanings are
in `DICTIONNAIRE.md`.

---

## 1. How the label is made: two thresholds, the worse one wins

The DPE label is computed by the 3CL method from the inputs the diagnostician
records. It produces two numbers per m² per year, and a class on each scale:

| class | primary energy (kWh/m²/yr) | greenhouse gas (kgCO₂/m²/yr) |
|---|---|---|
| A | ≤ 70 | ≤ 6 |
| B | ≤ 110 | ≤ 11 |
| C | ≤ 180 | ≤ 30 |
| D | ≤ 250 | ≤ 50 |
| E | ≤ 330 | ≤ 70 |
| F | ≤ 420 | ≤ 100 |
| G | above | above |

The final label is the **worse** of the two classes. The target here is
`classe_efg = 1` for E, F or G.

**What it justifies.**
- The calculated consumption, emissions, costs and heat losses are *not* in the
  data: the label is a threshold on them, so they would be the answer in
  disguise. Every column you receive is something the diagnostician observed.
- The **energy** used for heating and hot water matters beyond the building
  itself, and it matters in two directions. Electricity is converted to primary
  energy with a factor above 1 (2.3 in the 2021 method), so electric resistance
  heating is penalised on the energy scale. Fuel oil and gas emit far more CO₂
  than electricity, wood or most heat networks, so they are penalised on the
  greenhouse-gas scale. An energy is therefore not "good" or "bad" along a single
  axis: treat it as a category, and consider what it interacts with.

*Source: arrêté du 31 mars 2021, annexe 1 (3CL-DPE 2021 method); Guide DPE for
diagnosticians.*

## 2. Construction periods are the thermal regulations

France's thermal regulations came in steps: the first in 1974 (applied from
1975), then RT 1988, RT 2000, RT 2005 and RT 2012. The 3CL method assigns default
performance values **by construction period**, and the periods in
`periode_construction` are cut at those dates.

`periode_construction` is always filled. `annee_construction` is filled only when
the year is known, which is roughly half the time.

**What it justifies.** A missing year is not a reason to drop the row or to fill
it with a global median: the period tells you which regulation the building was
built under, which is what the calculation uses. Recovering a year, or an era,
from the period is a domain-grounded imputation. Whether the relationship
between age and label is a straight line is a separate question worth looking at.

*Source: arrêté du 31 mars 2021, annexe 1; SDES working paper n°60 (construction
date "en tranches en suivant l'évolution des réglementations thermiques").*

## 3. Unknown insulation: absent before 1975, present after

When the diagnostician cannot establish whether a wall, floor or roof is
insulated, the method falls back on defaults that depend on the period: before
1975 (no thermal regulation) the component is treated as uninsulated, with a
deliberately conservative heat-loss value; from 1975 on, insulation is assumed.
The October 2021 revision of the method was largely about these defaults, because
they drove many pre-1975 dwellings into F and G.

**What it justifies.** The same recorded insulation quality does not mean the
same thing for a 1960 house and a 1990 house. The label responds to the
**combination** of period and insulation, which a linear model can only use if
you give it the combination as a feature.

*Source: notice explicative of the arrêté du 8 octobre 2021 (DPE method changes).*

## 4. Insulation quality is an order

The six `qualite_isolation_*` columns take the values `insuffisante`, `moyenne`,
`bonne`, `très bonne`. These are ordered levels of one quantity, not unrelated
labels, and they read as text.

**What it justifies.** Text must become numbers before the model sees it; for
these columns the order is information you can keep. Several components describe
one envelope, and a column is empty when the dwelling has no such component
(no flat roof, no converted attic).

*Source: ADEME, Dictionnaire de données (Isolation); arrêté du 31 mars 2021.*

## 5. Codes are categories, not magnitudes

Energies, installation types, generators, emitters, ventilation types, climate
zones, altitude classes, départements and regions are **codes**. Some read as text
(`Gaz naturel`); some read as numbers (`code_departement_ban` is 59, 67, 69…).
Département 69 is not "more" than département 59.

**What it justifies.** A numeric code given to a linear model as a number gets
one coefficient, which asserts an ordering and a spacing that do not exist.
Categories need an encoding that gives each level its own effect. Some of these
columns have many rare levels, and the 300-column limit makes that a choice too.

*Source: ADEME, Dictionnaire de données and Énumérateurs des tables.*

## 6. Small dwellings consume more per m²

A small dwelling has more wall, window and roof per m² of living area, and more
hot water per m² because it is more densely occupied, so its consumption *per
m²* is higher. SDES reports 34% of dwellings under 30 m² labelled F or G against
13% above 100 m². Since 1 July 2024, the class thresholds of dwellings under
40 m² have been adjusted to offset part of this; every DPE in this dataset was
issued after that date, so one rule applies to all of them.

**What it justifies.** Surface acts on the label non-linearly, and differently
for flats and houses (SDES defines its surface bands separately for the two).

*Source: arrêté du 25 mars 2024 (small surfaces); SDES working paper n°60.*

## 7. Near-duplicates: flats generated from a building DPE

A flat's DPE can be "généré à partir des données DPE immeuble" (see
`methode_application_dpe`): one diagnosis of the building is copied to each
flat, with the same envelope, the same systems and usually the same label. The
train/test split already keeps every building on one side (dwellings sharing a
building DPE, an address or a building identifier), so copies do not leak
across. Inside the training set, a building with many flats weighs as many
times.

**What it justifies.** Which rows to learn from is yours to choose: the scorer
fits on the train ids you submit (at least 20,000 of them), and scores every test
id. Dropping copies, or rows you judge implausible, is allowed. Remember that the
test set keeps its copies: whatever the model learns from the rows you keep must
still make sense for the rows you removed.

*Source: ADEME, Dictionnaire de données (`numero_dpe_immeuble_associe`,
`appartement_non_visite`).*

## 8. Structural missingness

Most empty cells here are not "unknown". They are "not applicable":
- floor, position in the building and number of flats exist only for flats;
- photovoltaics, renewables and cooling are filled only when present;
- the second heating installation, and the second generator of an installation,
  only when there is one (about a fifth of dwellings have a second installation);
- the COP only for heat-pump water heaters;
- hot-water installation dates are period strings (`Avant 2010`, `2010-2014`,
  `A partir de 2015`), not years.

**What it justifies.** Imputing a median into "this dwelling has no cooling"
invents a cooling system. For each column, decide whether empty means absent,
zero, or unknown, and whether the fact that it is empty is itself worth a feature.

*Source: ADEME, Dictionnaire de données; 3CL method.*

## 9. Manifest errors

The values are raw entries, and ADEME states it does not correct them. SDES,
before modelling the same table, "supprime les doublons et traite les erreurs
manifestes". You will find, among others:
- surfaces below 8 m² or above 1,000 m² for a single dwelling;
- construction years before 1700 (the oldest is 1300), or inconsistent with the
  construction period recorded for the same dwelling;
- ceiling heights above 5 m, up to 249 m; dwellings of 341 levels;
- surfaces heated, ventilated or served by the hot water that are many times the
  dwelling's own, because the installation serves a whole building.

**What it justifies.** A single absurd value moves a linear model's coefficient
and, unscaled, its solver. Detecting, flagging and correcting implausible values
is a decision with a physical basis: you know what a dwelling looks like.

*Source: ADEME dataset notes ("données brutes saisies par les diagnostiqueurs");
SDES working paper n°60, annexe 2.*

## 10. Labels that embed an era

Many system labels carry a date inside the text: `Chaudière gaz à condensation
après 2015`, `VMC SF Hygro B après 2012`, `PAC air/air installée entre 2008 et
2014`, `insert installé avant 1990`. The method assigns efficiencies by
generation of equipment, so the date is part of what the label means.

**What it justifies.** Each such value is a category *and* a date. You can keep
it as a category, extract the era, extract the kind of equipment, or do several
of these. The 300-column limit and the rare levels are the trade-off.

*Source: ADEME, Énumérateurs des tables; arrêté du 31 mars 2021, annexe 1.*

## 11. The official model is a logit

The statistical service of the Ministry (SDES) estimates the energy labels of the
whole French housing stock from this very table with a **multinomial logit**:
construction date in thermal-regulation bands, surface bands defined separately
for flats and houses, energy, dwelling type, and a climate zone or département
effect. "Les périodes de construction et les tranches de surface ont un fort
effet explicatif."

**What it justifies.** The fixed logistic regression of this challenge is the
professionals' tool on this problem, and the features they give it are
constructed, not raw. It also tells you which constructions the experts rely on.

*Source: SDES working paper n°60, annexe 2.*

---

## Sources

- 3CL method, annexe 1 of the arrêté du 31 mars 2021:
  <https://rt-re-batiment.developpement-durable.gouv.fr/IMG/pdf/consolide_annexe_1_arrete_du_31_03_2021_relatif_aux_methodes_et_procedures_applicables.pdf>
- Notice of the October 2021 changes to the method:
  <https://www.ecologie.gouv.fr/sites/default/files/documents/notice_DPE.pdf>
- SDES working paper n°60, *Le parc de logements par classe de performance
  énergétique au 1er janvier 2022*:
  <https://www.statistiques.developpement-durable.gouv.fr/sites/default/files/2022-07/document_travail_60_parc_logements_dpe_juillet2022.pdf>
- Arrêté du 25 mars 2024 (thresholds for small surfaces):
  <https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000049446315>
- Guide DPE for diagnosticians:
  <https://www.ecologie.gouv.fr/sites/default/files/documents/Guide_pour_les_diagnostiqueurs_DPE.pdf>
- ADEME, *DPE Logements existants (depuis juillet 2021)*, with its data
  dictionary and enumerations (Licence Ouverte 2.0):
  <https://data.ademe.fr/datasets/dpe03existant>
