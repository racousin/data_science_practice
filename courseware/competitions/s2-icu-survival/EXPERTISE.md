# EXPERTISE — what clinicians know about these columns

The scorer's model is fixed: a default `LogisticRegression()`. It draws one straight line
through whatever numbers you give it. It cannot decide that an age of 999 is not an age,
that a diagnosis is not a scale, or that an empty albumin cell means "the lab was not
ordered". You decide that, and the reasons are written down: SUPPORT was a prognostic study,
and its investigators documented how each value was collected, what a normal value is, and
what they did when a value was missing.

This brief collects the twelve facts that matter most for preprocessing, with the choice
each one justifies. It does not rank them: measuring them is your job. Column meanings are
in `DICTIONARY.pdf`.

---

## 1. The cohort and the outcome

SUPPORT (Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments)
followed 9,105 seriously ill adults admitted to five US academic medical centres between
1989 and 1994. A patient entered the study on the hospital day when one of the qualifying
diagnoses was met (fact 6). The physiology you receive is the **worst value recorded on the
third day after study entry**, as the investigators collected it for their prognostic model.
The label is the patient's vital status two months (60 days) after entry: `dead = 1` if the
patient died within that window.

**What it justifies.** Every column is something known at the bedside on day 3, with one
exception (fact 10). Each row is one patient; the site is a hospital, not a quantity.

*Source: Knaus et al., Ann Intern Med 1995; hbiostat.org/data/repo/supportdesc.*

## 2. A lab that was not ordered is presumed normal

A test is ordered when the physician wants its result. When none was ordered, the
investigators did not treat the value as unknown: they filled it with a normal value, and
they published the values they used.

| lab | SUPPORT fill-in | unit |
|---|---|---|
| albumin (`alb`) | 3.5 | g/dL |
| PaO2/FiO2 (`pafi`) | 333.3 | mmHg |
| bilirubin (`bili`) | 1.01 | mg/dL |
| creatinine (`crea`) | 1.01 | mg/dL |
| blood urea nitrogen (`bun`) | 6.51 | mg/dL |
| white blood cells (`wblc`) | 9 | ×10⁹/L |
| urine output (`urine`) | 2502 | mL/day |

Arterial pH and glucose are not in their list. The textbook normals are pH 7.40 (reference
range 7.35 to 7.45) and glucose about 100 mg/dL (fasting 70 to 100).

The median of the cohort is not a normal value, because the cohort is sick: in the source
file the median albumin is 2.9 g/dL (normal 3.5 to 5.0), the median BUN is 23 mg/dL (normal
7 to 20), the median PaO2/FiO2 is 224 (above 400 on room air in a healthy adult).

**What it justifies.** Filling an empty lab with the cohort's median asserts that the
untested patient looks like the tested ones. The investigators' own choice was the opposite.
The table gives the domain fill, the median gives the generic one; which to use is yours to
measure.

*Source: hbiostat.org/data/repo/supportdesc, "Baseline physiologic variables"; Harrison's
Principles of Internal Medicine, appendix of laboratory values.*

## 3. Which absences carry information

Not every empty cell is the same kind of empty.

- **The patient could not answer.** `adlp` is the number of activities of daily living the
  *patient* said they could not do. It is empty when the patient could not be interviewed on
  day 3: intubated, comatose, delirious. `adls` is the same count reported by a *surrogate*
  (usually a family member), which is why it is filled far more often. The SUPPORT team
  built a single ADL score by taking the patient's answer when it existed and the
  surrogate's otherwise.
- **The arterial blood gas.** `pafi` and `ph` come from the same arterial sample, drawn when
  there is a respiratory or acid-base concern. They are missing together, and the absence
  means no such concern was raised on day 3.
- **The liver panel.** `alb` and `bili` are ordered as one panel.
- **The metabolic work-up.** `glucose`, `bun` and the 24-hour `urine` collection are missing
  together, as a set.
- **The interview.** `edu` and `income` are filled only when the social interview took
  place.

**What it justifies.** For each column, decide whether empty means "unknown", "not ordered
because nothing suggested it", or "the patient could not speak", and whether the fact of the
absence deserves a column of its own.

*Source: hbiostat.org/data/repo/supportdesc (adlp, adls, adlsc); Knaus et al. 1995.*

## 4. Real extremes versus impossible values

Seriously ill patients have extreme values, and most extremes in this file are real. A few
are not measurements at all. The reference range and the physiology tell them apart.

| column | normal (adult) | extreme but real | not a measurement |
|---|---|---|---|
| `age` | 18 and over in this cohort | 100 | 999 (placeholder for "unknown") |
| `temp` | 36.5 to 37.5 °C | 31.7 °C (severe hypothermia); 41.7 °C | 0.0 |
| `hrt` | 60 to 100 /min | 0 (arrest); 300 (a flutter conducted 1:1 is about the ceiling) | well above 300 |
| `meanbp` | 70 to 100 mmHg | 0 (circulatory arrest); 195 (hypertensive crisis) | |
| `wblc` | 4 to 11 ×10⁹/L | 0 (marrow aplasia after chemotherapy); 200 (leukaemia) | |
| `bili` | 0.1 to 1.2 mg/dL | 60 (fulminant liver failure) | |
| `crea` | 0.6 to 1.2 mg/dL | 20 (untreated end-stage kidney failure, dialysis level) | |
| `glucose` | 70 to 100 mg/dL fasting | 1000 (hyperosmolar hyperglycaemic state) | |
| `sod` | 135 to 145 mEq/L | about 110 to 180 is the range seen in the living | 1370 (a decimal slip for 137.0) |
| `urine` | 800 to 2000 mL/day | 0 (anuria); 9000 (polyuria) | |

**What it justifies.** A clipping rule written without the physiology removes real patients:
the zero mean arterial pressure and the white cell count of 200 belong to the sickest people
in the cohort. A placeholder left in place is a different error: one 999 in `age` is a lever on
a straight line.

*Source: Harrison's Principles of Internal Medicine, appendix; MSD Manual Professional
Edition, "Normal laboratory values".*

## 5. Sites chart in different units

The five hospitals chart in their own conventions.

- **Creatinine**: sites A, B and C report mg/dL; **sites D and E report µmol/L**. 1 mg/dL =
  88.4 µmol/L. A normal creatinine reads 1.0 at site A and 88 at site D.
- **Temperature**: sites A, B, D and E chart °C; **site C charts °F**. °F = °C × 9/5 + 32. A
  fever of 38.5 °C reads 101.3 at site C.

**What it justifies.** Any threshold, ratio or comparison across sites is meaningless until
the units agree; the SIRS cut-offs and the BUN/creatinine ratio of fact 7 assume °C and
mg/dL. Harmonise first, then look for extremes: a creatinine of 700 is an impossible mg/dL
value and an ordinary µmol/L one.

*Source: site conventions in `DICTIONARY.pdf`; conversion factors, MSD Manual.*

## 6. Diagnoses are categories, severities are orders, sex and race are neither

`dzgroup` is the qualifying diagnosis, in eight groups: ARF/MOSF w/Sepsis (acute respiratory
failure, or multiple organ system failure with sepsis); MOSF w/Malig (multiple organ system
failure in a patient with a malignancy); Coma (non-traumatic); CHF (congestive heart
failure); COPD (chronic obstructive pulmonary disease, acute exacerbation); Cirrhosis (with
a decompensation); Colon Cancer (with metastases); Lung Cancer (non-small-cell, advanced
stage).

These are not points on one scale: a coma and a lung cancer kill through different
mechanisms and at different speeds. `dzclass` groups the same column into four: ARF/MOSF =
{ARF/MOSF w/Sepsis, MOSF w/Malig}; COPD/CHF/Cirrhosis = {COPD, CHF, Cirrhosis}; Cancer =
{Colon Cancer, Lung Cancer}; Coma = {Coma}. It contains nothing that `dzgroup` does not.

Two columns *are* orders. `ca` is cancer status: none < non-metastatic < metastatic.
`income` has four brackets: under $11k < $11-$25k < $25-$50k < over $50k. `sex` and `race`
are nominal, and for a prognostic score they are not risk factors in themselves: the SUPPORT
model did not use them.

**What it justifies.** A diagnosis needs an encoding that gives each group its own effect. A
severity can keep its order as one number, or be split into categories; both cost different
numbers of columns. Encoding a nominal column as an order (male = 1, female = 2) asserts a
direction that does not exist.

*Source: Knaus et al. 1995 (entry criteria, model variables); hbiostat supportdesc.*

## 7. What clinicians compute at the bedside

Each of these is a formula on columns you have, with a published threshold.

- **SIRS** (ACCP/SCCM 1992): one point each for temperature > 38 or < 36 °C, heart rate > 90
  /min, respiratory rate > 20 /min, white cells > 12 or < 4 ×10⁹/L. Two or more points
  define SIRS; the count runs 0 to 4.
- **Shock index** = heart rate / systolic pressure (normal 0.5 to 0.7). Systolic pressure is
  not in this file; its variant, the **modified shock index** = heart rate / MAP, is (Liu et
  al. 2012: normal about 0.7 to 1.3, mortality rises above 1.3).
- **BUN/creatinine ratio**: above 20 suggests prerenal azotaemia (the kidney is
  under-perfused rather than damaged). Both values in mg/dL.
- **Berlin definition of ARDS** by PaO2/FiO2: mild 200 to 300, moderate 100 to 200, severe
  below 100; above 300 is not ARDS.
- **Oliguria**: fewer than 500 mL of urine in 24 hours; anuria fewer than 100 (KDIGO 2012
  states its criterion per hour, < 0.5 mL/kg/h, which for an adult lands in the same
  region).
- **Hypoalbuminaemia**: albumin below 3.5 g/dL (below 2.5 is severe). **Acidaemia**: pH
  below 7.35; alkalaemia above 7.45. **Dysnatraemia**: sodium outside 135 to 145 mEq/L, in
  either direction.
- **The ADL count** (`adlp`, `adls`): 0 to 7 basic self-care activities the patient cannot
  do unaided (bathing, dressing, eating, getting out of bed, walking). Higher is more
  dependent.
- **The SUPPORT coma score** (`scoma`): the Glasgow Coma Scale rescaled to 0 (fully
  conscious) to 100 (deep coma).

Many labs are **log-normal**: bilirubin, creatinine, BUN and white cells span two orders of
magnitude, and clinicians think in doublings, not differences (KDIGO stages kidney injury by
the *ratio* of creatinine to its baseline: 1.5×, 2×, 3×). A white cell count is abnormal in
both directions (fact 4).

**What it justifies.** A straight line cannot compute a ratio, a threshold or a logarithm
from raw columns. Each formula above is a candidate feature with a clinical reason behind
it; which ones pay, on this cohort, is what you measure.

*Source: Bone et al., Chest 1992; Liu et al., World J Emerg Med 2012; ARDS Definition Task
Force, JAMA 2012; KDIGO 2012; Knaus et al. 1995.*

## 8. Scales differ by five orders of magnitude

Hospital charges are in dollars (thousands to over a million), urine output in mL (0 to
9,000), PaO2/FiO2 in the hundreds, glucose up to 1,100, creatinine in µmol/L in the hundreds
at two sites. Arterial pH lives between 6.8 and 7.8, albumin between 0.4 and 5.

**What it justifies.** The scorer's solver (lbfgs) takes steps that the widest-scaled column
dominates, and it stops after 100 iterations whether or not it has arrived; its L2 penalty
also weighs a coefficient by the unit its column happens to be in. Scaling is a decision
about the solver, not about medicine, and a scaled placeholder is still a placeholder.

*Source: scikit-learn documentation, `LogisticRegression` (`lbfgs`, `max_iter=100`,
`C=1.0`).*

## 9. The same information twice

- `dzclass` is a function of `dzgroup` (fact 6).
- `bun` and `crea` both measure kidney function and rise together; their *ratio* is its own
  signal (fact 7).
- `adlp` and `adls` measure the same dependence from two reporters.

**What it justifies.** Two copies of one signal do not give a regularised line more to work
with: the penalty splits one coefficient between them, and the fit gets less stable. Where
two columns say the same thing, keep the better one, or a combination that says something
new.

*Source: hbiostat.org/data/repo/supportdesc; Knaus et al. 1995.*

## 10. Information that exists only after the outcome

`charges` is the total hospital bill for the stay. It is computed at discharge, and it grows
with the length of the stay and the intensity of care, so it also reflects how the stay
ended. For a patient still in the ward there is no bill yet. In `train.csv.gz` the column is
filled for 98 % of the patients; in `test.csv.gz` it is empty for every patient.

**What it justifies.** A model cannot use at prediction time what does not exist at
prediction time. A column that is strong on the training rows and absent on the test rows
makes the train AUC rise and the test AUC fall; the scorer reports both, and warns when they
part.

*Source: hbiostat.org/data/repo/supportdesc (charges); Knaus et al. 1995 (the model uses
day-3 data only).*

## 11. Age, comorbidities and the day of entry

`age` is the age at study entry. `num_co` counts the comorbid conditions recorded on entry
(0 to 9); `diabetes` and `dementia` are two of them singled out. `hday` is the hospital day
on which the patient met the entry criteria: most patients qualified on the day of
admission, a minority after weeks in hospital, up to 148 days.

**What it justifies.** Age and the comorbidity count are already numbers on sensible scales.
`hday` is not: the step from day 1 to day 3 does not mean what the step from day 101 to day
103 means, and a few long stays sit far from the rest.

*Source: hbiostat.org/data/repo/supportdesc; Knaus et al. 1995.*

## 12. The official model is a regression on these very columns

The SUPPORT prognostic model estimated each patient's probability of surviving two and six
months from the qualifying diagnosis, age, the number of days in hospital before entry, the
coma score, the comorbidities and the day-3 physiology. It was a Cox proportional-hazards
regression (the survival cousin of a logistic regression) with the transformation of each
variable chosen by the statisticians together with the clinicians. Sex and race were not
inputs.

**What it justifies.** The fixed logistic regression of this challenge is the professionals'
tool on this problem, given the professionals' variables. The features they gave it were
constructed, not raw; the raw columns are what the chart contained.

*Source: Knaus et al., Ann Intern Med 1995;122:191-203.*

---

## Sources

- Knaus WA, Harrell FE, Lynn J, et al. *The SUPPORT prognostic model: objective estimates of
  survival for seriously ill hospitalized adults.* Ann Intern Med 1995;122:191-203.
- Harrell FE. *SUPPORT dataset description* (variable definitions, fill-in values):
  <https://hbiostat.org/data/repo/supportdesc>
- Bone RC, Balk RA, Cerra FB, et al. *Definitions for sepsis and organ failure* (ACCP/SCCM
  consensus, SIRS criteria). Chest 1992;101:1644-1655.
- Liu YC, Liu JH, Fang ZA, et al. *Modified shock index and mortality rate of emergency
  patients.* World J Emerg Med 2012;3:114-117.
- ARDS Definition Task Force. *Acute respiratory distress syndrome: the Berlin definition.*
  JAMA 2012;307:2526-2533.
- KDIGO. *Clinical practice guideline for acute kidney injury.* Kidney Int Suppl
  2012;2:1-138.
- Harrison's Principles of Internal Medicine, 21st ed. (2022), appendix "Laboratory values
  of clinical importance"; MSD Manual Professional Edition, *Normal laboratory values*:
  <https://www.msdmanuals.com/professional/resources/normal-laboratory-values>
- SUPPORT2 on the UCI repository: <https://archive.ics.uci.edu/dataset/880/support2>
