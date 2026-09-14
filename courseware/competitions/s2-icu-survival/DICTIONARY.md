# DICTIONARY — the columns of `train.csv.gz` and `test.csv.gz`

Both files carry `id` and the 31 columns below, in this order; `train.csv.gz`
adds the target `dead` (1 if the patient died within 60 days of study entry,
else 0).

Units differ by site, spellings vary, a few cells hold placeholders, one
numeric column holds a text token, and labs are empty when nobody ordered
them. `EXPERTISE.pdf` gives the clinical reasons behind each column.

- **Meaning:** from the SUPPORT documentation
  ([hbiostat.org/data/repo/supportdesc](https://hbiostat.org/data/repo/supportdesc))
  and the charting conventions of the five sites.
- **Read as:** the dtype `pd.read_csv` gives the column. `number` columns can
  still be codes or flags; `text` columns can still hold an order, or numbers.
- **Filled:** share of the 6,373 rows of `train.csv.gz` with a value.
- **Distinct:** number of distinct values among the filled rows.
- **Examples:** for text, the distinct spellings actually present (all of them
  when there are few, else the most frequent) with their share of the filled
  rows; for numbers, the minimum, median and maximum.


## Identification & site

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `site` | Hospital site, A to E. SUPPORT ran in five US academic medical centres; each site charts in its own units (see `crea`, `temp`). | code | text | 100.0% | 5 | `A` (29%) · `B` (26%) · `D` (18%) · `C` (15%) · `E` (12%) |
| `hday` | Hospital day on which the patient entered the study (1 = the day of admission). | days | number | 100.0% | 80 | min 1 · median 1 · max 148 |

## Demographics & social

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `age` | Age at study entry. Some sites record 999 when the age is unknown. | years | number | 100.0% | 3,844 | min 18.04 · median 65.04 · max 999 |
| `sex` | Sex of the patient. Free-text entry: the spellings differ by site and by clerk (full word or initial, upper or lower case). | — | text | 100.0% | 8 | `male` (27%) · `female` (20%) · `Male` (14%) · `Female` (11%) · `M` (8%) · `m` (7%) · `F` (6%) · `f` (5%) |
| `race` | Race as recorded at admission: white, black, hispanic, asian, other. Free-text entry: case and trailing spaces vary. Site E writes 'unknown' where the other sites leave the cell empty. | — | text | 99.6% | 11 | `white` (68%) · `black` (13%) · `White ` (12%) · `hispanic` (3%) · *+7 more* |
| `edu` | Years of education. Empty when the social interview did not take place. | years | number | 82.0% | 31 | min 0 · median 12 · max 31 |
| `income` | Household income bracket, from the interview: under $11k, $11-$25k, $25-$50k, >$50k. Empty when the patient could not be asked or declined. | — | text | 66.7% | 4 | `under $11k` (47%) · `$11-$25k` (25%) · `$25-$50k` (17%) · `>$50k` (11%) |

## Diagnosis & comorbidity

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `dzgroup` | Qualifying diagnosis, one of eight groups: ARF/MOSF w/Sepsis, CHF, COPD, Cirrhosis, Colon Cancer, Coma, Lung Cancer, MOSF w/Malig (MOSF = multiple organ system failure; ARF = acute respiratory failure). | — | text | 100.0% | 8 | `ARF/MOSF w/Sepsis` (39%) · `CHF` (15%) · `COPD` (10%) · `Lung Cancer` (10%) · `MOSF w/Malig` (8%) · `Coma` (6%) · `Colon Cancer` (6%) · `Cirrhosis` (6%) |
| `dzclass` | Disease class: ARF/MOSF, COPD/CHF/Cirrhosis, Cancer, Coma. A grouping of `dzgroup` into four. | — | text | 100.0% | 4 | `ARF/MOSF` (47%) · `COPD/CHF/Cirrhosis` (31%) · `Cancer` (15%) · `Coma` (6%) |
| `num_co` | Number of comorbid conditions recorded at entry, 0 to 9. | count | number | 100.0% | 9 | min 0 · median 2 · max 8 |
| `ca` | Cancer status: no, yes (non-metastatic), metastatic. | — | text | 100.0% | 3 | `no` (66%) · `metastatic` (20%) · `yes` (14%) |
| `diabetes` | 1 if diabetes is among the comorbidities. | 0/1 | number | 100.0% | 2 | min 0 · median 0 · max 1 |
| `dementia` | 1 if dementia is among the comorbidities. | 0/1 | number | 100.0% | 2 | min 0 · median 0 · max 1 |

## Vital signs (day 3)

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `scoma` | SUPPORT coma score on day 3, derived from the Glasgow Coma Scale: 0 = fully conscious, 100 = deep coma. | 0-100 | number | 100.0% | 11 | min 0 · median 0 · max 100 |
| `meanbp` | Mean arterial blood pressure on day 3 (worst value of the day). | mmHg | number | 100.0% | 159 | min 0 · median 77 · max 195 |
| `hrt` | Heart rate on day 3 (worst value of the day). | /min | number | 100.0% | 171 | min 0 · median 100 · max 300 |
| `resp` | Respiratory rate on day 3 (worst value of the day). | /min | number | 100.0% | 65 | min 0 · median 24 · max 90 |
| `temp` | Body temperature on day 3 (worst value of the day). Charted in °C at sites A, B, D and E and in °F at site C. Some rows record 0.0 when the temperature was not taken. | °C or °F by site | number | 100.0% | 153 | min 0 · median 37.0938 · max 105.8 |

## Laboratory (day 3)

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `wblc` | White blood cell count on day 3. | ×10⁹/L (thousands/µL) | number | 97.7% | 453 | min 0 · median 10.5 · max 200 |
| `pafi` | PaO2/FiO2 ratio on day 3, from the arterial blood gas. Empty when no blood gas was drawn. | mmHg | number | 74.4% | 1,236 | min 31 · median 223.312 · max 890.375 |
| `alb` | Serum albumin on day 3. Empty when the liver panel was not ordered. | g/dL | number | 62.8% | 57 | min 0.4 · median 2.8999 · max 29 |
| `bili` | Serum bilirubin on day 3. Empty when the liver panel was not ordered. | mg/dL | number | 71.8% | 249 | min 0.1 · median 0.8999 · max 63 |
| `crea` | Serum creatinine on day 3. Reported in mg/dL at sites A, B and C and in µmol/L at sites D and E (1 mg/dL = 88.4 µmol/L). | mg/dL or µmol/L by site | number | 99.3% | 212 | min 0.2 · median 1.7 · max 1,140 |
| `sod` | Serum sodium on day 3. A few rows carry the value with the decimal point dropped (ten times too large). | mEq/L | number | 100.0% | 76 | min 110 · median 137 · max 1,530 |
| `ph` | Arterial pH on day 3, from the arterial blood gas. Empty when no blood gas was drawn. | — | number | 74.9% | 70 | min 6.9092 · median 7.4297 · max 7.7695 |
| `glucose` | Serum glucose on day 3. Empty when the chemistry panel was not ordered, except at site B, which writes the text 'not done' instead; pandas therefore reads the whole column as text. | mg/dL | text | 62.6% | 395 | min 0 · median 133 · max 1,092 · `not done` (20.1%) |
| `bun` | Blood urea nitrogen on day 3. Empty when the chemistry panel was not ordered. | mg/dL | number | 51.7% | 153 | min 1 · median 23 · max 300 |
| `urine` | Urine output on day 3, 24-hour collection. Empty when not collected; 0 is a measured value (anuria). | mL/24h | number | 46.1% | 1,223 | min 0 · median 1,975 · max 9,000 |

## Functional status

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `adlp` | Activities of daily living the patient reported being unable to do on day 3, 0 to 7. Empty when the patient could not be interviewed. | count 0-7 | number | 37.8% | 8 | min 0 · median 0 · max 7 |
| `adls` | The same count reported by the surrogate (family) on day 3, 0 to 7. Empty when no surrogate was interviewed. | count 0-7 | number | 68.2% | 8 | min 0 · median 1 · max 7 |

## Billing

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `charges` | Total hospital charges for the stay, billed at discharge; empty for patients not yet discharged. | USD | number | 98.1% | 6,083 | min 2,238 · median 42,748 · max 1,845,060 |

## Target (`train.csv.gz` only)

| column | meaning | unit | read as | filled | distinct | examples |
|---|---|---|---|---|---|---|
| `dead` | 1 if the patient died within 60 days (2 months) of study entry, 0 if alive at 60 days. | 0/1 | number | 100.0% | 2 | 46.2% are 1 |
