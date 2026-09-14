"""Renders DICTIONNAIRE.md: every shipped column, its meaning, unit, fill rate and examples.

Meanings are translated from ADEME's "DPE - Dictionnaire de données des jeux de
données" (the `DPE_dictionnaire_de_données_JDD.xlsx` attachment of the dataset)
and completed from the 3CL method where that dictionary is silent. The fill
rates and example values are computed by `prepare_data.py` from the shipped
`train.csv.gz`, so they describe exactly the file a student downloads.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from columns import JITTERED_SURFACES, KEPT, KEPT_GROUPS

_FLAG = "0/1"
_TEXT = "—"

_GEN = {
    "type_generateur": ("Type of heating generator {g} of installation {i} "
                        "(boiler, heat pump, electric heater, stove…).", _TEXT),
    "type_energie_generateur": ("Energy used by heating generator {g} of installation {i}.", _TEXT),
    "usage_generateur": ("Uses served by heating generator {g} of installation {i} "
                         "(heating, hot water, or both).", _TEXT),
    "description_generateur_chauffage": ("Label describing heating generator {g} of "
                                         "installation {i}.", _TEXT),
}
_INST = {
    "type_installation_chauffage": ("Type of heating installation {i} (individual, "
                                    "collective, collective multi-building).", _TEXT),
    "type_emetteur_installation_chauffage": ("Emitters of heating installation {i}: the "
                                             "emission and distribution system together "
                                             "(radiators, underfloor heating, air vents…).", _TEXT),
    "configuration_installation_chauffage": ("Configuration of heating installation {i} "
                                             "(single system, with base + backup, with "
                                             "solar…).", _TEXT),
    "description_installation_chauffage": ("Label describing heating installation {i}.", _TEXT),
    "surface_chauffee_installation_chauffage": ("Floor area heated by installation {i}.", "m²"),
    "facteur_couverture_solaire_saisi_installation_chauffage": (
        "Solar coverage factor of heating installation {i}, as entered by the "
        "diagnostician (share of the heating need covered by solar).", "fraction"),
}
_ECS_GEN = {
    "type_generateur": ("Type of hot-water generator {g} (water heater, boiler, heat pump…).", _TEXT),
    "type_energie_generateur": ("Energy used by hot-water generator {g}.", _TEXT),
    "usage_generateur": ("Uses served by hot-water generator {g} (hot water, or heating + "
                         "hot water).", _TEXT),
    "description_generateur": ("Label describing hot-water generator {g}.", _TEXT),
    "volume_stockage_generateur": ("Storage volume of hot-water generator {g}.", "L"),
    "cop_generateur": ("COP of the thermodynamic water heater {g}, storage efficiency "
                       "included (0–20). Only for heat-pump water heaters.", "—"),
    "date_installation_generateur": ("Installation period of the thermodynamic water "
                                     "heater {g}.", _TEXT),
}

MEANINGS: dict[str, tuple[str, str]] = {
    # dwelling and building
    "type_batiment": ("Building type. Here `maison` (house) or `appartement` (flat); "
                      "building-level DPEs are excluded.", _TEXT),
    "methode_application_dpe": ("Method used to establish the dwelling's DPE: house, flat, flat "
                                "generated from the building's DPE, flat from an RT2012 study.", _TEXT),
    "annee_construction": ("Year of construction of the property, when known.", "year"),
    "periode_construction": ("Construction period of the property (bands cut at the thermal "
                             "regulations).", _TEXT),
    "typologie_logement": ("Dwelling typology (T1 … T6: number of main rooms).", _TEXT),
    "surface_habitable_logement": ("Living area of the dwelling.", "m²"),
    "surface_habitable_immeuble": ("Total living area of the building, for a flat DPE with "
                                   "collective systems.", "m²"),
    "surface_tertiaire_immeuble": ("Total non-residential floor area of the building, counted "
                                   "when a collective installation also serves it.", "m²"),
    "hauteur_sous_plafond": ("Average ceiling height of the dwelling.", "m"),
    "nombre_niveau_logement": ("Number of levels of the dwelling.", "count"),
    "nombre_niveau_immeuble": ("Total number of levels of the building.", "count"),
    "nombre_appartement": ("Number of flats in the building, for a flat DPE with collective "
                           "systems.", "count"),
    "numero_etage_appartement": ("Floor number of the flat.", "floor"),
    "position_logement_dans_immeuble": ("Position of the dwelling in the building, in terms of "
                                        "floor (ground floor, intermediate, top).", _TEXT),
    "appartement_non_visite": ("1 if the flat was not visited, in a DPE generated from the "
                               "building's DPE (the least efficient individual systems of the "
                               "building are then applied).", _FLAG),
    "logement_traversant": ("1 if the dwelling is dual-aspect (windows on opposite façades).", _FLAG),
    "protection_solaire_exterieure": ("1 if the glazed façades (north excepted) have external "
                                      "solar protection.", _FLAG),
    "presence_brasseur_air": ("1 if the dwelling has ceiling fans.", _FLAG),
    "classe_inertie_batiment": ("Thermal inertia class of the building (light, medium, heavy, "
                                "very heavy).", _TEXT),
    "inertie_lourde": ("1 if the dwelling has heavy or very heavy thermal inertia.", _FLAG),
    # location and climate
    "code_departement_ban": ("Département of the geocoded address (a code, not a quantity).", "code"),
    "code_region_ban": ("Region of the geocoded address (a code, not a quantity).", "code"),
    "zone_climatique": ("Climate zone of the dwelling (H1a … H3), which sets the weather the "
                        "calculation assumes.", _TEXT),
    "classe_altitude": ("Altitude class of the dwelling.", _TEXT),
    # envelope
    "isolation_toiture": ("1 if the roof is insulated.", _FLAG),
    "qualite_isolation_murs": ("Insulation quality of the walls.", _TEXT),
    "qualite_isolation_plancher_haut_comble_amenage": ("Insulation quality of the roof / upper "
                                                       "floor, converted-attic part.", _TEXT),
    "qualite_isolation_plancher_haut_comble_perdu": ("Insulation quality of the roof / upper "
                                                     "floor, unconverted-attic part.", _TEXT),
    "qualite_isolation_plancher_haut_toit_terrasse": ("Insulation quality of the roof / upper "
                                                      "floor, flat-roof part.", _TEXT),
    "qualite_isolation_plancher_bas": ("Insulation quality of the lower floor.", _TEXT),
    "qualite_isolation_menuiseries": ("Insulation quality of the windows and doors.", _TEXT),
    # heating, whole dwelling
    "type_energie_principale_chauffage": ("Energy used by the main heating.", _TEXT),
    "type_generateur_chauffage_principal": ("Type of the main heating generator.", _TEXT),
    "type_installation_chauffage": ("Heating installation: individual, collective or mixed.", _TEXT),
    # hot water, whole dwelling
    "type_installation_ecs": ("Hot-water (ECS) installation: individual, collective or mixed.", _TEXT),
    "type_energie_principale_ecs": ("Energy used by the main hot-water system.", _TEXT),
    "type_generateur_chauffage_principal_ecs": ("Type of the main hot-water generator.", _TEXT),
    "type_installation_ecs_n1": ("Type of hot-water installation 1 (individual or collective).", _TEXT),
    "configuration_installation_ecs_n1": ("Configuration of hot-water installation 1 (single "
                                          "system, with solar, several systems).", _TEXT),
    "description_installation_ecs_n1": ("Label describing hot-water installation 1.", _TEXT),
    "nombre_logements_desservis_par_installation_ecs_n1": ("Number of dwellings served by "
                                                           "hot-water installation 1.", "count"),
    "surface_habitable_desservie_par_installation_ecs_n1": ("Living area served by hot-water "
                                                            "installation 1.", "m²"),
    "type_installation_solaire_n1": ("Type of solar installation (solar hot water only, solar "
                                     "hot water + heating…).", _TEXT),
    "facteur_couverture_solaire_saisi_n1": ("Solar coverage factor of the hot water, entered "
                                            "directly when it can be justified.", "fraction"),
    # ventilation, cooling, renewables
    "type_ventilation": ("Type of ventilation.", _TEXT),
    "surface_ventilee": ("Ventilated area; with a single ventilation system, the total living "
                         "area.", "m²"),
    "ventilation_posterieure_2012": ("1 if the ventilation system was installed after 2012.", _FLAG),
    "type_generateur_froid": ("Type of cooling generator.", _TEXT),
    "type_energie_climatisation": ("Energy used by the cooling generator.", _TEXT),
    "periode_installation_generateur_froid": ("Installation period of the cooling system.", _TEXT),
    "description_generateur_froid": ("Label describing the cooling generator.", _TEXT),
    "surface_climatisee": ("Air-conditioned area.", "m²"),
    "categorie_enr": ("Category of renewable-energy system present.", _TEXT),
    "systeme_production_electricite_origine_renouvelable": ("Renewable electricity production "
                                                            "systems present in the building.", _TEXT),
    "presence_production_pv": ("1 if there is photovoltaic production.", _FLAG),
    "surface_totale_capteurs_pv": ("Total area of photovoltaic panels (pro-rated for a flat on a "
                                   "collective installation).", "m²"),
    "nombre_module": ("Number of standard photovoltaic modules installed.", "count"),
}
for _i, _inum in (("n1", "1"), ("n2", "2")):
    for _stem, (_text, _unit) in _INST.items():
        MEANINGS[f"{_stem}_{_i}"] = (_text.format(i=_inum), _unit)
    for _g, _gnum in (("n1", "1"), ("n2", "2")):
        for _stem, (_text, _unit) in _GEN.items():
            MEANINGS[f"{_stem}_{_g}_installation_{_i}"] = (_text.format(g=_gnum, i=_inum), _unit)
for _g, _gnum in (("n1", "1"), ("n2", "2")):
    for _stem, (_text, _unit) in _ECS_GEN.items():
        MEANINGS[f"{_stem}_{_g}_ecs_n1"] = (_text.format(g=_gnum), _unit)

_HEADER = """# DICTIONNAIRE — the columns of `train.csv.gz` and `test.csv.gz`

Both files carry `id` and the {n} columns below, in this order; `train.csv.gz`
adds the target `classe_efg` (1 if the energy label is E, F or G, else 0).

The values are **exactly as ADEME publishes them**: French labels, codes stored
as numbers, empty cells, typing errors. Nothing has been cleaned or recoded.
Deciding what each column needs is the exercise; `EXPERTISE.md` gives the
reasons.

- **Meaning:** translated from ADEME's *Dictionnaire de données des jeux de
  données* (an attachment of the dataset
  [DPE Logements existants](https://data.ademe.fr/datasets/dpe03existant)).
- **Read as:** the dtype `pd.read_csv` gives the column. `number` columns can
  still be codes or flags; `text` columns can still hold an order.
- **Filled:** share of the {rows:,} rows of `train.csv.gz` with a value. An
  empty cell is often *structural* (a second installation that does not exist)
  rather than unknown.
- **Examples:** for text, the most frequent values and their share of the
  filled rows; for numbers, the minimum, median and maximum.
- Surfaces (marked †) have been multiplied, per row, by one random factor between
  0.98 and 1.02 and rounded to 0.1 m², so that rows cannot be looked up in the
  public base. Ratios between the surfaces of a row are kept (up to that rounding).
"""


def _fmt_num(x) -> str:
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.1f}" if abs(x) >= 1000 else f"{x:g}"


def _examples(s: pd.Series) -> str:
    filled = s.dropna()
    if filled.empty:
        return "*(always empty in this sample)*"
    if pd.api.types.is_numeric_dtype(s):
        return (f"min {_fmt_num(filled.min())} · median {_fmt_num(filled.median())} · "
                f"max {_fmt_num(filled.max())}")
    top = filled.astype(str).value_counts(normalize=True).head(3)
    parts = []
    for value, share in top.items():
        value = value if len(value) <= 60 else value[:57] + "…"
        value = value.replace("|", "\\|")
        parts.append(f"`{value}` ({share:.0%})")
    more = filled.nunique() - len(top)
    return " · ".join(parts) + (f" · *+{more} more*" if more > 0 else "")


def render(train: pd.DataFrame, out: Path) -> None:
    missing = [c for c in KEPT if c not in MEANINGS]
    extra = [c for c in MEANINGS if c not in KEPT]
    if missing or extra:
        raise SystemExit(f"dictionary out of sync with columns.KEPT: missing {missing}, extra {extra}")
    lines = [_HEADER.format(n=len(KEPT), rows=len(train))]
    for group, cols in KEPT_GROUPS.items():
        lines.append(f"\n## {group[0].upper() + group[1:]}\n")
        lines.append("| column | meaning | unit | read as | filled | examples |")
        lines.append("|---|---|---|---|---|---|")
        for col in cols:
            meaning, unit = MEANINGS[col]
            s = train[col]
            kind = "number" if pd.api.types.is_numeric_dtype(s) else "text"
            mark = " †" if col in JITTERED_SURFACES else ""
            lines.append(f"| `{col}`{mark} | {meaning} | {unit} | {kind} | "
                         f"{s.notna().mean():.1%} | {_examples(s)} |")
    lines.append("\n## Target (`train.csv.gz` only)\n")
    lines.append("| column | meaning | unit | read as | filled | examples |")
    lines.append("|---|---|---|---|---|---|")
    y = train["classe_efg"]
    lines.append(f"| `classe_efg` | 1 if the DPE label (`etiquette_dpe`) is E, F or G; 0 for A–D. "
                 f"The label is the worse of the energy and the greenhouse-gas classes. | {_FLAG} | "
                 f"number | {y.notna().mean():.1%} | {y.mean():.1%} are 1 |")
    out.write_text("\n".join(lines) + "\n")
