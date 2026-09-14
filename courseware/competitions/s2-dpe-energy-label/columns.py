"""The leakage audit of every column of ADEME's "DPE Logements existants".

Imported by `prepare_data.py` (the drop audit and its assertions) and by the
local tests. Every source column must land in exactly one of KEPT or DROPPED;
`classify()` raises on an unclassified or doubly classified column, and on a
listed name that the source does not have (a typo would otherwise pass as
"dropped").

The rule behind the split: **ship what the diagnostician observed; drop what
the 3CL software calculated.** The label is a threshold on the calculated
consumption and emissions, so anything downstream of the inputs is the label
in disguise. The reasons per group are in the challenge spec
(`tests-dummy/catalog/dpe_energy_label/COMPETITION.md` in the platform repo).
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# kept — what the diagnostician observed, shipped raw, in this order
# --------------------------------------------------------------------------- #
KEPT_GROUPS: dict[str, list[str]] = {
    "dwelling and building": [
        "type_batiment", "methode_application_dpe", "annee_construction",
        "periode_construction", "typologie_logement", "surface_habitable_logement",
        "surface_habitable_immeuble", "surface_tertiaire_immeuble",
        "hauteur_sous_plafond", "nombre_niveau_logement", "nombre_niveau_immeuble",
        "nombre_appartement", "numero_etage_appartement",
        "position_logement_dans_immeuble", "appartement_non_visite",
        "logement_traversant", "protection_solaire_exterieure",
        "presence_brasseur_air", "classe_inertie_batiment", "inertie_lourde",
    ],
    "location and climate": [
        "code_departement_ban", "code_region_ban", "zone_climatique", "classe_altitude",
    ],
    "envelope": [
        "isolation_toiture", "qualite_isolation_murs",
        "qualite_isolation_plancher_haut_comble_amenage",
        "qualite_isolation_plancher_haut_comble_perdu",
        "qualite_isolation_plancher_haut_toit_terrasse",
        "qualite_isolation_plancher_bas", "qualite_isolation_menuiseries",
    ],
    "heating": [
        "type_energie_principale_chauffage", "type_generateur_chauffage_principal",
        "type_installation_chauffage",
        *[c for inst in ("n1", "n2") for c in (
            f"type_installation_chauffage_{inst}",
            f"type_emetteur_installation_chauffage_{inst}",
            f"configuration_installation_chauffage_{inst}",
            f"description_installation_chauffage_{inst}",
            f"surface_chauffee_installation_chauffage_{inst}",
            f"facteur_couverture_solaire_saisi_installation_chauffage_{inst}",
            *[g for gen in ("n1", "n2") for g in (
                f"type_generateur_{gen}_installation_{inst}",
                f"type_energie_generateur_{gen}_installation_{inst}",
                f"usage_generateur_{gen}_installation_{inst}",
                f"description_generateur_chauffage_{gen}_installation_{inst}",
            )],
        )],
    ],
    "hot water": [
        "type_installation_ecs", "type_energie_principale_ecs",
        "type_generateur_chauffage_principal_ecs",
        "type_installation_ecs_n1", "configuration_installation_ecs_n1",
        "description_installation_ecs_n1",
        "nombre_logements_desservis_par_installation_ecs_n1",
        "surface_habitable_desservie_par_installation_ecs_n1",
        "type_installation_solaire_n1", "facteur_couverture_solaire_saisi_n1",
        *[g for gen in ("n1", "n2") for g in (
            f"type_generateur_{gen}_ecs_n1",
            f"type_energie_generateur_{gen}_ecs_n1",
            f"usage_generateur_{gen}_ecs_n1",
            f"description_generateur_{gen}_ecs_n1",
            f"volume_stockage_generateur_{gen}_ecs_n1",
            f"cop_generateur_{gen}_ecs_n1",
            f"date_installation_generateur_{gen}_ecs_n1",
        )],
    ],
    "ventilation, cooling, renewables": [
        "type_ventilation", "surface_ventilee", "ventilation_posterieure_2012",
        "type_generateur_froid", "type_energie_climatisation",
        "periode_installation_generateur_froid", "description_generateur_froid",
        "surface_climatisee", "categorie_enr",
        "systeme_production_electricite_origine_renouvelable",
        "presence_production_pv", "surface_totale_capteurs_pv", "nombre_module",
    ],
}
KEPT: list[str] = [c for cols in KEPT_GROUPS.values() for c in cols]

# Every surface column of a row is multiplied by ONE random factor in
# [0.98, 1.02] (re-identification cost; ratios and equalities survive).
JITTERED_SURFACES: list[str] = [
    "surface_habitable_logement", "surface_habitable_immeuble",
    "surface_chauffee_installation_chauffage_n1",
    "surface_chauffee_installation_chauffage_n2",
    "surface_ventilee", "surface_habitable_desservie_par_installation_ecs_n1",
    "surface_climatisee",
]

# --------------------------------------------------------------------------- #
# dropped — the label, calculation outputs, identifiers, dates, location
# --------------------------------------------------------------------------- #
_USES = ("chauffage", "ecs", "refroidissement", "eclairage", "auxiliaires")

DROPPED_GROUPS: dict[str, list[str]] = {
    # final class = worse of the energy and GHG classes
    "label": ["etiquette_dpe", "etiquette_ges"],
    # the energy scale is a threshold on these
    "consumption": [
        "conso_5_usages_ep", "conso_5_usages_par_m2_ep",
        *[f"conso_{u}_ep" for u in _USES],
        "conso_5_usages_ef", "conso_5_usages_par_m2_ef",
        *[f"conso_{u}_ef" for u in _USES],
        *[f"conso_{what}_ef_energie_{n}" for n in ("n1", "n2", "n3")
          for what in ("5_usages", "chauffage", "ecs")],
        "conso_chauffage_installation_chauffage_n1",
        "conso_chauffage_installation_chauffage_n2",
        *[f"conso_chauffage_generateur_{g}_installation_{i}"
          for i in ("n1", "n2") for g in ("n1", "n2")],
        "conso_ef_installation_ecs_n1",
        "conso_ef_generateur_n1_ecs_n1", "conso_ef_generateur_n2_ecs_n1",
        "conso_refroidissement_annuel",
    ],
    # the GHG scale is a threshold on these
    "emissions": [
        "emission_ges_5_usages", "emission_ges_5_usages_par_m2",
        *[f"emission_ges_{u}" for u in _USES],
        *[f"emission_ges_{what}_energie_{n}" for n in ("n1", "n2", "n3")
          for what in ("5_usages", "chauffage", "ecs")],
    ],
    # consumption x tariffs
    "costs": [
        "cout_total_5_usages", *[f"cout_{u}" for u in _USES],
        *[f"cout_{what}_energie_{n}" for n in ("n1", "n2", "n3")
          for what in ("total_5_usages", "chauffage", "ecs")],
    ],
    # intermediate results, one step before consumption
    "heat balance": [
        "deperditions_enveloppe", "deperditions_ponts_thermiques",
        "deperditions_murs", "deperditions_planchers_hauts",
        "deperditions_planchers_bas", "deperditions_portes",
        "deperditions_baies_vitrees", "deperditions_renouvellement_air",
        "ubat_w_par_m2_k", "besoin_chauffage", "besoin_ecs",
        "besoin_refroidissement", "besoin_ecs_batiment", "besoin_ecs_logement",
        "apport_interne_saison_chauffe", "apport_interne_saison_froide",
        "apport_solaire_saison_chauffe", "apport_solaire_saison_froide",
    ],
    # computed outputs of the engine
    "engine syntheses": [
        "qualite_isolation_enveloppe", "indicateur_confort_ete",
        "facteur_couverture_solaire_installation_chauffage_n1",
        "facteur_couverture_solaire_installation_chauffage_n2",
        "facteur_couverture_solaire_n1", "production_ecs_solaire_installation_n1",
        "production_electricite_pv_kwhep_par_an", "electricite_pv_autoconsommee",
    ],
    # ordered by consumption share, so an output
    "energy ranking": ["type_energie_n1", "type_energie_n2", "type_energie_n3"],
    # re-identification: identifiers
    "identifiers": [
        "numero_dpe", "numero_dpe_remplace", "numero_dpe_immeuble_associe",
        "id_rnb", "provenance_id_rnb", "numero_rpls_logement",
        "numero_immatriculation_copropriete",
    ],
    "dates": [
        "date_etablissement_dpe", "date_visite_diagnostiqueur", "date_reception_dpe",
        "date_derniere_modification_dpe", "date_fin_validite_dpe",
    ],
    "constant": ["version_dpe", "modele_dpe"],
    "address and geocoding": [
        "adresse_ban", "numero_voie_ban", "nom_rue_ban", "nom_commune_ban",
        "code_postal_ban", "code_insee_ban", "identifiant_ban",
        "coordonnee_cartographique_x_ban", "coordonnee_cartographique_y_ban",
        "score_ban", "statut_geocodage", "adresse_brut", "adresse_complete_brut",
        "nom_commune_brut", "code_postal_brut", "nom_residence",
        "complement_adresse_batiment", "complement_adresse_logement",
    ],
    # data-fair fields. `_score` is not in the schema: the /lines endpoint adds
    # it to every result row (the relevance of a full-text query, null here).
    "api fields": ["_geopoint", "_id", "_i", "_rand", "_score"],
}
DROPPED: list[str] = [c for cols in DROPPED_GROUPS.values() for c in cols]

# Fields that exist only in API responses, never in the published schema.
RESPONSE_ONLY = {"_score"}


def classify(source_columns) -> dict[str, str]:
    """Map every source column to 'kept' or 'dropped:<group>'. Fail fast.

    Raises SystemExit on: a duplicate inside KEPT or DROPPED, a column in both,
    a source column in neither, or a listed column the source does not have.
    """
    source = list(source_columns)
    problems = []
    for name, cols in (("KEPT", KEPT), ("DROPPED", DROPPED)):
        dup = sorted({c for c in cols if cols.count(c) > 1})
        if dup:
            problems.append(f"{name} lists these twice: {dup}")
    both = sorted(set(KEPT) & set(DROPPED))
    if both:
        problems.append(f"in both KEPT and DROPPED: {both}")
    unclassified = sorted(set(source) - set(KEPT) - set(DROPPED))
    if unclassified:
        problems.append(f"source columns in neither list (classify them): {unclassified}")
    unknown = sorted((set(KEPT) | set(DROPPED)) - set(source) - RESPONSE_ONLY)
    if unknown:
        problems.append(f"listed columns the source does not have (typo?): {unknown}")
    if problems:
        raise SystemExit("column audit failed:\n  " + "\n  ".join(problems))

    verdict = {c: "kept" for c in KEPT if c in source}
    for group, cols in DROPPED_GROUPS.items():
        for c in cols:
            if c in source:
                verdict[c] = f"dropped:{group}"
    return verdict
