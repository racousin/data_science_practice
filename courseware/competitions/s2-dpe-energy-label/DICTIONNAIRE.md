# DICTIONNAIRE — the columns of `train.csv.gz` and `test.csv.gz`

Both files carry `id` and the 99 columns below, in this order; `train.csv.gz`
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
- **Filled:** share of the 69,854 rows of `train.csv.gz` with a value. An
  empty cell is often *structural* (a second installation that does not exist)
  rather than unknown.
- **Examples:** for text, the most frequent values and their share of the
  filled rows; for numbers, the minimum, median and maximum.
- Surfaces (marked †) have been multiplied, per row, by one random factor between
  0.98 and 1.02 and rounded to 0.1 m², so that rows cannot be looked up in the
  public base. Ratios between the surfaces of a row are kept (up to that rounding).


## Dwelling and building

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `type_batiment` | Building type. Here `maison` (house) or `appartement` (flat); building-level DPEs are excluded. | — | text | 100.0% | `appartement` (72%) · `maison` (28%) |
| `methode_application_dpe` | Method used to establish the dwelling's DPE: house, flat, flat generated from the building's DPE, flat from an RT2012 study. | — | text | 100.0% | `dpe appartement individuel` (46%) · `dpe maison individuelle` (28%) · `dpe appartement généré à partir des données DPE immeuble` (26%) |
| `annee_construction` | Year of construction of the property, when known. | year | number | 50.9% | min 1400 · median 1983 · max 2025 |
| `periode_construction` | Construction period of the property (bands cut at the thermal regulations). | — | text | 100.0% | `1948-1974` (27%) · `avant 1948` (26%) · `2013-2021` (12%) · *+7 more* |
| `typologie_logement` | Dwelling typology (T1 … T6: number of main rooms). | — | text | 1.9% | `T2` (35%) · `T1` (27%) · `T3` (19%) · *+3 more* |
| `surface_habitable_logement` † | Living area of the dwelling. | m² | number | 100.0% | min 4.9 · median 65.8 · max 834.2 |
| `surface_habitable_immeuble` † | Total living area of the building, for a flat DPE with collective systems. | m² | number | 41.0% | min 14.7 · median 1975.5 · max 1456722.6 |
| `surface_tertiaire_immeuble` | Total non-residential floor area of the building, counted when a collective installation also serves it. | m² | number | 0.1% | min 9.2 · median 358 · max 792 |
| `hauteur_sous_plafond` | Average ceiling height of the dwelling. | m | number | 100.0% | min 0.4 · median 2.5 · max 210 |
| `nombre_niveau_logement` | Number of levels of the dwelling. | count | number | 100.0% | min 1 · median 1 · max 104 |
| `nombre_niveau_immeuble` | Total number of levels of the building. | count | number | 28.0% | min 1 · median 5 · max 52 |
| `nombre_appartement` | Number of flats in the building, for a flat DPE with collective systems. | count | number | 67.4% | min 1 · median 1 · max 1501 |
| `numero_etage_appartement` | Floor number of the flat. | floor | number | 90.2% | min -2 · median 0 · max 29 |
| `position_logement_dans_immeuble` | Position of the dwelling in the building, in terms of floor (ground floor, intermediate, top). | — | text | 1.9% | `RDC` (75%) · `étage intermédiaire` (22%) · `dernier étage` (3%) |
| `appartement_non_visite` | 1 if the flat was not visited, in a DPE generated from the building's DPE (the least efficient individual systems of the building are then applied). | 0/1 | number | 26.8% | min 0 · median 1 · max 1 |
| `logement_traversant` | 1 if the dwelling is dual-aspect (windows on opposite façades). | 0/1 | number | 75.3% | min 0 · median 1 · max 1 |
| `protection_solaire_exterieure` | 1 if the glazed façades (north excepted) have external solar protection. | 0/1 | number | 75.3% | min 0 · median 1 · max 1 |
| `presence_brasseur_air` | 1 if the dwelling has ceiling fans. | 0/1 | number | 75.3% | min 0 · median 0 · max 1 |
| `classe_inertie_batiment` | Thermal inertia class of the building (light, medium, heavy, very heavy). | — | text | 100.0% | `Moyenne` (33%) · `Lourde` (28%) · `Légère` (28%) · *+1 more* |
| `inertie_lourde` | 1 if the dwelling has heavy or very heavy thermal inertia. | 0/1 | number | 75.3% | min 0 · median 0 · max 1 |

## Location and climate

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `code_departement_ban` | Département of the geocoded address (a code, not a quantity). | code | number | 100.0% | min 13 · median 44 · max 84 |
| `code_region_ban` | Region of the geocoded address (a code, not a quantity). | code | number | 100.0% | min 32 · median 75 · max 93 |
| `zone_climatique` | Climate zone of the dwelling (H1a … H3), which sets the weather the calculation assumes. | — | text | 100.0% | `H1a` (13%) · `H2c` (13%) · `H1c` (13%) · *+5 more* |
| `classe_altitude` | Altitude class of the dwelling. | — | text | 100.0% | `inférieur à 400m` (99%) · `400-800m` (1%) · `supérieur à 800m` (0%) |

## Envelope

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `isolation_toiture` | 1 if the roof is insulated. | 0/1 | number | 74.7% | min 0 · median 1 · max 1 |
| `qualite_isolation_murs` | Insulation quality of the walls. | — | text | 100.0% | `insuffisante` (51%) · `bonne` (18%) · `très bonne` (18%) · *+1 more* |
| `qualite_isolation_plancher_haut_comble_amenage` | Insulation quality of the roof / upper floor, converted-attic part. | — | text | 16.0% | `insuffisante` (33%) · `bonne` (27%) · `très bonne` (27%) · *+1 more* |
| `qualite_isolation_plancher_haut_comble_perdu` | Insulation quality of the roof / upper floor, unconverted-attic part. | — | text | 52.3% | `très bonne` (65%) · `insuffisante` (16%) · `moyenne` (11%) · *+1 more* |
| `qualite_isolation_plancher_haut_toit_terrasse` | Insulation quality of the roof / upper floor, flat-roof part. | — | text | 26.1% | `insuffisante` (40%) · `très bonne` (39%) · `bonne` (15%) · *+1 more* |
| `qualite_isolation_plancher_bas` | Insulation quality of the lower floor. | — | text | 95.3% | `très bonne` (38%) · `insuffisante` (34%) · `bonne` (17%) · *+1 more* |
| `qualite_isolation_menuiseries` | Insulation quality of the windows and doors. | — | text | 100.0% | `moyenne` (41%) · `très bonne` (23%) · `bonne` (23%) · *+1 more* |

## Heating

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `type_energie_principale_chauffage` | Energy used by the main heating. | — | text | 100.0% | `Gaz naturel` (42%) · `Électricité` (41%) · `Réseau de Chauffage urbain` (11%) · *+9 more* |
| `type_generateur_chauffage_principal` | Type of the main heating generator. | — | text | 20.3% | `Chaudière gaz à condensation après 2015` (16%) · `Chaudière gaz à condensation 2001-2015` (12%) · `Panneau rayonnant électrique NFC, NF** et NF***` (9%) · *+104 more* |
| `type_installation_chauffage` | Heating installation: individual, collective or mixed. | — | text | 71.6% | `individuel` (64%) · `collectif` (30%) · `mixte (collectif-individuel)` (6%) |
| `type_installation_chauffage_n1` | Type of heating installation 1 (individual, collective, collective multi-building). | — | text | 100.0% | `installation individuelle` (76%) · `installation collective` (20%) · `installation collective multi-bâtiment : modélisée comme …` (3%) · *+1 more* |
| `type_emetteur_installation_chauffage_n1` | Emitters of heating installation 1: the emission and distribution system together (radiators, underfloor heating, air vents…). | — | text | 100.0% | `Radiateur bitube avec robinet thermostatique sur réseau i…` (13%) · `radiateur électrique NFC, NF** et NF***` (13%) · `Panneau rayonnant NFC, NF** et NF***` (9%) · *+45 more* |
| `configuration_installation_chauffage_n1` | Configuration of heating installation 1 (single system, with base + backup, with solar…). | — | text | 100.0% | `Installation de chauffage simple` (93%) · `Installation de chauffage avec insert ou poêle bois en ap…` (5%) · `Installation de chauffage avec une chaudière ou une PAC e…` (1%) · *+8 more* |
| `description_installation_chauffage_n1` | Label describing heating installation 1. | — | text | 99.9% | `Chauffage sans solaire` (9%) · `Convecteur électrique NFC, NF** et NF*** (système individ…` (3%) · `Radiateur électrique à inertie (modélisé comme un radiate…` (3%) · *+5708 more* |
| `surface_chauffee_installation_chauffage_n1` † | Floor area heated by installation 1. | m² | number | 100.0% | min 2.6 · median 74.6 · max 20172.5 |
| `facteur_couverture_solaire_saisi_installation_chauffage_n1` | Solar coverage factor of heating installation 1, as entered by the diagnostician (share of the heating need covered by solar). | fraction | number | 0.0% | min 0 · median 0.5 · max 0.8 |
| `type_generateur_n1_installation_n1` | Type of heating generator 1 of installation 1 (boiler, heat pump, electric heater, stove…). | — | text | 100.0% | `Chaudière gaz à condensation après 2015` (16%) · `Chaudière gaz à condensation 2001-2015` (10%) · `Panneau rayonnant électrique NFC, NF** et NF***` (9%) · *+133 more* |
| `type_energie_generateur_n1_installation_n1` | Energy used by heating generator 1 of installation 1. | — | text | 100.0% | `Gaz naturel` (42%) · `Électricité` (40%) · `Réseau de Chauffage urbain` (11%) · *+9 more* |
| `usage_generateur_n1_installation_n1` | Uses served by heating generator 1 of installation 1 (heating, hot water, or both). | — | text | 100.0% | `chauffage` (56%) · `chauffage + ecs` (44%) · `ecs` (0%) |
| `description_generateur_chauffage_n1_installation_n1` | Label describing heating generator 1 of installation 1. | — | text | 100.0% | `Gaz Naturel - Chaudière gaz à condensation installée à pa…` (13%) · `Gaz Naturel - Chaudière gaz à condensation installée entr…` (8%) · `Réseau de chaleur isolé` (7%) · *+266 more* |
| `type_generateur_n2_installation_n1` | Type of heating generator 2 of installation 1 (boiler, heat pump, electric heater, stove…). | — | text | 8.1% | `Chaudière gaz à condensation 2001-2015` (12%) · `insert installé entre 1990 et 2004` (7%) · `insert installé avant 1990` (7%) · *+86 more* |
| `type_energie_generateur_n2_installation_n1` | Energy used by heating generator 2 of installation 1. | — | text | 8.1% | `Bois – Bûches` (47%) · `Gaz naturel` (26%) · `Électricité` (18%) · *+3 more* |
| `usage_generateur_n2_installation_n1` | Uses served by heating generator 2 of installation 1 (heating, hot water, or both). | — | text | 8.1% | `chauffage` (83%) · `chauffage + ecs` (17%) |
| `description_generateur_chauffage_n2_installation_n1` | Label describing heating generator 2 of installation 1. | — | text | 8.1% | `Gaz Naturel - Chaudière gaz à condensation installée entr…` (11%) · `Bois - Insert installé avant 1990` (5%) · `Bois - Insert installé entre 1990 et 2004` (5%) · *+133 more* |
| `type_installation_chauffage_n2` | Type of heating installation 2 (individual, collective, collective multi-building). | — | text | 23.6% | `installation individuelle` (99%) · `installation collective` (1%) · `installation hybride collective-individuelle (chauffage b…` (0%) · *+1 more* |
| `type_emetteur_installation_chauffage_n2` | Emitters of heating installation 2: the emission and distribution system together (radiators, underfloor heating, air vents…). | — | text | 23.6% | `radiateur électrique NFC, NF** et NF***` (18%) · `Soufflage d'air chaud (air soufflé) avec distribution par…` (12%) · `Convecteur électrique NFC, NF** et NF***` (12%) · *+32 more* |
| `configuration_installation_chauffage_n2` | Configuration of heating installation 2 (single system, with base + backup, with solar…). | — | text | 23.6% | `Installation de chauffage simple` (98%) · `Installation de chauffage avec insert ou poêle bois en ap…` (2%) · `Installation de chauffage avec en appoint un insert ou po…` (0%) · *+5 more* |
| `description_installation_chauffage_n2` | Label describing heating installation 2. | — | text | 23.6% | `Chauffage sans solaire` (8%) · `Convecteur électrique NFC, NF** et NF*** (système individ…` (6%) · `Panneau rayonnant électrique NFC, NF** et NF*** (système …` (4%) · *+938 more* |
| `surface_chauffee_installation_chauffage_n2` † | Floor area heated by installation 2. | m² | number | 23.6% | min 0.1 · median 37.6 · max 2680.4 |
| `facteur_couverture_solaire_saisi_installation_chauffage_n2` | Solar coverage factor of heating installation 2, as entered by the diagnostician (share of the heating need covered by solar). | fraction | number | 0.0% | *(always empty in this sample)* |
| `type_generateur_n1_installation_n2` | Type of heating generator 1 of installation 2 (boiler, heat pump, electric heater, stove…). | — | text | 23.6% | `PAC air/air installée à partir de 2015` (16%) · `Convecteur électrique NFC, NF** et NF***` (12%) · `Radiateur électrique à accumulation` (11%) · *+96 more* |
| `type_energie_generateur_n1_installation_n2` | Energy used by heating generator 1 of installation 2. | — | text | 23.6% | `Électricité` (71%) · `Gaz naturel` (25%) · `Bois – Bûches` (3%) · *+6 more* |
| `usage_generateur_n1_installation_n2` | Uses served by heating generator 1 of installation 2 (heating, hot water, or both). | — | text | 23.6% | `chauffage` (88%) · `chauffage + ecs` (12%) |
| `description_generateur_chauffage_n1_installation_n2` | Label describing heating generator 1 of installation 2. | — | text | 23.6% | `Electrique - PAC air/air sans réseau de distribution inst…` (13%) · `Electrique - Radiateur électrique à accumulation` (11%) · `Gaz Naturel - Chaudière gaz à condensation installée à pa…` (10%) · *+160 more* |
| `type_generateur_n2_installation_n2` | Type of heating generator 2 of installation 2 (boiler, heat pump, electric heater, stove…). | — | text | 0.5% | `PAC air/air installée à partir de 2015` (24%) · `insert installé avant 1990` (8%) · `PAC air/air installée entre 2008 et 2014` (6%) · *+37 more* |
| `type_energie_generateur_n2_installation_n2` | Energy used by heating generator 2 of installation 2. | — | text | 0.5% | `Bois – Bûches` (43%) · `Électricité` (41%) · `Bois – Granulés (pellets) ou briquettes` (9%) · *+2 more* |
| `usage_generateur_n2_installation_n2` | Uses served by heating generator 2 of installation 2 (heating, hot water, or both). | — | text | 0.5% | `chauffage` (95%) · `chauffage + ecs` (5%) |
| `description_generateur_chauffage_n2_installation_n2` | Label describing heating generator 2 of installation 2. | — | text | 0.5% | `Electrique - PAC air/air sans réseau de distribution inst…` (20%) · `Générateur 2` (7%) · `Pompe à chaleur Air/Air` (6%) · *+49 more* |

## Hot water

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `type_installation_ecs` | Hot-water (ECS) installation: individual, collective or mixed. | — | text | 71.6% | `individuel` (75%) · `collectif` (25%) · `mixte (collectif-individuel)` (0%) |
| `type_energie_principale_ecs` | Energy used by the main hot-water system. | — | text | 100.0% | `Électricité` (50%) · `Gaz naturel` (39%) · `Réseau de Chauffage urbain` (8%) · *+9 more* |
| `type_generateur_chauffage_principal_ecs` | Type of the main hot-water generator. | — | text | 20.3% | `Ballon électrique à accumulation vertical Catégorie B ou …` (25%) · `Chaudière gaz à condensation après 2015` (15%) · `Ballon électrique à accumulation vertical Autres ou inconnue` (11%) · *+75 more* |
| `type_installation_ecs_n1` | Type of hot-water installation 1 (individual or collective). | — | text | 99.5% | `installation individuelle` (82%) · `installation collective` (18%) · `installation collective multi-bâtiment : modélisée comme …` (0%) |
| `configuration_installation_ecs_n1` | Configuration of hot-water installation 1 (single system, with solar, several systems). | — | text | 99.5% | `Un seul système d'ECS sans solaire` (97%) · `Un seul système d'ECS avec solaire` (2%) · `Deux systèmes d'ECS dans une maison ou un appartement` (1%) |
| `description_installation_ecs_n1` | Label describing hot-water installation 1. | — | text | 99.5% | `Combiné au système de chauffage` (34%) · `Ballon électrique à accumulation vertical (catégorie B ou…` (6%) · `Ballon électrique à accumulation vertical (catégorie B ou…` (6%) · *+1595 more* |
| `nombre_logements_desservis_par_installation_ecs_n1` | Number of dwellings served by hot-water installation 1. | count | number | 99.5% | min 1 · median 1 · max 1501 |
| `surface_habitable_desservie_par_installation_ecs_n1` † | Living area served by hot-water installation 1. | m² | number | 99.5% | min 0 · median 79.1 · max 20172.5 |
| `type_installation_solaire_n1` | Type of solar installation (solar hot water only, solar hot water + heating…). | — | text | 99.5% | `Non affecté` (97%) · `ECS solaire seule sup 5 ans` (2%) · `ECS solaire seule inf 5 ans` (0%) · *+1 more* |
| `facteur_couverture_solaire_saisi_n1` | Solar coverage factor of the hot water, entered directly when it can be justified. | fraction | number | 0.1% | min 0 · median 0.4 · max 1 |
| `type_generateur_n1_ecs_n1` | Type of hot-water generator 1 (water heater, boiler, heat pump…). | — | text | 99.5% | `Ballon électrique à accumulation vertical Catégorie B ou …` (25%) · `Chaudière gaz à condensation après 2015` (15%) · `Ballon électrique à accumulation vertical Autres ou inconnue` (10%) · *+98 more* |
| `type_energie_generateur_n1_ecs_n1` | Energy used by hot-water generator 1. | — | text | 99.5% | `Électricité` (50%) · `Gaz naturel` (40%) · `Réseau de Chauffage urbain` (8%) · *+8 more* |
| `usage_generateur_n1_ecs_n1` | Uses served by hot-water generator 1 (hot water, or heating + hot water). | — | text | 99.5% | `ecs` (53%) · `chauffage + ecs` (47%) |
| `description_generateur_n1_ecs_n1` | Label describing hot-water generator 1. | — | text | 95.0% | `Electrique - Ballon électrique à accumulation vertical (c…` (21%) · `Gaz Naturel - Chaudière gaz à condensation installée à pa…` (13%) · `Electrique - Ballon électrique à accumulation vertical (a…` (9%) · *+160 more* |
| `volume_stockage_generateur_n1_ecs_n1` | Storage volume of hot-water generator 1. | L | number | 99.5% | min 0 · median 70 · max 23200 |
| `cop_generateur_n1_ecs_n1` | COP of the thermodynamic water heater 1, storage efficiency included (0–20). Only for heat-pump water heaters. | — | number | 4.3% | min 2 · median 2.5 · max 8 |
| `date_installation_generateur_n1_ecs_n1` | Installation period of the thermodynamic water heater 1. | — | text | 9.2% | `A partir de 2015` (63%) · `Avant 2010` (25%) · `2010-2014` (12%) |
| `type_generateur_n2_ecs_n1` | Type of hot-water generator 2 (water heater, boiler, heat pump…). | — | number | 0.0% | *(always empty in this sample)* |
| `type_energie_generateur_n2_ecs_n1` | Energy used by hot-water generator 2. | — | number | 0.0% | *(always empty in this sample)* |
| `usage_generateur_n2_ecs_n1` | Uses served by hot-water generator 2 (hot water, or heating + hot water). | — | number | 0.0% | *(always empty in this sample)* |
| `description_generateur_n2_ecs_n1` | Label describing hot-water generator 2. | — | number | 0.0% | *(always empty in this sample)* |
| `volume_stockage_generateur_n2_ecs_n1` | Storage volume of hot-water generator 2. | L | number | 0.0% | *(always empty in this sample)* |
| `cop_generateur_n2_ecs_n1` | COP of the thermodynamic water heater 2, storage efficiency included (0–20). Only for heat-pump water heaters. | — | number | 0.0% | *(always empty in this sample)* |
| `date_installation_generateur_n2_ecs_n1` | Installation period of the thermodynamic water heater 2. | — | number | 0.0% | *(always empty in this sample)* |

## Ventilation, cooling, renewables

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `type_ventilation` | Type of ventilation. | — | text | 100.0% | `VMC SF Hygro B après 2012` (16%) · `Ventilation par ouverture des fenêtres` (15%) · `VMC SF Auto réglable après 2012` (9%) · *+33 more* |
| `surface_ventilee` † | Ventilated area; with a single ventilation system, the total living area. | m² | number | 100.0% | min 4.9 · median 81 · max 20172.5 |
| `ventilation_posterieure_2012` | 1 if the ventilation system was installed after 2012. | 0/1 | number | 100.0% | min 0 · median 0 · max 1 |
| `type_generateur_froid` | Type of cooling generator. | — | text | 12.0% | `PAC air/air installée à partir de 2015` (64%) · `PAC air/air installée avant 2008` (18%) · `PAC air/air installée entre 2008 et 2014` (14%) · *+14 more* |
| `type_energie_climatisation` | Energy used by the cooling generator. | — | text | 11.4% | `Électricité` (100%) · `Réseau de Chauffage urbain` (0%) · `Gaz naturel` (0%) · *+1 more* |
| `periode_installation_generateur_froid` | Installation period of the cooling system. | — | text | 12.0% | `A partir de 2015` (67%) · `Avant 2008` (19%) · `2008-2014` (14%) |
| `description_generateur_froid` | Label describing the cooling generator. | — | text | 12.0% | `Refroidissement - Electrique - Pompe à chaleur (divisé) -…` (50%) · `Refroidissement - Electrique - Pompe à chaleur air/air` (24%) · `Pac air / air` (12%) · *+66 more* |
| `surface_climatisee` † | Air-conditioned area. | m² | number | 12.0% | min 2.8 · median 52.9 · max 5570 |
| `categorie_enr` | Category of renewable-energy system present. | — | text | 29.9% | `pompe à chaleur` (33%) · `réseau de chaleur ou de froid vertueux` (23%) · `chauffage au bois` (18%) · *+6 more* |
| `systeme_production_electricite_origine_renouvelable` | Renewable electricity production systems present in the building. | — | text | 4.9% | `Solaire Photovoltaïque` (100%) · `Eolien` (0%) |
| `presence_production_pv` | 1 if there is photovoltaic production. | 0/1 | number | 4.9% | min 0 · median 0 · max 1 |
| `surface_totale_capteurs_pv` | Total area of photovoltaic panels (pro-rated for a flat on a collective installation). | m² | number | 1.2% | min 0.1 · median 14.4 · max 2145 |
| `nombre_module` | Number of standard photovoltaic modules installed. | count | number | 0.9% | min 1 · median 10 · max 320 |

## Target (`train.csv.gz` only)

| column | meaning | unit | read as | filled | examples |
|---|---|---|---|---|---|
| `classe_efg` | 1 if the DPE label (`etiquette_dpe`) is E, F or G; 0 for A–D. The label is the worse of the energy and the greenhouse-gas classes. | 0/1 | number | 100.0% | 17.5% are 1 |
