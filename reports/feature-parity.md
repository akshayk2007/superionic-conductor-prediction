# IonLab CSV–CIF feature parity audit

**All 131 features use the same calculation and units in both paths.** The CSV path supplies reported unit-cell composition and crystallographic metadata. The upload path supplies composition, lattice, and symmetry derived from the CIF. Those inputs produce different feature values for 104 of 247 accepted training pairs and 36 of 67 test pairs at the tolerances below.

The audit uses the checksummed OBELiX CSVs and CIF archives at revision `4eaac889809dd489e3bae468c45d5c0e02b9147a`. It reads the saved random forest and applies its existing featurizer and classifier. No model was fit, selected, or saved. The artifact SHA-256 before and after was `aa83b2cc696cd7b2a481c4801398ae5dc1660e6a823bcf3fdcd343cd74bbda23`. [Official OBELiX field definitions](https://github.com/NRC-Mila/OBELiX#labels-and-features)

## Code paths and input sources

| Input | CSV training path | Uploaded CIF path |
| --- | --- | --- |
| Per-element atom counts | Parses the row’s `True Composition` as a reported unit-cell formula. | Sums each element’s fractional occupancy over the symmetry-expanded atom sites. The formula header is parser metadata. Atom counts come from the sites. |
| Lattice | Reads `a`, `b`, `c`, `alpha`, `beta`, and `gamma` as Å and degrees. | Reads lattice lengths and angles from the parsed non-primitive cell, in Å and degrees. |
| Symmetry | Reads the reported international `Space group #`. | Infers the international space group from the parsed structure with a 0.01 Å symmetry tolerance and 5° angle tolerance. |

Both paths run `MaterialFeatures.transform`, which calls `material_features`. CSV formulas enter as strings and CIF atom counts enter as dictionaries. Both become pymatgen elemental compositions. The upload parser preserves partial occupancies and proportionally normalizes site totals above 1 and at most 1.002 before featurization. The saved random forest consumes the 131 features directly. The audit uses OBELiX IDs only to pair archive members with CSV rows. [Feature code](../electrolyte_screening/features.py) · [Training and upload entry points](../electrolyte_screening/models.py) · [pymatgen CIF parser](https://pymatgen.org/pymatgen.io.html#pymatgen.io.cif.CifParser)

## Paired value mismatches

The denominators are the 247 accepted training CIFs and 67 accepted test CIFs. The fractional-composition criterion is an element appearing in only one source or an absolute atomic-fraction change greater than 0.0001. Volume-per-atom differences use a relative cutoff of 1% with the CSV value as reference. Weighted summaries use `numpy.isclose` with the CSV value as reference, relative tolerance 0.0001, and absolute tolerance 0.000001. Element count and crystal-system flags are compared exactly.

| Difference | Training pairs / 247 | Test pairs / 67 |
| --- | ---: | ---: |
| Any model feature | 104 | 36 |
| Any elemental fraction | 95 | 34 |
| Lithium fraction | 90 | 24 |
| Any atomic-number or mass summary | 98 | 35 |
| Cell volume per atom | 21 | 4 |
| Number of elements | 1 | 0 |
| Crystal system | 1 | 0 |
| Exact space group input | 3 | 1 |

- **Fractional composition.** For test entry `duf`, the lithium atomic fraction rose from 0.2797 in the CSV to 0.4024 in the archived CIF. OBELiX flags this structure as a close match. For training entry `hx7`, the CSV includes H and the archived structure has no parsed H sites. Among accepted CIFs, 189 train and 54 test structures have partial occupancies.
- **Cell volume per atom.** For test entry `24k`, the cell volume is 941.745 Å³ in both sources. The atom count changes from 92 to 46, so the feature doubles from 10.236 to 20.473 Å³ per atom. For training entry `vy5`, all elemental counts scale by three at the same cell volume, taking the feature from 45.366 to 15.122 Å³ per atom. This points to a mismatch between the cells described by the lattice and composition inputs. A consistent cell expansion would scale both atom count and volume together.
- **Symmetry.** Training entry `019` switches from reported group 168 (hexagonal) to inferred group 167 (trigonal), changing two one-hot flags. Test entry `um1` switches from group 167 to 148. Both groups are trigonal, so all seven symmetry feature values agree.

## Effect on the fitted model

On the same 67 held-out rows, switching from CSV features to CIF features changes one predicted class. Entry `duf` is a labelled fast conductor whose estimated fast-class probability drops from 89.8% to 26.3%. On this fixed cohort, fast-class recall moves from 0.55 to 0.50 and F1 from 0.688 to 0.645. Several other scores changed without crossing the class boundary. Cell volume per atom has the largest mean decrease in impurity in the fitted forest, at 9.7% of total global feature importance.

The table below scores hybrid inputs with the same saved forest. The composition-only column replaces the 118 fractions, element count, and four weighted summaries with CIF values. The volume-only column replaces only the volume-per-atom feature. Values are estimated fast-class probabilities. Some hybrid inputs can be physically inconsistent because composition and volume share an atom-count denominator. These checks measure sensitivity to the input changes.

| Held-out ID | All CSV | CIF composition only | CIF volume per atom only | All CIF |
| --- | ---: | ---: | ---: | ---: |
| `duf` | 0.898 | 0.268 | 0.828 | 0.263 |
| `24k` | 0.243 | 0.245 | 0.345 | 0.347 |
| `4xq` | 0.201 | 0.185 | 0.216 | 0.197 |
| `9o6` | 0.910 | 0.726 | 0.889 | 0.732 |

## CIF-only sample-size tradeoff

| Cohort within the official split | Train | Train fast / slow | Test | Test fast / slow | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full CSV benchmark | 478 | 218 / 260 | 121 | 53 / 68 | 599 |
| Archived CIFs | 254 | 93 / 161 | 67 | 20 / 47 | 321 |
| Accepted by the upload path | 247 | 88 / 159 | 67 | 20 / 47 | 314 |
| Accepted, excluding flagged close matches | 195 | 67 / 128 | 58 | 15 / 43 | 253 |

Seven training CIFs have zero Li in their parsed sites despite lithium-bearing CSV formulas. The raw atom-site type lists also omit Li in every case. Those entries are `b4c`, `cy2`, `5tl`, `u2y`, `hta`, `s3h`, and `u8r`. The current upload path rejects them. All 67 archived test CIFs were accepted.

A CIF-only model would retain **51.7% of training rows** and **55.4% of test rows**. Fast conductors drop from 45.6% to 35.6% of training and from 43.8% to 29.9% of testing. Distinct normalized training compositions drop from 390 in the full CSV to 230 in the accepted CIFs. The training element set drops from 54 to 51, losing Cd, Lu, and Te. No accepted CIF train and test pair shares the same normalized CIF composition at eight decimal places.

**Recommendation.** Evaluate a CIF-only model as an upload-aligned candidate before replacing the current forest. The same parser and input source would improve train–inference consistency. Predictive improvement is unproven, and the smaller, less fast-conductor-rich cohort changes the data available to learn from. The 52 training and nine test CIFs flagged as close matches are potential sources of feature–label mismatch because conductivity labels come from the corresponding experimental CSV rows. Compare the current saved model, an equal-size model trained on CSV features from the same 247 accepted training rows, and a CIF-only model on the same 67 held-out CIFs. Use composition-grouped cross-validation within the accepted training split. With 20 fast test cases, a single case moves recall by five percentage points.

## All 131 model features

`n_E` is the unit-cell atom count for element `E`, and `N` is the sum of elemental counts for that cell. `x_E = n_E/N`, `Z_E` is atomic number, `m_E` is atomic mass, and `V` is cell volume computed from the six lattice parameters. `μ_Z` and `μ_m` are the weighted means below. The counts show paired differences at the stated tolerances. Zero means no observed difference at those tolerances.

| Saved feature | Shared calculation | Units in both paths | Same calculation / units | Train differences / 247 | Test differences / 67 |
| --- | --- | --- | --- | ---: | ---: |
| `fraction_H` | `n_H / N` | dimensionless fraction | Yes / Yes | 2 | 0 |
| `fraction_He` | `n_He / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Li` | `n_Li / N` | dimensionless fraction | Yes / Yes | 90 | 24 |
| `fraction_Be` | `n_Be / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_B` | `n_B / N` | dimensionless fraction | Yes / Yes | 2 | 0 |
| `fraction_C` | `n_C / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_N` | `n_N / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_O` | `n_O / N` | dimensionless fraction | Yes / Yes | 49 | 16 |
| `fraction_F` | `n_F / N` | dimensionless fraction | Yes / Yes | 4 | 1 |
| `fraction_Ne` | `n_Ne / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Na` | `n_Na / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Mg` | `n_Mg / N` | dimensionless fraction | Yes / Yes | 0 | 3 |
| `fraction_Al` | `n_Al / N` | dimensionless fraction | Yes / Yes | 13 | 5 |
| `fraction_Si` | `n_Si / N` | dimensionless fraction | Yes / Yes | 2 | 0 |
| `fraction_P` | `n_P / N` | dimensionless fraction | Yes / Yes | 26 | 1 |
| `fraction_S` | `n_S / N` | dimensionless fraction | Yes / Yes | 35 | 8 |
| `fraction_Cl` | `n_Cl / N` | dimensionless fraction | Yes / Yes | 3 | 1 |
| `fraction_Ar` | `n_Ar / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_K` | `n_K / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ca` | `n_Ca / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Sc` | `n_Sc / N` | dimensionless fraction | Yes / Yes | 1 | 0 |
| `fraction_Ti` | `n_Ti / N` | dimensionless fraction | Yes / Yes | 7 | 5 |
| `fraction_V` | `n_V / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cr` | `n_Cr / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Mn` | `n_Mn / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Fe` | `n_Fe / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Co` | `n_Co / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ni` | `n_Ni / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cu` | `n_Cu / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Zn` | `n_Zn / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ga` | `n_Ga / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Ge` | `n_Ge / N` | dimensionless fraction | Yes / Yes | 14 | 4 |
| `fraction_As` | `n_As / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Se` | `n_Se / N` | dimensionless fraction | Yes / Yes | 1 | 6 |
| `fraction_Br` | `n_Br / N` | dimensionless fraction | Yes / Yes | 5 | 0 |
| `fraction_Kr` | `n_Kr / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Rb` | `n_Rb / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Sr` | `n_Sr / N` | dimensionless fraction | Yes / Yes | 7 | 2 |
| `fraction_Y` | `n_Y / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Zr` | `n_Zr / N` | dimensionless fraction | Yes / Yes | 6 | 4 |
| `fraction_Nb` | `n_Nb / N` | dimensionless fraction | Yes / Yes | 2 | 4 |
| `fraction_Mo` | `n_Mo / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Tc` | `n_Tc / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ru` | `n_Ru / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Rh` | `n_Rh / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Pd` | `n_Pd / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ag` | `n_Ag / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cd` | `n_Cd / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_In` | `n_In / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Sn` | `n_Sn / N` | dimensionless fraction | Yes / Yes | 3 | 3 |
| `fraction_Sb` | `n_Sb / N` | dimensionless fraction | Yes / Yes | 0 | 1 |
| `fraction_Te` | `n_Te / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_I` | `n_I / N` | dimensionless fraction | Yes / Yes | 3 | 0 |
| `fraction_Xe` | `n_Xe / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cs` | `n_Cs / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ba` | `n_Ba / N` | dimensionless fraction | Yes / Yes | 1 | 0 |
| `fraction_La` | `n_La / N` | dimensionless fraction | Yes / Yes | 10 | 12 |
| `fraction_Ce` | `n_Ce / N` | dimensionless fraction | Yes / Yes | 1 | 0 |
| `fraction_Pr` | `n_Pr / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Nd` | `n_Nd / N` | dimensionless fraction | Yes / Yes | 1 | 0 |
| `fraction_Pm` | `n_Pm / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Sm` | `n_Sm / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Eu` | `n_Eu / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Gd` | `n_Gd / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Tb` | `n_Tb / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Dy` | `n_Dy / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ho` | `n_Ho / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Er` | `n_Er / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Tm` | `n_Tm / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Yb` | `n_Yb / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Lu` | `n_Lu / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Hf` | `n_Hf / N` | dimensionless fraction | Yes / Yes | 2 | 0 |
| `fraction_Ta` | `n_Ta / N` | dimensionless fraction | Yes / Yes | 9 | 0 |
| `fraction_W` | `n_W / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Re` | `n_Re / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Os` | `n_Os / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ir` | `n_Ir / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Pt` | `n_Pt / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Au` | `n_Au / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Hg` | `n_Hg / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Tl` | `n_Tl / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Pb` | `n_Pb / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Bi` | `n_Bi / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Po` | `n_Po / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_At` | `n_At / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Rn` | `n_Rn / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Fr` | `n_Fr / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ra` | `n_Ra / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ac` | `n_Ac / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Th` | `n_Th / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Pa` | `n_Pa / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_U` | `n_U / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Np` | `n_Np / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Pu` | `n_Pu / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Am` | `n_Am / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cm` | `n_Cm / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Bk` | `n_Bk / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cf` | `n_Cf / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Es` | `n_Es / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Fm` | `n_Fm / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Md` | `n_Md / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_No` | `n_No / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Lr` | `n_Lr / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Rf` | `n_Rf / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Db` | `n_Db / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Sg` | `n_Sg / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Bh` | `n_Bh / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Hs` | `n_Hs / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Mt` | `n_Mt / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ds` | `n_Ds / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Rg` | `n_Rg / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Cn` | `n_Cn / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Nh` | `n_Nh / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Fl` | `n_Fl / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Mc` | `n_Mc / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Lv` | `n_Lv / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Ts` | `n_Ts / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `fraction_Og` | `n_Og / N` | dimensionless fraction | Yes / Yes | 0 | 0 |
| `element_count` | Count of elements with positive atom count | element count | Yes / Yes | 1 | 0 |
| `mean_atomic_number` | ? x_E ? Z_E | dimensionless | Yes / Yes | 94 | 35 |
| `std_atomic_number` | ?(? x_E ? (Z_E ? ?_Z)?) | dimensionless | Yes / Yes | 70 | 30 |
| `mean_atomic_mass_amu` | ? x_E ? m_E | u (amu) | Yes / Yes | 94 | 35 |
| `std_atomic_mass_amu` | ?(? x_E ? (m_E ? ?_m)?) | u (amu) | Yes / Yes | 71 | 29 |
| `volume_per_atom_angstrom3` | V / N | ?? per atom | Yes / Yes | 21 | 4 |
| `crystal_system_triclinic` | 1 for space groups 1?2, else 0 | dimensionless flag | Yes / Yes | 0 | 0 |
| `crystal_system_monoclinic` | 1 for space groups 3?15, else 0 | dimensionless flag | Yes / Yes | 0 | 0 |
| `crystal_system_orthorhombic` | 1 for space groups 16?74, else 0 | dimensionless flag | Yes / Yes | 0 | 0 |
| `crystal_system_tetragonal` | 1 for space groups 75?142, else 0 | dimensionless flag | Yes / Yes | 0 | 0 |
| `crystal_system_trigonal` | 1 for space groups 143?167, else 0 | dimensionless flag | Yes / Yes | 1 | 0 |
| `crystal_system_hexagonal` | 1 for space groups 168?194, else 0 | dimensionless flag | Yes / Yes | 1 | 0 |
| `crystal_system_cubic` | 1 for space groups 195?230, else 0 | dimensionless flag | Yes / Yes | 0 | 0 |
