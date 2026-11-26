# OBELiX data audit

Official train and test CSVs and CIF archives at revision [`4eaac889809d`](https://github.com/NRC-Mila/OBELiX/tree/4eaac889809dd489e3bae468c45d5c0e02b9147a/data/downloads).
The audit counts blank or whitespace-only fields as missing.

## Rows, labels, and CIF coverage

| Measure | Train | Test |
| --- | ---: | ---: |
| Rows | 478 | 121 |
| Columns | 28 | 28 |
| Unique entry IDs | 478 | 121 |
| Slow labels (0) | 260 | 68 |
| Fast labels (1) | 218 | 53 |
| Exact values at the threshold | 2 | 1 |
| Exact conductivity range (S/cm) | 4.2E-18 to 0.025 | 4.4E-18 to 0.025 |
| Upper bounds (S/cm) | `<1E-10` × 28, `<1E-8` × 1 | `<1E-10` × 8 |
| Repeated reduced formula strings | 51 groups / 131 rows | 8 groups / 17 rows |
| Repeated unit-cell formula strings | 47 groups / 124 rows | 4 groups / 8 rows |
| Repeated normalized reduced compositions | 56 groups / 144 rows | 8 groups / 17 rows |
| Repeated normalized unit-cell compositions | 56 groups / 144 rows | 8 groups / 17 rows |
| Normalized unit-cell compositions with both labels | 14 | 1 |
| CIF members in archive | 254 | 67 |
| CSV rows with an archived CIF | 254 | 67 |
| Archive CIFs without a matching CSV ID | 0 | 0 |
| CIF archive entries marked as close matches | 52 | 9 |
| CIF flags | `blank` × 7, `done` × 254, `not available` × 217 | `blank` × 2, `available` × 2, `done` × 67, `not available` × 50 |
| Positive CIF flags with no archived file | None | dco, tqb |

The target is the primary experimental room-temperature ionic conductivity in S/cm.
The auxiliary total and bulk conductivity columns also use S/cm.
Cell lengths use angstroms, angles use degrees, and `Z` counts formula units per cell.
The reported space group number is dimensionless.

## Cross-split overlap

| Grouping | Shared values |
| --- | ---: |
| Exact `ID` | 0 |
| Exact `DOI` | 0 |
| Exact `Reduced Composition` | 0 |
| Exact `True Composition` | 0 |
| Normalized `Reduced Composition` | 2 |
| Normalized `True Composition` | 2 |

Normalization parses the formulas and compares elemental fractions rounded to eight decimal places.
Repeated-group counts include every row in each repeated group.

## Missing values by column

| Column | Train blanks | Test blanks |
| --- | ---: | ---: |
| `ID` | 0 | 0 |
| `Reduced Composition` | 0 | 0 |
| `Z` | 0 | 0 |
| `True Composition` | 0 | 0 |
| `Ionic conductivity (S cm-1)` | 0 | 0 |
| `IC (Total)` | 449 | 121 |
| `IC (Bulk)` | 450 | 121 |
| `Space group` | 0 | 0 |
| `Space group #` | 0 | 0 |
| `a` | 0 | 0 |
| `b` | 0 | 0 |
| `c` | 0 | 0 |
| `alpha` | 0 | 0 |
| `beta` | 0 | 0 |
| `gamma` | 0 | 0 |
| `Family` | 18 | 15 |
| `DOI` | 0 | 0 |
| `Checked` | 0 | 0 |
| `Ref` | 15 | 14 |
| `Cif ID` | 7 | 2 |
| `Cif ref_1` | 274 | 62 |
| `Cif ref_2` | 464 | 120 |
| `note` | 371 | 92 |
| `close match` | 426 | 112 |
| `close match DOI` | 427 | 112 |
| `ICSD ID` | 249 | 58 |
| `Laskowski ID` | 105 | 26 |
| `Liion ID` | 345 | 90 |
