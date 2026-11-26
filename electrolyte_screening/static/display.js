export const number = (value, digits = 4) => new Intl.NumberFormat("en", {
  maximumSignificantDigits: digits,
}).format(value);
export const proportion = value => Number(value).toFixed(2);
export const probability = value => (value * 100).toFixed(1) + "%";
export const classLabel = result => result.fast_conductor === 1 ? "Fast" : "Slow";
export const capitalize = value => value.charAt(0).toUpperCase() + value.slice(1);
export const reportedConductivity = material => material.reported_conductivity === null ?
  "Not reported" : material.reported_conductivity + " S/cm";
export const exactMatchLabel = material => material.match_type === "same_file" ?
  "Identical archived training CIF" : "Equivalent archived training structure";

export function neighborReason(material) {
  const overlap = (100 * material.composition_overlap).toFixed(1);
  const chemistry = overlap + "% elemental-fraction overlap. Largest shared contributions: " +
    material.shared_elements.join(", ") + ".";
  if (material.same_crystal_system === null) return chemistry + " Crystal system unavailable for the reference.";
  const system = capitalize(material.crystal_system);
  return chemistry + (material.same_crystal_system ?
    " Both are " + system.toLowerCase() + "." : " The reference is " + system.toLowerCase() + ".");
}

const featureLabels = {
  element_count: "Element count",
  mean_atomic_number: "Mean atomic number",
  std_atomic_number: "Atomic number standard deviation",
  mean_atomic_mass_amu: "Mean atomic mass (amu)",
  std_atomic_mass_amu: "Atomic mass standard deviation (amu)",
  volume_per_atom_angstrom3: "Cell volume per atom (Å³)",
};

export function featureLabel(name) {
  if (name.startsWith("fraction_")) return name.slice(9) + " atomic fraction";
  if (name.startsWith("crystal_system_")) return capitalize(name.slice(15)) + " crystal system";
  return featureLabels[name] ?? name;
}

export function coverageMessages(coverage) {
  return coverage.reasons.map(reason => {
    switch (reason.code) {
      case "unseen_elements":
        return "Elements absent from training: " + reason.elements.join(", ") + ".";
      case "distant_composition":
        return "Nearest training composition distance " + number(coverage.nearest_composition_distance) +
          " exceeds the training cutoff of " + number(coverage.composition_distance_threshold) + ".";
      case "volume_outside_training_range":
        return "Cell volume per atom " + number(reason.volume_per_atom_angstrom3) +
          " Å³ is outside the training range of " + reason.training_range_angstrom3.map(value => number(value)).join("–") + " Å³.";
      case "unseen_crystal_system":
        return "Crystal system absent from training: " + reason.crystal_system + ".";
      case "coverage_profile_missing":
        return "The training coverage profile is unavailable.";
      default:
        return "The material triggered a training-coverage warning.";
    }
  });
}
