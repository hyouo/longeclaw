"""
Control Theory of Aging - Implementation based on arxiv 2605.16781

This module implements the mathematical framework viewing aging through control theory:
- Biological state as point in hallmark space
- Interventions as vector fields
- Biological age as accumulated control cost
- Synergy analysis via Lie brackets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

# Lopez-Otin Hallmarks of Aging (state space dimensions)
HALLMARKS = [
    "genomic_instability",
    "telomere_attrition",
    "epigenetic_alterations",
    "loss_of_proteostasis",
    "disabled_macroautophagy",
    "deregulated_nutrient_sensing",
    "mitochondrial_dysfunction",
    "cellular_senescence",
    "stem_cell_exhaustion",
    "altered_intercellular_communication",
    "chronic_inflammation",
    "dysbiosis",
]


@dataclass
class BiologicalState:
    """
    Represents biological state as a point in hallmark space.
    Each dimension corresponds to a Lopez-Otin hallmark of aging.
    Values range from 0 (youthful) to 1 (severely aged).
    """
    hallmarks: Dict[str, float] = field(default_factory=dict)
    chronological_age: float = 0.0

    def __post_init__(self):
        # Initialize missing hallmarks to 0
        for h in HALLMARKS:
            if h not in self.hallmarks:
                self.hallmarks[h] = 0.0

    def as_vector(self) -> List[float]:
        """Return state as ordered vector."""
        return [self.hallmarks.get(h, 0.0) for h in HALLMARKS]

    def biological_age(self) -> float:
        """
        Compute biological age as weighted sum of hallmark states.
        This is the "control cost" in the theory.
        """
        weights = {
            "genomic_instability": 1.2,
            "telomere_attrition": 1.1,
            "epigenetic_alterations": 1.3,
            "loss_of_proteostasis": 1.0,
            "disabled_macroautophagy": 0.9,
            "deregulated_nutrient_sensing": 1.1,
            "mitochondrial_dysfunction": 1.2,
            "cellular_senescence": 1.4,
            "stem_cell_exhaustion": 1.0,
            "altered_intercellular_communication": 0.8,
            "chronic_inflammation": 1.3,
            "dysbiosis": 0.7,
        }
        total = sum(self.hallmarks.get(h, 0) * weights.get(h, 1.0) for h in HALLMARKS)
        # Scale to approximate biological age
        return self.chronological_age * (1 + total / len(HALLMARKS))

    def distance_to(self, other: "BiologicalState") -> float:
        """Euclidean distance in hallmark space."""
        v1 = self.as_vector()
        v2 = other.as_vector()
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))


@dataclass
class Intervention:
    """
    An intervention (drug, lifestyle change) as a vector field.
    Specifies how it changes each hallmark dimension.
    """
    name: str
    description: str
    # Effect on each hallmark: negative = improvement, positive = worsening
    effects: Dict[str, float] = field(default_factory=dict)
    # Confidence in effect estimates (0-1)
    confidence: float = 0.5
    # Evidence level: preclinical, clinical, meta-analysis
    evidence_level: str = "preclinical"
    # Drug class for synergy analysis
    drug_class: str = "unknown"

    def apply(self, state: BiologicalState, magnitude: float = 1.0) -> BiologicalState:
        """Apply intervention to state, returning new state."""
        new_hallmarks = dict(state.hallmarks)
        for h, effect in self.effects.items():
            if h in new_hallmarks:
                new_hallmarks[h] = max(0.0, min(1.0, new_hallmarks[h] + effect * magnitude))
        return BiologicalState(
            hallmarks=new_hallmarks,
            chronological_age=state.chronological_age
        )

    def as_vector(self) -> List[float]:
        """Return effects as ordered vector."""
        return [self.effects.get(h, 0.0) for h in HALLMARKS]


# Known interventions database
KNOWN_INTERVENTIONS: Dict[str, Intervention] = {
    "rapamycin": Intervention(
        name="Rapamycin",
        description="mTOR inhibitor, extends lifespan in multiple species",
        effects={
            "deregulated_nutrient_sensing": -0.3,
            "disabled_macroautophagy": -0.25,
            "cellular_senescence": -0.15,
            "loss_of_proteostasis": -0.1,
        },
        confidence=0.8,
        evidence_level="clinical",
        drug_class="mtor_inhibitor",
    ),
    "metformin": Intervention(
        name="Metformin",
        description="AMPK activator, diabetes drug with longevity benefits",
        effects={
            "deregulated_nutrient_sensing": -0.2,
            "mitochondrial_dysfunction": -0.15,
            "chronic_inflammation": -0.1,
            "cellular_senescence": -0.1,
        },
        confidence=0.75,
        evidence_level="clinical",
        drug_class="ampk_activator",
    ),
    "senolytics_dq": Intervention(
        name="Dasatinib + Quercetin",
        description="Senolytic combination targeting senescent cells",
        effects={
            "cellular_senescence": -0.4,
            "chronic_inflammation": -0.2,
            "altered_intercellular_communication": -0.15,
            "stem_cell_exhaustion": -0.1,
        },
        confidence=0.7,
        evidence_level="clinical",
        drug_class="senolytic",
    ),
    "nad_precursors": Intervention(
        name="NAD+ Precursors (NMN/NR)",
        description="Boost NAD+ levels, support mitochondrial function",
        effects={
            "mitochondrial_dysfunction": -0.2,
            "epigenetic_alterations": -0.1,
            "stem_cell_exhaustion": -0.1,
            "genomic_instability": -0.05,
        },
        confidence=0.6,
        evidence_level="preclinical",
        drug_class="nad_booster",
    ),
    "spermidine": Intervention(
        name="Spermidine",
        description="Natural polyamine that induces autophagy",
        effects={
            "disabled_macroautophagy": -0.25,
            "loss_of_proteostasis": -0.15,
            "mitochondrial_dysfunction": -0.1,
            "epigenetic_alterations": -0.05,
        },
        confidence=0.65,
        evidence_level="clinical",
        drug_class="autophagy_inducer",
    ),
    "caloric_restriction": Intervention(
        name="Caloric Restriction",
        description="30% calorie reduction, robust lifespan extension",
        effects={
            "deregulated_nutrient_sensing": -0.35,
            "disabled_macroautophagy": -0.2,
            "mitochondrial_dysfunction": -0.15,
            "chronic_inflammation": -0.15,
            "cellular_senescence": -0.1,
        },
        confidence=0.9,
        evidence_level="meta-analysis",
        drug_class="dietary",
    ),
    "exercise": Intervention(
        name="Regular Exercise",
        description="Moderate aerobic + resistance training",
        effects={
            "mitochondrial_dysfunction": -0.2,
            "chronic_inflammation": -0.15,
            "stem_cell_exhaustion": -0.1,
            "loss_of_proteostasis": -0.1,
            "deregulated_nutrient_sensing": -0.1,
        },
        confidence=0.95,
        evidence_level="meta-analysis",
        drug_class="lifestyle",
    ),
    "fisetin": Intervention(
        name="Fisetin",
        description="Natural senolytic flavonoid",
        effects={
            "cellular_senescence": -0.3,
            "chronic_inflammation": -0.15,
        },
        confidence=0.5,
        evidence_level="preclinical",
        drug_class="senolytic",
    ),
}


def compute_lie_bracket(
    intervention_a: Intervention,
    intervention_b: Intervention,
) -> Dict[str, float]:
    """
    Compute Lie bracket [A, B] = AB - BA as measure of non-commutativity.
    Non-zero bracket indicates order of interventions matters.

    In biological terms: synergistic/antagonistic interactions.
    """
    # Simplified model: interaction strength based on overlapping targets
    bracket = {}
    for h in HALLMARKS:
        effect_a = intervention_a.effects.get(h, 0.0)
        effect_b = intervention_b.effects.get(h, 0.0)
        # Non-linearity coefficient (would be derived from biological data)
        # Positive = synergistic, negative = antagonistic
        interaction = effect_a * effect_b * 0.5  # Simplified
        if abs(interaction) > 0.001:
            bracket[h] = interaction
    return bracket


def compute_control_cost(
    initial_state: BiologicalState,
    target_state: BiologicalState,
    interventions: List[Intervention],
) -> float:
    """
    Compute total control cost to move from initial to target state.
    Lower cost = more efficient intervention strategy.
    """
    current = initial_state
    total_cost = 0.0

    for intervention in interventions:
        # Apply intervention
        new_state = intervention.apply(current)
        # Cost is proportional to intervention strength and confidence penalty
        intervention_strength = sum(abs(e) for e in intervention.effects.values())
        confidence_penalty = 1.0 / max(intervention.confidence, 0.1)
        total_cost += intervention_strength * confidence_penalty
        current = new_state

    # Add penalty for not reaching target
    distance_penalty = current.distance_to(target_state) * 10.0
    total_cost += distance_penalty

    return total_cost


def rank_interventions_by_cost_reduction(
    current_state: BiologicalState,
    available_interventions: Optional[List[str]] = None,
) -> List[Tuple[str, float, float]]:
    """
    Rank interventions by their ability to reduce biological age.
    Returns: List of (intervention_name, bio_age_reduction, confidence)
    """
    if available_interventions is None:
        available_interventions = list(KNOWN_INTERVENTIONS.keys())

    current_bio_age = current_state.biological_age()
    results = []

    for name in available_interventions:
        if name not in KNOWN_INTERVENTIONS:
            continue
        intervention = KNOWN_INTERVENTIONS[name]
        new_state = intervention.apply(current_state)
        new_bio_age = new_state.biological_age()
        reduction = current_bio_age - new_bio_age
        results.append((name, reduction, intervention.confidence))

    # Sort by reduction * confidence (expected value)
    results.sort(key=lambda x: x[1] * x[2], reverse=True)
    return results


def analyze_combination_synergy(
    intervention_names: List[str],
    current_state: BiologicalState,
) -> Dict:
    """
    Analyze synergy between multiple interventions using Lie brackets.
    Returns analysis with synergy scores and recommended order.
    """
    interventions = [
        KNOWN_INTERVENTIONS[name]
        for name in intervention_names
        if name in KNOWN_INTERVENTIONS
    ]

    if len(interventions) < 2:
        return {"error": "Need at least 2 interventions for synergy analysis"}

    # Compute pairwise Lie brackets
    synergies = []
    for i, int_a in enumerate(interventions):
        for j, int_b in enumerate(interventions):
            if i >= j:
                continue
            bracket = compute_lie_bracket(int_a, int_b)
            total_interaction = sum(bracket.values())
            synergies.append({
                "pair": (int_a.name, int_b.name),
                "bracket": bracket,
                "total_interaction": total_interaction,
                "synergistic": total_interaction < 0,  # Negative = both reduce hallmarks more together
            })

    # Compute combined effect
    combined_state = current_state
    for intervention in interventions:
        combined_state = intervention.apply(combined_state)

    # Compute individual effects sum (without synergy)
    individual_reductions = []
    for intervention in interventions:
        temp_state = intervention.apply(current_state)
        reduction = current_state.biological_age() - temp_state.biological_age()
        individual_reductions.append(reduction)

    actual_reduction = current_state.biological_age() - combined_state.biological_age()
    sum_individual = sum(individual_reductions)
    synergy_bonus = actual_reduction - sum_individual

    return {
        "interventions": [i.name for i in interventions],
        "pairwise_synergies": synergies,
        "individual_reductions": dict(zip([i.name for i in interventions], individual_reductions)),
        "combined_reduction": actual_reduction,
        "sum_of_individual": sum_individual,
        "synergy_bonus": synergy_bonus,
        "synergy_ratio": actual_reduction / sum_individual if sum_individual > 0 else 1.0,
        "is_synergistic": synergy_bonus > 0,
        "final_biological_age": combined_state.biological_age(),
    }


def create_typical_aging_state(chronological_age: float) -> BiologicalState:
    """
    Create a typical biological state for a given chronological age.
    Hallmarks increase roughly linearly with age.
    """
    # Normalize age effect (0 at 20, 1 at 100)
    age_factor = max(0, min(1, (chronological_age - 20) / 80))

    # Different hallmarks accumulate at different rates
    rates = {
        "genomic_instability": 0.8,
        "telomere_attrition": 1.0,
        "epigenetic_alterations": 0.9,
        "loss_of_proteostasis": 0.7,
        "disabled_macroautophagy": 0.6,
        "deregulated_nutrient_sensing": 0.75,
        "mitochondrial_dysfunction": 0.85,
        "cellular_senescence": 0.95,
        "stem_cell_exhaustion": 0.65,
        "altered_intercellular_communication": 0.5,
        "chronic_inflammation": 0.9,
        "dysbiosis": 0.4,
    }

    hallmarks = {h: age_factor * rate for h, rate in rates.items()}
    return BiologicalState(hallmarks=hallmarks, chronological_age=chronological_age)


def optimal_intervention_sequence(
    current_state: BiologicalState,
    intervention_names: List[str],
    max_steps: int = 10,
) -> List[str]:
    """
    Find optimal ordering of interventions using greedy control-theoretic approach.
    At each step, pick intervention that maximally reduces biological age.
    """
    remaining = set(intervention_names)
    sequence = []
    state = current_state

    for _ in range(min(max_steps, len(intervention_names))):
        if not remaining:
            break

        best_name = None
        best_reduction = -float('inf')

        for name in remaining:
            if name not in KNOWN_INTERVENTIONS:
                continue
            intervention = KNOWN_INTERVENTIONS[name]
            new_state = intervention.apply(state)
            reduction = state.biological_age() - new_state.biological_age()
            if reduction > best_reduction:
                best_reduction = reduction
                best_name = name

        if best_name is None or best_reduction <= 0:
            break

        sequence.append(best_name)
        remaining.remove(best_name)
        state = KNOWN_INTERVENTIONS[best_name].apply(state)

    return sequence


# Agent tool interface
def control_law_analysis(
    chronological_age: float = 50,
    intervention_names: Optional[List[str]] = None,
) -> Dict:
    """
    Main entry point for control law analysis.
    Creates typical aging state and analyzes interventions.
    """
    state = create_typical_aging_state(chronological_age)

    if intervention_names is None:
        intervention_names = list(KNOWN_INTERVENTIONS.keys())

    # Rank single interventions
    rankings = rank_interventions_by_cost_reduction(state, intervention_names)

    # Analyze combinations if multiple specified
    synergy_analysis = None
    if len(intervention_names) >= 2:
        synergy_analysis = analyze_combination_synergy(intervention_names, state)

    # Optimal sequence
    optimal_seq = optimal_intervention_sequence(state, intervention_names)

    return {
        "chronological_age": chronological_age,
        "initial_biological_age": state.biological_age(),
        "hallmark_state": state.hallmarks,
        "intervention_rankings": [
            {"name": name, "bio_age_reduction": red, "confidence": conf}
            for name, red, conf in rankings
        ],
        "synergy_analysis": synergy_analysis,
        "optimal_sequence": optimal_seq,
        "available_interventions": list(KNOWN_INTERVENTIONS.keys()),
    }
