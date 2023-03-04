import numpy as np
from typing import List


class SelectionOperator:
    def roulette_wheel(
        fitness_scores: List[float],
        n: int,
        replace: bool = False,
        epsilon: float = 0.01,
    ) -> List[int]:
        # When fitness is zero, the probability distribution may only allow
        # selection of fewer individuals than required when replace = False.
        # To solve this, we add a small epsilon. We choose 0.01 to be small
        # enough (0.5% relative to max score of 2).
        
        perturbed_fitness = np.array(fitness_scores) + epsilon
        norm_fitness_scores = perturbed_fitness / sum(perturbed_fitness)
        return np.random.choice(
            len(fitness_scores), n, p=norm_fitness_scores, replace=replace
        )

    def tournament(
        fitness_scores: List[float], n: int, k: int = 3, replace: bool = False
    ) -> List[int]:
        ranked_indices = np.argsort(fitness_scores)
        population = len(ranked_indices)
        parents = []

        for _ in range(n):
            chosen = max(
                np.random.choice(population, min(k, population), replace=replace)
            )
            parents.append(ranked_indices[chosen])

            if not replace:
                np.delete(ranked_indices, chosen)
                population -= 1

        return parents
