from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.runner import GeneticRunner
from models.generator import ModelGenerator
from tests.test import TestWithData
import numpy as np


class GeneticOperatorTests(TestWithData):
    def __init__(self):
        super().__init__()

    def test_crossover_no_side_effects(self):
        """
        Test the crossover operator reuses a model if it
        is an elite.

        If a model is no longer an elite, its data can be
        copied by reference as it does not persist in the
        next generation.
        """
        repeats = 1000
        model1 = ModelGenerator.get_random_model(2)
        model2 = ModelGenerator.get_random_model(2)

        initial_hashes = (model1.hash(), model2.hash())

        mutations = [
            MutationOperator.add_noise,
            MutationOperator.add_edge,
            MutationOperator.edit_edge,
            MutationOperator.flip_tf,
            MutationOperator.add_activity_noise,
            MutationOperator.flip_activity,
        ]
        crossover = CrossoverOperator.one_point_triangular_row_swap
        select = SelectionOperator.tournament
        runner = GeneticRunner(self.data, mutations, crossover, select)

        # Model1 can change hash but model2 must remain the same
        children = runner.crossover(model1, model2, False, True)
        for child in children:
            for _ in range(repeats):
                runner.mutate(child)

        assert (
            initial_hashes[1] == model2.hash()
        ), "Second model's hash has unexpectedly changed"

    def test_models_generated_are_valid(self):
        """
        Test randomly generated models, their mutations, and their
        crossover offsprings all remain valid.
        """
        repeats = 1000

        # Random models are valid
        for num_states in range(2, 10):
            for _ in range(repeats):
                model = ModelGenerator.get_random_model(num_states, p_edge=0.5)
                assert ModelGenerator.is_valid(
                    model, verbose=True
                ), f"Randomly generated model is invalid"

        # Crossover maintains validity
        mutations = [
            MutationOperator.add_noise,
            MutationOperator.add_edge,
            MutationOperator.edit_edge,
            MutationOperator.flip_tf,
        ]

        crossover = CrossoverOperator.subgraph_swap
        select = SelectionOperator.tournament
        runner = GeneticRunner(self.data, mutations, crossover, select)

        for _ in range(repeats):
            models = [
                ModelGenerator.get_random_model(states=states, p_edge=p_edge)
                for states, p_edge in zip(
                    2 + np.random.choice(20, size=2), np.random.uniform(size=2)
                )
            ]
            models = runner.crossover(*models, False, False)
            for model in models:
                assert ModelGenerator.is_valid(
                    model, verbose=True
                ), "Crossover offspring is invalid"

        # Mutations maintain validity
        model = ModelGenerator.get_random_model(10, p_edge=0.1)
        for _ in range(repeats):
            model = runner.mutate(model)
            assert ModelGenerator.is_valid(
                model, verbose=True
            ), "Mutated offspring is invalid"
