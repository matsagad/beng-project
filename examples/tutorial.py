from evolution.genetic.operators.mutation import MutationOperator
from evolution.genetic.operators.crossover import CrossoverOperator
from evolution.genetic.operators.selection import SelectionOperator
from evolution.genetic.penalty import ModelPenalty
from evolution.genetic.runner import GeneticRunner
from models.generator import ModelGenerator
from models.model import PromoterModel
from models.preset import Preset
from models.rates.function import RateFunction as RF
from mi_estimation.decoding import DecodingEstimator
from pipeline.one_step_decoding import OneStepDecodingPipeline
from ssa.one_step import OneStepSimulator
from typing import Tuple
from utils.data import ClassWithData
import numpy as np


class TutorialExamples:
    """
    This is a tutorial on how to use the library of classes and functions
    within the project together with some explanations on how they work.
    This is intended to allow anyone to easily make adjustments as they see fit.

    Whenever a function is introduced for the first time, the parameters
    are explicitly typed for clarity. After that, only keyword arguments
    are specified to be succinct.
    """

    class DataHandling(ClassWithData):
        def __init__(self):
            super().__init__()

        def data_format(self):
            """
            The exogenous data used throughout is a multidimensional array
            containing TF concentrations for each cell replicate as stress is
            induced.

            An origin index is used to mark at which point in time the cell
            replicates were introduced to stress.

            A time delta must also be provided to set a timescale for the simulation.

            Note: this class inherits the data from its singleton superclass,
            and it is accessible through self.data. In general, the data should
            be a numpy array of shape: (num_envs, num_tfs, num_cells, num_timestamps).

            All of the pre-processing is done in `utils/process.py`. Simply populate
            the `data` folder and all these parameters are automatically derived, cached,
            and accessed as appropriate.
            """
            print(self.data.shape)
            print(self.origin)
            print(self.time_delta)

    class PromoterModels:
        def creating_a_model(self):
            """
            A promoter model given by the PromoterModel class takes in
            a (generator) matrix which specifies the transition rates
            between states. It is a continuous-time Markov chain, and,
            as such, the diagonal entries are kept as None (i.e. 0).

            Each entry in the matrix is of instance RateFunction. Some
            defined examples are: Constant, Linear, and Hill. Each of
            them take two lists containing the rates and the TFs associated
            with it.

            Each state is given an activity weight. These are the values
            they output in the trajectory when such a state is chosen during
            the simulation. By default, the first state is set to 1 and the
            rest are set to zero. During the simulation, these values are all
            scaled to have unit sum.

            Below, we construct a simple two-state model:

                     ┌───────k1────>──┐
                (A: 0.75)        (I: 0.25)
                     └──<──k0·TF1─────┘

            """
            # Rates (between 10^-2 and 10^2)
            k0, k1 = 1, 1
            # Transcription factor index
            tf1 = 1

            model = PromoterModel(
                rate_fn_matrix=[
                    [None, RF.Constant([k1])],
                    [RF.Linear([k0], [tf1]), None],
                ]
            ).with_activity_weights([0.75, 0.25])

        def randomly_generating_models(self):
            """
            To create randomly generated models, the ModelGenerator class'
            `get_random_model` function can be called.

            It begins by creating a uniform spanning tree (Wilson, 1996) with
            number of nodes corresponding to the number states specified. This way,
            the initial graph (the model's skeleton) is entirely connected.

            If reactions are set to be strictly reversible (i.e. if a directed edge
            exists from u to v, there must be an edge from v to u), then it first
            assumes the generated graph is undirected, and later populates twice the
            number of edges.

            It randomly connects two nodes by some probability `p_edge`. When all the
            edges are mapped, it initialises them by some random rate function
            with random rates and TFs to depend on (see ModelGenerator.get_random_rate_fn).

            It also allows having exactly one active state. If not set, then the
            activity weights of each node are initialised by a standard uniform.
            """
            model = ModelGenerator.get_random_model(
                states=100, p_edge=0.5, reversible=True, one_active_state=False
            )

        def visualising_models(self):
            """
            Models can be visualised through their visualise method.

            With small models, a background on the edge labels make for clear
            illustrations. To turn this off, the `transparent` flag can be set to
            True.

            When models become relatively large and analyses focus mainly
            on which TFs it depends on, the `with_rates` flag can be set to False.
            This hides the specific constants associated.

            When models become extremely large and analyses focus mainly
            on the architecture and connectedness of the model, the `small_size`
            flag can be set to True. This hides edge labels entirely.

            When a `target_ax` is provided, it can plot the graph on a matplotlib
            subplot - perfect for comparing models side-by-side!

            The model can be saved by setting the `save` flag to True and providing
            a file name via the `fname` parameter.
            """
            # A small-sized model
            ModelGenerator.get_random_model(2).visualise()

            # A medium-sized model
            ModelGenerator.get_random_model(4).visualise(transparent=True)

            # A large-sized model
            ModelGenerator.get_random_model(8).visualise(with_rates=False)

            # An extremely large-sized model
            ModelGenerator.get_random_model(16).visualise(small_size=True)

    class TrajectorySimulation(ClassWithData):
        def __init__(self):
            super().__init__()

        def producing_a_trajectory(self):
            """
            A simulator (of instance Simulator, e.g. OneStepSimulator)
            produces the activity trajectories of a model given the exogenous data
            provided as means of quantifying the concentration of TFs at any time point.

            As this process is entirely stochastic, one may wish to run multiple
            trials for each cell. This can be adjusted by the `replicates` parameter.

            The trajectory produced has shape: (num_timepoints, num_envs, num_cells, num_states).
            This indicates at each time point, within each environment, for each cell,
            the state which is currently active.

            The simulation heavily relies on matrix exponential calculation (see
            `model.get_matrix_exp`) and tensor multiplication (between 4D and 5D tensors).
            This accounts for the most of its execution time.

            Discounting the above two, another important section within it uses a
            vectorised O(n) approach by default over a non-vectorised O(log(n)) one.
            To toggle this, `sim.binary_search` can be set to True. However, the returns
            of the logarithmic approach only present themselves when the number of states
            is ~1000+, which is beyond the study's scope.
            """
            model = ModelGenerator.get_random_model(4)
            sim = OneStepSimulator(
                exogenous_data=self.data, tau=self.time_delta, replicates=10
            )

            trajectories = sim.simulate(model)
            print(trajectories.shape)

        def producing_the_probability_distribution_trajectory(self):
            """
            Instead of realising decisions made by each cell, one can opt for the
            entirely probabilistic trajectory of a model. That is, at each time point,
            it specifies the probability of being in any of the states.

            This can be accessed by setting the `realised` parameter to False.
            Note: when this is False, the `replicates` parameter does nothing.

            As this simply involves multiplying matrix exponentials over and over
            (the Kolmogorov forward equation), it is much faster than a realised approach.
            Although this is not conducive to an experimental setting on single-cell
            decision marking and is thus not used, it offers a way to visualise the
            behaviour of the cells on a population-level.
            """
            model = ModelGenerator.get_random_model(4)
            sim = OneStepSimulator(
                self.data, self.time_delta, realised=False, replicates=10
            )

            trajectories = sim.simulate(model)

        def visualising_trajectories(self):
            """
            The trajectories produced can be visualised by the simulator's
            `visualise_trajectory` method.

            The average trajectories across all cell replicates for each environment
            is plotted when the `average` parameter is set to True. Note that as the
            number of replicates is increased, it converges to the probability
            distribution as specified above!

            A single cell's trajectories for each environment is plotted when the
            `average` parameter is set to False. The cell corresponding to the
            `batch_num` specified is chosen.

            The probability distribution can also be visualised by simply passing
            the trajectories produced in any scheme (average/single-cell).

            Similar to model visuations, the `fname` parameter can be set in which
            the plot is saved to that file.
            """
            model = ModelGenerator.get_random_model(3)
            sim = OneStepSimulator(self.data, self.time_delta, replicates=10)

            trajectories = sim.simulate(model)

            # Average trajectory
            sim.visualise_trajectory(trajectory=trajectories, model=model, average=True)

            # Single trajectory (e.g. cell #27)
            sim.visualise_trajectory(
                trajectory=trajectories, model=model, average=False, batch_num=27
            )

            # Probability distribution trajectory
            sim.realised = False
            prob_trajectories = sim.simulate(model)
            sim.visualise_trajectory(trajectory=prob_trajectories, model=model)

    class MIEstimation(ClassWithData):
        def __init__(self):
            super().__init__()

            self.DEFAULT_INTERVAL = 30

        def estimating_mutual_information(self):
            """
            A decoding-based approach is employed within the study. That is,
            a machine learning classifier's performance on distinguishing the
            trajectories within each environment is indicative of the mutual
            information by the model. In fact, it is a lower-bound estimate.

            This is done by passing the trajectories produced to an estimator
            (of instance Estimator, e.g. DecodingEstimator). To perform splitting
            and other inferences on the trajectories, it needs to take in the
            origin, model, and replicates used.

            The trajectories are first split into their respective environments
            given some interval away from the origin. This is illustrated below:

                A promoter trajectory           A promoter trajectory
                in environment E1.              in environment E2.
                (where X is the origin)         (note X is the same)
                            ┌─┐                              ┌─┐
                         ┌──┘ │                    ┌─┐       │ │
                ─────────┘    └────             ───┘ └───────┘ └───
                0 ..... X ..... 100             0 ..... X ..... 100

                Given an interval d <= X, we cut the trajectories:

                  rich    E1 stress               rich    E2 stress
                        │    ┌─┐                        │    ┌─┐
                        │ ┌──┘ │                   ┌─┐  │    │ │
                  ───── │ ┘    └─                 ─┘ └─ │ ───┘ └─
                X-d ... X ... X+d               X-d ... X ... X+d

                All the trajectory cuts are collected and labelled according
                to whether they are in a rich state or some stressful
                environment.

                If we have N environments. Note that we have N times as
                many rich states. To resolve this, we downsample them.

            The classifer is first validated on the trajectories, then, in a
            bootstrapping manner, trained and tested. The confusion matrix
            produced in each bootstrap is used to calculate the MI estimate.
            The mean is then returned.

            The `halving` parameter specifies whether the experimental but
            much faster sklearn HalvingGridSearch is used over regular GridSearch
            for hyperparameter tuning.
            """
            model = ModelGenerator.get_random_model(4)
            trajectories = OneStepSimulator(
                self.data, self.time_delta, replicates=10
            ).simulate(model)

            est = DecodingEstimator(
                origin=self.origin, interval=self.DEFAULT_INTERVAL, replicates=10
            )
            mi_score = est.estimate(
                model=model, trajectory=trajectories, verbose=False, halving=True
            )

        def choosing_classifiers(self):
            """
            As with this approach, the classifier used to estimate the MI can
            be changed. Ultimately, a classifier that can be validated, trained,
            and make accurate predictions quickly is desired. This estimation step
            is the bottleneck in the entire procedure. As such, an appropriate choice
            for the classifier can make or break the runtime and quality of results.

            By default, the Naive Bayes classifier is used. Multiple others have been
            pre-loaded along with their respective hyperparameters for tuning (see
            DecodingEstimator for the complete list). For these classifiers, switching
            is as simple as specifying the `classifier_name` attribute.

            If one has a specific set of parameters to use, this can be passed to the
            `classifier_params` argument.

            For a custom model, one can add to the dictionary of classifiers or pass
            in the model object itself to the `classifier` argument. Note that the
            object must implement the sklearn Estimator interface.
            """
            # An SVM estimator
            DecodingEstimator(
                self.origin,
                self.DEFAULT_INTERVAL,
                classifier_name="svm",
            )

            # A random forest estimator with specific parameters
            DecodingEstimator(
                self.origin,
                self.DEFAULT_INTERVAL,
                classifier_name="random_forest",
                classifier_params={
                    "criterion": "entropy",
                    "max_depth": 32,
                    "n_estimators": 100,
                },
            )

            # An estimator using a custom classifier
            my_own_classifier = None
            DecodingEstimator(
                self.origin, self.DEFAULT_INTERVAL, classifier=my_own_classifier
            )

        def multiprocessing_decoding(self):
            """
            To speed up hyperparameter tuning, sklearn's `n_jobs` parameter can
            be set to the number of processors one wishes to use. By setting the
            decoder's `parallel` property to True, this is set to -1 which
            corresponds to using all available processors.

            Note: when using multiprocessing for estimating multiple methods by
            way of python's concurrent.futures.ProcessPoolExecutor, a deadlock
            occurs after all futures are completed. As such, nested parallelism
            does not work with this scheme. Using a dask backend, this is possible
            (see examples.benchmarking.sklearn_nested_parallelism).
            """
            est = DecodingEstimator(self.origin, self.DEFAULT_INTERVAL)
            est.parallel = True

        def experimental_trajectory_processing(self):
            """
            The estimator also allows the addition of synthetic noise by way of
            bit-flipping under a Bernoulli(p) scheme and the smoothing of the
            time series. Although, these are only used to demonstrate the
            degradation of MI as noise is added and to test other ways to
            capture temporal features in the time series.
            """
            model = ModelGenerator.get_random_model(4)
            trajectories = OneStepSimulator(
                self.data, self.time_delta, replicates=10
            ).simulate(model)

            est = DecodingEstimator(self.origin, self.DEFAULT_INTERVAL)
            mi_score = est.estimate(model, trajectories, add_noise=True, smoothen=False)

        def wrapping_into_a_pipeline(self):
            """
            If you are still reading all of this, thank you! Very much appreciated.
            Here is a gold star for your efforts: ★

            As the simulator and estimator are often used in synchronison, a
            pipeline can instead be used. To combine both the OneStepSimulator
            and the DecodingEstimator, the OneStepDecodingPipeline can be used.

            By default, for this study, the origin, time delta, and interval are
            all populated by values indicative of the Granados et al. data set.

            The pipeline's decoder can be set to use multiprocessing by calling
            the `set_parallel` method.

            As the trajectories are cut between (origin-interval, origin+interval),
            the pipeline also prematurely terminates the simulator when the
            origin+interval time point is reached.
            """
            pip = OneStepDecodingPipeline(
                exogenous_data=self.data,
                tau=self.time_delta,
                realised=True,
                replicates=10,
                origin=self.origin,
                interval=self.DEFAULT_INTERVAL,
                classifier_name="naive_bayes",
            )
            pip.set_parallel()

            model = ModelGenerator.get_random_model(4)
            mi_score = pip.evaluate(model, verbose=True)

    class GeneticAlgorithm(ClassWithData):
        def __init__(self):
            super().__init__()

        def running_evolution(self):
            """
            The genetic algorithm runner can be initialised with the chosen
            crossover, mutation, and selection operators (see CrossoverOperator,
            MutationOperator, and SelectionOperator for a full list of operators
            and how they work).

            The `scale_fitness` parameter takes in a function with the model and
            MI as its arguments and returns a fitness value by scaling the MI
            based on the model's properties, e.g. number of states or edges (see
            ModelPenalty for a full list of penalties available). By default, it
            is the identity function.

            To run the algorithm, the `run` method is typically called with parameters:
            `states`, the number of states for models in the initial population;
            `population`, the number of models to consider at any one time;
            `elite_ratio`, the proportion of models kept as elites;
            `iterations`, the number of iterations to run the algorithm; and
            `n_processors`, the number of processes to be delegated futures.

            Other parameters include: `runs_per_model`, specifying how many times
            a model is to be estimated for MI (at any point, its running average is
            used for comparison), `initial_population`, a list containing an output
            of models from a previous run to continue the simulation, and
            `model_generator_params`, a dictionary containing arguments for how the
            initial population is randomly generated (see arguments of
            `ModelGeneration.get_random_model`).

            For a HPC, the algorithm can be run as a job from the command line (see
            `jobs.routine_jobs.GeneticAlgorithmJob` for a complete list of parameters
            and their descriptions).

            This returns a list of tuples: (fitness, mi, runs_left, model), and a
            dictionary containing statistics on the trends during runtime, e.g.
            average number of states, average fitness, standard deviation of MI, etc.
            """
            # Genetic operators
            mutations = [
                MutationOperator.edit_edge,
                MutationOperator.add_edge,
                MutationOperator.flip_tf,
                MutationOperator.add_noise,
                MutationOperator.flip_activity,
                MutationOperator.add_activity_noise,
            ]
            crossover = CrossoverOperator.subgraph_swap
            select = SelectionOperator.tournament

            # Fitness scaler
            scale_fitness = ModelPenalty.balanced_state_penalty

            # Initialise the runner
            runner = GeneticRunner(
                data=self.data,
                mutations=mutations,
                crossover=crossover,
                select=select,
                scale_fitness=scale_fitness,
            )

            # Run the genetic algorithm
            models, stats = runner.run(
                states=5,
                population=100,
                elite_ratio=0.1,
                iterations=100,
                n_processors=10,
            )

        def crossover_models(self):
            """
            It may be worth testing how models produce offspring given
            a certain crossover operator. Simply pass the two models as arguments
            to the crossover function and get back their two children.

            Additional parameters `model_is_elite`, `other_is_elite` are used
            during the genetic algorithm to determine whether a model's
            rate function matrix can be reused or must be deep-copied. When
            a model is elite, it persists onto the next generation and thereby
            copying rate functions by reference may cause unwanted side-effects
            if mutations affect them.
            """
            parent1 = ModelGenerator.get_random_model(3)
            parent2 = ModelGenerator.get_random_model(5)
            child1, child2 = CrossoverOperator.subgraph_swap(parent1, parent2)

        def mutate_models(self):
            """
            To mutate models, simply pass them as an argument to the mutation
            function.

            Note: models are directly modified and the returned model is the same
            object reference. It is returned here to allow ease in function
            composition when there are multiple mutation operators to be used
            in the genetic algorithm.
            """
            model = ModelGenerator.get_random_model(4)
            mutated_model = MutationOperator.add_activity_noise(model)
