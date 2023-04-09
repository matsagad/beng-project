def main():
    """
    Examples are given below. Uncomment them and their respective
    imports to get started. Note that the data folder must first
    be populated by the single-cell microscopy data by Granados et al.

    Alternatively, jobs can be run from the command line
    (see `jobs/README.md` for syntax and list of supported jobs).
    """

    """
    Tutorial to get started with accessing the library.
    """
    # from examples.tutorial import TutorialExamples

    # tut_dh_ex = TutorialExamples.DataHandling()
    # tut_dh_ex.raw_data_structure()
    # tut_dh_ex.data_format()

    # tut_pm_ex = TutorialExamples.PromoterModels()
    # tut_pm_ex.creating_a_model()
    # tut_pm_ex.randomly_generating_models()
    # tut_pm_ex.visualising_models()

    # tut_ts_ex = TutorialExamples.TrajectorySimulation()
    # tut_ts_ex.producing_a_trajectory()
    # tut_ts_ex.producing_the_probability_distribution_trajectory()
    # tut_ts_ex.visualising_trajectories()

    # tut_mie_ex = TutorialExamples.MIEstimation()
    # tut_mie_ex.estimating_mutual_information()
    # tut_mie_ex.choosing_classifiers()
    # tut_mie_ex.multiprocessing_decoding()
    # tut_mie_ex.experimental_trajectory_processing()
    # tut_mie_ex.wrapping_into_a_pipeline()

    # tut_ga_ex = TutorialExamples.GeneticAlgorithm()
    # tut_ga_ex.running_evolution()
    # tut_ga_ex.crossover_models()
    # tut_ga_ex.mutate_models()

    """
    Benchmarking simulations, classifier speeds, and multiprocessing.
    """
    # from examples.benchmarking import BenchmarkingExamples
    # bm_ex = BenchmarkingExamples()
    # bm_ex.matrix_exponentials()
    # bm_ex.trajectory_simulation()
    # bm_ex.mi_estimation()
    # bm_ex.mi_estimation_table()
    # bm_ex.sklearn_nested_parallelism()
    # bm_ex.genetic_multiprocessing_overhead()

    """
    Exploring changes in MI estimates as methodology is tweaked.
    """
    # from examples.mi_trends import MITrendsExamples
    # mit_ex = MITrendsExamples()
    # mit_ex.mi_vs_interval()
    # mit_ex.mi_distribution()
    # mit_ex.max_mi_estimation()
    # mit_ex.mi_vs_repeated_intervals()

    """
    Visualisation of models, trajectories, activities, etc.
    """
    # from examples.visualisation import VisualisationExamples
    # vis_ex = VisualisationExamples()
    # vis_ex.visualise_model()
    # vis_ex.visualise_trajectory()
    # vis_ex.visualise_activity()
    # vis_ex.visualise_tf_concentration()
    # vis_ex.visualise_activity_grid()
    # vis_ex.visualise_tf_concentration_grid()
    # vis_ex.visualise_random_model_generation()
    # vis_ex.visualise_crossover()
    # vis_ex.visualise_crossover_chart()
    # vis_ex.visualise_crossbreeding()

    """
    Optimisation of (simple and) arbitrary model weights.
    """
    # from examples.optimisation import OptimisationExamples
    # op_ex = OptimisationExamples()
    # op_ex.grid_search_simple()
    # op_ex.particle_swarm_simple()
    # op_ex.particle_swarm_optimise()

    """
    Analysis of genetic algorithm results.
    """
    # from examples.genetic_analysis import GeneticAnalysisExamples
    # ga_ex = GeneticAnalysisExamples()
    # ga_ex.evaluate_models()
    # ga_ex.visualise_models()
    # ga_ex.examine_run_stats()
    # ga_ex.evaluate_tf_presence_in_models()


if __name__ == "__main__":
    main()
