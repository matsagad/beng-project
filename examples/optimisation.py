from models.preset import Preset
from optimisation.grid_search import GridSearch
from optimisation.particle_swarm import ParticleSwarm
from utils.data import ClassWithData


class OptimisationExamples(ClassWithData):
    def __init__(self):
        super().__init__()
        self.model = Preset.simple(2, 1, 1)

    def grid_search_simple(self):
        """
        Perform a grid search to find the best (simple) two-state
        model using any of the TFs available. Here, the rate function
        from active to inactive is assumed to be constant. A heatmap
        is plotted for each TF.
        """
        gs = GridSearch()
        gs.optimise_simple(
            self.data,
            self.tf_names,
            fname=f"{self.SAVE_FOLDER}/optimise_simple.png",
            cache_folder=self.CACHING_FOLDER,
        )

    def particle_swarm_simple(self):
        """
        Perform a particle swarm optimisation to find the best rate
        parameters given a (simple) two-state model with constant and
        linear rate functions and a TF to depend on.

        This uses a legacy method `_optimise_simple` which is no longer
        of use. A more capable and non-discriminating method is `optimise`
        as shown below, which can handle arbitrary models.
        """
        ps = ParticleSwarm()
        ps._optimise_simple(self.data, tf=2)

    def particle_swarm_optimise(self):
        """
        Perform a particle swarm optimisation to find the best rate
        parameters given an arbitrary model. The type of rate functions
        and the TFs they depend on are kept constants, but constant rates
        are tweaked.
        """
        ps = ParticleSwarm()
        ps.optimise(self.data, self.model, start_at_pos=False)
