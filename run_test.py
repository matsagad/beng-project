from tests.distance_metric import DistanceMetricTests
from tests.genetic_operator import GeneticOperatorTests
from tests.mi_estimation import MIEstimationTests

def main():
  DistanceMetricTests().run_all_tests()
  GeneticOperatorTests().run_all_tests()
  MIEstimationTests().run_all_tests()



if __name__ == "__main__":
  main()