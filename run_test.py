from tests.mi_estimation import MIEstimationTests
from tests.genetic_operator import GeneticOperatorTests

def main():
  MIEstimationTests().run_all_tests()
  GeneticOperatorTests().run_all_tests()


if __name__ == "__main__":
  main()