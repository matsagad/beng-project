from jobs.common_jobs import GeneticAlgorithmJob
import sys

jobs = {"genetic_algorithm": GeneticAlgorithmJob(verbose=True)}


def main():
    _args = sys.argv[1:]
    if not _args:
        print("No arguments provided. Expecting: {job_type} {arg_name=arg_value}*")
        return

    job_type, *args = sys.argv[1:]
    if job_type not in jobs:
        print(
            f"Incorrect job type provided. Expecting any of: {' '.join(jobs.keys())}."
        )
        return

    arg_dict = arg_dict = dict(arg.split("=") for arg in args if "=" in arg)
    jobs[job_type].run(arg_dict)


if __name__ == "__main__":
    main()
