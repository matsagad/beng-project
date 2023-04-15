from jobs.routine_jobs import (
    GeneticAlgorithmJob,
    ParticleSwarmWeightOptimisationJob,
    NoveltySearchJob,
)
import sys
import signal

HOURS = 3600
jobs = {
    "genetic_algorithm": GeneticAlgorithmJob(verbose=True),
    "pso_weight_optimisation": ParticleSwarmWeightOptimisationJob(verbose=True),
    "novelty_search": NoveltySearchJob(verbose=True),
}


class Alarm(Exception):
    pass


def alarm_handler(*args):
    raise Alarm


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

    max_runtime = int(arg_dict.get("max_runtime", 2 * HOURS))

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(max_runtime)

    try:
        jobs[job_type].run(arg_dict)
        signal.alarm(0)
    except Alarm:
        jobs[job_type].on_interrupted()


if __name__ == "__main__":
    main()
