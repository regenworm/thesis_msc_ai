import time
import os


def get_results_dir():
    current_dir = os.path.realpath(__file__)
    return os.path.join(current_dir, '..', 'results')


def create_run_dir(results_dir=None):
    if results_dir is None:
        results_dir = get_results_dir()

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(results_dir, timestr)
    os.mkdir(run_dir)

    # create embedding, data, and plots folders
    os.mkdir(os.path.join(run_dir, 'model_output'))
    os.mkdir(os.path.join(run_dir, 'data'))
    os.mkdir(os.path.join(run_dir, 'plots'))

    return run_dir
