HEADER = [
    'dataset',
    'method',
    'fold_no',
    'additional_info',
    'no_states',
    'maxiter',
    'accuracy',
    'mcc',
    'degenerated_share',
    'mean_no_iterations',
    'max_no_iterations',
    'complete_execution_time',
    'cmeans_time',
    'no_random_initializations',
    'covariance_type',
    'mutation',
    'recombination',
    'popsize']


def get_header():
    return HEADER

def get_meaningful_columns():
    return [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17]