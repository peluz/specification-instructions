import numpy as np
from collections import OrderedDict


def fix_run_idxs(suite, ids):
    i=0
    for func, _ in suite.tests.items():
        suite.tests[func].result_indexes = ids[func]
        suite.tests[func].run_idxs = np.array(list(OrderedDict.fromkeys(ids[func])))
        suite.test_ranges[func] = (i, i+len(ids[func]))
        i+=len(ids[func])

def get_pass_rates(suite):
    results = {} 
    all_hits = {}
    for k, v in suite.tests.items():
        if k not in ['"used to" should reduce', 'reducers']:
            score = 100 - v.get_stats()['fail_rate']
            results[k] = score
            filtered = v.filtered_idxs()
            fails = v.fail_idxs()
            n_tests = v.get_stats().testcases_run
            hits = np.ones(n_tests)
            hits[fails] = 0
            hits[filtered] = np.nan
            all_hits[k] = hits.tolist()
        else:
            continue
    return results, all_hits

def get_test_hits(suite):
    results = OrderedDict()
    for k, v in suite.tests.items():
        if k in ['"used to" should reduce', "reducers"]:
            continue
        filtered = v.filtered_idxs()
        fails = v.fail_idxs()
        n_tests = v.get_stats().testcases_run
        hits = np.ones(n_tests)
        hits[fails] = 0
        hits[filtered] = np.nan
        results[k] = hits
    return results