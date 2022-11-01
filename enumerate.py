import itertools

# method for making combinations of possible answers
def make_hyperparam_combos(config, ordered_config_keys):
  for k in ordered_config_keys:
    a = list(config.values())
    # print(len(list(itertools.product(*a))))
    return list(itertools.product(*a))
