from comb_opt.factory import task_factory

if __name__ == '__main__':
    tkwargs = {"ntk_name": 'sqrt'}
    task, search_space = task_factory('mig_optimization', seq_len=2, **tkwargs)

    x = search_space.sample(5)
    y = task(x)
    print(y)