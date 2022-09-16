from comb_opt.factory import task_factory

if __name__ == '__main__':
    task, search_space = task_factory('mig_optimization', seq_len=2)

    x = search_space.sample(3)
    y = task(x)
    print(y)