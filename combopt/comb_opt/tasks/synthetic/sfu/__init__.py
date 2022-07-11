# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from comb_opt.tasks.synthetic.sfu.ackley import Ackley
from comb_opt.tasks.synthetic.sfu.griewank import Griewank
from comb_opt.tasks.synthetic.sfu.langermann import Langermann
from comb_opt.tasks.synthetic.sfu.levy import Levy
from comb_opt.tasks.synthetic.sfu.rastrigin import Rastrigin
from comb_opt.tasks.synthetic.sfu.schwefel import Schwefel
from comb_opt.tasks.synthetic.sfu.perm0 import Perm0
from comb_opt.tasks.synthetic.sfu.rotated_hyper_ellipsoid import RotHyp
from comb_opt.tasks.synthetic.sfu.sphere import Sphere
from comb_opt.tasks.synthetic.sfu.modified_sphere import ModifiedSphere
from comb_opt.tasks.synthetic.sfu.sum_pow import SumPow
from comb_opt.tasks.synthetic.sfu.sum_squares import SumSquares
from comb_opt.tasks.synthetic.sfu.trid import Trid
from comb_opt.tasks.synthetic.sfu.power_sum import PowSum
from comb_opt.tasks.synthetic.sfu.zakharov import Zakharov
from comb_opt.tasks.synthetic.sfu.dixon_prince import DixonPrince
from comb_opt.tasks.synthetic.sfu.rosenbrock import Rosenbrock
from comb_opt.tasks.synthetic.sfu.michalewicz import Michalewicz
from comb_opt.tasks.synthetic.sfu.perm import Perm
from comb_opt.tasks.synthetic.sfu.powell import Powell
from comb_opt.tasks.synthetic.sfu.styblinski_tang import StyblinskiTang

from .default_params_factory import default_sfu_params_factory

MANY_LOCAL_MINIMA = {'ackley': Ackley,
                     'griewank': Griewank,
                     'langermann': Langermann,
                     'levy': Levy,
                     'rastrigin': Rastrigin,
                     'schwefel': Schwefel}

BOWL_SHAPED = {'perm0': Perm0,
               'rot_hyp': RotHyp,
               'sphere': Sphere,
               'modified_sphere': ModifiedSphere,
               'sum_pow': SumPow,
               'sum_squares': SumSquares,
               'trid': Trid}

PLATE_SHAPED = {'power_sum': PowSum,
                'zakharov': Zakharov}

VALLEY_SHAPED = {'dixon_prince': DixonPrince,
                 'rosenbrock': Rosenbrock}

STEEP_RIDGES = {'michalewicz': Michalewicz}

OTHER = {'perm': Perm,
         'powell': Powell,
         'styblinski_tang': StyblinskiTang}

SFU_FUNCTIONS = dict(**MANY_LOCAL_MINIMA, **BOWL_SHAPED, **PLATE_SHAPED, **VALLEY_SHAPED, **STEEP_RIDGES, **OTHER)
