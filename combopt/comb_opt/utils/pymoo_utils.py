from typing import Dict

import numpy as np
import pandas as pd
import torch
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.variable import Real, Integer, Choice, Binary, Variable

from comb_opt.search_space import SearchSpace
from comb_opt.search_space.params.bool_param import BoolPara
from comb_opt.search_space.params.integer_param import IntegerPara
from comb_opt.search_space.params.nominal_param import NominalPara
from comb_opt.search_space.params.numeric_param import NumericPara
from comb_opt.search_space.params.ordinal_param import OrdinalPara
from comb_opt.search_space.params.pow_param import PowPara
from comb_opt.trust_region.tr_manager_base import TrManagerBase
from comb_opt.utils.discrete_vars_utils import get_discrete_choices
from comb_opt.utils.distance_metrics import hamming_distance


class PymooProblem(Problem):
    def __init__(self, search_space: SearchSpace):

        self.search_space = search_space

        vars: Dict[str, Variable] = {}

        for i, name in enumerate(search_space.params):
            param = search_space.params[name]
            if isinstance(param, NumericPara):
                vars[name] = Real(bounds=(param.lb, param.ub))
            elif isinstance(param, PowPara):
                vars[name] = Real(bounds=(
                param.transform(param.param_dict.get('lb')).item(), param.transform(param.param_dict.get('ub')).item()))
            elif isinstance(param, IntegerPara):
                vars[name] = Integer(bounds=(param.lb, param.ub))
            elif isinstance(param, (NominalPara, OrdinalPara)):
                vars[name] = Choice(options=np.arange(len(param.categories)))
            elif isinstance(param, BoolPara):
                vars[name] = Binary()  # TODO: debug this
            else:
                raise Exception(
                    f' The Genetic Algorithm optimizer can only work with numeric,'
                    f' integer, nominal and ordinal variables. Not with {type(param)}')

        super().__init__(vars=vars, n_obj=1, n_ieq_constr=0)

    def pymoo_to_comb_opt(self, x):

        # Convert X to a dictionary compatible with pandas
        x_pd_dict = {}
        for i, var_name in enumerate(self.search_space.param_names):
            param = self.search_space.params[var_name]
            x_pd_dict[self.search_space.param_names[i]] = []
            for j in range(len(x)):
                val = x[j][var_name]
                if isinstance(param, (OrdinalPara, NominalPara)):
                    val = param.categories[val]
                if isinstance(param, PowPara):
                    val = param.inverse_transform(torch.tensor([val])).item()
                x_pd_dict[self.search_space.param_names[i]].append(val)

        return pd.DataFrame(x_pd_dict)

    def comb_opt_to_pymoo(self, x):
        x_pymoo = []
        for i in range(len(x)):
            x_pymoo.append({})
            for j, param_name in enumerate(self.search_space.param_names):
                val = x.iloc[i][param_name]
                param = self.search_space.params[param_name]
                if isinstance(param, (OrdinalPara, NominalPara)):
                    val = param.categories.index(val)
                if isinstance(param, PowPara):
                    val = param.transform(val).item()
                x_pymoo[i][param_name] = val

        return np.array(x_pymoo)

    def _evaluate(self, x, out, *args, **kwargs):
        pass


class TrRepair(Repair):

    def __init__(self, search_space: SearchSpace, tr_manager: TrManagerBase, pymoo_problem: PymooProblem):
        self.search_space = search_space
        self.tr_manager = tr_manager
        self.pymoo_problem = pymoo_problem

        self.nominal_dims = self.search_space.nominal_dims
        self.numeric_dims = self.search_space.cont_dims + self.search_space.disc_dims

        # Dimensions of discrete variables in tensors containing only numeric variables
        self.disc_dims_in_numeric = [i + len(self.search_space.cont_dims) for i in
                                     range(len(self.search_space.disc_dims))]

        self.discrete_choices = get_discrete_choices(search_space)

        self.inverse_mapping = [(self.numeric_dims + self.search_space.nominal_dims).index(i) for i in
                                range(self.search_space.num_dims)]

        self.tr_centre = self.tr_manager.center.unsqueeze(0)
        self.tr_centre_numeric = self.tr_centre[:, self.numeric_dims]
        self.tr_centre_nominal = self.tr_centre[:, self.nominal_dims]

        super(TrRepair, self).__init__()

    def _reconstruct_x(self, x_numeric: torch.FloatTensor, x_nominal: torch.FloatTensor) -> torch.FloatTensor:

        return torch.cat((x_numeric, x_nominal), dim=1)[:, self.inverse_mapping]

    def _do(self, problem, x, **kwargs):
        x_comb_opt = self.pymoo_problem.pymoo_to_comb_opt(x)
        x_normalised = self.search_space.transform(x_comb_opt)

        x_numeric = x_normalised[:, self.numeric_dims]
        x_nominal = x_normalised[:, self.nominal_dims]

        # Repair numeric variables
        if len(self.tr_centre_numeric[0]) > 0:
            delta_numeric = self.tr_centre_numeric[0] - x_numeric
            mask = torch.abs(delta_numeric) > self.tr_manager.radii['numeric']
            x_numeric[mask] += delta_numeric[mask]  # project back to x_centre

        # Repair nominal variables
        if len(self.tr_centre_nominal[0]) > 1:

            # Calculate the hamming distance of all nominal variables from the trust region centre
            d_hamming = hamming_distance(self.tr_centre_nominal, x_nominal, False)

            nominal_valid = d_hamming <= self.tr_manager.radii['nominal']

            # repair all invalid samples
            for sample_num in range(len(nominal_valid)):
                if not nominal_valid[sample_num]:
                    mask = x_nominal[sample_num] != self.tr_centre_nominal[0]
                    indices = np.random.choice([idx for idx, x in enumerate(mask) if x],
                                               size=d_hamming[sample_num].item() - self.tr_manager.radii['nominal'],
                                               replace=False)
                    x_nominal[sample_num, indices] = self.tr_centre_nominal[0, indices]

        x_normalised = self._reconstruct_x(x_numeric, x_nominal)
        x_comb_opt = self.search_space.inverse_transform(x_normalised)
        x_pymoo = self.pymoo_problem.comb_opt_to_pymoo(x_comb_opt)

        return x_pymoo
