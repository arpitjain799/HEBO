# Comes from BOiLS
import glob
import os
from typing import List, Dict

import numpy as np

from .utils import get_circuits_path_root

EPFL_ARITHMETIC = ['hyp', 'div', 'log2', 'multiplier', 'sqrt', 'square', 'sin', 'bar', 'adder', 'max']
EPFL_CONTROL = ['arbiter', 'cavlc', 'ctrl', 'dec', 'i2c', 'int2float', 'mem_ctrl', 'priority', 'router', 'voter']
EPFL_MTM = ['sixteen', 'twenty', 'twentythree']

DESIGN_GROUPS: Dict[str, List[str]] = {
    'epfl_arithmetic': EPFL_ARITHMETIC,
    'epfl_control': EPFL_CONTROL,
    'epfl_mtm': EPFL_MTM,
}

for file in glob.glob(f"{get_circuits_path_root()}/*.blif"):
    circ_name = os.path.basename(file[:-5])
    DESIGN_GROUPS[circ_name] = [circ_name]

EPFLS = [EPFL_ARITHMETIC, EPFL_CONTROL, EPFL_MTM]
for epfl in EPFLS:
    for design in epfl:
        DESIGN_GROUPS[design] = [design]

AUX_TEST_GP = ['adder', 'bar']
AUX_TEST_ABC_GRAPH = ['adder', 'sin']

DESIGN_GROUPS['aux_test_designs_group'] = AUX_TEST_GP
DESIGN_GROUPS['aux_test_abc_graph'] = AUX_TEST_ABC_GRAPH


def get_designs_path(designs_id: str, frac_part: str = None) -> List[str]:
    """ Get list of filepaths to designs """

    designs_filepath: List[str] = []
    if designs_id in DESIGN_GROUPS:
        group = DESIGN_GROUPS[designs_id]
    else:
        try:
            from comb_opt.tasks.eda_seq_opt.utils.utils_design_groups_perso import DESIGN_GROUPS_PERSO
            if designs_id in DESIGN_GROUPS_PERSO:
                group = DESIGN_GROUPS_PERSO[designs_id]
        except ModuleNotFoundError:
            raise
    for design_id in group:
        designs_filepath.append(os.path.join(get_circuits_path_root(), f'{design_id}.blif'))
    if frac_part is None:
        s = slice(0, len(designs_filepath))
    else:
        i, j = map(int, frac_part.split('/'))
        assert j > 0 and i > 0, (i, j)
        step = int(np.ceil(len(designs_filepath) / j))
        s = slice((i - 1) * step, i * step)

    return designs_filepath[s]


if __name__ == '__main__':

    designs_id_ = 'test_designs_group'
    N = 6
    for n in range(1, N + 1):
        frac = f'{n}/{N}'
        print(f'{frac} -----> ', end='')
        print(get_designs_path(designs_id=designs_id_, frac_part=frac))
