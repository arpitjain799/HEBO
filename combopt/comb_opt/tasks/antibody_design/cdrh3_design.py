# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

from comb_opt.tasks import TaskBase
from comb_opt.tasks.antibody_design.utils import get_AbsolutNoLib_dir, download_precomputed_antigen_structure, \
    get_valid_antigens, compute_developability_scores, check_constraint_satisfaction


class CDRH3Design(TaskBase):

    @property
    def name(self) -> str:
        return f'{self.antigen} Antibody Design'

    def __init__(self, antigen: str = '1ADQ_A', cdrh3_length: int = 11, num_cpus: int = 10, first_cpu: int = 0,
                 absolut_dir: Optional[str] = None):
        super(CDRH3Design, self).__init__()
        self.num_cpus = num_cpus
        self.first_cpu = first_cpu
        self.antigen = antigen
        self.cdrh3_length = cdrh3_length
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']
        self.amino_acid_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_amino_acid = {value: key for key, value in self.amino_acid_to_idx.items()}

        self.AbsolutNoLib_dir = get_AbsolutNoLib_dir(absolut_dir)
        self.valid_antigens = get_valid_antigens(self.AbsolutNoLib_dir)
        self.need_to_check_precomputed_antigen_structure = True
        assert antigen in self.valid_antigens, f'Specified antigen is not valid. Please choose of from: \n\n {self.valid_antigens}'

    def evaluate(self, x: pd.DataFrame) -> np.ndarray:

        assert os.path.exists(os.path.join(self.AbsolutNoLib_dir, 'antigen_data', f'{self.antigen}'))

        if self.need_to_check_precomputed_antigen_structure:
            download_precomputed_antigen_structure(self.AbsolutNoLib_dir, self.antigen)
            self.need_to_check_precomputed_antigen_structure = False

        # Change working directory
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.AbsolutNoLib_dir, 'antigen_data', f'{self.antigen}'))
        pid = os.getpid()

        sequences = []
        with open(f'TempCDR3_{self.antigen}_pid_{pid}.txt', 'w') as f:
            for i in range(len(x)):
                seq = x.iloc[i]
                seq = ''.join(aa for aa in seq)
                line = f"{i + 1}\t{seq}\n"
                f.write(line)
                sequences.append(seq)

        _ = subprocess.run(
            ['taskset', '-c', f"{self.first_cpu}-{self.first_cpu + self.num_cpus}",
             "./../../AbsolutNoLib", 'repertoire', self.antigen, f"TempCDR3_{self.antigen}_pid_{pid}.txt",
             str(self.num_cpus)], capture_output=True, text=True)

        data = pd.read_csv(f"{self.antigen}FinalBindings_Process_1_Of_1.txt", sep='\t', skiprows=1)

        # Add an extra column to ensure that ordering will be ok after groupby operation
        data['sequence_idx'] = data.apply(lambda row: int(row.ID_slide_Variant.split("_")[0]), axis=1)
        energy = data.groupby(by=['sequence_idx']).min(['Energy'])
        min_energy = energy['Energy'].values.reshape(-1, 1)

        # Remove all created files and change the working directory to what it was
        for i in range(self.num_cpus):
            os.remove(f"TempBindingsFor{self.antigen}_t{i}_Part1_of_1.txt")
        os.remove(f"TempCDR3_{self.antigen}_pid_{pid}.txt")

        os.remove(f"{self.antigen}FinalBindings_Process_1_Of_1.txt")
        os.chdir(current_dir)
        return min_energy

    @staticmethod
    def compute_developability_scores(x: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
        charge, n_gly_seq, max_count = compute_developability_scores(x)
        return charge, n_gly_seq, max_count

    @staticmethod
    def check_constraint_satisfaction(x: pd.DataFrame) -> np.ndarray:
        return check_constraint_satisfaction(x)
