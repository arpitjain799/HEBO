# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import pandas as pd


# TODO 'Result logger supports n_suggests==1 for now'
class ResultsLogger:

    def __init__(self):
        self.columns = ["Eval Num", 'f(x)', 'f(x*)', 'Elapsed Time']

        self.data = []

    def append(self, eval_num: int, y: float, y_star: float, elapsed_time: float):
        self.data.append([int(eval_num), y, y_star, elapsed_time])

    def save(self, save_path):
        if save_path.split('.')[-1] != 'csv':
            save_path = save_path + '.csv'

        pd.DataFrame(self.data, columns=self.columns).to_csv(save_path, index=False)

    def restart(self):
        self.data = []
