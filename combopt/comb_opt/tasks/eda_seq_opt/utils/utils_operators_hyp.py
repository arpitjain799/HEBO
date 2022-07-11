# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary
# forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Dict, List, Any, Optional

from comb_opt.tasks.eda_seq_opt.utils.utils_operators import BOILS_MAPPING_OPERATORS, BOILS_PRE_MAPPING_OPERATORS, \
    BOILS_POST_MAPPING_OPERATORS

# ------------ Pre-mapping --------------- #

BOILS_PRE_MAPPING_ALGO_PARAMS = {
    op.op_id: [] for op in BOILS_PRE_MAPPING_OPERATORS
}

# ------------ Mapping --------------- #

BOILS_MAPPING_ALGO_PARAMS = {
    op.op_id: [] for op in BOILS_MAPPING_OPERATORS
}

# ------------ Post-mapping --------------- #

BOILS_POST_MAPPING_ALGO_PARAMS = {
    op.op_id: [] for op in BOILS_POST_MAPPING_OPERATORS
}


# ------------------------------------------------------------------------------------ #


class OperatorHypSpace:

    def __init__(self,
                 pre_mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 post_mapping_operator_hyps: Dict[str, List[Dict[str, Any]]],
                 ):
        self.pre_mapping_operator_hyps = pre_mapping_operator_hyps
        self.mapping_operator_hyps = mapping_operator_hyps
        self.post_mapping_operator_hyps = post_mapping_operator_hyps
        self.all_hyps: Dict[str, List[Dict[str, Any]]] = {}
        self.all_hyps.update(self.pre_mapping_operator_hyps)
        self.all_hyps.update(self.mapping_operator_hyps)
        self.all_hyps.update(self.post_mapping_operator_hyps)


HYPERPARAMS_SPACES: Dict[str, OperatorHypSpace] = {
    'boils_hyp_op_space': OperatorHypSpace(
        pre_mapping_operator_hyps=BOILS_PRE_MAPPING_ALGO_PARAMS,
        mapping_operator_hyps=BOILS_MAPPING_ALGO_PARAMS,
        post_mapping_operator_hyps=BOILS_POST_MAPPING_ALGO_PARAMS
    ),
}


def get_operator_hyperparms_space(operator_hyperparams_space_id: Optional[str]) -> Optional[OperatorHypSpace]:
    if operator_hyperparams_space_id is None or operator_hyperparams_space_id == "":
        return None
    return HYPERPARAMS_SPACES[operator_hyperparams_space_id]
