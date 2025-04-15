# SPDX-License-Identifier: Apache-2.0
class Llama70BH100:
    target_c_kv = 6.24369264e-05
    target_c_compute = 6.40216301e-02
    target_c_batch_size = -3.97382298e-02
    target_c_enable_spec_decode = -1.12291902e-01
    target_c_fixed = 18.34535
    draft_percentage = 0.04
    overhead_percentage = 0.05


class Llama8BH100:
    target_c_kv = 4.85907942e-05
    target_c_compute = 1.84383603e-02
    target_c_batch_size = -4.33144783e-02
    target_c_fixed = 6.938699571857084
    target_c_enable_spec_decode = 0
    draft_percentage = 0.04
    overhead_percentage = 0.05

    draft_c_kv = 2.89700974e-06
    draft_c_compute = 4.38694070e-05
    draft_c_num_spec_tokens = 4.52175870e-01
    draft_c_batch_size = 5.52805909e-03
    draft_c_fixed = 0.4471109584339972


dsd_model_config = {
    "llama-70b-h100": Llama70BH100,
    "llama-8b-h100": Llama8BH100,
}
