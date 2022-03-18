# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import popart
import poptorch


def get_options(config):
    opts = poptorch.Options()
    opts._Popart.set('swapLimitScheduler', 60)
    ipu_list = [0] if config.IPU.PIPELINE is None else config.IPU.PIPELINE
    mem_prop = {f'IPU{i}': 0.15 for i, _ in enumerate(ipu_list)}
    opts.autoRoundNumIPUs(True)
    opts.Training.gradientAccumulation(config.IPU.GA)
    opts.deviceIterations(config.IPU.DEVIterations)
    opts.replicationFactor(config.IPU.REPLIC)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(
            poptorch.AutoStage.SameAsIpu))
    opts.Training.accumulationAndReplicationReductionType(
        poptorch.ReductionType.Mean)
    opts.setAvailableMemoryProportion(mem_prop)
    opts.enableExecutableCaching('./cachedir')
    opts.outputMode(poptorch.OutputMode.All)
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(
        popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True,
                              "TiedGatherAccumulate": True,
                              "UpdateInplacePrioritiesForIpu": True})
    opts.Precision.enableStochasticRounding(True)
    if config.PRECISION[1] == 'half':
        opts.Precision.setPartialsType(torch.half)
    else:
        opts.Precision.setPartialsType(torch.float32)

    return opts
