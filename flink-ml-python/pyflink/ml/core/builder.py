################################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from typing import List, TypeVar, Dict, Any

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import Table

from pyflink.ml.core.api import Estimator, Model, AlgoOperator, Stage
from pyflink.ml.core.param import Param

E = TypeVar('E')


class PipelineModel(Model):
    """
    A PipelineModel acts as a Model. It consists of an ordered list of stages, each of which could
    be a Model, Transformer or AlgoOperator.
    """

    def __init__(self, stages: List[Stage]):
        self._stages = stages
        self._param_map = {}  # type: Dict[Param[Any], Any]

    def transform(self, *inputs: Table) -> List[Table]:
        """
        Applies all stages in this PipelineModel on the input tables in order. The output of one
        stage is used as the input of the next stage (if any). The output of the last stage is
        returned as the result of this method.

        :param inputs: A list of tables.
        :return: A list of tables.
        """
        for stage in self._stages:
            if isinstance(stage, AlgoOperator):
                inputs = stage.transform(*inputs)
            else:
                raise TypeError(f"The stage {stage} must be an AlgoOperator.")
        return list(inputs)

    def save(self, path: str) -> None:
        from pyflink.ml.util import read_write_utils
        read_write_utils.save_pipeline(self, self._stages, path)

    @classmethod
    def load(cls, env: StreamExecutionEnvironment, path: str) -> 'PipelineModel':
        from pyflink.ml.util import read_write_utils
        return PipelineModel(read_write_utils.load_pipeline(env, path))

    def get_param_map(self):
        return self._param_map


class Pipeline(Estimator[E, PipelineModel]):
    """
    A Pipeline acts as an Estimator. It consists of an ordered list of stages, each of which could
    be an Estimator, Model, Transformer or AlgoOperator.
    """

    def __init__(self, stages: List[Stage]):
        self._stages = stages
        self._param_map = {}  # type: Dict[Param[Any], Any]

    def fit(self, *inputs: Table) -> PipelineModel:
        """
        Trains the pipeline to fit on the given tables.

        This method goes through all stages of this pipeline in order and does the following on
        each stage until the last Estimator (inclusive).

        <ul>
            <li> If a stage is an Estimator, invoke :func:`~Estimator.fit` with the input
                tables to generate a Model. And if there is Estimator after this stage, transform
                the input tables using the generated Model to get result tables, then pass the
                result tables to the next stage as inputs.
            <li> If a stage is an AlgoOperator AND there is Estimator after this stage, transform
                the input tables using this stage to get result tables, then pass the result tables
                to the next stage as inputs.
        </ul>

        After all the Estimators are trained to fit their input tables, a new PipelineModel will
        be created with the same stages in this pipeline, except that all the Estimators in the
        PipelineModel are replaced with the models generated in the above process.

        :param inputs: A list of tables.
        :return: A PipelineModel.
        """
        last_estimator_idx = -1
        for i, stage in enumerate(self._stages):
            if isinstance(stage, Estimator):
                last_estimator_idx = i

        model_stages = []
        last_inputs = inputs
        for i, stage in enumerate(self._stages):
            if isinstance(stage, Estimator):
                model_stage = stage.fit(*last_inputs)
            else:
                model_stage = stage
            model_stages.append(model_stage)

            if i < last_estimator_idx:
                last_inputs = model_stage.transform(*last_inputs)

        return PipelineModel(model_stages)

    def save(self, path: str) -> None:
        from pyflink.ml.util import read_write_utils
        read_write_utils.save_pipeline(self, self._stages, path)

    @classmethod
    def load(cls, env: StreamExecutionEnvironment, path: str) -> 'Pipeline':
        from pyflink.ml.util import read_write_utils
        return Pipeline(read_write_utils.load_pipeline(env, path))

    def get_param_map(self):
        return self._param_map
