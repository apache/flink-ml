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
import os
from typing import Dict, Any, List

import pandas as pd
from pandas._testing import assert_frame_equal
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import Table

from pyflink.ml.core.api import Model
from pyflink.ml.core.builder import PipelineModel, Pipeline
from pyflink.ml.core.param import Param
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class PipelineTest(PyFlinkMLTestCase):

    def test_pipeline_model(self):
        input_table = self.t_env.from_elements([(1,), (2,), (3,)], ['a'])

        model_a = Add10Model()
        model_b = Add10Model()
        model_c = Add10Model()
        model = PipelineModel([model_a, model_b, model_c])
        output_table = model.transform(input_table)[0]

        assert_frame_equal(output_table.to_pandas(),
                           pd.DataFrame([[31], [32], [33]], columns=['a']))

        # Saves and loads the PipelineModel.
        path = os.path.join(self.temp_dir, "test_pipeline_model")
        model.save(path)
        loaded_model = PipelineModel.load(self.env, path)

        output_table2 = loaded_model.transform(input_table)[0]
        assert_frame_equal(output_table2.to_pandas(),
                           pd.DataFrame([[31], [32], [33]], columns=['a']))

    def test_pipeline(self):
        input_table = self.t_env.from_elements([(1,), (2,), (3,)], ['a'])

        model_a = Add10Model()
        model_b = Add10Model()
        estimator = Pipeline([model_a, model_b])
        model = estimator.fit(input_table)
        output_table = model.transform(input_table)[0]

        assert_frame_equal(output_table.to_pandas(),
                           pd.DataFrame([[21], [22], [23]], columns=['a']))

        # Saves and loads the PipelineModel.
        path = os.path.join(self.temp_dir, "test_pipeline")
        estimator.save(path)
        loaded_estimator = Pipeline.load(self.env, path)

        model = loaded_estimator.fit(input_table)
        output_table = model.transform(input_table)[0]
        assert_frame_equal(output_table.to_pandas(),
                           pd.DataFrame([[21], [22], [23]], columns=['a']))


class Add10Model(Model):
    def transform(self, *inputs: Table) -> List[Table]:
        assert len(inputs) == 1
        return [inputs[0].select("a + 10 as a")]

    def save(self, path: str) -> None:
        from pyflink.ml.util import read_write_utils
        read_write_utils.save_metadata(self, path)

    @classmethod
    def load(cls, env: StreamExecutionEnvironment, path: str):
        from pyflink.ml.util import read_write_utils
        return read_write_utils.load_stage_param(path)

    def get_param_map(self) -> Dict['Param[Any]', Any]:
        return {}
