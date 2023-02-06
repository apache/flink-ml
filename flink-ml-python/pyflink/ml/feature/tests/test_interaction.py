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

from pyflink.common import Types

from pyflink.ml.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.feature.interaction import Interaction
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class InteractionTest(PyFlinkMLTestCase):
    def setUp(self):
        super(InteractionTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1,
                 Vectors.dense(1, 2),
                 Vectors.dense(3, 4)),
                (2,
                 Vectors.dense(2, 8),
                 Vectors.dense(3, 4))
            ],
                type_info=Types.ROW_NAMED(
                    ['f0', 'f1', 'f2'],
                    [Types.INT(), DenseVectorTypeInfo(), DenseVectorTypeInfo()])))

        self.expected_output_data_1 = Vectors.dense(3.0, 4.0, 6.0, 8.0)
        self.expected_output_data_2 = Vectors.dense(12.0, 16.0, 48.0, 64.0)

    def test_param(self):
        interaction = Interaction()
        self.assertEqual('output', interaction.output_col)

        interaction.set_input_cols('f0', 'f1', 'f2') \
            .set_output_col('interaction_vec')

        self.assertEqual(('f0', 'f1', 'f2'), interaction.input_cols)
        self.assertEqual('interaction_vec', interaction.output_col)

    def test_save_load_transform(self):
        interaction = Interaction() \
            .set_input_cols('f0', 'f1', 'f2') \
            .set_output_col('interaction_vec')

        path = os.path.join(self.temp_dir, 'test_save_load_transform_interaction')
        interaction.save(path)
        interaction = Interaction.load(self.t_env, path)

        output_table = interaction.transform(self.input_data_table)[0]
        actual_outputs = [(result[0], result[3]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(2, len(actual_outputs))
        for actual_output in actual_outputs:
            self.assertEqual(4, len(actual_output[1]))
            if actual_output[0] == 1:
                for i in range(len(actual_output[1])):
                    self.assertAlmostEqual(self.expected_output_data_1.get(i),
                                           actual_output[1].get(i), delta=1e-5)
            else:
                for i in range(len(actual_output[1])):
                    self.assertAlmostEqual(self.expected_output_data_2.get(i),
                                           actual_output[1].get(i), delta=1e-5)
