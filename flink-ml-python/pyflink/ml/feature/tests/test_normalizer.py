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
from pyflink.ml.feature.normalizer import Normalizer
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class NormalizerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(NormalizerTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (Vectors.dense(2.1, 3.1, 2.3, 3.4, 5.3, 5.1),),
                (Vectors.dense(2.3, 4.1, 1.3, 2.4, 5.1, 4.1),),
            ],
                type_info=Types.ROW_NAMED(
                    ["intput_vec"],
                    [DenseVectorTypeInfo()])))
        self.expected_output_data = [
            Vectors.dense(
                0.17386300895299714,
                0.25665491797823387,
                0.19042139075804446,
                0.28149249068580484,
                0.43879711783375464,
                0.42223873602870726),
            Vectors.dense(
                0.20785190042726007,
                0.3705186051094636,
                0.11748150893714701,
                0.2168889395762714,
                0.4608889965995767,
                0.3705186051094636)]

    def test_param(self):
        normalizer = Normalizer()

        self.assertEqual('input', normalizer.get_input_col())
        self.assertEqual('output', normalizer.get_output_col())
        self.assertEqual(2.0, normalizer.get_p())

        normalizer.set_input_col("intput_vec") \
            .set_output_col('output_vec') \
            .set_p(1.5)

        self.assertEqual("intput_vec", normalizer.get_input_col())
        self.assertEqual(1.5, normalizer.get_p())
        self.assertEqual(float, type(normalizer.get_p()))
        self.assertEqual('output_vec', normalizer.get_output_col())

    def test_save_load_transform(self):
        normalizer = Normalizer() \
            .set_input_col("intput_vec") \
            .set_output_col('output_vec') \
            .set_p(1.5)

        path = os.path.join(self.temp_dir, 'test_save_load_transform_normalizer')
        normalizer.save(path)
        normalizer = Normalizer.load(self.t_env, path)

        output_table = normalizer.transform(self.input_data_table)[0]
        actual_outputs = [(result[1]) for result in
                          self.t_env.to_data_stream(output_table).execute_and_collect()]

        self.assertEqual(2, len(actual_outputs))
        actual_outputs.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
        self.expected_output_data.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))
        self.assertEqual(self.expected_output_data, actual_outputs)
