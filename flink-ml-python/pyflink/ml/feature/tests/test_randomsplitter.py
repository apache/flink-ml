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

from pyflink.ml.feature.randomsplitter import RandomSplitter
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class RandomSplitterTest(PyFlinkMLTestCase):
    def setUp(self):
        super(RandomSplitterTest, self).setUp()
        data = []
        for i in range(1, 10000):
            data.append((i, ))
        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection(
                data,
                type_info=Types.ROW_NAMED(
                    ['f0', ],
                    [Types.INT(), ])))

    def test_param(self):
        splitter = RandomSplitter()
        splitter.set_weights(0.2, 0.8).set_seed(5)
        self.assertEqual(0.2, splitter.weights[0])
        self.assertEqual(0.8, splitter.weights[1])
        self.assertEqual(5, splitter.seed)

    def test_output_schema(self):
        splitter = RandomSplitter()
        input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                ('', ''),
            ],
                type_info=Types.ROW_NAMED(
                    ['test_input', 'dummy_input'],
                    [Types.STRING(), Types.STRING()])))
        output = splitter.set_weights(0.5, 0.5).set_seed(0) \
            .transform(input_data_table)[0]

        self.assertEqual(
            ['test_input', 'dummy_input'],
            output.get_schema().get_field_names())

    def test_transform(self):
        splitter = RandomSplitter().set_weights(0.4, 0.6).set_seed(0)

        output = splitter.transform(self.input_table)
        results = [result for result in self.t_env.to_data_stream(output[0]).execute_and_collect()]
        self.assertAlmostEqual(len(results) / 4000.0, 1.0, delta=0.1)

    def test_save_load_transform(self):
        splitter = RandomSplitter().set_weights(0.4, 0.6).set_seed(0)
        path = os.path.join(self.temp_dir, 'test_save_load_random_splitter')
        splitter.save(path)
        splitter = RandomSplitter.load(self.t_env, path)

        output = splitter.transform(self.input_table)
        results = [result for result in self.t_env.to_data_stream(output[0]).execute_and_collect()]
        self.assertAlmostEqual(len(results) / 4000.0, 1.0, delta=0.1)
