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

from pyflink.common import Types
from pyflink.common.time import Time, Instant
from pyflink.java_gateway import get_gateway
from pyflink.table import Schema
from pyflink.table.types import DataTypes
from pyflink.table.expressions import col

from pyflink.ml.core.linalg import Vectors, DenseVectorTypeInfo
from pyflink.ml.lib.feature.onlinestandardscaler import OnlineStandardScaler, \
    OnlineStandardScalerModel
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase, update_existing_params
from pyflink.ml.core.windows import GlobalWindows, EventTimeTumblingWindows


class OnlineStandardScalerTest(PyFlinkMLTestCase):
    def setUp(self):
        super(OnlineStandardScalerTest, self).setUp()

        dense_vector_serializer = get_gateway().jvm.org.apache.flink.table.types.logical.RawType(
            get_gateway().jvm.org.apache.flink.ml.linalg.DenseVector(0).getClass(),
            get_gateway().jvm.org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer()
        ).getSerializerString()

        schema = Schema.new_builder() \
            .column("ts_in_long", DataTypes.BIGINT()) \
            .column("ts", "TIMESTAMP_LTZ(3)") \
            .column("input", "RAW('org.apache.flink.ml.linalg.DenseVector', '{serializer}')"
                    .format(serializer=dense_vector_serializer)) \
            .watermark("ts", "ts - INTERVAL '1' SECOND") \
            .build()

        self.input_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (0, Instant.of_epoch_milli(0), Vectors.dense(-2.5, 9, 1),),
                (1000, Instant.of_epoch_milli(1000), Vectors.dense(1.4, -5, 1),),
                (2000, Instant.of_epoch_milli(2000), Vectors.dense(2, -1, -2),),
                (6000, Instant.of_epoch_milli(6000), Vectors.dense(0.7, 3, 1),),
                (7000, Instant.of_epoch_milli(7000), Vectors.dense(0, 1, 1),),
                (8000, Instant.of_epoch_milli(8000), Vectors.dense(0.5, 0, -2),),
                (9000, Instant.of_epoch_milli(9000), Vectors.dense(0.4, 1, 1),),
                (10000, Instant.of_epoch_milli(10000), Vectors.dense(0.3, 2, 1),),
                (11000, Instant.of_epoch_milli(11000), Vectors.dense(0.5, 1, -2),)
            ],
                type_info=Types.ROW_NAMED(
                    ['ts_in_long', 'ts', 'input'],
                    [Types.LONG(), Types.INSTANT(), DenseVectorTypeInfo()])),
            schema)

        self.window_size_ms = 3000

        self.expected_model_data = [
            [
                Vectors.dense(0.3, 1, 0),
                Vectors.dense(2.4433583, 7.2111026, 1.7320508),
                0,
                2999
            ],
            [
                Vectors.dense(0.35, 1.1666667, 0),
                Vectors.dense(1.5630099, 4.6654760, 1.5491933),
                1,
                8999
            ],
            [
                Vectors.dense(0.3666667, 1.2222222, 0),
                Vectors.dense(1.2369316, 3.7006005, 1.5),
                2,
                11999
            ]
        ]

        self.tolerance = 1e-7

    def test_param(self):
        standard_scaler = OnlineStandardScaler()

        self.assertEqual('input', standard_scaler.input_col)
        self.assertEqual(False, standard_scaler.with_mean)
        self.assertEqual(True, standard_scaler.with_std)
        self.assertEqual('output', standard_scaler.output_col)
        self.assertEqual('version', standard_scaler.model_version_col)
        self.assertEqual(GlobalWindows(), standard_scaler.windows)
        self.assertEqual(0, standard_scaler.max_allowed_model_delay_ms)

        standard_scaler.set_input_col('test_input') \
            .set_with_mean(True) \
            .set_with_std(False) \
            .set_output_col('test_output') \
            .set_model_version_col('test_version') \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(3000))) \
            .set_max_allowed_model_delay_ms(3000)

        self.assertEqual('test_input', standard_scaler.input_col)
        self.assertEqual(True, standard_scaler.with_mean)
        self.assertEqual(False, standard_scaler.with_std)
        self.assertEqual('test_output', standard_scaler.output_col)
        self.assertEqual("test_version", standard_scaler.model_version_col)
        self.assertEqual(EventTimeTumblingWindows.of(Time.milliseconds(3000)),
                         standard_scaler.windows)
        self.assertEqual(3000, standard_scaler.max_allowed_model_delay_ms)

    def test_output_schema(self):
        temp_table = self.input_table.alias('ts_in_long', 'ts', "test_input")

        # Tests the case when modelVersionCol is not null.
        standard_scaler = OnlineStandardScaler() \
            .set_input_col('test_input') \
            .set_output_col('test_output')

        output = standard_scaler.fit(temp_table).transform(temp_table)[0]

        self.assertEqual(['ts_in_long', 'ts', 'test_input', 'test_output', 'version'],
                         output.get_schema().get_field_names())

        # Tests the case when modelVersionCol is null.
        standard_scaler.set_model_version_col(None)

        output = standard_scaler.fit(temp_table).transform(temp_table)[0]

        self.assertEqual(['ts_in_long', 'ts', 'test_input', 'test_output'],
                         output.get_schema().get_field_names())

    def test_fit_and_predict(self):
        standard_scaler = OnlineStandardScaler()

        standard_scaler \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(self.window_size_ms)))
        output = standard_scaler.fit(self.input_table).transform(self.input_table)[0]
        self.verify_used_model_version(output, standard_scaler.model_version_col,
                                       standard_scaler.max_allowed_model_delay_ms)

    def test_get_model_data(self):
        standard_scaler = OnlineStandardScaler()

        standard_scaler \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(self.window_size_ms)))

        model_data = standard_scaler.fit(self.input_table).get_model_data()[0]
        self.assertEqual(["mean", "std", "version", "timestamp"],
                         model_data.get_schema().get_field_names())

        model_rows = [result for result in
                      self.t_env.to_data_stream(model_data).execute_and_collect()]
        self.assertEqual(len(self.expected_model_data), len(model_rows))
        for idx in range(len(self.expected_model_data)):
            self.assertListAlmostEqual(self.expected_model_data[idx][0].to_array(),
                                       model_rows[idx][0].to_array(),
                                       delta=self.tolerance)
            self.assertListAlmostEqual(self.expected_model_data[idx][1].to_array(),
                                       model_rows[idx][1].to_array(),
                                       delta=self.tolerance)
            self.assertEqual(self.expected_model_data[idx][2], model_rows[idx][2])
            self.assertEqual(self.expected_model_data[idx][3], model_rows[idx][3])

    def test_set_model_data(self):
        standard_scaler = OnlineStandardScaler()

        model = standard_scaler \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(self.window_size_ms))) \
            .fit(self.input_table)
        model_data = model.get_model_data()[0]

        new_model = OnlineStandardScalerModel().set_model_data(model_data)
        update_existing_params(new_model, model)
        output = new_model.transform(self.input_table)[0]

        self.verify_used_model_version(output, model.model_version_col,
                                       model.max_allowed_model_delay_ms)

    def test_save_load_and_predict(self):
        standard_scaler = OnlineStandardScaler() \
            .set_windows(EventTimeTumblingWindows.of(Time.milliseconds(self.window_size_ms)))

        reloaded_standard_scaler = self.save_and_reload(standard_scaler)
        model = reloaded_standard_scaler.fit(self.input_table)
        reloaded_model = self.save_and_reload(model)
        model_data = model.get_model_data()[0]
        reloaded_model.set_model_data(model_data)
        output = reloaded_model.transform(self.input_table)[0]
        self.verify_used_model_version(output, reloaded_model.model_version_col,
                                       reloaded_model.max_allowed_model_delay_ms)

    def verify_used_model_version(self, output_table, model_version_col,
                                  max_allowed_model_delay_ms):
        # TODO: remove this line when Instant values can be acquired without errors.
        output_table = output_table.select(col("ts_in_long"), col(model_version_col))
        collected_results = [result for result in
                             self.t_env.to_data_stream(output_table).execute_and_collect()]
        for row in collected_results:
            data_timestamp = row[0]
            model_version = row[1]
            model_time_stamp = self.expected_model_data[model_version][3]
            self.assertTrue(data_timestamp - model_time_stamp <= max_allowed_model_delay_ms)
