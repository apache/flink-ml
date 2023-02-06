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
from pyflink.ml.evaluation.binaryclassification import BinaryClassificationEvaluator
from pyflink.ml.tests.test_utils import PyFlinkMLTestCase


class BinaryClassificationEvaluatorTest(PyFlinkMLTestCase):
    def setUp(self):
        super(BinaryClassificationEvaluatorTest, self).setUp()
        self.input_data_table = self.t_env.from_data_stream(
            self.env.from_collection([
                (1.0, Vectors.dense(0.1, 0.9)),
                (1.0, Vectors.dense(0.2, 0.8)),
                (1.0, Vectors.dense(0.3, 0.7)),
                (0.0, Vectors.dense(0.25, 0.75)),
                (0.0, Vectors.dense(0.4, 0.6)),
                (1.0, Vectors.dense(0.35, 0.65)),
                (1.0, Vectors.dense(0.45, 0.55)),
                (0.0, Vectors.dense(0.6, 0.4)),
                (0.0, Vectors.dense(0.7, 0.3)),
                (1.0, Vectors.dense(0.65, 0.35)),
                (0.0, Vectors.dense(0.8, 0.2)),
                (1.0, Vectors.dense(0.9, 0.1))
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'rawPrediction'],
                    [Types.DOUBLE(), DenseVectorTypeInfo()]))
        )

        self.input_data_table_score = self.t_env.from_data_stream(
            self.env.from_collection([
                (1, 0.9),
                (1, 0.8),
                (1, 0.7),
                (0, 0.75),
                (0, 0.6),
                (1, 0.65),
                (1, 0.55),
                (0, 0.4),
                (0, 0.3),
                (1, 0.35),
                (0, 0.2),
                (1, 0.1)
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'rawPrediction'],
                    [Types.INT(), Types.DOUBLE()]))
        )

        self.input_data_table_with_multi_score = self.t_env.from_data_stream(
            self.env.from_collection([
                (1.0, Vectors.dense(0.1, 0.9)),
                (1.0, Vectors.dense(0.1, 0.9)),
                (1.0, Vectors.dense(0.1, 0.9)),
                (0.0, Vectors.dense(0.25, 0.75)),
                (0.0, Vectors.dense(0.4, 0.6)),
                (1.0, Vectors.dense(0.1, 0.9)),
                (1.0, Vectors.dense(0.1, 0.9)),
                (0.0, Vectors.dense(0.6, 0.4)),
                (0.0, Vectors.dense(0.7, 0.3)),
                (1.0, Vectors.dense(0.1, 0.9)),
                (0.0, Vectors.dense(0.8, 0.2)),
                (1.0, Vectors.dense(0.9, 0.1))
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'rawPrediction'],
                    [Types.DOUBLE(), DenseVectorTypeInfo()]))
        )

        self.input_data_table_with_weight = self.t_env.from_data_stream(
            self.env.from_collection([
                (1.0, Vectors.dense(0.1, 0.9), 0.8),
                (1.0, Vectors.dense(0.1, 0.9), 0.7),
                (1.0, Vectors.dense(0.1, 0.9), 0.5),
                (0.0, Vectors.dense(0.25, 0.75), 1.2),
                (0.0, Vectors.dense(0.4, 0.6), 1.3),
                (1.0, Vectors.dense(0.1, 0.9), 1.5),
                (1.0, Vectors.dense(0.1, 0.9), 1.4),
                (0.0, Vectors.dense(0.6, 0.4), 0.3),
                (0.0, Vectors.dense(0.7, 0.3), 0.5),
                (1.0, Vectors.dense(0.1, 0.9), 1.9),
                (0.0, Vectors.dense(0.8, 0.2), 1.2),
                (1.0, Vectors.dense(0.9, 0.1), 1.0)
            ],
                type_info=Types.ROW_NAMED(
                    ['label', 'rawPrediction', 'weight'],
                    [Types.DOUBLE(), DenseVectorTypeInfo(), Types.DOUBLE()]))
        )

        self.expected_data = [0.7691481137909708, 0.3714285714285714, 0.6571428571428571]

        self.expected_data_m = [0.8571428571428571, 0.9377705627705628,
                                0.8571428571428571, 0.6488095238095237]

        self.expected_data_w = 0.8911680911680911

        self.eps = 1e-5

    def test_param(self):
        binary_classification_evaluator = BinaryClassificationEvaluator()

        self.assertEqual('label', binary_classification_evaluator.label_col)
        self.assertIsNone(binary_classification_evaluator.weight_col)
        self.assertEqual('rawPrediction', binary_classification_evaluator.raw_prediction_col)
        self.assertEqual(("areaUnderROC", "areaUnderPR"),
                         binary_classification_evaluator.metrics_names)

        binary_classification_evaluator.set_label_col('labelCol') \
            .set_raw_prediction_col('raw') \
            .set_metrics_names("areaUnderROC") \
            .set_weight_col("weight")

        self.assertEqual('labelCol', binary_classification_evaluator.label_col)
        self.assertEqual('weight', binary_classification_evaluator.weight_col)
        self.assertEqual('raw', binary_classification_evaluator.raw_prediction_col)
        self.assertEqual(("areaUnderROC",), binary_classification_evaluator.metrics_names)

    def test_evaluate(self):
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderPR", "ks", "areaUnderROC")
        output = evaluator.transform(self.input_data_table)[0]
        self.assertEqual(
            ["areaUnderPR", "ks", "areaUnderROC"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        for i in range(len(self.expected_data)):
            self.assertAlmostEqual(self.expected_data[i], result[i], delta=self.eps)

    def test_evaluate_with_double_raw(self):
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderPR", "ks", "areaUnderROC")
        output = evaluator.transform(self.input_data_table_score)[0]
        self.assertEqual(
            ["areaUnderPR", "ks", "areaUnderROC"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        for i in range(len(self.expected_data)):
            self.assertAlmostEqual(self.expected_data[i], result[i], delta=self.eps)

    def test_more_subtask_than_data(self):
        self.env.set_parallelism(15)
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderPR", "ks", "areaUnderROC")
        output = evaluator.transform(self.input_data_table)[0]
        self.assertEqual(
            ["areaUnderPR", "ks", "areaUnderROC"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        for i in range(len(self.expected_data)):
            self.assertAlmostEqual(self.expected_data[i], result[i], delta=self.eps)

    def test_evaluate_with_multi_score(self):
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderROC", "areaUnderPR", "ks", "areaUnderLorenz")
        output = evaluator.transform(self.input_data_table_with_multi_score)[0]
        self.assertEqual(
            ["areaUnderROC", "areaUnderPR", "ks", "areaUnderLorenz"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        for i in range(len(self.expected_data_m)):
            self.assertAlmostEqual(self.expected_data_m[i], result[i], delta=self.eps)

    def test_evaluate_with_weight(self):
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderROC") \
            .set_weight_col("weight")
        output = evaluator.transform(self.input_data_table_with_weight)[0]
        self.assertEqual(
            ["areaUnderROC"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        self.assertAlmostEqual(self.expected_data_w, result[0], delta=self.eps)

    def test_save_load_and_evaluate(self):
        evaluator = BinaryClassificationEvaluator() \
            .set_metrics_names("areaUnderPR", "ks", "areaUnderROC")
        path = os.path.join(self.temp_dir, 'test_save_load_and_evaluate_binary_classification')
        evaluator.save(path)
        evaluator = BinaryClassificationEvaluator.load(self.t_env, path)
        output = evaluator.transform(self.input_data_table)[0]
        self.assertEqual(
            ["areaUnderPR", "ks", "areaUnderROC"],
            output.get_schema().get_field_names())
        results = [result for result in output.execute().collect()]
        result = results[0]
        for i in range(len(self.expected_data)):
            self.assertAlmostEqual(self.expected_data[i], result[i], delta=self.eps)
