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

from typing import Tuple

from pyflink.ml.param import Param, StringArrayParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.evaluation.common import JavaEvaluationAlgoOperator
from pyflink.ml.common.param import HasLabelCol, HasRawPredictionCol, HasWeightCol


class _BinaryClassificationEvaluatorParams(
    JavaWithParams,
    HasLabelCol,
    HasRawPredictionCol,
    HasWeightCol
):
    """
    Params for :class:`BinaryClassificationEvaluator`.
    """

    METRICS_NAMES: Param[Tuple[str, ...]] = StringArrayParam(
        "metrics_names",
        "Names of output metrics.",
        ["areaUnderROC", "areaUnderPR"],
        ParamValidators.is_sub_set(["areaUnderROC", "areaUnderPR", "areaUnderLorenz", "ks"]))

    def __init__(self, java_params):
        super(_BinaryClassificationEvaluatorParams, self).__init__(java_params)

    def set_metrics_names(self, *value: str):
        return self.set(self.METRICS_NAMES, value)

    def get_metrics_names(self) -> Tuple[str, ...]:
        return self.get(self.METRICS_NAMES)

    @property
    def metrics_names(self) -> Tuple[str, ...]:
        return self.get_metrics_names()


class BinaryClassificationEvaluator(
    JavaEvaluationAlgoOperator,
    _BinaryClassificationEvaluatorParams
):
    """
    An AlgoOperator which calculates the evaluation metrics for binary classification. The input
    data has columns rawPrediction, label and an optional weight column. The rawPrediction can be
    of type double (binary 0/1 prediction, or probability of label 1) or of type vector (length-2
    vector of raw predictions, scores, or label probabilities). The output may contain different
    metrics which will be defined by parameter metrics_names.
    """

    def __init__(self, java_algo_operator=None):
        super(BinaryClassificationEvaluator, self).__init__(java_algo_operator)

    @classmethod
    def _java_algo_operator_package_name(cls) -> str:
        return "binaryclassification"

    @classmethod
    def _java_algo_operator_class_name(cls) -> str:
        return "BinaryClassificationEvaluator"
