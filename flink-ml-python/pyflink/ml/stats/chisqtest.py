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

from pyflink.ml.param import Param, BooleanParam
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.stats.common import JavaStatsAlgoOperator
from pyflink.ml.common.param import HasFeaturesCol, HasLabelCol


class _ChiSqTestParams(
    JavaWithParams,
    HasFeaturesCol,
    HasLabelCol
):
    """
    Params for :class:`ChiSqTest`.
    """

    FLATTEN: Param[bool] = BooleanParam(
        "flatten",
        "If false, the returned table contains only a single row, otherwise, one row per feature.",
        False)

    def set_flatten(self, value: bool):
        return self.set(self.FLATTEN, value)

    def get_flatten(self) -> bool:
        return self.get(self.FLATTEN)

    @property
    def flatten(self) -> bool:
        return self.get_flatten()


class ChiSqTest(JavaStatsAlgoOperator, _ChiSqTestParams):
    """
    An AlgoOperator which implements the Chi-square test algorithm.

    Chi-square Test computes the statistics of independence of variables in a contingency table,
    e.g., p-value, and DOF(number of degrees of freedom) for each input feature. The contingency
    table is constructed from the observed categorical values.

    The input of this algorithm is a table containing a labelColumn of numerical type and a
    featuresColumn of vector type. Each index in the input vector represents a feature to be tested.
    By default, the output of this algorithm is a table containing a single row with the following
    columns, each of which has one value per feature.

    - "pValues": vector
    - "degreesOfFreedom": int array
    - "statistics": vector

    The output of this algorithm can be flattened to multiple rows by setting
    HasFlatten#FLATTEN to True, which would contain the following columns:

    - "featureIndex": int
    - "pValue": double
    - "degreeOfFreedom": int
    - "statistic": double
    """

    def __init__(self, java_algo_operator=None):
        super(ChiSqTest, self).__init__(java_algo_operator)

    @classmethod
    def _java_algo_operator_package_name(cls) -> str:
        return "chisqtest"

    @classmethod
    def _java_algo_operator_class_name(cls) -> str:
        return "ChiSqTest"
