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
import typing

from pyflink.ml.param import IntParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.common.param import HasInputCol, HasOutputCol


class _IDFModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`IDFModel`.
    """

    def __init__(self, java_params):
        super(_IDFModelParams, self).__init__(java_params)


class _IDFParams(_IDFModelParams):
    """
    Params for :class:`IDF`.
    """

    MIN_DOC_FREQ: IntParam = IntParam(
        "min_doc_freq",
        "Minimum number of documents that a term should appear for filtering.",
        0,
        ParamValidators.gt_eq(0))

    def __init__(self, java_params):
        super(_IDFParams, self).__init__(java_params)

    def set_min_doc_freq(self, value: int):
        return typing.cast(_IDFParams, self.set(self.MIN_DOC_FREQ, value))

    def get_min_doc_freq(self) -> int:
        return self.get(self.MIN_DOC_FREQ)

    @property
    def min_doc_freq(self):
        return self.get_min_doc_freq()


class IDFModel(JavaFeatureModel, _IDFModelParams):
    """
    A Model which transforms data using the model data computed by :class::IDF.
    """

    def __init__(self, java_model=None):
        super(IDFModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "idf"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "IDFModel"


class IDF(JavaFeatureEstimator, _IDFParams):
    """
    An Estimator that computes the inverse document frequency (IDF) for the input documents.
    IDF is computed following `idf = log((m + 1) / (d(t) + 1))`, where `m` is the total
    number of documents and `d(t)` is the number of documents that contains `t`.

    <p>Users could filter out terms that appeared in little documents by setting
    {@link IDFParams#getMinDocFreq()}.

    <p>See https://en.wikipedia.org/wiki/Tf%E2%80%93idf.
    """

    def __init__(self):
        super(IDF, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> IDFModel:
        return IDFModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "idf"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "IDF"
