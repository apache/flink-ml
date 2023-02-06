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
from pyflink.ml.common.param import HasOutputCol, HasInputCol

from pyflink.ml.wrapper import JavaWithParams

from pyflink.ml.param import FloatParam, BooleanParam, ParamValidators, IntParam

from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator


class _CountVectorizerModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol,
):
    """
    Params for :class:`CountVectorizerModel`.
    """
    MIN_TF: FloatParam = FloatParam(
        "min_t_f",
        "Filter to ignore rare words in a document. For each document, "
        "terms with frequency/count less than the given threshold are ignored."
        "If this is an integer >= 1, then this specifies a count (of times "
        "the term must appear in the document); if this is a double in [0,1), "
        "then this specifies a fraction (out of the document's token count).",
        1.0,
        ParamValidators.gt_eq(0.0)
    )

    BINARY: BooleanParam = BooleanParam(
        "binary",
        "Binary toggle to control the output vector values. If True, all "
        "nonzero counts (after minTF filter applied) are set to 1.0.",
        False
    )

    def __init__(self, java_params):
        super(_CountVectorizerModelParams, self).__init__(java_params)

    def set_min_tf(self, value: float):
        return typing.cast(_CountVectorizerModelParams,
                           self.set(self.MIN_TF, float(value)))

    def get_min_tf(self):
        return self.get(self.MIN_TF)

    def set_binary(self, value: bool):
        return typing.cast(_CountVectorizerModelParams, self.set(self.BINARY, value))

    def get_binary(self):
        return self.get(self.BINARY)

    @property
    def min_tf(self):
        return self.get_min_tf()

    @property
    def binary(self):
        return self.get_binary()


class _CountVectorizerParams(_CountVectorizerModelParams):
    """
    Params for :class:`CountVectorizer`.
    """
    VOCABULARY_SIZE: IntParam = IntParam(
        "vocabulary_size",
        "Max size of the vocabulary. CountVectorizer will build a vocabulary "
        "that only considers the top vocabularySize terms ordered by term "
        "frequency across the corpus.",
        1 << 18,
        ParamValidators.gt(0)
    )

    MIN_DF: FloatParam = FloatParam(
        "min_d_f",
        "Specifies the minimum number of different documents a term must"
        "appear in to be included in the vocabulary. If this is an "
        "integer >= 1, this specifies the number of documents the term must "
        "appear in; if this is a double in [0,1), then this specifies the "
        "fraction of documents.",
        1.0,
        ParamValidators.gt_eq(0.0)
    )

    MAX_DF: FloatParam = FloatParam(
        "max_d_f",
        "Specifies the maximum number of different documents a term could "
        "appear in to be included in the vocabulary. A term that appears "
        "more than the threshold will be ignored. If this is an integer >= 1,"
        "this specifies the maximum number of documents the term could "
        "appear in; if this is a double in [0,1), then this specifies the "
        "maximum fraction of documents the term could appear in.",
        float(2**63 - 1),
        ParamValidators.gt_eq(0.0)
    )

    def __init__(self, java_params):
        super(_CountVectorizerParams, self).__init__(java_params)

    def set_vocabulary_size(self, value: str):
        return typing.cast(_CountVectorizerParams, self.set(self.VOCABULARY_SIZE, value))

    def get_vocabulary_size(self) -> str:
        return self.get(self.VOCABULARY_SIZE)

    def set_min_df(self, value: float):
        return typing.cast(_CountVectorizerParams, self.set(self.MIN_DF, float(value)))

    def get_min_df(self):
        return self.get(self.MIN_DF)

    def set_max_df(self, value: float):
        return typing.cast(_CountVectorizerParams, self.set(self.MAX_DF, float(value)))

    def get_max_df(self):
        return self.get(self.MAX_DF)

    @property
    def vocabulary_size(self):
        return self.get_vocabulary_size()

    @property
    def min_df(self):
        return self.get_min_df()

    @property
    def max_df(self):
        return self.get_max_df()


class CountVectorizerModel(JavaFeatureModel, _CountVectorizerModelParams):
    """
    A Model which transforms data using the model data computed by CountVectorizer.
    """

    def __init__(self, java_model=None):
        super(CountVectorizerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "countvectorizer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "CountVectorizerModel"


class CountVectorizer(JavaFeatureEstimator, _CountVectorizerParams):
    """
    An Estimator which converts a collection of text documents
    to vectors of token counts. When an a-priori dictionary is not available,
    CountVectorizer can be used as an estimator to extract the vocabulary,
    and generates a CountVectorizerModel. The model produces sparse
    representations for the documents over the vocabulary, which can then
    be passed to other algorithms like LDA.
    """

    def __init__(self):
        super(CountVectorizer, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> CountVectorizerModel:
        return CountVectorizerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "countvectorizer"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "CountVectorizer"
