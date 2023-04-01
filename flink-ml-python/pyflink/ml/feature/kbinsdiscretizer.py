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

from pyflink.ml.common.param import HasInputCol, HasOutputCol
from pyflink.ml.feature.common import JavaFeatureModel, JavaFeatureEstimator
from pyflink.ml.param import IntParam, StringParam, ParamValidators
from pyflink.ml.wrapper import JavaWithParams


class _KBinsDiscretizerModelParams(
    JavaWithParams,
    HasInputCol,
    HasOutputCol
):
    """
    Params for :class:`KBinsDiscretizerModel`.
    """

    def __init__(self, java_params):
        super(_KBinsDiscretizerModelParams, self).__init__(java_params)


class _KBinsDiscretizerParams(_KBinsDiscretizerModelParams):
    """
    Params for :class:`KBinsDiscretizer`.
    """

    """
    Supported options to define the widths of the bins are listed as follows.
    <ul>
        <li>uniform: all bins in each feature have identical widths.
        <li>quantile: all bins in each feature have the same number of points.
        <li>kmeans: values in each bin have the same nearest center of a 1D kmeans cluster.
    </ul>
    """
    STRATEGY: StringParam = StringParam(
        "strategy",
        "Strategy used to define the width of the bin.",
        'quantile',
        ParamValidators.in_array(['uniform', 'quantile', 'kmeans']))

    NUM_BINS: IntParam = IntParam(
        "num_bins",
        "Number of bins to produce.",
        5,
        ParamValidators.gt_eq(2)
    )

    SUB_SAMPLES: IntParam = IntParam(
        "sub_samples",
        "Maximum number of samples used to fit the model.",
        200000,
        ParamValidators.gt_eq(2)
    )

    def __init__(self, java_params):
        super(_KBinsDiscretizerParams, self).__init__(java_params)

    def set_strategy(self, value: str):
        return typing.cast(_KBinsDiscretizerParams, self.set(self.STRATEGY, value))

    def get_strategy(self) -> str:
        return self.get(self.STRATEGY)

    def set_num_bins(self, value: int):
        return typing.cast(_KBinsDiscretizerParams, self.set(self.NUM_BINS, value))

    def get_num_bins(self) -> int:
        return self.get(self.NUM_BINS)

    def set_sub_samples(self, value: int):
        return typing.cast(_KBinsDiscretizerParams, self.set(self.SUB_SAMPLES, value))

    def get_sub_samples(self) -> int:
        return self.get(self.SUB_SAMPLES)

    @property
    def strategy(self):
        return self.get_strategy()

    @property
    def num_bins(self):
        return self.get_num_bins()

    @property
    def sub_samples(self):
        return self.get_sub_samples()


class KBinsDiscretizerModel(JavaFeatureModel, _KBinsDiscretizerModelParams):
    """
    A Model which transforms continuous features into discrete features using the model data
    computed by :class::KBinsDiscretizer.

    <p>A feature value `v` should be mapped to a bin with edges as `{left, right}` if `v` is
    in `[left, right)`. If `v` does not fall into any of the bins, it is mapped to the
    closest bin. For example uppose the bin edges are `{-1, 0, 1}` for one column, then
    we have two bins `{-1, 0}` and `{0, 1}`. In this case, -2 is mapped into 0-th bin,
    0 is mapped into the 1-st bin and 2 is mapped into the 1-st bin.
    """

    def __init__(self, java_model=None):
        super(KBinsDiscretizerModel, self).__init__(java_model)

    @classmethod
    def _java_model_package_name(cls) -> str:
        return "kbinsdiscretizer"

    @classmethod
    def _java_model_class_name(cls) -> str:
        return "KBinsDiscretizerModel"


class KBinsDiscretizer(JavaFeatureEstimator, _KBinsDiscretizerParams):
    """
    An Estimator which implements discretization (also known as quantization or binning) to
    transform continuous features into discrete ones. The output values are in [0, numBins).

    <p>KBinsDiscretizer implements three different binning strategies, and it can be set by {@link
    KBinsDiscretizerParams#STRATEGY}. If the strategy is set as
    {@link KBinsDiscretizerParams#KMEANS} or {@link KBinsDiscretizerParams#QUANTILE},
    users should further set {@link KBinsDiscretizerParams#SUB_SAMPLES} for
    better performance.

    <p>There are several corner cases for different inputs as listed below:

    <ul>
        <li>When the input values of one column are all the same, then they should be mapped
        to the same bin (i.e., the zero-th bin). Thus the corresponding bin edges are
        {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY}.
        <li>When the number of distinct values of one column is less than the specified
        number of bins and the {@link KBinsDiscretizerParams#STRATEGY} is set as {@link
        KBinsDiscretizerParams#KMEANS}, we switch to {@link KBinsDiscretizerParams#UNIFORM}.
        <li>When the width of one output bin is zero, i.e., the left edge equals to the
        right edge of the bin, we replace the right edge as the average value of its two
        neighbors. One exception is that the last two edges are the same --- in this case,
        the left edge is updated as the average of its two neighbors. For example,
        the bin edges {0, 1, 1, 2, 2} are transformed into {0, 1, 1.5, 1.75, 2}.
    </ul>
    """

    def __init__(self):
        super(KBinsDiscretizer, self).__init__()

    @classmethod
    def _create_model(cls, java_model) -> KBinsDiscretizerModel:
        return KBinsDiscretizerModel(java_model)

    @classmethod
    def _java_estimator_package_name(cls) -> str:
        return "kbinsdiscretizer"

    @classmethod
    def _java_estimator_class_name(cls) -> str:
        return "KBinsDiscretizer"
