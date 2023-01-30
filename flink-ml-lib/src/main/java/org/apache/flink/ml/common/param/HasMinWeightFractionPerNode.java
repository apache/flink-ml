/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.flink.ml.common.param;

import org.apache.flink.ml.param.DoubleParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.WithParams;

/** Interface for shared param minWeightFractionPerNode. */
public interface HasMinWeightFractionPerNode<T> extends WithParams<T> {
    Param<Double> MIN_WEIGHT_FRACTION_PER_NODE =
            new DoubleParam(
                    "minWeightFractionPerNode",
                    "Minimum fraction of the weighted sample count that each node must have. If a split causes the left or right child to have a smaller fraction of the total weight than minWeightFractionPerNode, the split is invalid.",
                    0.,
                    ParamValidators.gtEq(0.));

    default double getMinWeightFractionPerNode() {
        return get(MIN_WEIGHT_FRACTION_PER_NODE);
    }

    default T setMinWeightFractionPerNode(Double value) {
        return set(MIN_WEIGHT_FRACTION_PER_NODE, value);
    }
}
