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

package org.apache.flink.ml.feature.bucketizer;

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasOutputCols;
import org.apache.flink.ml.param.DoubleArrayArrayParam;
import org.apache.flink.ml.param.ParamValidator;

/**
 * Params for {@link Bucketizer}.
 *
 * @param <T> The class type of this instance.
 */
public interface BucketizerParams<T>
        extends HasInputCols<T>, HasOutputCols<T>, HasHandleInvalid<T> {
    /**
     * The array of split points for mapping continuous features into buckets for multiple columns.
     *
     * <p>Each input column is supposed to be mapped into {numberOfSplitPoints - 1} buckets. A
     * bucket is defined by two split points. For example, bucket(x,y) contains values in the range
     * [x,y). An exception is that the last bucket also contains y. The array should contain at
     * least three split points and be strictly increasing.
     */
    DoubleArrayArrayParam SPLITS_ARRAY =
            new DoubleArrayArrayParam(
                    "splitsArray",
                    "Array of split points for mapping continuous features into buckets.",
                    null,
                    new SplitsArrayValidator());

    default Double[][] getSplitsArray() {
        return get(SPLITS_ARRAY);
    }

    default T setSplitsArray(Double[][] value) {
        set(SPLITS_ARRAY, value);
        return (T) this;
    }

    /** Param validator for splitsArray. */
    class SplitsArrayValidator implements ParamValidator<Double[][]> {

        @Override
        public boolean validate(Double[][] splitsArray) {
            if (null == splitsArray || splitsArray.length == 0) {
                return false;
            }
            for (Double[] splits : splitsArray) {
                if (splits.length < 3) {
                    return false;
                }
                for (int j = 1; j < splits.length; j++) {
                    if (splits[j] <= splits[j - 1]) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}
