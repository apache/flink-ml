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

package org.apache.flink.ml.feature.vectorslicer;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.IntArrayParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidator;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Params of {@link VectorSlicer}.
 *
 * @param <T> The class type of this instance.
 */
public interface VectorSlicerParams<T> extends HasInputCol<T>, HasOutputCol<T> {
    Param<Integer[]> INDICES =
            new IntArrayParam(
                    "indices",
                    "An array of indices to select features from a vector column.",
                    null,
                    indicesValidator());

    default Integer[] getIndices() {
        return get(INDICES);
    }

    default T setIndices(Integer... value) {
        return set(INDICES, value);
    }

    // Checks the indices parameter.
    static ParamValidator<Integer[]> indicesValidator() {
        return indices -> {
            if (indices == null) {
                return false;
            }
            for (Number ele : indices) {
                if (ele.doubleValue() < 0) {
                    return false;
                }
            }
            Set<Integer> set = new HashSet<>(Arrays.asList(indices));
            if (set.size() != indices.length) {
                return false;
            }
            return set.size() != 0;
        };
    }
}
