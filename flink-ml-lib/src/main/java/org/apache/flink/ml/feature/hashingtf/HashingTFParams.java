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

package org.apache.flink.ml.feature.hashingtf;

import org.apache.flink.ml.common.param.HasInputCol;
import org.apache.flink.ml.common.param.HasNumFeatures;
import org.apache.flink.ml.common.param.HasOutputCol;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.Param;

/**
 * Params of {@link HashingTF}.
 *
 * @param <T> The class type of this instance.
 */
public interface HashingTFParams<T> extends HasInputCol<T>, HasOutputCol<T>, HasNumFeatures<T> {

    /**
     * Supported options to decide whether each dimension of the output vector is binary or not.
     *
     * <ul>
     *   <li>true: the value at one dimension is set as 1 if there are some features hashed to this
     *       column.
     *   <li>false: the value at one dimension is set as number of features that has been hashed to
     *       this column.
     * </ul>
     */
    Param<Boolean> BINARY =
            new BooleanParam(
                    "binary",
                    "Whether each dimension of the output vector is binary or not.",
                    false);

    default boolean getBinary() {
        return get(BINARY);
    }

    default T setBinary(boolean value) {
        return set(BINARY, value);
    }
}
