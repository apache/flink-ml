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

package org.apache.flink.ml.feature.onehotencoder;

import org.apache.flink.ml.common.param.HasHandleInvalid;
import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasOutputCols;
import org.apache.flink.ml.param.BooleanParam;
import org.apache.flink.ml.param.Param;

/**
 * Params of OneHotEncoderModel.
 *
 * <p>The `keep` and `skip` option of {@link HasHandleInvalid} is not supported in {@link
 * OneHotEncoderParams}.
 *
 * @param <T> The class type of this instance.
 */
public interface OneHotEncoderParams<T>
        extends HasInputCols<T>, HasOutputCols<T>, HasHandleInvalid<T> {
    Param<Boolean> DROP_LAST =
            new BooleanParam("dropLast", "Whether to drop the last category.", true);

    default boolean getDropLast() {
        return get(DROP_LAST);
    }

    default T setDropLast(boolean value) {
        return set(DROP_LAST, value);
    }
}
