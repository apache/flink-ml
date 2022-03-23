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

package org.apache.flink.ml.feature.stringindexer;

import org.apache.flink.ml.common.param.HasInputCols;
import org.apache.flink.ml.common.param.HasOutputCols;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params of {@link StringIndexerModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface StringIndexerModelParams<T> extends HasInputCols<T>, HasOutputCols<T> {
    String ERROR_INVALID = "error";
    String SKIP_INVALID = "skip";
    String KEEP_INVALID = "keep";

    /**
     * Supported options and the corresponding behavior to handle invalid entries in
     * StringIndexerModel are listed as follows.
     *
     * <ul>
     *   <li>error: raise an exception.
     *   <li>skip: filter out rows with bad values.
     *   <li>keep: put the invalid entries in a special bucket, whose index is the number of
     *       distinct values in this column.
     * </ul>
     */
    Param<String> HANDLE_INVALID =
            new StringParam(
                    "handleInvalid",
                    "Strategy to handle invalid entries.",
                    ERROR_INVALID,
                    ParamValidators.inArray(ERROR_INVALID, SKIP_INVALID, KEEP_INVALID));

    default String getHandleInvalid() {
        return get(HANDLE_INVALID);
    }

    default T setHandleInvalid(String value) {
        return set(HANDLE_INVALID, value);
    }
}
