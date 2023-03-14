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

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

/** Interface for the shared model version column param. */
public interface HasModelVersionCol<T> extends WithParams<T> {
    Param<String> MODEL_VERSION_COL =
            new StringParam(
                    "modelVersionCol",
                    "The name of the column which contains the version of the model "
                            + "data that the input data is predicted with. The version "
                            + "should be a 64-bit integer.",
                    "version");

    default String getModelVersionCol() {
        return get(MODEL_VERSION_COL);
    }

    default T setModelVersionCol(String value) {
        return set(MODEL_VERSION_COL, value);
    }
}
