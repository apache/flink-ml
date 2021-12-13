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

/** Interface for the shared raw prediction column param. */
public interface HasRawPredictionCol<T> extends WithParams<T> {

    Param<String> RAW_PREDICTION_COL =
            new StringParam("rawPredictionCol", "Raw prediction column name.", "rawPrediction");

    default String getRawPredictionCol() {
        return get(RAW_PREDICTION_COL);
    }

    default T setRawPredictionCol(String value) {
        return set(RAW_PREDICTION_COL, value);
    }
}
