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

package org.apache.flink.ml.recommendation.als;

import org.apache.flink.ml.common.param.HasPredictionCol;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.param.StringParam;

/**
 * Params for {@link AlsModel}.
 *
 * @param <T> The class type of this instance.
 */
public interface AlsModelParams<T> extends HasPredictionCol<T> {
    Param<String> USER_COL =
            new StringParam("userCol", "Name of user column.", "user", ParamValidators.notNull());

    Param<String> ITEM_COL =
            new StringParam("itemCol", "Name of item column.", "item", ParamValidators.notNull());

    default String getUserCol() {
        return get(USER_COL);
    }

    default T setUserCol(String value) {
        return set(USER_COL, value);
    }

    default String getItemCol() {
        return get(ITEM_COL);
    }

    default T setItemCol(String value) {
        return set(ITEM_COL, value);
    }
}
