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

package org.apache.flink.ml.api.core;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.table.api.Table;

/**
 * Estimators are responsible for training and generating Models.
 *
 * @param <E> class type of the Estimator implementation itself.
 * @param <M> class type of the Model this Estimator produces.
 */
@PublicEvolving
public interface Estimator<E extends Estimator<E, M>, M extends Model<M>> extends Stage<E> {
    /**
     * Trains on the given inputs and produces a Model.
     *
     * @param inputs a list of tables
     * @return a Model
     */
    M fit(Table... inputs) throws Exception;

    // TODO: remote this. This is only needed when the algorithm still uses DataSet.
    default M fitDataSet(DataSet... inputs) throws Exception {
        return null;
    }
}
