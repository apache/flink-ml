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

package org.apache.flink.ml.benchmark;

import org.apache.flink.ml.api.Stage;
import org.apache.flink.table.api.Table;

/** Interface for specifying a machine learning algorithm (i.e., {@link Stage}) for benchmark. */
public interface BenchmarkStage<S extends Stage<S>> {
    /**
     * Returns the input tables for training.
     *
     * @param context The context information of this benchmark.
     * @return The train data.
     */
    Table[] getTrainData(BenchmarkContext context);

    /**
     * Returns the input tables for testing. It treats the training data as the test data by
     * default.
     *
     * @param context The context information of this benchmark.
     * @return The test data.
     */
    default Table[] getTestData(BenchmarkContext context) {
        return getTrainData(context);
    }

    /**
     * Returns the stage that is to be benchmarked.
     *
     * @param context The context information of this benchmark.
     * @return The stage to be benchmarked.
     */
    Stage<S> getStage(BenchmarkContext context);
}
