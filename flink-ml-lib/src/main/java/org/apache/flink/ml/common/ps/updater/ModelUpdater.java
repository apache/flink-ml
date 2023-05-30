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

package org.apache.flink.ml.common.ps.updater;

import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;

import java.io.Serializable;
import java.util.Iterator;

/**
 * A model updater that could be used to handle push/pull request from workers.
 *
 * <p>Note that model updater should also ensure that model data is robust to failures, by writing
 * model data to snapshots.
 */
public interface ModelUpdater extends Serializable {

    /** Initializes the model data. */
    void open(long startFeatureIndex, long endFeatureIndex);

    /** Applies the push to update the model data, e.g., using gradient to update model. */
    void handlePush(long[] keys, double[] values);

    /** Applies the pull and return the retrieved model data. */
    double[] handlePull(long[] keys);

    /**
     * Returns model segments with the format of (startFeatureIdx, endFeatureIdx, modelValues). The
     * model segments are continuously updated/retrieved by push/pull(i.e., `handlePush` and
     * `handlePull`).
     */
    Iterator<Tuple3<Long, Long, double[]>> getModelSegments();

    /** Recovers the model data from state. */
    void initializeState(StateInitializationContext context) throws Exception;

    /** Snapshots the model data to state. */
    void snapshotState(StateSnapshotContext context) throws Exception;
}
