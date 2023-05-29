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

package org.apache.flink.ml.common.ps.training;

import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.runtime.util.ResettableIterator;

import java.io.Serializable;

/**
 * Stores the context information that is alive during the training process. Note that the context
 * information will be updated by each {@link IterationStage}.
 *
 * <p>Note that subclasses should take care of the snapshot of object stored in {@link
 * TrainingContext} if the object satisfies that: the write-process is followed by an {@link
 * PullStage}, which is later again read by other stages.
 */
public interface TrainingContext extends Serializable {
    /** Sets the current iteration ID. */
    default void setIterationId(int iterationId) {}

    /** Sets the worker id and total number of workers. */
    default void setWorldInfo(int workerId, int numWorkers) {}

    /** Sets the training data. */
    default void setInputData(ResettableIterator<?> inputData) {}

    /** Recover from state. */
    default void initializeState(StateInitializationContext context) throws Exception {}

    /** Snapshots to state. */
    default void snapshotState(StateSnapshotContext context) throws Exception {}
}
