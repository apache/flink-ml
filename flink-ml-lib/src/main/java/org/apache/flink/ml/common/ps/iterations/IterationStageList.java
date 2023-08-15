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

package org.apache.flink.ml.common.ps.iterations;

import org.apache.flink.util.Preconditions;
import org.apache.flink.util.function.SerializableFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * A list of iteration stages to express the logic of an iterative machine learning process.
 *
 * <p>Note that there should be at least one stage (e.g., {@link PullStage}, {@link AllReduceStage}
 * or {@link ReduceScatterStage}) that needs to wait for responses from servers.
 */
public class IterationStageList<T extends MLSession> implements Serializable {
    /** The session on each worker. */
    public final T session;
    /** The termination criteria. */
    public Function<T, Boolean> shouldTerminate;
    /** The stage list that describes the iterative process. */
    public List<IterationStage> stageList;

    public IterationStageList(T session) {
        this.stageList = new ArrayList<>();
        this.session = session;
    }

    /** Sets the criteria of termination. */
    public IterationStageList<T> setTerminationCriteria(
            SerializableFunction<T, Boolean> shouldTerminate) {
        boolean waitServer = false;
        for (IterationStage stage : stageList) {
            if (stage instanceof PullStage
                    || stage instanceof AllReduceStage
                    || stage instanceof ReduceScatterStage) {
                waitServer = true;
                break;
            }
        }
        Preconditions.checkState(
                waitServer,
                String.format(
                        "There should be at least one stage that needs to receive response from servers (i.e., %s, %s, %s).\n",
                        PullStage.class.getSimpleName(),
                        AllReduceStage.class.getSimpleName(),
                        ReduceScatterStage.class.getSimpleName()));
        this.shouldTerminate = shouldTerminate;
        return this;
    }

    /** Adds an iteration stage into the stage list. */
    public IterationStageList<T> addStage(IterationStage stage) {
        stageList.add(stage);
        return this;
    }
}
