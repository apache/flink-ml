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

import org.apache.flink.runtime.util.ResettableIterator;
import org.apache.flink.util.OutputTag;

import java.util.List;

/**
 * The default implementation of {@link MLSession}.
 *
 * @param <DT> Data type of input data.
 */
public class MLSessionImpl<DT> implements MLSession {
    /** Current iteration id. */
    public int iterationId;
    /** Index of this worker. */
    public int workerId;
    /** Number of workers in total for this distributed ML job. */
    public int numWorkers;
    /** The input data. */
    public ResettableIterator<DT> inputData;

    public List<OutputTag<?>> outputTags;

    /** Constructs an instance with side outputs. */
    public MLSessionImpl(List<OutputTag<?>> outputTags) {
        this.outputTags = outputTags;
    }

    /** Constructs an instance without side outputs. */
    public MLSessionImpl() {}

    @Override
    public List<OutputTag<?>> getOutputTags() {
        return outputTags;
    }

    @Override
    public void setIterationId(int iterationId) {
        this.iterationId = iterationId;
    }

    @Override
    public void setWorldInfo(int workerId, int numWorkers) {
        this.workerId = workerId;
        this.numWorkers = numWorkers;
    }

    @Override
    public void setInputData(ResettableIterator<?> inputData) {
        this.inputData = (ResettableIterator<DT>) inputData;
    }
}
