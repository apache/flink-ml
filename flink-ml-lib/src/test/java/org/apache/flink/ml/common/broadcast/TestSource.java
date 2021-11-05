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

package org.apache.flink.ml.common.broadcast;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;

import java.util.Iterator;

/**
 * Utility class that generates int stream and also throws exceptions to test the fail over. In
 * detail, given ${numElementsPerPartition}, this class generate a number sequence with elements
 * range in [0, StreamExecutionEnvironment.getParallelism() * numElementsPerPartition).
 *
 * <p>For example, when the parallelism is 2 and the ${numElementsPerPartition} is 5, this class
 * generates {0,1,2,3,4,5,6,7,8,9}.
 */
public class TestSource extends RichParallelSourceFunction<Integer>
        implements CheckpointedFunction {

    private static volatile boolean hasThrown = false;

    private ListState<Integer> currentIdxState;

    private Integer currentIdx;

    private Integer mod, numPartitions, numElementsPerPartition;

    private transient volatile boolean running = true;

    public TestSource(int numElementsPerPartition) {
        this.numElementsPerPartition = numElementsPerPartition;
    }

    @Override
    public void open(Configuration parameters) {
        this.mod = getRuntimeContext().getIndexOfThisSubtask();
        this.numPartitions = getRuntimeContext().getNumberOfParallelSubtasks();
        running = true;
    }

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        this.currentIdxState.clear();
        this.currentIdxState.add(currentIdx);
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        currentIdxState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "currentIdx", BasicTypeInfo.INT_TYPE_INFO));
        Iterator<Integer> iterator = currentIdxState.get().iterator();
        currentIdx = 0;
        if (iterator.hasNext()) {
            currentIdx = iterator.next();
        }
    }

    @Override
    public void run(SourceContext<Integer> sourceContext) throws Exception {
        while (running && currentIdx < numElementsPerPartition) {
            synchronized (sourceContext.getCheckpointLock()) {
                sourceContext.collect(currentIdx * numPartitions + mod);
                currentIdx++;
            }
            Thread.sleep(1);
            if (currentIdx == numElementsPerPartition / 2 && (!hasThrown)) {
                hasThrown = true;
                throw new RuntimeException("Failing source");
            }
        }
    }

    @Override
    public void cancel() {
        running = false;
    }
}
