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

package org.apache.flink.ml.benchmark.datagenerator.common;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import org.apache.flink.types.Row;

import java.util.Random;

/** A parallel source to generate user defined rows. */
public abstract class RowGenerator extends RichParallelSourceFunction<Row> {
    /** Random instance to generate data. */
    protected Random random;
    /** Number of values to generate in total. */
    private final long numValues;
    /** The init seed to generate data. */
    private final long initSeed;
    /** Number of tasks to generate in one local task. */
    private long numValuesOnThisTask;
    /** Whether this source is still running. */
    private volatile boolean isRunning = true;

    public RowGenerator(long numValues, long initSeed) {
        this.numValues = numValues;
        this.initSeed = initSeed;
    }

    @Override
    public final void open(Configuration parameters) throws Exception {
        super.open(parameters);
        int taskIdx = getRuntimeContext().getIndexOfThisSubtask();
        int numTasks = getRuntimeContext().getNumberOfParallelSubtasks();
        random = new Random(Tuple2.of(initSeed, taskIdx).hashCode());
        long div = numValues / numTasks;
        long mod = numValues % numTasks;
        numValuesOnThisTask = mod > taskIdx ? div + 1 : div;
    }

    @Override
    public final void run(SourceContext<Row> ctx) throws Exception {
        long cnt = 0;
        while (isRunning && cnt < numValuesOnThisTask) {
            ctx.collect(nextRow());
            cnt++;
        }
    }

    @Override
    public final void cancel() {
        isRunning = false;
    }

    /** Generates a new data point. */
    protected abstract Row nextRow();

    /** Returns the output type information for this generator. */
    protected abstract RowTypeInfo getRowTypeInfo();
}
