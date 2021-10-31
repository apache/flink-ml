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

package org.apache.flink.test.iteration.operators;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.util.Collector;

/**
 * A proxy of {@link ReduceAllRoundProcessFunction} to two-inputs. It assumes the input 1 is empty.
 */
public class TwoInputReduceAllRoundProcessFunction
        extends CoProcessFunction<EpochRecord, EpochRecord, EpochRecord>
        implements IterationListener<EpochRecord> {

    private final ReduceAllRoundProcessFunction reduceAllRoundProcessFunction;

    public TwoInputReduceAllRoundProcessFunction(boolean sync, int maxRound) {
        this.reduceAllRoundProcessFunction = new ReduceAllRoundProcessFunction(sync, maxRound);
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        reduceAllRoundProcessFunction.open(parameters);
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark,
            IterationListener.Context context,
            Collector<EpochRecord> collector) {
        reduceAllRoundProcessFunction.onEpochWatermarkIncremented(
                epochWatermark, context, collector);
    }

    @Override
    public void onIterationTerminated(
            IterationListener.Context context, Collector<EpochRecord> collector) {
        reduceAllRoundProcessFunction.onIterationTerminated(context, collector);
    }

    @Override
    public void processElement1(
            EpochRecord record,
            CoProcessFunction<EpochRecord, EpochRecord, EpochRecord>.Context ctx,
            Collector<EpochRecord> out)
            throws Exception {

        // Processing the following round of messages.
        reduceAllRoundProcessFunction.processRecord(record, ctx::output, out);
    }

    @Override
    public void processElement2(
            EpochRecord record,
            CoProcessFunction<EpochRecord, EpochRecord, EpochRecord>.Context ctx,
            Collector<EpochRecord> out)
            throws Exception {

        // Processing the first round of messages.
        reduceAllRoundProcessFunction.processRecord(record, ctx::output, out);
    }
}
