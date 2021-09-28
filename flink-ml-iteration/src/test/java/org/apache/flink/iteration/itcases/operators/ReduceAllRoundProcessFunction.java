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

package org.apache.flink.iteration.itcases.operators;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.iteration.IterationListener;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * An operator that reduces the received numbers and emit the result into the output, and also emit
 * the received numbers to the next operator.
 */
public class ReduceAllRoundProcessFunction extends ProcessFunction<EpochRecord, EpochRecord>
        implements IterationListener<EpochRecord> {

    private final boolean sync;

    private final int maxRound;

    private transient Map<Integer, Integer> sumByEpochs;

    private transient List<EpochRecord> cachedRecords;

    private transient OutputTag<OutputRecord<Integer>> outputTag;

    public ReduceAllRoundProcessFunction(boolean sync, int maxRound) {
        this.sync = sync;
        this.maxRound = maxRound;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        sumByEpochs = new HashMap<>();
        cachedRecords = new ArrayList<>();
        outputTag = new OutputTag<OutputRecord<Integer>>("output") {};
    }

    @Override
    public void processElement(
            EpochRecord record,
            ProcessFunction<EpochRecord, EpochRecord>.Context ctx,
            Collector<EpochRecord> out)
            throws Exception {
        processRecord(record, ctx::output, out);
    }

    protected void processRecord(
            EpochRecord record,
            BiConsumer<OutputTag<OutputRecord<Integer>>, OutputRecord<Integer>> sideOutput,
            Collector<EpochRecord> out) {
        sumByEpochs.compute(
                record.getEpoch(), (k, v) -> v == null ? record.getValue() : v + record.getValue());

        if (record.getEpoch() < maxRound) {
            if (!sync) {
                out.collect(record);
            } else {
                cachedRecords.add(record);
            }
        }

        if (!sync) {
            sideOutput.accept(
                    outputTag,
                    new OutputRecord<>(
                            OutputRecord.Event.PROCESS_ELEMENT,
                            record.getEpoch(),
                            sumByEpochs.get(record.getEpoch())));
        }
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark,
            IterationListener.Context context,
            Collector<EpochRecord> collector) {
        if (sync) {
            context.output(
                    outputTag,
                    new OutputRecord<>(
                            OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED,
                            epochWatermark,
                            sumByEpochs.get(epochWatermark)));
            cachedRecords.forEach(collector::collect);
            cachedRecords.clear();
        }
    }

    @Override
    public void onIterationTerminated(
            IterationListener.Context context, Collector<EpochRecord> collector) {
        context.output(outputTag, new OutputRecord<>(OutputRecord.Event.TERMINATED, -1, -1));
    }
}
