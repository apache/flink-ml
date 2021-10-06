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

package org.apache.flink.ml.iteration.itcases.operators;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.typeutils.MapTypeInfo;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.iteration.EpochWatermarkAware;
import org.apache.flink.ml.iteration.IterationListener;
import org.apache.flink.ml.iteration.operator.OperatorStateUtils;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Supplier;

/**
 * An operators that reduce the received numbers and emit the result into the output, and also emit
 * the received numbers to the next operator.
 */
public class ReduceAllRoundProcessFunction extends ProcessFunction<Integer, Integer>
        implements IterationListener<Integer>, EpochWatermarkAware, CheckpointedFunction {

    private final boolean sync;

    private final int maxRound;

    private transient Map<Integer, Integer> sumByRounds;

    private transient List<Integer> cachedRecords;

    private transient OutputTag<OutputRecord<Integer>> outputTag;

    private transient Supplier<Integer> epochWatermarkSupplier;

    private transient ListState<Map<Integer, Integer>> sumByRoundsState;

    private transient ListState<Integer> cachedRecordsState;

    public ReduceAllRoundProcessFunction(boolean sync, int maxRound) {
        this.sync = sync;
        this.maxRound = maxRound;
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext)
            throws Exception {
        this.sumByRounds = new HashMap<>();
        cachedRecords = new ArrayList<>();

        sumByRoundsState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(
                                new ListStateDescriptor<>(
                                        "test",
                                        new MapTypeInfo<Integer, Integer>(
                                                BasicTypeInfo.INT_TYPE_INFO,
                                                BasicTypeInfo.INT_TYPE_INFO)));
        Optional<Map<Integer, Integer>> old =
                OperatorStateUtils.getUniqueElement(sumByRoundsState, "test");
        old.ifPresent(v -> sumByRounds.putAll(v));

        cachedRecordsState =
                functionInitializationContext
                        .getOperatorStateStore()
                        .getListState(new ListStateDescriptor<>("cache", Integer.class));
        cachedRecordsState.get().forEach(v -> cachedRecords.add(v));
    }

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        sumByRoundsState.clear();
        sumByRoundsState.update(Collections.singletonList(new HashMap<>(sumByRounds)));

        cachedRecordsState.clear();
        cachedRecordsState.addAll(cachedRecords);
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        super.open(parameters);
        this.outputTag = new OutputTag<OutputRecord<Integer>>("output") {};
    }

    @Override
    public void processElement(
            Integer value, ProcessFunction<Integer, Integer>.Context ctx, Collector<Integer> out)
            throws Exception {
        processRecord(value, ctx::output, out);
    }

    protected void processRecord(
            Integer value,
            BiConsumer<OutputTag<OutputRecord<Integer>>, OutputRecord<Integer>> sideOutput,
            Collector<Integer> out) {
        int round = epochWatermarkSupplier.get();
        sumByRounds.compute(round, (k, v) -> v == null ? value : v + value);
        System.out.println("Process " + value + "@" + round);

        if (round < maxRound) {
            if (!sync) {
                out.collect(value);
            } else {
                cachedRecords.add(value);
            }
        }

        if (!sync) {
            sideOutput.accept(
                    outputTag,
                    new OutputRecord<>(
                            OutputRecord.Event.PROCESS_ELEMENT, round, sumByRounds.get(round)));
        }
    }

    @Override
    public void onEpochWatermarkIncremented(
            int epochWatermark, IterationListener.Context context, Collector<Integer> collector) {
        if (sync) {
            context.output(
                    outputTag,
                    new OutputRecord<>(
                            OutputRecord.Event.EPOCH_WATERMARK_INCREMENTED,
                            epochWatermark,
                            sumByRounds.get(epochWatermark)));
            cachedRecords.forEach(collector::collect);
            cachedRecords.clear();
        }
    }

    @Override
    public void onIterationTerminated(
            IterationListener.Context context, Collector<Integer> collector) {
        context.output(outputTag, new OutputRecord<>(OutputRecord.Event.TERMINATED, -1, -1));
    }

    @Override
    public void setEpochWatermarkSupplier(Supplier<Integer> epochWatermarkSupplier) {
        this.epochWatermarkSupplier = epochWatermarkSupplier;
    }
}
