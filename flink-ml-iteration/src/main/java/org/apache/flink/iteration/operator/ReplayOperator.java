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

package org.apache.flink.iteration.operator;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.iteration.IterationRecord;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheReader;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheWriter;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTracker;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTrackerFactory;
import org.apache.flink.iteration.progresstrack.OperatorEpochWatermarkTrackerListener;
import org.apache.flink.iteration.typeinfo.IterationRecordSerializer;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.BoundedMultiInput;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.api.operators.TwoInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.util.ExceptionUtils;

import java.io.IOException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

import static org.apache.flink.util.Preconditions.checkState;

/** Replays the data received in the round 0 in the following round. */
public class ReplayOperator<T> extends AbstractStreamOperator<IterationRecord<T>>
        implements TwoInputStreamOperator<
                        IterationRecord<T>, IterationRecord<Void>, IterationRecord<T>>,
                OperatorEpochWatermarkTrackerListener,
                BoundedMultiInput {

    private OperatorEpochWatermarkTracker progressTracker;

    private Path basePath;

    private FileSystem fileSystem;

    private TypeSerializer<T> typeSerializer;

    private Executor dataReplayerExecutor;

    private DataCacheWriter<T> dataCacheWriter;

    private AtomicReference<DataCacheReader<T>> currentDataCacheReader;

    @Override
    public void setup(
            StreamTask<?, ?> containingTask,
            StreamConfig config,
            Output<StreamRecord<IterationRecord<T>>> output) {
        super.setup(containingTask, config, output);
        progressTracker = OperatorEpochWatermarkTrackerFactory.create(config, containingTask, this);

        try {
            basePath =
                    OperatorUtils.getDataCachePath(
                            containingTask.getEnvironment().getTaskManagerInfo().getConfiguration(),
                            containingTask
                                    .getEnvironment()
                                    .getIOManager()
                                    .getSpillingDirectoriesPaths());
            fileSystem = basePath.getFileSystem();

            IterationRecordSerializer<T> iterationRecordSerializer =
                    (IterationRecordSerializer)
                            config.getTypeSerializerOut(getClass().getClassLoader());
            typeSerializer = iterationRecordSerializer.getInnerSerializer();
            dataReplayerExecutor =
                    Executors.newSingleThreadExecutor(
                            runnable -> {
                                Thread thread = new Thread(runnable);
                                thread.setName(
                                        "Replay-"
                                                + getOperatorID()
                                                + "-"
                                                + containingTask.getIndexInSubtaskGroup());
                                return thread;
                            });
            dataCacheWriter =
                    new DataCacheWriter<>(
                            typeSerializer,
                            fileSystem,
                            OperatorUtils.createDataCacheFileGenerator(
                                    basePath, "replay", config.getOperatorID()));

            currentDataCacheReader = new AtomicReference<>();
        } catch (Exception e) {
            ExceptionUtils.rethrow(e);
        }
    }

    @Override
    public void initializeState(StateInitializationContext context) throws Exception {
        super.initializeState(context);
    }

    @Override
    public void processElement1(StreamRecord<IterationRecord<T>> element) throws Exception {
        switch (element.getValue().getType()) {
            case RECORD:
                dataCacheWriter.addRecord(element.getValue().getValue());
                output.collect(element);
                break;
            case EPOCH_WATERMARK:
                progressTracker.onEpochWatermark(
                        0, element.getValue().getSender(), element.getValue().getEpoch());
                break;
            default:
                throw new UnsupportedOperationException(
                        "Not supported element type: " + element.getValue());
        }
    }

    @Override
    public void processElement2(StreamRecord<IterationRecord<Void>> element) throws Exception {
        if (element.getValue().getType() == IterationRecord.Type.EPOCH_WATERMARK) {
            progressTracker.onEpochWatermark(
                    1, element.getValue().getSender(), element.getValue().getEpoch());
        } else {
            throw new UnsupportedOperationException(
                    "Not supported element type: " + element.getValue());
        }
    }

    @Override
    public void endInput(int i) throws Exception {
        // The notification ranges from 1 to N while the track uses 0 to N -1.
        progressTracker.finish(i - 1);
    }

    @Override
    public void onEpochWatermarkIncrement(int epochWatermark) throws IOException {
        if (epochWatermark == 0) {
            // No need to replay for the round 0, it is output directly.
            dataCacheWriter.finish();
            output.collect(
                    new StreamRecord<>(
                            IterationRecord.newEpochWatermark(
                                    epochWatermark,
                                    OperatorUtils.getUniqueSenderId(
                                            config.getOperatorID(),
                                            getContainingTask().getIndexInSubtaskGroup()))));
            return;
        } else if (epochWatermark == Integer.MAX_VALUE) {
            output.collect(
                    new StreamRecord<>(
                            IterationRecord.newEpochWatermark(
                                    epochWatermark,
                                    OperatorUtils.getUniqueSenderId(
                                            config.getOperatorID(),
                                            getContainingTask().getIndexInSubtaskGroup()))));
            return;
        }

        checkState(currentDataCacheReader.get() == null, "Concurrent replay is not supported");
        currentDataCacheReader.set(
                new DataCacheReader<>(
                        typeSerializer, fileSystem, dataCacheWriter.getFinishSegments()));
        dataReplayerExecutor.execute(
                () -> {
                    DataCacheReader<T> reader = currentDataCacheReader.get();
                    StreamRecord<IterationRecord<T>> reusable =
                            new StreamRecord<>(IterationRecord.newRecord(null, epochWatermark));
                    while (reader.hasNext()) {
                        T next = reader.next();
                        reusable.getValue().setValue(next);
                        output.collect(reusable);
                    }
                    currentDataCacheReader.set(null);

                    reusable.replace(
                            IterationRecord.newEpochWatermark(
                                    epochWatermark,
                                    OperatorUtils.getUniqueSenderId(
                                            config.getOperatorID(),
                                            getContainingTask().getIndexInSubtaskGroup())));
                    output.collect(reusable);
                });
    }
}
