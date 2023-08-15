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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.ManagedMemoryUseCase;
import org.apache.flink.iteration.operator.OperatorUtils;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.runtime.state.StateInitializationContext;
import org.apache.flink.runtime.state.StatePartitionStreamProvider;
import org.apache.flink.runtime.state.StateSnapshotContext;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.runtime.tasks.StreamTask;
import org.apache.flink.table.runtime.util.LazyMemorySegmentPool;
import org.apache.flink.table.runtime.util.MemorySegmentPool;
import org.apache.flink.util.Preconditions;

import org.apache.commons.collections.IteratorUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ListState} child class that records data and replays them when required.
 *
 * <p>This class basically stores data in file system, and provides the option to cache them in
 * memory. In order to use the memory caching option, users need to allocate certain managed memory
 * for the wrapper operator through {@link
 * org.apache.flink.api.dag.Transformation#declareManagedMemoryUseCaseAtOperatorScope} and set
 * `memorySubFraction` with a value larger than 0.
 *
 * <p>NOTE: Users need to explicitly invoke this class's {@link
 * #snapshotState(StateSnapshotContext)} method in order to store the recorded data in snapshot.
 */
public class ListStateWithCache<T> implements ListState<T> {

    /** The tool to serialize/deserialize records. */
    private final TypeSerializer<T> serializer;

    /** The path of the directory that holds the files containing recorded data. */
    private final Path basePath;

    /** The data cache writer for the received records. */
    private final DataCacheWriter<T> dataCacheWriter;

    /**
     * Creates an instance of {@link ListStateWithCache}.
     *
     * @param serializer The type serializer of data.
     * @param manager Operator-scope managed memory manager.
     * @param key The key registered in the manager.
     * @param containingTask The container task.
     * @param runtimeContext The runtime context.
     * @param stateInitializationContext The state initialization state.
     * @param operatorID The operator ID.
     */
    @SuppressWarnings("unchecked")
    public ListStateWithCache(
            TypeSerializer<T> serializer,
            OperatorScopeManagedMemoryManager manager,
            String key,
            StreamTask<?, ?> containingTask,
            StreamingRuntimeContext runtimeContext,
            StateInitializationContext stateInitializationContext,
            OperatorID operatorID)
            throws IOException {
        double memorySubFraction = manager.getFraction(key);
        this.serializer = serializer;

        MemorySegmentPool segmentPool = null;
        double fraction =
                containingTask
                        .getConfiguration()
                        .getManagedMemoryFractionOperatorUseCaseOfSlot(
                                ManagedMemoryUseCase.OPERATOR,
                                runtimeContext.getTaskManagerRuntimeInfo().getConfiguration(),
                                runtimeContext.getUserCodeClassLoader());
        if (fraction * memorySubFraction > 0) {
            MemoryManager memoryManager = containingTask.getEnvironment().getMemoryManager();
            segmentPool =
                    new LazyMemorySegmentPool(
                            containingTask,
                            memoryManager,
                            memoryManager.computeNumberOfPages(fraction * memorySubFraction));
        }

        basePath =
                OperatorUtils.getDataCachePath(
                        containingTask.getEnvironment().getTaskManagerInfo().getConfiguration(),
                        containingTask
                                .getEnvironment()
                                .getIOManager()
                                .getSpillingDirectoriesPaths());

        List<StatePartitionStreamProvider> inputs =
                IteratorUtils.toList(
                        stateInitializationContext.getRawOperatorStateInputs().iterator());
        Preconditions.checkState(
                inputs.size() < 2, "The input from raw operator state should be one or zero.");

        List<Segment> priorFinishedSegments = new ArrayList<>();
        if (!inputs.isEmpty()) {
            DataCacheSnapshot dataCacheSnapshot =
                    DataCacheSnapshot.recover(
                            inputs.get(0).getStream(),
                            basePath.getFileSystem(),
                            OperatorUtils.createDataCacheFileGenerator(
                                    basePath, "cache", operatorID));

            if (segmentPool != null) {
                dataCacheSnapshot.tryReadSegmentsToMemory(serializer, segmentPool);
            }

            priorFinishedSegments = dataCacheSnapshot.getSegments();
        }

        this.dataCacheWriter =
                new DataCacheWriter<>(
                        serializer,
                        basePath.getFileSystem(),
                        OperatorUtils.createDataCacheFileGenerator(basePath, "cache", operatorID),
                        segmentPool,
                        priorFinishedSegments);
    }

    public void snapshotState(StateSnapshotContext context) throws Exception {
        dataCacheWriter.writeSegmentsToFiles();
        DataCacheSnapshot dataCacheSnapshot =
                new DataCacheSnapshot(
                        basePath.getFileSystem(), null, dataCacheWriter.getSegments());
        context.getRawOperatorStateOutput().startNewPartition();
        dataCacheSnapshot.writeTo(context.getRawOperatorStateOutput());
    }

    @Override
    public Iterable<T> get() throws Exception {
        List<Segment> segments = dataCacheWriter.getSegments();
        return () -> new DataCacheReader<>(serializer, segments);
    }

    @Override
    public void add(T t) throws Exception {
        dataCacheWriter.addRecord(t);
    }

    @Override
    public void update(List<T> list) throws Exception {
        dataCacheWriter.clear();
        addAll(list);
    }

    @Override
    public void addAll(List<T> list) throws Exception {
        for (T t : list) {
            add(t);
        }
    }

    @Override
    public void clear() {
        try {
            dataCacheWriter.clear();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
