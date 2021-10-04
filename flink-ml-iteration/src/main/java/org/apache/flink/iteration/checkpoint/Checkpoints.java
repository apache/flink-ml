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

package org.apache.flink.iteration.checkpoint;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheSnapshot;
import org.apache.flink.iteration.datacache.nonkeyed.DataCacheWriter;
import org.apache.flink.runtime.state.OperatorStateCheckpointOutputStream;
import org.apache.flink.util.ResourceGuard;
import org.apache.flink.util.function.SupplierWithException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.SortedMap;
import java.util.TreeMap;

/** Maintains the pending checkpoints. */
public class Checkpoints<T> implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(Checkpoints.class);

    private final TypeSerializer<T> typeSerializer;
    private final FileSystem fileSystem;
    private final SupplierWithException<Path, IOException> pathSupplier;

    private final TreeMap<Long, PendingCheckpoint> uncompletedCheckpoints = new TreeMap<>();

    public Checkpoints(
            TypeSerializer<T> typeSerializer,
            FileSystem fileSystem,
            SupplierWithException<Path, IOException> pathSupplier) {
        this.typeSerializer = typeSerializer;
        this.fileSystem = fileSystem;
        this.pathSupplier = pathSupplier;
    }

    public void startLogging(long checkpointId, OperatorStateCheckpointOutputStream outputStream)
            throws IOException {
        DataCacheWriter<T> dataCacheWriter =
                new DataCacheWriter<>(typeSerializer, fileSystem, pathSupplier);
        ResourceGuard.Lease snapshotLease = outputStream.acquireLease();
        uncompletedCheckpoints.put(
                checkpointId, new PendingCheckpoint(dataCacheWriter, outputStream, snapshotLease));
    }

    public void append(T element) throws IOException {
        for (PendingCheckpoint pendingCheckpoint : uncompletedCheckpoints.values()) {
            pendingCheckpoint.dataCacheWriter.addRecord(element);
        }
    }

    public void commitCheckpointsUntil(long checkpointId) {
        SortedMap<Long, PendingCheckpoint> completedCheckpoints =
                uncompletedCheckpoints.headMap(checkpointId, true);
        completedCheckpoints
                .values()
                .forEach(
                        pendingCheckpoint -> {
                            try {
                                pendingCheckpoint.dataCacheWriter.finish();
                                DataCacheSnapshot snapshot =
                                        new DataCacheSnapshot(
                                                fileSystem,
                                                null,
                                                pendingCheckpoint.dataCacheWriter
                                                        .getFinishSegments());
                                pendingCheckpoint.checkpointOutputStream.startNewPartition();
                                snapshot.writeTo(pendingCheckpoint.checkpointOutputStream);
                                pendingCheckpoint.dataCacheWriter.cleanup();
                            } catch (Exception e) {
                                LOG.error("Failed to commit checkpoint until " + checkpointId, e);
                            } finally {
                                pendingCheckpoint.snapshotLease.close();
                            }
                        });

        completedCheckpoints.clear();
    }

    @Override
    public void close() {
        uncompletedCheckpoints.forEach(
                (checkpointId, pendingCheckpoint) -> {
                    pendingCheckpoint.snapshotLease.close();
                    try {
                        pendingCheckpoint.dataCacheWriter.cleanup();
                    } catch (IOException e) {
                        LOG.error("Failed to cleanup " + checkpointId, e);
                    }
                });
        uncompletedCheckpoints.clear();
    }

    private class PendingCheckpoint {
        final DataCacheWriter<T> dataCacheWriter;

        final OperatorStateCheckpointOutputStream checkpointOutputStream;

        final ResourceGuard.Lease snapshotLease;

        public PendingCheckpoint(
                DataCacheWriter<T> dataCacheWriter,
                OperatorStateCheckpointOutputStream checkpointOutputStream,
                ResourceGuard.Lease snapshotLease) {
            this.dataCacheWriter = dataCacheWriter;
            this.checkpointOutputStream = checkpointOutputStream;
            this.snapshotLease = snapshotLease;
        }
    }
}
