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

package org.apache.flink.ml.common.ps;

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.common.ps.sarray.SharedDoubleArray;
import org.apache.flink.ml.common.ps.sarray.SharedLongArray;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Preconditions;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.longs.LongArrayList;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.Function;

/**
 * ServerAgent resides on each worker. It serves as an agent for {@link WorkerOperator} to talk with
 * {@link ServerOperator}.
 */
class ServerAgent {
    /** Index of the worker that this agent resides on. */
    private final int workerId;
    /** Number of servers to talk to. */
    private final int numServers;
    /** Hash function to partition keys to different servers. */
    private final Function<Long, Integer> hashFunc;
    /** The collector on this worker. */
    private final Output<StreamRecord<byte[]>> output;

    ServerAgent(
            int workerId,
            int numServers,
            Function<Long, Integer> hashFunc,
            Output<StreamRecord<byte[]>> output) {
        this.workerId = workerId;
        this.numServers = numServers;
        this.output = output;
        this.hashFunc = hashFunc;
    }

    /** Pushes a key-value arrays to servers. */
    void push(SharedLongArray keys, SharedDoubleArray values, int stageId) {
        Tuple2<LongArrayList[], DoubleArrayList[]> slicedRequests = sliceRequest(keys, values);
        LongArrayList[] splitKeys = slicedRequests.f0;
        DoubleArrayList[] splitValues = slicedRequests.f1;
        for (int serverId = 0; serverId < splitKeys.length; serverId++) {
            Message message =
                    new Message(
                            workerId,
                            serverId,
                            stageId,
                            splitKeys[serverId].toLongArray(),
                            splitValues[serverId].toDoubleArray());
            output.collect(new StreamRecord<>(message.bytes));
        }
    }

    /** Pulls the values from servers with the specified keys. */
    void pull(SharedLongArray keys, int stageId) {
        Tuple2<LongArrayList[], DoubleArrayList[]> slicedRequests = sliceRequest(keys, null);
        LongArrayList[] splitKeys = slicedRequests.f0;
        for (int serverId = 0; serverId < splitKeys.length; serverId++) {
            Message message =
                    new Message(
                            workerId,
                            serverId,
                            stageId,
                            splitKeys[serverId].toLongArray(),
                            new double[0]);
            output.collect(new StreamRecord<>(message.bytes));
        }
    }

    /**
     * Pushes the values to servers to apply all-reduce/reduce-scatter operation.
     *
     * <p>Note that the values pushed by this function are not going to update the model, but just
     * perform an reduce operation.
     */
    <V> void reduce(V[] values, TypeSerializer<V> typeSerializer, int stageId) throws IOException {
        int shardSize = values.length / numServers + 1;
        for (int serverId = 0; serverId < numServers; serverId++) {
            int s = Math.min(serverId * shardSize, values.length);
            int e = Math.min(s + shardSize, values.length);
            V[] segment = Arrays.copyOfRange(values, s, e);
            Message message =
                    new Message(workerId, serverId, stageId, new long[0], segment, typeSerializer);
            output.collect(new StreamRecord<>(message.bytes));
        }
    }

    /**
     * Splits the push/pull request according to the given sorted keys and the corresponding values.
     *
     * @param keys keys of push/pull request.
     * @param values the push values if not null.
     * @return the split requests for each server.
     */
    private Tuple2<LongArrayList[], DoubleArrayList[]> sliceRequest(
            SharedLongArray keys, @Nullable SharedDoubleArray values) {
        LongArrayList[] splitKeys = new LongArrayList[numServers];
        DoubleArrayList[] splitValues = new DoubleArrayList[numServers];
        for (int i = 0; i < numServers; i++) {
            splitKeys[i] = new LongArrayList();
            splitValues[i] = new DoubleArrayList();
        }

        int numDoublesPerKey = 0;
        if (values != null) {
            Preconditions.checkState(
                    values.size() % keys.size() == 0, "The length of each key should be the same.");
            numDoublesPerKey = values.size() / keys.size();
        }

        long[] keyArray = keys.elements();
        for (int i = 0; i < keys.size(); i++) {
            int serverId = hashFunc.apply(keyArray[i]);
            splitKeys[serverId].add(keyArray[i]);
            if (values != null) {
                for (int j = 0; j < numDoublesPerKey; j++) {
                    splitValues[serverId].add(values.get(i * numDoublesPerKey + j));
                }
            }
        }

        return Tuple2.of(splitKeys, splitValues);
    }
}
