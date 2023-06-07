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
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.common.ps.message.Message;
import org.apache.flink.ml.common.ps.message.MessageType;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

/** ServerAgent resides on each worker. It serves as an agent for workers to talk with servers. */
public class ServerAgent {
    /** Index of the worker that this agent resides on. */
    private final int workerId;
    /** Number of servers to talk to. */
    private int numServers;
    /** Key ranges of each server. */
    private long[] ranges;
    /** The collector on this worker. */
    private final Output<StreamRecord<Tuple2<Integer, byte[]>>> output;

    ServerAgent(int workerId, Output<StreamRecord<Tuple2<Integer, byte[]>>> output) {
        this.workerId = workerId;
        this.output = output;
    }

    void open(int numServers, long maxKey) {
        this.numServers = numServers;
        this.ranges = new long[numServers + 1];
        long shardSize = (maxKey + 1) / numServers;

        for (int serverId = 0; serverId < numServers; serverId++) {
            ranges[serverId] = shardSize * serverId;
        }
        ranges[numServers] = maxKey + 1;
    }

    /** Sends a request to servers to initialize key range on each server. */
    void initializeModel() {
        for (int serverId = 0; serverId < numServers; serverId++) {
            long start = ranges[serverId];
            long end = ranges[serverId + 1];
            Message message =
                    new Message(
                            serverId,
                            workerId,
                            MessageType.INITIALIZE,
                            new long[] {start, end},
                            new double[0]);
            output.collect(new StreamRecord<>(Tuple2.of(serverId, message.bytes)));
        }
    }

    /** Pushes a key-value arrays to servers. */
    void push(long[] indices, double[] values) {
        Iterator<Tuple3<Integer, long[], double[]>> requests = sliceRequest(indices, values);
        while (requests.hasNext()) {
            Tuple3<Integer, long[], double[]> request = requests.next();
            Message message =
                    new Message(request.f0, workerId, MessageType.PUSH, request.f1, request.f2);
            output.collect(new StreamRecord<>(Tuple2.of(request.f0, message.bytes)));
        }
    }

    /** Pulls the values from servers with the specified indices. */
    void pull(long[] indices) {
        Iterator<Tuple3<Integer, long[], double[]>> requests = sliceRequest(indices, null);
        while (requests.hasNext()) {
            Tuple3<Integer, long[], double[]> request = requests.next();
            Message message =
                    new Message(request.f0, workerId, MessageType.PULL, request.f1, new double[0]);
            output.collect(new StreamRecord<>(Tuple2.of(request.f0, message.bytes)));
        }
    }

    /**
     * Pushes the values to servers to apply all reduce operation.
     *
     * <p>Note that the values pushed by this function are not going to update the model, but just
     * perform an all reduce operation.
     */
    <V> void allReducePush(V[] values, TypeSerializer<V> typeSerializer) throws IOException {
        final int MIN_MESSAGE_SIZE = 1024;
        int messageSize = Math.max(MIN_MESSAGE_SIZE, values.length / numServers + 1);
        for (int serverId = 0; serverId < numServers; serverId++) {
            int s = Math.min(serverId * messageSize, values.length);
            int e = Math.min(s + messageSize, values.length);
            V[] segment;
            if (s == e) {
                segment = (V[]) new Object[0];
            } else {
                segment = Arrays.copyOfRange(values, s, e);
            }
            Message message =
                    new Message(
                            workerId,
                            serverId,
                            MessageType.ALL_REDUCE,
                            new long[0],
                            segment,
                            typeSerializer);
            output.collect(new StreamRecord<>(Tuple2.of(serverId, message.bytes)));
        }
    }

    /**
     * Splits the push/pull request according to the given sorted indices and the corresponding
     * values.
     *
     * @param indices sorted indices of push/pull request.
     * @param values the push values if not null.
     * @return the split requests for each server.
     */
    private Iterator<Tuple3<Integer, long[], double[]>> sliceRequest(
            long[] indices, @Nullable double[] values) {
        return new RequestsIterator(numServers, indices, values, ranges);
    }

    private static class RequestsIterator implements Iterator<Tuple3<Integer, long[], double[]>> {
        private final int numServers;
        private final long[] indices;
        private final double[] values;
        /**
         * Number of values per key. If the model data is a vector, numValuesPerKey is one. If the
         * model data is a matrix, numValuesPerKey is the number of columns.
         */
        private final int numValuesPerKey;

        private final long[] ranges;

        private int serverId = 0;

        private int s = 0;

        public RequestsIterator(
                int numServers, long[] indices, @Nullable double[] values, long[] ranges) {
            this.numServers = numServers;
            this.indices = indices;
            this.values = values;
            this.ranges = ranges;
            if (indices.length != 0 && values != null) {
                numValuesPerKey = values.length / indices.length;
                Preconditions.checkArgument(
                        numValuesPerKey * indices.length == values.length,
                        String.format(
                                "The size of values [%d] cannot be divided by size of keys [%d].",
                                values.length, indices.length));
            } else {
                numValuesPerKey = 1;
            }
        }

        @Override
        public boolean hasNext() {
            return serverId < numServers;
        }

        @Override
        public Tuple3<Integer, long[], double[]> next() {
            int e = s;
            while (e < indices.length && indices[e] < ranges[serverId + 1]) {
                e++;
            }

            long[] splitIndices = new long[0];
            double[] splitValues = values == null ? null : new double[0];
            if (s < e) {
                splitIndices = Arrays.copyOfRange(indices, s, e);
                splitValues =
                        values == null
                                ? null
                                : Arrays.copyOfRange(
                                        values, s * numValuesPerKey, e * numValuesPerKey);
            }
            s = e;
            serverId++;
            return Tuple3.of(serverId - 1, splitIndices, splitValues);
        }
    }
}
