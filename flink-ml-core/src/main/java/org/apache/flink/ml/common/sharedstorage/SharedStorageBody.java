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

package org.apache.flink.ml.common.sharedstorage;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * The builder of the subgraph that will be executed with a common shared storage. Users can only
 * create data streams from {@code inputs}. Users can not refer to data streams outside, and can not
 * add sources/sinks.
 *
 * <p>The shared storage body requires all streams accessing the shared storage, i.e., {@link
 * SharedStorageBodyResult#accessors} have same parallelism and can be co-located.
 */
@Experimental
@FunctionalInterface
public interface SharedStorageBody extends Serializable {

    /**
     * This method creates the subgraph for the shared storage body.
     *
     * @param inputs Input data streams.
     * @return Result of the subgraph, including output data streams, data streams with access to
     *     the shared storage, and a mapping from share items to their owners.
     */
    SharedStorageBodyResult process(List<DataStream<?>> inputs);

    /**
     * The result of a {@link SharedStorageBody}, including output data streams, data streams with
     * access to the shared storage, and a mapping from descriptors of share items to their owners.
     */
    @Experimental
    class SharedStorageBodyResult {
        /** A list of output streams. */
        private final List<DataStream<?>> outputs;

        /**
         * A list of data streams which access to the shared storage. All data streams in the list
         * should implement {@link SharedStorageStreamOperator}.
         */
        private final List<DataStream<?>> accessors;

        /**
         * A mapping from descriptors of shared items to their owners. The owner is specified by
         * {@link SharedStorageStreamOperator#getSharedStorageAccessorID()}, which must be kept
         * unchanged for an instance of {@link SharedStorageStreamOperator}.
         */
        private final Map<ItemDescriptor<?>, SharedStorageStreamOperator> ownerMap;

        public SharedStorageBodyResult(
                List<DataStream<?>> outputs,
                List<DataStream<?>> accessors,
                Map<ItemDescriptor<?>, SharedStorageStreamOperator> ownerMap) {
            this.outputs = outputs;
            this.accessors = accessors;
            this.ownerMap = ownerMap;
        }

        public List<DataStream<?>> getOutputs() {
            return outputs;
        }

        public List<DataStream<?>> getAccessors() {
            return accessors;
        }

        public Map<ItemDescriptor<?>, SharedStorageStreamOperator> getOwnerMap() {
            return ownerMap;
        }
    }
}
