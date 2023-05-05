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

package org.apache.flink.ml.common.sharedobjects;

import org.apache.flink.annotation.Experimental;
import org.apache.flink.api.dag.Transformation;
import org.apache.flink.streaming.api.datastream.DataStream;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * The builder of the subgraph that will be executed with a common shared objects. Users can only
 * create data streams from {@code inputs}. Users can not refer to data streams outside, and can not
 * add sources/sinks.
 *
 * <p>The shared objects body requires all transformations accessing the shared objects, i.e.,
 * {@link SharedObjectsBodyResult#coLocatedTransformations}, to have same parallelism and can be
 * co-located.
 */
@Experimental
@FunctionalInterface
public interface SharedObjectsBody extends Serializable {

    /**
     * This method creates the subgraph for the shared objects body.
     *
     * @param inputs Input data streams.
     * @return Result of the subgraph, including output data streams, data streams with access to
     *     the shared objects, and a mapping from share items to their owners.
     */
    SharedObjectsBodyResult process(List<DataStream<?>> inputs);

    /**
     * The result of a {@link SharedObjectsBody}, including output data streams, data streams with
     * access to the shared objects, and a mapping from descriptors of share items to their owners.
     */
    @Experimental
    class SharedObjectsBodyResult {
        /** A list of output streams. */
        private final List<DataStream<?>> outputs;

        /** A list of {@link Transformation}s that should be co-located. */
        private final List<Transformation<?>> coLocatedTransformations;

        /**
         * A mapping from descriptors of shared items to their owners. The owner is specified by
         * {@link SharedObjectsStreamOperator#getSharedObjectsAccessorID()}, which must be kept
         * unchanged for an instance of {@link SharedObjectsStreamOperator}.
         */
        private final Map<ItemDescriptor<?>, SharedObjectsStreamOperator> ownerMap;

        public SharedObjectsBodyResult(
                List<DataStream<?>> outputs,
                List<Transformation<?>> coLocatedTransformations,
                Map<ItemDescriptor<?>, SharedObjectsStreamOperator> ownerMap) {
            this.outputs = outputs;
            this.coLocatedTransformations = coLocatedTransformations;
            this.ownerMap = ownerMap;
        }

        public List<DataStream<?>> getOutputs() {
            return outputs;
        }

        public List<Transformation<?>> getCoLocatedTransformations() {
            return coLocatedTransformations;
        }

        public Map<ItemDescriptor<?>, SharedObjectsStreamOperator> getOwnerMap() {
            return ownerMap;
        }
    }
}
