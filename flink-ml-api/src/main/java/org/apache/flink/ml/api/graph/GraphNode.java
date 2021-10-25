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

package org.apache.flink.ml.api.graph;

import org.apache.flink.ml.api.core.Stage;

import java.util.Objects;

/** The Graph node class. */
class GraphNode {
    public final int nodeId;
    public final Stage<?> stage;
    public final TableId[] estimatorInputs;
    public final TableId[] modelInputs;
    public final TableId[] outputs;

    public GraphNode(
            int nodeId,
            Stage<?> stage,
            TableId[] estimatorInputs,
            TableId[] modelInputs,
            TableId[] outputs) {
        this.nodeId = nodeId;
        this.stage = stage;
        this.estimatorInputs = estimatorInputs;
        this.modelInputs = modelInputs;
        this.outputs = outputs;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (!(obj instanceof GraphNode)) {
            return false;
        }
        GraphNode other = (GraphNode) obj;
        return nodeId == other.nodeId;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(nodeId);
    }

    @Override
    public String toString() {
        return "NodeId(" + nodeId + ")";
    }
}
