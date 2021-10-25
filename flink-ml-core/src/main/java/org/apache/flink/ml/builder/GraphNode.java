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

package org.apache.flink.ml.builder;

import org.apache.flink.ml.api.Stage;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** The Graph node class. */
public class GraphNode {
    /** This class specifies whether a node should be used as Estimator or AlgoOperator. */
    public enum StageType {
        ESTIMATOR,
        ALGO_OPERATOR;
    }

    public final int nodeId;
    public @Nullable Stage<?> stage;
    public final StageType stageType;
    public final @Nullable TableId[] estimatorInputIds;
    public final TableId[] algoOpInputIds;
    public final TableId[] outputIds;
    public @Nullable TableId[] inputModelDataIds;
    public @Nullable TableId[] outputModelDataIds;

    public GraphNode(
            int nodeId,
            Stage<?> stage,
            StageType stageType,
            TableId[] estimatorInputIds,
            TableId[] algoOpInputIds,
            TableId[] outputIds,
            TableId[] inputModelDataIds,
            TableId[] outputModelDataIds) {
        this.nodeId = Preconditions.checkNotNull(nodeId);
        this.stage = stage;
        this.stageType = Preconditions.checkNotNull(stageType);
        this.estimatorInputIds = estimatorInputIds;
        this.algoOpInputIds = Preconditions.checkNotNull(algoOpInputIds);
        this.outputIds = Preconditions.checkNotNull(outputIds);
        this.inputModelDataIds = inputModelDataIds;
        this.outputModelDataIds = outputModelDataIds;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> result = new HashMap<>();
        result.put("nodeId", nodeId);
        result.put("stageType", stageType.name());
        if (estimatorInputIds != null) {
            result.put("estimatorInputIds", TableId.toList(estimatorInputIds));
        }
        result.put("algoOpInputIds", TableId.toList(algoOpInputIds));
        result.put("outputIds", TableId.toList(outputIds));
        if (inputModelDataIds != null) {
            result.put("inputModelDataIds", TableId.toList(inputModelDataIds));
        }
        if (outputModelDataIds != null) {
            result.put("outputModelDataIds", TableId.toList(outputModelDataIds));
        }
        return result;
    }

    public static GraphNode fromMap(Map<String, Object> map) {
        int nodeId = (Integer) map.get("nodeId");
        StageType stageType = StageType.valueOf((String) map.get("stageType"));
        TableId[] estimatorInputIds = null;
        if (map.containsKey("estimatorInputIds")) {
            estimatorInputIds = TableId.fromList((List<Integer>) map.get("estimatorInputIds"));
        }
        TableId[] algoOpInputIds = TableId.fromList((List<Integer>) map.get("algoOpInputIds"));
        TableId[] outputIds = TableId.fromList((List<Integer>) map.get("outputIds"));
        TableId[] inputModelDataIds = null;
        if (map.containsKey("inputModelDataIds")) {
            inputModelDataIds = TableId.fromList((List<Integer>) map.get("inputModelDataIds"));
        }
        TableId[] outputModelDataIds = null;
        if (map.containsKey("outputModelDataIds")) {
            outputModelDataIds = TableId.fromList((List<Integer>) map.get("outputModelDataIds"));
        }
        return new GraphNode(
                nodeId,
                null,
                stageType,
                estimatorInputIds,
                algoOpInputIds,
                outputIds,
                inputModelDataIds,
                outputModelDataIds);
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
        return String.format(
                "GraphNode(nodeId=%d, stageType=%s, estimatorInputIds=%s, algoOpInputIds=%s, outputIds=%s, inputModelDataIds=%s, outputModelDataIds=%s)",
                nodeId,
                stageType.name(),
                Arrays.toString(estimatorInputIds),
                Arrays.toString(algoOpInputIds),
                Arrays.toString(outputIds),
                Arrays.toString(inputModelDataIds),
                Arrays.toString(outputModelDataIds));
    }
}
