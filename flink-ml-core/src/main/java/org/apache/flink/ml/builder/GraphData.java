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

import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** This class contains fields that can be used to re-construct Graph and GraphModel. */
public class GraphData {
    public final List<GraphNode> nodes;
    public final @Nullable TableId[] estimatorInputIds;
    public final TableId[] modelInputIds;
    public final TableId[] outputIds;
    public final @Nullable TableId[] inputModelDataIds;
    public final @Nullable TableId[] outputModelDataIds;

    public GraphData(
            List<GraphNode> nodes,
            TableId[] estimatorInputIds,
            TableId[] modelInputIds,
            TableId[] outputIds,
            TableId[] inputModelDataIds,
            TableId[] outputModelDataIds) {
        this.nodes = Preconditions.checkNotNull(nodes);
        this.estimatorInputIds = estimatorInputIds;
        this.modelInputIds = Preconditions.checkNotNull(modelInputIds);
        this.outputIds = Preconditions.checkNotNull(outputIds);
        this.inputModelDataIds = inputModelDataIds;
        this.outputModelDataIds = outputModelDataIds;
    }

    public Map<String, Object> toMap() {
        Map<String, Object> result = new HashMap<>();

        List<Map<String, Object>> nodeInfos = new ArrayList<>();
        for (GraphNode node : nodes) {
            nodeInfos.add(node.toMap());
        }
        result.put("nodes", nodeInfos);
        if (estimatorInputIds != null) {
            result.put("estimatorInputIds", TableId.toList(estimatorInputIds));
        }
        result.put("modelInputIds", TableId.toList(modelInputIds));
        result.put("outputIds", TableId.toList(outputIds));
        if (inputModelDataIds != null) {
            result.put("inputModelDataIds", TableId.toList(inputModelDataIds));
        }
        if (outputModelDataIds != null) {
            result.put("outputModelDataIds", TableId.toList(outputModelDataIds));
        }
        return result;
    }

    public static GraphData fromMap(Map<String, Object> map) {
        List<GraphNode> nodes = new ArrayList<>();
        List<Map<String, Object>> nodeInfos = (List<Map<String, Object>>) map.get("nodes");
        for (Map<String, Object> nodeInfo : nodeInfos) {
            nodes.add(GraphNode.fromMap(nodeInfo));
        }

        TableId[] estimatorInputIds = null;
        if (map.containsKey("estimatorInputIds")) {
            estimatorInputIds = TableId.fromList((List<Integer>) map.get("estimatorInputIds"));
        }
        TableId[] modelInputIds = TableId.fromList((List<Integer>) map.get("modelInputIds"));
        TableId[] outputIds = TableId.fromList((List<Integer>) map.get("outputIds"));
        TableId[] inputModelDataIds = null;
        if (map.containsKey("inputModelDataIds")) {
            inputModelDataIds = TableId.fromList((List<Integer>) map.get("inputModelDataIds"));
        }
        TableId[] outputModelDataIds = null;
        if (map.containsKey("outputModelDataIds")) {
            outputModelDataIds = TableId.fromList((List<Integer>) map.get("outputModelDataIds"));
        }
        return new GraphData(
                nodes,
                estimatorInputIds,
                modelInputIds,
                outputIds,
                inputModelDataIds,
                outputModelDataIds);
    }
}
