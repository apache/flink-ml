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

import org.apache.flink.table.api.Table;

import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * A container class that maintains the execution state of the graph (e.g. which nodes are ready to
 * run).
 */
class GraphReadyNodes {
    // A map from tableId to the list of nodes which take this table as input.
    private final Map<TableId, List<GraphNode>> dependentNodes = new HashMap<>();
    // A map from tableId to the actual table instance.
    private final Map<TableId, Table> readyTables = new HashMap<>();
    // A map that maintains the number of pending inputs for every node in the graph.
    private final Map<GraphNode, Integer> numPendingInputs = new HashMap<>();
    // An ordered list of nodes which are ready for execution.
    private final Deque<GraphNode> readyNodes = new LinkedList<>();

    public GraphReadyNodes(List<GraphNode> nodes) {
        // Initialize dependentNodes and numPendingInputs.
        for (GraphNode node : nodes) {
            int numInputs = node.modelInputs.length;

            if (node.estimatorInputs != null) {
                numInputs += node.estimatorInputs.length;
                for (TableId tableId : node.estimatorInputs) {
                    dependentNodes.putIfAbsent(tableId, new ArrayList<>());
                    dependentNodes.get(tableId).add(node);
                }
            }
            for (TableId tableId : node.modelInputs) {
                dependentNodes.putIfAbsent(tableId, new ArrayList<>());
                dependentNodes.get(tableId).add(node);
            }
            numPendingInputs.put(node, numInputs);
        }
    }

    public void setReadyTable(TableId tableId, Table table) {
        if (readyTables.containsKey(tableId)) {
            throw new IllegalStateException(
                    "The table with id=" + tableId + " has already been computed.");
        }

        readyTables.put(tableId, table);

        if (!dependentNodes.containsKey(tableId)) {
            return;
        }

        for (GraphNode node : dependentNodes.get(tableId)) {
            int prevNum = numPendingInputs.get(node);
            numPendingInputs.put(node, prevNum - 1);
            if (prevNum == 1) {
                readyNodes.addLast(node);
            }
        }
    }

    public Table getReadyTable(TableId tableId) {
        if (!readyTables.containsKey(tableId)) {
            throw new IllegalStateException(
                    "The table with id=" + tableId + " has not been computed yet.");
        }

        return readyTables.get(tableId);
    }

    public GraphNode pollNextReadyNode() {
        return readyNodes.pollFirst();
    }
}
