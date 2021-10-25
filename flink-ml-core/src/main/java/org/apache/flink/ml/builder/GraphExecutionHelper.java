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

import org.apache.flink.table.api.Table;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * A container class that maintains the execution state of the graph (e.g. which nodes are ready to
 * run).
 */
class GraphExecutionHelper {
    /** A map from tableId to the list of nodes which take this table as input. */
    private final Map<TableId, List<GraphNode>> consumerNodes = new HashMap<>();
    /**
     * A map from tableId to the corresponding table. A TableId would be mapped iff its
     * corresponding Table has been constructed.
     */
    private final Map<TableId, Table> constructedTables = new HashMap<>();
    /**
     * A map that maintains the number of input tables that have not been constructed for each node
     * in the graph.
     */
    private final Map<GraphNode, Integer> numUnConstructedInputTables = new HashMap<>();
    /**
     * An ordered list of nodes whose input tables have all been constructed AND who has not been
     * fetch via pollNextReadyNode.
     */
    private final Deque<GraphNode> unFetchedReadyNodes = new LinkedList<>();

    public GraphExecutionHelper(List<GraphNode> nodes) {
        // Initializes dependentNodes and numUnConstructedInputs.
        for (GraphNode node : nodes) {
            List<TableId> inputs = new ArrayList<>();
            inputs.addAll(Arrays.asList(node.algoOpInputIds));
            if (node.estimatorInputIds != null) {
                inputs.addAll(Arrays.asList(node.estimatorInputIds));
            }
            if (node.inputModelDataIds != null) {
                inputs.addAll(Arrays.asList(node.inputModelDataIds));
            }
            numUnConstructedInputTables.put(node, inputs.size());
            for (TableId tableId : inputs) {
                consumerNodes.putIfAbsent(tableId, new ArrayList<>());
                consumerNodes.get(tableId).add(node);
            }
        }
    }

    public void setTables(TableId[] tableIds, Table[] tables) {
        // The length of tableIds could be larger than the length of tables because we over-allocate
        // the number of tableIds (which is 20 by default) as placeholder of the stage's output
        // tables when adding a stage in GraphBuilder.
        Preconditions.checkArgument(
                tableIds.length >= tables.length,
                "the length of tablesIds %s is less than the length of tables %s",
                tableIds.length,
                tables.length);
        for (int i = 0; i < tables.length; i++) {
            setTable(tableIds[i], tables[i]);
        }
    }

    private void setTable(TableId tableId, Table table) {
        Preconditions.checkArgument(
                !constructedTables.containsKey(tableId),
                "the table with id=%s has already been constructed",
                tableId.toString());
        constructedTables.put(tableId, table);

        for (GraphNode node : consumerNodes.getOrDefault(tableId, new ArrayList<>())) {
            int prevNum = numUnConstructedInputTables.get(node);
            if (prevNum == 1) {
                unFetchedReadyNodes.addLast(node);
                numUnConstructedInputTables.remove(node);
            } else {
                numUnConstructedInputTables.put(node, prevNum - 1);
            }
        }
    }

    public Table[] getTables(TableId[] tableIds) {
        Table[] tables = new Table[tableIds.length];
        for (int i = 0; i < tableIds.length; i++) {
            tables[i] = getTable(tableIds[i]);
        }
        return tables;
    }

    private Table getTable(TableId tableId) {
        Preconditions.checkArgument(
                constructedTables.containsKey(tableId),
                "the table with id=%s has not been constructed yet",
                tableId.toString());
        return constructedTables.get(tableId);
    }

    public GraphNode pollNextReadyNode() {
        if (unFetchedReadyNodes.isEmpty() && !numUnConstructedInputTables.isEmpty()) {
            throw new RuntimeException("there exists node whose input can not be constructed");
        }
        return unFetchedReadyNodes.pollFirst();
    }
}
