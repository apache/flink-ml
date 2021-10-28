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

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.api.core.AlgoOperator;
import org.apache.flink.ml.api.core.Estimator;
import org.apache.flink.ml.api.core.Stage;
import org.apache.flink.ml.param.Param;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Graph acts as an Estimator. A Graph consists of a DAG of stages, each of which could be an
 * Estimator, Model, Transformer or AlgoOperator. When `Graph::fit` is called, the stages are
 * executed in a topologically-sorted order. If a stage is an Estimator, its `Estimator::fit` method
 * will be called on the input tables (from the input edges) to fit a Model. Then the Model will be
 * used to transform the input tables and produce output tables to the output edges. If a stage is
 * an AlgoOperator, its `AlgoOperator::transform` method will be called on the input tables and
 * produce output tables to the output edges. The GraphModel fitted from a Graph consists of the
 * fitted Models and AlgoOperators, corresponding to the Graph's stages.
 */
@PublicEvolving
public final class Graph implements Estimator<Graph, GraphModel> {
    private static final long serialVersionUID = 6354253958813529308L;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private final List<GraphNode> nodes;
    private final TableId[] estimatorInputIds;
    private final TableId[] modelInputIds;
    private final TableId[] outputIds;
    private final TableId[] inputModelData;
    private final TableId[] outputModelData;

    public Graph(
            List<GraphNode> nodes,
            TableId[] estimatorInputIds,
            TableId[] modelInputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        this.nodes = nodes;
        this.estimatorInputIds = estimatorInputIds;
        this.modelInputIds = modelInputs;
        this.outputIds = outputs;
        this.inputModelData = inputModelData;
        this.outputModelData = outputModelData;
    }

    @Override
    public GraphModel fit(Table... inputTables) {
        if (estimatorInputIds.length != inputTables.length) {
            throw new IllegalArgumentException(
                    "number of provided inputs "
                            + inputTables.length
                            + " does not match the expected number of inputs "
                            + estimatorInputIds.length);
        }
        List<GraphNode> modelNodes = new ArrayList<>();

        GraphReadyNodes readyNodes = new GraphReadyNodes(nodes);
        // Update states using the user-provided inputs.
        for (int i = 0; i < estimatorInputIds.length; i++) {
            readyNodes.setReadyTable(estimatorInputIds[i], inputTables[i]);
        }

        // Iterate until we have executed all ready nodes.
        GraphNode node;
        while ((node = readyNodes.pollNextReadyNode()) != null) {
            Stage<?> stage = node.stage;

            if (stage instanceof Estimator) {
                Table[] nodeInputs = new Table[node.estimatorInputs.length];
                for (int i = 0; i < node.estimatorInputs.length; i++) {
                    nodeInputs[i] = readyNodes.getReadyTable(node.estimatorInputs[i]);
                }
                stage = ((Estimator<?, ?>) stage).fit(nodeInputs);
            }

            Table[] nodeInputs = new Table[node.modelInputs.length];
            for (int i = 0; i < node.modelInputs.length; i++) {
                nodeInputs[i] = readyNodes.getReadyTable(node.modelInputs[i]);
            }
            Table[] nodeOutputs = ((AlgoOperator<?>) stage).transform(nodeInputs);

            for (int i = 0; i < nodeOutputs.length; i++) {
                readyNodes.setReadyTable(node.outputs[i], nodeOutputs[i]);
            }
            modelNodes.add(new GraphNode(node.nodeId, stage, null, node.modelInputs, node.outputs));
        }

        return new GraphModel(
                modelNodes, modelInputIds, outputIds, inputModelData, outputModelData);
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        throw new UnsupportedOperationException("this operation is not supported");
    }

    public static Graph load(String path) throws IOException {
        throw new UnsupportedOperationException();
    }
}
