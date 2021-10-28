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
import org.apache.flink.ml.api.core.Model;
import org.apache.flink.ml.api.core.Stage;
import org.apache.flink.ml.param.Param;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A GraphModel acts as a Model. A GraphModel consists of a DAG of stages, each of which could be an
 * Estimator, Model, Transformer or AlgoOperators. When `GraphModel::transform` is called, the
 * stages are executed in a topologically-sorted order. When a stage is executed, its
 * `AlgoOperator::transform` method will be called on the input tables (from the input edges) and
 * produce output tables to the output edges.
 */
@PublicEvolving
public final class GraphModel implements Model<GraphModel> {
    private static final long serialVersionUID = 6354856913812529398L;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private final List<GraphNode> nodes;
    private final TableId[] inputIds;
    private final TableId[] outputIds;
    private final TableId[] inputModelData;
    private final TableId[] outputModelData;

    public GraphModel(
            List<GraphNode> nodes,
            TableId[] inputIds,
            TableId[] outputIds,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        this.nodes = nodes;
        this.inputIds = inputIds;
        this.outputIds = outputIds;
        this.inputModelData = inputModelData;
        this.outputModelData = outputModelData;
    }

    @Override
    public Table[] transform(Table... inputTables) {
        if (inputIds.length != inputTables.length) {
            throw new IllegalArgumentException(
                    "number of provided inputs "
                            + inputTables.length
                            + " does not match the expected number of inputs "
                            + inputIds.length);
        }

        GraphReadyNodes readyNodes = new GraphReadyNodes(nodes);
        // Update states using the user-provided inputs.
        for (int i = 0; i < inputIds.length; i++) {
            readyNodes.setReadyTable(inputIds[i], inputTables[i]);
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
        }

        Table[] results = new Table[outputIds.length];
        for (int i = 0; i < outputIds.length; i++) {
            results[i] = readyNodes.getReadyTable(outputIds[i]);
        }
        return results;
    }

    @Override
    public void setModelData(Table... inputs) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Table[] getModelData() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        throw new UnsupportedOperationException("this operation is not supported");
    }

    public static GraphModel load(String path) throws IOException {
        throw new UnsupportedOperationException();
    }
}
