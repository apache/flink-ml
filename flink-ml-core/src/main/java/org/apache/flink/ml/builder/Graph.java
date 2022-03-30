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

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.builder.GraphNode.StageType;

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
    private final @Nullable TableId[] inputModelDataIds;
    private final @Nullable TableId[] outputModelDataIds;

    public Graph(
            List<GraphNode> nodes,
            TableId[] estimatorInputIds,
            TableId[] modelInputs,
            TableId[] outputs,
            TableId[] inputModelDataIds,
            TableId[] outputModelDataIds) {
        this.nodes = Preconditions.checkNotNull(nodes);
        this.estimatorInputIds = Preconditions.checkNotNull(estimatorInputIds);
        this.modelInputIds = Preconditions.checkNotNull(modelInputs);
        this.outputIds = Preconditions.checkNotNull(outputs);
        this.inputModelDataIds = inputModelDataIds;
        this.outputModelDataIds = outputModelDataIds;
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public GraphModel fit(Table... inputTables) {
        Preconditions.checkArgument(
                estimatorInputIds.length == inputTables.length,
                "number of provided tables %s does not match the expected number of tables %s",
                inputTables.length,
                estimatorInputIds.length);
        List<GraphNode> modelNodes = new ArrayList<>();
        GraphExecutionHelper executionHelper = new GraphExecutionHelper(nodes);
        // Maps estimatorInputIds to inputTables.
        executionHelper.setTables(estimatorInputIds, inputTables);
        // Iterates until we have executed all ready nodes.
        GraphNode node;
        while ((node = executionHelper.pollNextReadyNode()) != null) {
            Stage<?> stage = node.stage;
            // Invokes fit(...) if stageType == ESTIMATOR.
            if (node.stageType == StageType.ESTIMATOR) {
                stage =
                        ((Estimator<?, ?>) stage)
                                .fit(executionHelper.getTables(node.estimatorInputIds));
            }
            // Invokes setModelData(...).
            if (node.inputModelDataIds != null) {
                Table[] nodeInputModelData = executionHelper.getTables(node.inputModelDataIds);
                ((Model<?>) stage).setModelData(nodeInputModelData);
            }
            // Invokes transform(...).
            Table[] nodeOutputs =
                    ((AlgoOperator<?>) stage)
                            .transform(executionHelper.getTables(node.algoOpInputIds));
            executionHelper.setTables(node.outputIds, nodeOutputs);
            // Invokes getModelData().
            if (node.outputModelDataIds != null) {
                Table[] nodeOutputModelData = ((Model<?>) stage).getModelData();
                executionHelper.setTables(node.outputModelDataIds, nodeOutputModelData);
            }

            modelNodes.add(
                    new GraphNode(
                            node.nodeId,
                            stage,
                            StageType.ALGO_OPERATOR,
                            null,
                            node.algoOpInputIds,
                            node.outputIds,
                            node.inputModelDataIds,
                            node.outputModelDataIds));
        }
        return new GraphModel(
                modelNodes, modelInputIds, outputIds, inputModelDataIds, outputModelDataIds);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        GraphData graphData =
                new GraphData(
                        nodes,
                        estimatorInputIds,
                        modelInputIds,
                        outputIds,
                        inputModelDataIds,
                        outputModelDataIds);
        ReadWriteUtils.saveGraph(this, graphData, path);
    }

    public static Graph load(StreamTableEnvironment tEnv, String path) throws IOException {
        return (Graph) ReadWriteUtils.loadGraph(tEnv, path, Graph.class.getName());
    }
}
