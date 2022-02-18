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
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.builder.GraphNode.StageType;

/**
 * A GraphModel acts as a Model. A GraphModel consists of a DAG of stages, each of which could be an
 * Estimator, Model, Transformer or AlgoOperator. When `GraphModel::transform` is called, the stages
 * are executed in a topologically-sorted order. When a stage is executed, its
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
    private final @Nullable TableId[] inputModelDataIds;
    private final @Nullable TableId[] outputModelDataIds;
    private final GraphExecutionHelper executionHelper;

    public GraphModel(
            List<GraphNode> nodes,
            TableId[] inputIds,
            TableId[] outputIds,
            TableId[] inputModelDataIds,
            TableId[] outputModelDataIds) {
        this.nodes = Preconditions.checkNotNull(nodes);
        this.inputIds = Preconditions.checkNotNull(inputIds);
        this.outputIds = Preconditions.checkNotNull(outputIds);
        this.inputModelDataIds = inputModelDataIds;
        this.outputModelDataIds = outputModelDataIds;
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
        executionHelper = new GraphExecutionHelper(nodes);
    }

    @Override
    public Table[] transform(Table... inputTables) {
        Preconditions.checkArgument(
                inputIds.length == inputTables.length,
                "number of provided tables %s does not match the expected number of tables %s",
                inputTables.length,
                inputIds.length);
        // Maps inputIds to inputTables.
        executionHelper.setTables(inputIds, inputTables);
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
        }
        return executionHelper.getTables(outputIds);
    }

    @Override
    public GraphModel setModelData(Table... inputTables) {
        Preconditions.checkArgument(inputModelDataIds != null, "setModelData() is not supported");
        Preconditions.checkArgument(
                inputModelDataIds.length == inputTables.length,
                "number of provided tables %s does not match the expected number of tables %s",
                inputTables.length,
                inputIds.length);
        // Maps inputModelDataIds to inputTables.
        executionHelper.setTables(inputModelDataIds, inputTables);
        return this;
    }

    @Override
    public Table[] getModelData() {
        Preconditions.checkArgument(outputModelDataIds != null);
        return executionHelper.getTables(outputModelDataIds);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        GraphData graphData =
                new GraphData(
                        nodes, null, inputIds, outputIds, inputModelDataIds, outputModelDataIds);
        ReadWriteUtils.saveGraph(this, graphData, path);
    }

    public static GraphModel load(StreamExecutionEnvironment env, String path) throws IOException {
        return (GraphModel) ReadWriteUtils.loadGraph(env, path, GraphModel.class.getName());
    }
}
