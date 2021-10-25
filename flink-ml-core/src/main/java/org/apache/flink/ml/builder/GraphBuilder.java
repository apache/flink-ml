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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.flink.ml.builder.GraphNode.StageType;

/**
 * A GraphBuilder provides APIs to build Estimator/Model/AlgoOperator from a DAG of stages, each of
 * which could be an Estimator, Model, Transformer or AlgoOperator.
 */
@PublicEvolving
public final class GraphBuilder {

    private int maxOutputLength = 20;

    private int nextTableId = 0;

    private int nextNodeId = 0;

    /** An ordered list of nodes in the graph. */
    private final List<GraphNode> nodes = new ArrayList<>();
    /** A map from stage instance to the corresponding node in the graph. */
    private final Map<Stage<?>, GraphNode> existingNodes = new HashMap<>();

    public GraphBuilder() {}

    /**
     * Specifies the loose upper bound of the number of output tables that can be returned by the
     * Model::getModelData() and AlgoOperator::transform() methods, for any stage involved in this
     * Graph.
     *
     * <p>The default upper bound is 20.
     */
    public GraphBuilder setMaxOutputTableNum(int maxOutputLength) {
        this.maxOutputLength = maxOutputLength;
        return this;
    }

    /**
     * Creates a TableId associated with this GraphBuilder. It can be used to specify the passing of
     * tables between stages, as well as the input/output tables of the Graph/GraphModel generated
     * by this builder.
     *
     * @return A TableId.
     */
    public TableId createTableId() {
        return new TableId(nextTableId++);
    }

    /**
     * Adds an AlgoOperator in the graph.
     *
     * <p>When the graph runs as Estimator, the transform() of the given AlgoOperator would be
     * invoked with the given inputs. Then when the GraphModel fitted by this graph runs, the
     * transform() of the given AlgoOperator would be invoked with the given inputs.
     *
     * <p>When the graph runs as AlgoOperator or Model, the transform() of the given AlgoOperator
     * would be invoked with the given inputs.
     *
     * <p>NOTE: the number of the returned TableIds does not represent the actual number of Tables
     * outputted by transform(). This number could be configured using {@link
     * #setMaxOutputTableNum(int)}. Users should make sure that this number >= the actual number of
     * Tables outputted by transform().
     *
     * @param algoOp An AlgoOperator instance.
     * @param inputs A list of TableIds which represents inputs to transform() of the given
     *     AlgoOperator.
     * @return A list of TableIds which represents the outputs of transform() of the given
     *     AlgoOperator.
     */
    public TableId[] addAlgoOperator(AlgoOperator<?> algoOp, TableId... inputs) {
        return addStage(algoOp, StageType.ALGO_OPERATOR, null, inputs);
    }

    /**
     * Adds an Estimator in the graph.
     *
     * <p>When the graph runs as Estimator, the fit() of the given Estimator would be invoked with
     * the given inputs. Then when the GraphModel fitted by this graph runs, the transform() of the
     * Model fitted by the given Estimator would be invoked with the given inputs.
     *
     * <p>When the graph runs as AlgoOperator or Model, the fit() of the given Estimator would be
     * invoked with the given inputs, then the transform() of the Model fitted by the given
     * Estimator would be invoked with the given inputs.
     *
     * <p>NOTE: the number of the returned TableIds does not represent the actual number of Tables
     * outputted by transform(). This number could be configured using {@link
     * #setMaxOutputTableNum(int)}. Users should make sure that this number >= the actual number of
     * Tables outputted by transform().
     *
     * @param estimator An Estimator instance.
     * @param inputs A list of TableIds which represents inputs to fit() of the given Estimator as
     *     well as inputs to transform() of the Model fitted by the given Estimator.
     * @return A list of TableIds which represents the outputs of transform() of the Model fitted by
     *     the given Estimator.
     */
    public TableId[] addEstimator(Estimator<?, ?> estimator, TableId... inputs) {
        return addEstimator(estimator, inputs, inputs);
    }

    /**
     * Adds an Estimator in the graph.
     *
     * <p>When the graph runs as Estimator, the fit() of the given Estimator would be invoked with
     * estimatorInputs. Then when the GraphModel fitted by this graph runs, the transform() of the
     * Model fitted by the given Estimator would be invoked with modelInputs.
     *
     * <p>When the graph runs as AlgoOperator or Model, the fit() of the given Estimator would be
     * invoked with estimatorInputs, then the transform() of the Model fitted by the given Estimator
     * would be invoked with modelInputs.
     *
     * <p>NOTE: the number of the returned TableIds does not represent the actual number of Tables
     * outputted by transform(). This number could be configured using {@link
     * #setMaxOutputTableNum(int)}. Users should make sure that this number >= the actual number of
     * Tables outputted by transform().
     *
     * @param estimator An Estimator instance.
     * @param estimatorInputs A list of TableIds which represents inputs to fit() of the given
     *     Estimator.
     * @param modelInputs A list of TableIds which represents inputs to transform() of the Model
     *     fitted by the given Estimator.
     * @return A list of TableIds which represents the outputs of transform() of the Model fitted by
     *     the given Estimator.
     */
    public TableId[] addEstimator(
            Estimator<?, ?> estimator, TableId[] estimatorInputs, TableId[] modelInputs) {
        return addStage(estimator, StageType.ESTIMATOR, estimatorInputs, modelInputs);
    }

    /**
     * When the graph runs as Estimator, it first generates a GraphModel that contains the Model
     * fitted by the given Estimator. Then when this GraphModel runs, the setModelData() of the
     * fitted Model would be invoked with the given inputs before its transform() is invoked.
     *
     * <p>When the graph runs as AlgoOperator or Model, the setModelData() of the Model fitted by
     * the given Estimator would be invoked with the given inputs before its transform() is invoked.
     *
     * @param estimator An Estimator instance.
     * @param inputs A list of TableIds which represents inputs to setModelData() of the Model
     *     fitted by the given Estimator.
     */
    public void setModelDataOnEstimator(Estimator<?, ?> estimator, TableId... inputs) {
        GraphNode node = existingNodes.get(estimator);
        if (node == null) {
            throw new RuntimeException("the Estimator has not been added to the graph");
        }
        if (node.stageType != StageType.ESTIMATOR) {
            throw new RuntimeException("the Estimator was previously added as an AlgoOperator");
        }
        if (node.inputModelDataIds != null) {
            throw new RuntimeException("the model data of this Estimator has already been set");
        }
        node.inputModelDataIds = inputs;
    }

    /**
     * When the graph runs as Estimator, the setModelData() of the given Model would be invoked with
     * the given inputs before its transform() is invoked. Then when the GraphModel fitted by this
     * graph runs, the setModelData() of the given Model would be invoked with the given inputs.
     *
     * <p>When the graph runs as AlgoOperator or Model, the setModelData() of the given Model would
     * be invoked with the given inputs before its transform() is invoked.
     *
     * @param model A Model instance.
     * @param inputs A list of TableIds which represents inputs to setModelData() of the given
     *     Model.
     */
    public void setModelDataOnModel(Model<?> model, TableId... inputs) {
        GraphNode node = existingNodes.get(model);
        if (node == null) {
            throw new RuntimeException("the Model has not been added to the graph");
        }
        if (node.stageType != StageType.ALGO_OPERATOR) {
            throw new RuntimeException("the Model was previously added as an Estimator");
        }
        if (node.inputModelDataIds != null) {
            throw new RuntimeException("the model data of this Model has already been set");
        }
        node.inputModelDataIds = inputs;
    }

    /**
     * When the graph runs as Estimator, it first generates a GraphModel that contains the Model
     * fitted by the given Estimator. Then when this GraphModel runs, the getModelData() of the
     * fitted Model would be invoked.
     *
     * <p>When the graph runs as AlgoOperator or Model, the getModelData() of the Model fitted by
     * the given Estimator would be invoked.
     *
     * <p>NOTE: the number of the returned TableIds does not represent the actual number of Tables
     * outputted by getModelData(). This number could be configured using {@link
     * #setMaxOutputTableNum(int)}. Users should make sure that this number >= the actual number of
     * Tables outputted by getModelData().
     *
     * @param estimator An Estimator instance.
     * @return A list of TableIds which represents the outputs of getModelData() of the Model fitted
     *     by the given Estimator.
     */
    public TableId[] getModelDataFromEstimator(Estimator<?, ?> estimator) {
        GraphNode node = existingNodes.get(estimator);
        if (node == null) {
            throw new RuntimeException("the Estimator has not been added to the graph");
        }
        if (node.stageType != StageType.ESTIMATOR) {
            throw new RuntimeException("the Estimator was previously added as an AlgoOperator");
        }
        if (node.outputModelDataIds != null) {
            throw new RuntimeException("the model data of this Estimator has already been fetched");
        }
        node.outputModelDataIds = createTableIds(maxOutputLength);
        return node.outputModelDataIds;
    }

    /**
     * When the graph runs as Estimator, the getModelData() of the given Model would be invoked.
     * Then when the GraphModel fitted by this graph runs, the getModelData() of the given Model
     * would be invoked.
     *
     * <p>When the graph runs as AlgoOperator or Model, the getModelData() of the given Model would
     * be invoked.
     *
     * <p>NOTE: the number of the returned TableIds does not represent the actual number of Tables
     * outputted by getModelData(). This number could be configured using {@link
     * #setMaxOutputTableNum(int)}. Users should make sure that this number >= the actual number of
     * Tables outputted by getModelData().
     *
     * @param model A Model instance.
     * @return A list of TableIds which represents the outputs of getModelData() of the given Model.
     */
    public TableId[] getModelDataFromModel(Model<?> model) {
        GraphNode node = existingNodes.get(model);
        if (node == null) {
            throw new RuntimeException("the Model has not been added to the graph");
        }
        if (node.stageType != StageType.ALGO_OPERATOR) {
            throw new RuntimeException("the Model was previously added as an Estimator");
        }
        if (node.outputModelDataIds != null) {
            throw new RuntimeException("the model data of this Model has already been fetched");
        }
        node.outputModelDataIds = createTableIds(maxOutputLength);
        return node.outputModelDataIds;
    }

    /**
     * Wraps nodes of the graph into an Estimator.
     *
     * <p>When the returned Estimator runs, and when the Model fitted by the returned Estimator
     * runs, the sequence of operations recorded by the {@code addAlgoOperator(...)}, {@code
     * addEstimator(...)}, {@code setModelData(...)} and {@code getModelData(...)} would be executed
     * as specified in the Java doc of the corresponding methods.
     *
     * @param inputs A list of TableIds which represents inputs to fit() of the returned Estimator
     *     as well as inputs to transform() of the Model fitted by the returned Estimator.
     * @param outputs A list of TableIds which represents outputs of transform() of the Model fitted
     *     by the returned Estimator.
     * @return An Estimator which wraps the nodes of this graph.
     */
    public Estimator<?, ?> buildEstimator(TableId[] inputs, TableId[] outputs) {
        return buildEstimator(inputs, inputs, outputs, null, null);
    }

    /**
     * Wraps nodes of the graph into an Estimator.
     *
     * <p>When the returned Estimator runs, and when the Model fitted by the returned Estimator
     * runs, the sequence of operations recorded by the {@code addAlgoOperator(...)}, {@code
     * addEstimator(...)}, {@code setModelData(...)} and {@code getModelData(...)} would be executed
     * as specified in the Java doc of the corresponding methods.
     *
     * @param inputs A list of TableIds which represents inputs to fit() of the returned Estimator
     *     as well as inputs to transform() of the Model fitted by the returned Estimator.
     * @param outputs A list of TableIds which represents outputs of transform() of the Model fitted
     *     by the returned Estimator.
     * @param inputModelData A list of TableIds which represents inputs to setModelData() of the
     *     Model fitted by the returned Estimator.
     * @param outputModelData A list of TableIds which represents outputs of getModelData() of the
     *     Model fitted by the returned Estimator.
     * @return An Estimator which wraps the nodes of this graph.
     */
    public Estimator<?, ?> buildEstimator(
            TableId[] inputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        return buildEstimator(inputs, inputs, outputs, inputModelData, outputModelData);
    }

    /**
     * Wraps nodes of the graph into an Estimator.
     *
     * <p>When the returned Estimator runs, and when the Model fitted by the returned Estimator
     * runs, the sequence of operations recorded by the {@code addAlgoOperator(...)}, {@code
     * addEstimator(...)}, {@code setModelData(...)} and {@code getModelData(...)} would be executed
     * as specified in the Java doc of the corresponding methods.
     *
     * @param estimatorInputs A list of TableIds which represents inputs to fit() of the returned
     *     Estimator.
     * @param modelInputs A list of TableIds which represents inputs to transform() of the Model
     *     fitted by the returned Estimator.
     * @param outputs A list of TableIds which represents outputs of transform() of the Model fitted
     *     by the returned Estimator.
     * @param inputModelData A list of TableIds which represents inputs to setModelData() of the
     *     Model fitted by the returned Estimator.
     * @param outputModelData A list of TableIds which represents outputs of getModelData() of the
     *     Model fitted by the returned Estimator.
     * @return An Estimator which wraps the nodes of this graph.
     */
    public Estimator<?, ?> buildEstimator(
            TableId[] estimatorInputs,
            TableId[] modelInputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        return new Graph(
                nodes, estimatorInputs, modelInputs, outputs, inputModelData, outputModelData);
    }

    /**
     * Wraps nodes of the graph into an AlgoOperator.
     *
     * <p>When the returned AlgoOperator runs, the sequence of operations recorded by the {@code
     * addAlgoOperator(...)} and {@code addEstimator(...)} would be executed as specified in the
     * Java doc of the corresponding methods.
     *
     * @param inputs A list of TableIds which represents inputs to transform() of the returned
     *     AlgoOperator.
     * @param outputs A list of TableIds which represents outputs of transform() of the returned
     *     AlgoOperator.
     * @return An AlgoOperator which wraps the nodes of this graph.
     */
    public AlgoOperator<?> buildAlgoOperator(TableId[] inputs, TableId[] outputs) {
        return buildModel(inputs, outputs, null, null);
    }

    /**
     * Wraps nodes of the graph into a Model.
     *
     * <p>When the returned Model runs, the sequence of operations recorded by the {@code
     * addAlgoOperator(...)} and {@code addEstimator(...)} would be executed as specified in the
     * Java doc of the corresponding methods.
     *
     * @param inputs A list of TableIds which represents inputs to transform() of the returned
     *     Model.
     * @param outputs A list of TableIds which represents outputs of transform() of the returned
     *     Model.
     * @return A Model which wraps the nodes of this graph.
     */
    public Model<?> buildModel(TableId[] inputs, TableId[] outputs) {
        return buildModel(inputs, outputs, null, null);
    }

    /**
     * Wraps nodes of the graph into a Model.
     *
     * <p>When the returned Model runs, the sequence of operations recorded by the {@code
     * addAlgoOperator(...)}, {@code addEstimator(...)}, {@code setModelData(...)} and {@code
     * getModelData(...)} would be executed as specified in the Java doc of the corresponding
     * methods.
     *
     * @param inputs A list of TableIds which represents inputs to transform() of the returned
     *     Model.
     * @param outputs A list of TableIds which represents outputs of transform() of the returned
     *     Model.
     * @param inputModelData A list of TableIds which represents inputs to setModelData() of the
     *     returned Model.
     * @param outputModelData A list of TableIds which represents outputs of getModelData() of the
     *     returned Model.
     * @return A Model which wraps the nodes of this graph.
     */
    public Model<?> buildModel(
            TableId[] inputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        return new GraphModel(nodes, inputs, outputs, inputModelData, outputModelData);
    }

    private TableId[] createTableIds(int count) {
        TableId[] result = new TableId[count];
        for (int i = 0; i < count; i++) {
            result[i] = createTableId();
        }
        return result;
    }

    private TableId[] addStage(
            Stage<?> stage, StageType stageType, TableId[] estimatorInputs, TableId[] modelInputs) {
        TableId[] outputs = createTableIds(maxOutputLength);
        if (existingNodes.containsKey(stage)) {
            throw new RuntimeException("The stage " + stage + " has already been added.");
        }
        GraphNode node =
                new GraphNode(
                        nextNodeId++,
                        stage,
                        stageType,
                        estimatorInputs,
                        modelInputs,
                        outputs,
                        null,
                        null);
        nodes.add(node);
        existingNodes.put(stage, node);
        return outputs;
    }
}
