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

import java.util.ArrayList;
import java.util.List;

/**
 * A GraphBuilder provides APIs to build Estimator/Model/AlgoOperator from a DAG of stages, each of
 * which could be an Estimator, Model, Transformer or AlgoOperator.
 */
@PublicEvolving
public final class GraphBuilder {

    private final List<GraphNode> nodes = new ArrayList<>();
    private int maxOutputLength = 20;
    private int nextTableId = 0;
    private int nextNodeId = 0;

    public GraphBuilder() {}

    /**
     * Specifies the upper bound (could be loose) of the number of output tables that can be
     * returned by the Model::getModelData and AlgoOperator::transform methods, for any stage
     * involved in this Graph.
     *
     * <p>The default upper bound is 20.
     */
    public GraphBuilder setMaxOutputLength(int maxOutputLength) {
        this.maxOutputLength = maxOutputLength;
        return this;
    }

    /**
     * Creates a TableId associated with this GraphBuilder. It can be used to specify the passing of
     * tables between stages, as well as the input/output tables of the Graph/GraphModel generated
     * by this builder.
     */
    public TableId createTableId() {
        return new TableId(nextTableId++);
    }

    /**
     * If the stage is an Estimator, both its fit method and the transform method of its fitted
     * Model would be invoked with the given inputs when the graph runs.
     *
     * <p>If this stage is a Model, Transformer or AlgoOperator, its transform method would be
     * invoked with the given inputs when the graph runs.
     *
     * <p>Returns a list of TableIds, which represents outputs of AlgoOperator::transform of the
     * given stage.
     */
    public TableId[] getOutputs(Stage<?> stage, TableId... inputs) {
        TableId[] outputs = new TableId[maxOutputLength];
        for (int i = 0; i < maxOutputLength; i++) {
            outputs[i] = createTableId();
        }
        if (stage instanceof Estimator) {
            nodes.add(new GraphNode(nextNodeId++, stage, inputs, inputs, outputs));
        } else {
            nodes.add(new GraphNode(nextNodeId++, stage, null, inputs, outputs));
        }
        return outputs;
    }

    /**
     * If this stage is an Estimator, its fit method would be invoked with estimatorInputs, and the
     * transform method of its fitted Model would be invoked with modelInputs.
     *
     * <p>This method throws Exception if the stage is not an Estimator.
     *
     * <p>This method is useful when the state is an Estimator AND the Estimator::fit needs to take
     * a different list of Tables from the Model::transform of the fitted Model.
     *
     * <p>Returns a list of TableIds, which represents outputs of Model::transform of the fitted
     * Model.
     */
    public TableId[] getOutputs(Stage<?> stage, TableId[] estimatorInputs, TableId[] modelInputs) {
        if (!(stage instanceof Estimator)) {
            throw new IllegalArgumentException(
                    "stage should be Estimator but it is " + stage.getClass().getName());
        }

        TableId[] outputs = new TableId[maxOutputLength];
        for (int i = 0; i < maxOutputLength; i++) {
            outputs[i] = createTableId();
        }
        nodes.add(new GraphNode(nextNodeId++, stage, estimatorInputs, modelInputs, outputs));
        return outputs;
    }

    /**
     * The setModelData() of the fitted GraphModel should invoke the setModelData() of the given
     * stage with the given inputs.
     */
    public void setModelData(Stage<?> stage, TableId... inputs) {
        throw new UnsupportedOperationException();
    }

    /**
     * The getModelData() of the fitted GraphModel should invoke the getModelData() of the given
     * stage.
     *
     * <p>Returns a list of TableIds, which represents the outputs of getModelData() of the given
     * stage.
     */
    public TableId[] getModelData(Stage<?> stage) {
        throw new UnsupportedOperationException();
    }

    /**
     * Returns an Estimator instance with the following behavior:
     *
     * <p>1) Estimator::fit should take the given inputs and return a Model with the following
     * behavior.
     *
     * <p>2) Model::transform should take the given inputs and return the given outputs.
     *
     * <p>The fit method of the returned Estimator and the transform method of the fitted Model
     * should invoke the corresponding methods of the internal stages as specified by the
     * GraphBuilder.
     */
    public Estimator<?, ?> buildEstimator(TableId[] inputs, TableId[] outputs) {
        return buildEstimator(inputs, inputs, outputs, null, null);
    }

    /**
     * Returns an Estimator instance with the following behavior:
     *
     * <p>1) Estimator::fit should take the given inputs and returns a Model with the following
     * behavior.
     *
     * <p>2) Model::transform should take the given inputs and return the given outputs.
     *
     * <p>3) Model::setModelData should take the given inputModelData.
     *
     * <p>4) Model::getModelData should return the given outputModelData.
     *
     * <p>The fit method of the returned Estimator and the transform/setModelData/getModelData
     * methods of the fitted Model should invoke the corresponding methods of the internal stages as
     * specified by the GraphBuilder.
     */
    public Estimator<?, ?> buildEstimator(
            TableId[] inputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        return buildEstimator(inputs, inputs, outputs, inputModelData, outputModelData);
    }

    /**
     * Returns an Estimator instance with the following behavior:
     *
     * <p>1) Estimator::fit should take the given estimatorInputs and returns a Model with the
     * following behavior.
     *
     * <p>2) Model::transform should take the given transformerInputs and return the given outputs.
     *
     * <p>3) Model::setModelData should take the given inputModelData.
     *
     * <p>4) Model::getModelData should return the given outputModelData.
     *
     * <p>The fit method of the returned Estimator and the transform/setModelData/getModelData
     * methods of the fitted Model should invoke the corresponding methods of the internal stages as
     * specified by the GraphBuilder.
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
     * Returns an AlgoOperator instance with the following behavior:
     *
     * <p>1) AlgoOperator::transform should take the given inputs and returns the given outputs.
     *
     * <p>The transform method of the returned AlgoOperator should invoke the corresponding methods
     * of the internal stages as specified by the GraphBuilder.
     */
    public AlgoOperator<?> buildAlgoOperator(TableId[] inputs, TableId[] outputs) {
        return buildModel(inputs, outputs, null, null);
    }

    /**
     * Returns a Model instance with the following behavior:
     *
     * <p>1) Model::transform should take the given inputs and returns the given outputs.
     *
     * <p>The transform method of the returned Model should invoke the corresponding methods of the
     * internal stages as specified by the GraphBuilder.
     */
    public Model<?> buildModel(TableId[] inputs, TableId[] outputs) {
        return buildModel(inputs, outputs, null, null);
    }

    /**
     * Returns a Model instance with the following behavior:
     *
     * <p>1) Model::transform should take the given inputs and returns the given outputs.
     *
     * <p>2) Model::setModelData should take the given inputModelData.
     *
     * <p>3) Model::getModelData should return the given outputModelData.
     *
     * <p>The transform/setModelData/getModelData methods of the returned Model should invoke the
     * corresponding methods of the internal stages as specified by the GraphBuilder.
     */
    public Model<?> buildModel(
            TableId[] inputs,
            TableId[] outputs,
            TableId[] inputModelData,
            TableId[] outputModelData) {
        return new GraphModel(nodes, inputs, outputs, inputModelData, outputModelData);
    }
}
