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

package org.apache.flink.ml.api.core;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Pipeline acts as an Estimator. It consists of an ordered list of stages, each of which could be
 * an Estimator, Model, Transformer or AlgoOperator.
 */
@PublicEvolving
public final class Pipeline implements Estimator<Pipeline, PipelineModel> {
    private static final long serialVersionUID = 6384850154817512318L;
    private final List<Stage<?>> stages;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public Pipeline(List<Stage<?>> stages) {
        this.stages = stages;
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    /**
     * Trains the pipeline to fit on the given tables.
     *
     * <p>This method goes through all stages of this pipeline in order and does the following on
     * each stage until the last Estimator (inclusive).
     *
     * <ul>
     *   <li>If a stage is an Estimator, invoke {@link Estimator#fit(Table...)} with the input
     *       tables to generate a Model. And if there is Estimator after this stage, transform the
     *       input tables using the generated Model to get result tables, then pass the result
     *       tables to the next stage as inputs.
     *   <li>If a stage is an AlgoOperator AND there is Estimator after this stage, transform the
     *       input tables using this stage to get result tables, then pass the result tables to the
     *       next stage as inputs.
     * </ul>
     *
     * <p>After all the Estimators are trained to fit their input tables, a new PipelineModel will
     * be created with the same stages in this pipeline, except that all the Estimators in the
     * PipelineModel are replaced with the models generated in the above process.
     *
     * @param inputs a list of tables
     * @return a PipelineModel
     */
    @Override
    public PipelineModel fit(Table... inputs) {
        int lastEstimatorIdx = -1;
        for (int i = 0; i < stages.size(); i++) {
            if (stages.get(i) instanceof Estimator) {
                lastEstimatorIdx = i;
            }
        }

        List<Stage<?>> modelStages = new ArrayList<>(stages.size());
        Table[] lastInputs = inputs;

        for (int i = 0; i < stages.size(); i++) {
            Stage<?> stage = stages.get(i);
            AlgoOperator<?> modelStage;
            if (stage instanceof AlgoOperator) {
                modelStage = (AlgoOperator<?>) stage;
            } else {
                modelStage = ((Estimator<?, ?>) stage).fit(lastInputs);
            }
            modelStages.add(modelStage);

            // Transforms inputs only if there exists Estimator stage after this stage.
            if (i < lastEstimatorIdx) {
                lastInputs = modelStage.transform(lastInputs);
            }
        }

        return new PipelineModel(modelStages);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.savePipeline(this, stages, path);
    }

    public static Pipeline load(String path) throws IOException {
        return new Pipeline(ReadWriteUtils.loadPipeline(path, Pipeline.class.getName()));
    }

    /**
     * Returns a list of all stages in this Pipeline in order. The list is immutable.
     *
     * @return an immutable list of stages.
     */
    @VisibleForTesting
    List<Stage<?>> getStages() {
        return Collections.unmodifiableList(stages);
    }
}
