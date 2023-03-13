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
import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.ml.api.AlgoOperator;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.TransformerServable;
import org.apache.flink.ml.servable.builder.PipelineModelServable;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A PipelineModel acts as a Model. It consists of an ordered list of stages, each of which could be
 * a Model, Transformer or AlgoOperator.
 */
@PublicEvolving
public final class PipelineModel implements Model<PipelineModel> {
    private static final long serialVersionUID = 6184950154217411318L;
    private final List<Stage<?>> stages;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public PipelineModel(List<Stage<?>> stages) {
        this.stages = Preconditions.checkNotNull(stages);
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    /**
     * Applies all stages in this PipelineModel on the input tables in order. The output of one
     * stage is used as the input of the next stage (if any). The output of the last stage is
     * returned as the result of this method.
     *
     * @param inputs a list of tables
     * @return a list of tables
     */
    @Override
    public Table[] transform(Table... inputs) {
        for (Stage<?> stage : stages) {
            inputs = ((AlgoOperator<?>) stage).transform(inputs);
        }
        return inputs;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.savePipeline(this, stages, path);
    }

    public static PipelineModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        return new PipelineModel(
                ReadWriteUtils.loadPipeline(tEnv, path, PipelineModel.class.getName()));
    }

    public static PipelineModelServable loadServable(String path) throws IOException {
        return PipelineModelServable.load(path);
    }

    /**
     * Whether all stages in the pipeline have corresponding {@link TransformerServable} so that the
     * PipelineModel can be turned into a TransformerServable and used in an online inference
     * program.
     *
     * @return true if all stages have corresponding TransformerServable, false if not.
     */
    public boolean supportServable() {
        for (Stage<?> stage : stages) {
            if (!(stage instanceof Transformer)) {
                return false;
            }
            Transformer<?> transformer = (Transformer<?>) stage;
            Class<?> clazz = transformer.getClass();
            try {
                clazz.getMethod("loadServable", String.class);
            } catch (NoSuchMethodException e) {
                return false;
            }
        }
        return true;
    }

    /**
     * Returns a list of all stages in this PipelineModel in order. The list is immutable.
     *
     * @return an immutable list of transformers.
     */
    @VisibleForTesting
    List<Stage<?>> getStages() {
        return Collections.unmodifiableList(stages);
    }
}
