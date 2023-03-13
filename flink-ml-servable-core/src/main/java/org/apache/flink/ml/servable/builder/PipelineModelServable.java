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

package org.apache.flink.ml.servable.builder;

import org.apache.flink.annotation.PublicEvolving;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.ModelServable;
import org.apache.flink.ml.servable.api.TransformerServable;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ServableReadWriteUtils;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A PipelineModelServable acts as a {@link ModelServable}. It consists of an ordered list of
 * servables, each of which could be a TransformerServable or ModelServable.
 */
@PublicEvolving
public final class PipelineModelServable implements ModelServable<PipelineModelServable> {

    private final List<TransformerServable<?>> servables;

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public PipelineModelServable(List<TransformerServable<?>> servables) {
        this.servables = Preconditions.checkNotNull(servables);
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public DataFrame transform(DataFrame input) {
        for (TransformerServable<?> servable : servables) {
            input = servable.transform(input);
        }
        return input;
    }

    public static PipelineModelServable load(String path) throws IOException {
        return new PipelineModelServable(ServableReadWriteUtils.loadPipeline(path));
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
