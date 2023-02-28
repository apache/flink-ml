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

package org.apache.flink.ml.regression.gbtregressor;

import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.common.gbt.GBTRunner;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.table.api.Expressions.$;

/** An Estimator which implements the gradient boosting trees regression algorithm. */
public class GBTRegressor
        implements Estimator<GBTRegressor, GBTRegressorModel>, GBTRegressorParams<GBTRegressor> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public GBTRegressor() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    public static GBTRegressor load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public GBTRegressorModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<GBTModelData> modelData = GBTRunner.trainRegressor(inputs[0], this);
        GBTRegressorModel model = new GBTRegressorModel();
        model.setModelData(tEnv.fromDataStream(modelData).renameColumns($("f0").as("modelData")));
        ReadWriteUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }
}