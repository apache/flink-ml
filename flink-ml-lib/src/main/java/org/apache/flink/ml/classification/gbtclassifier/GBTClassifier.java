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

package org.apache.flink.ml.classification.gbtclassifier;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.table.api.Expressions.$;

/** An Estimator which implements the gradient boosting trees classification algorithm. */
public class GBTClassifier
        implements Estimator<GBTClassifier, GBTClassifierModel>,
                GBTClassifierParams<GBTClassifier> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public GBTClassifier() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    public static GBTClassifier load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public GBTClassifierModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        // TODO: add training part in next PRs.
        DataStream<GBTModelData> modelData =
                tEnv.toDataStream(tEnv.fromValues(0)).map(d -> new GBTModelData("CLASSIFICATION"));
        GBTClassifierModel model = new GBTClassifierModel();
        // TODO: change to GBTModelDataTypeInformation in next PRs.
        model.setModelData(
                tEnv.fromDataStream(
                                modelData
                                        .map(Row::of)
                                        .returns(
                                                Types.ROW_NAMED(
                                                        new String[] {"f0"},
                                                        TypeInformation.of(GBTModelData.class))))
                        .renameColumns($("f0").as("modelData")));
        ReadWriteUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }
}
