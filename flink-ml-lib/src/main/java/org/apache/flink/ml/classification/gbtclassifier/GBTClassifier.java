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

/**
 * An Estimator which implements the gradient boosting trees classification algorithm (<a
 * href="http://en.wikipedia.org/wiki/Gradient_boosting">Gradient Boosting</a>).
 *
 * <p>The implementation has been inspired by advanced implementations like <a
 * href="https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf">XGBoost</a> and <a
 * href="https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">LightGBM</a>.
 * It supports features like regularized learning objective with second-order approximation,
 * histogram-based and sparsity-aware split-finding algorithm.
 *
 * <p>The implementation of distributed system takes <a
 * href="http://www.vldb.org/pvldb/vol12/p1357-fu.pdf">this work</a> as a reference. Right now, we
 * support horizontal partition of data and row-store storage of instances.
 *
 * <p>NOTE: Currently, some features are not supported yet: weighted input samples, early-stopping
 * with validation set, encoding with leaf ids, etc.
 */
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
        DataStream<GBTModelData> modelData = GBTRunner.train(inputs[0], this);
        DataStream<Map<String, Double>> featureImportance =
                GBTRunner.getFeatureImportance(modelData);
        GBTClassifierModel model = new GBTClassifierModel();
        model.setModelData(
                tEnv.fromDataStream(modelData).renameColumns($("f0").as("modelData")),
                tEnv.fromDataStream(featureImportance)
                        .renameColumns($("f0").as("featureImportance")));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }
}
