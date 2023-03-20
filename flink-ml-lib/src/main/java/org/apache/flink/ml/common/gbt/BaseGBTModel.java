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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifier;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressor;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/** Base model computed by {@link GBTClassifier} or {@link GBTRegressor}. */
public abstract class BaseGBTModel<T extends BaseGBTModel<T>> implements Model<T> {

    protected final Map<Param<?>, Object> paramMap = new HashMap<>();
    protected Table modelDataTable;
    protected Table featureImportanceTable;

    public BaseGBTModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable, featureImportanceTable};
    }

    @Override
    public T setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 2);
        modelDataTable = inputs[0];
        featureImportanceTable = inputs[1];
        //noinspection unchecked
        return (T) this;
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                GBTModelData.getModelDataStream(modelDataTable),
                path,
                new GBTModelData.ModelDataEncoder());
    }
}
