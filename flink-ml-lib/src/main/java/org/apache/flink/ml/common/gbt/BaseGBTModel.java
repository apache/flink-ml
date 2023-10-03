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

import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.common.typeutils.base.DoubleSerializer;
import org.apache.flink.api.common.typeutils.base.MapSerializer;
import org.apache.flink.api.common.typeutils.base.StringSerializer;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.FSDataInputStream;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputView;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.classification.gbtclassifier.GBTClassifier;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.regression.gbtregressor.GBTRegressor;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/** Base model computed by {@link GBTClassifier} or {@link GBTRegressor}. */
public abstract class BaseGBTModel<T extends BaseGBTModel<T>> implements Model<T> {
    protected static final String MODEL_DATA_PATH = "model_data";
    protected static final String FEATURE_IMPORTANCE_PATH = "feature_importance";

    protected final Map<Param<?>, Object> paramMap = new HashMap<>();
    protected Table modelDataTable;
    protected Table featureImportanceTable;

    public BaseGBTModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    protected static <T extends BaseGBTModel<T>> T load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        T model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(
                        tEnv,
                        new Path(path, MODEL_DATA_PATH).toString(),
                        new GBTModelData.ModelDataDecoder());
        Table featureImportanceTable =
                ReadWriteUtils.loadModelData(
                        tEnv,
                        new Path(path, FEATURE_IMPORTANCE_PATH).toString(),
                        new FeatureImportanceEncoderDecoder());
        return model.setModelData(modelDataTable, featureImportanceTable);
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
                new Path(path, MODEL_DATA_PATH).toString(),
                new GBTModelData.ModelDataEncoder());

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) featureImportanceTable).getTableEnvironment();
        ReadWriteUtils.saveModelData(
                tEnv.toDataStream(
                        featureImportanceTable,
                        DataTypes.MAP(DataTypes.STRING(), DataTypes.DOUBLE())),
                new Path(path, FEATURE_IMPORTANCE_PATH).toString(),
                new FeatureImportanceEncoderDecoder());
    }

    private static class FeatureImportanceEncoderDecoder
            extends SimpleStreamFormat<Map<String, Double>>
            implements Encoder<Map<String, Double>> {

        final MapSerializer<String, Double> serializer =
                new MapSerializer<>(StringSerializer.INSTANCE, DoubleSerializer.INSTANCE);

        @Override
        public void encode(Map<String, Double> element, OutputStream stream) throws IOException {
            DataOutputView dataOutputView = new DataOutputViewStreamWrapper(stream);
            serializer.serialize(element, dataOutputView);
        }

        @Override
        public Reader<Map<String, Double>> createReader(
                Configuration config, FSDataInputStream stream) throws IOException {
            return new Reader<Map<String, Double>>() {
                @Override
                public Map<String, Double> read() {
                    DataInputView source = new DataInputViewStreamWrapper(stream);
                    try {
                        return serializer.deserialize(source);
                    } catch (IOException e) {
                        return null;
                    }
                }

                @Override
                public void close() throws IOException {
                    stream.close();
                }
            };
        }

        @Override
        public TypeInformation<Map<String, Double>> getProducedType() {
            return Types.MAP(Types.STRING, Types.DOUBLE);
        }
    }
}
