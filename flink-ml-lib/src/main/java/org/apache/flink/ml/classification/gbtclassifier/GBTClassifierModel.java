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

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.BaseGBTModel;
import org.apache.flink.ml.common.gbt.GBTModelData;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.analysis.function.Sigmoid;
import org.eclipse.collections.impl.map.mutable.primitive.IntDoubleHashMap;

import java.io.IOException;
import java.util.Collections;

/** A Model computed by {@link GBTClassifier}. */
public class GBTClassifierModel extends BaseGBTModel<GBTClassifierModel>
        implements GBTClassifierModelParams<GBTClassifierModel> {

    /**
     * Loads model data from path.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path Model path.
     * @return GBT classification model.
     */
    public static GBTClassifierModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        GBTClassifierModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new GBTModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> inputStream = tEnv.toDataStream(inputs[0]);
        final String broadcastModelKey = "broadcastModelKey";
        DataStream<GBTModelData> modelDataStream = GBTModelData.getModelDataStream(modelDataTable);
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(),
                                Types.DOUBLE,
                                DenseVectorTypeInfo.INSTANCE,
                                DenseVectorTypeInfo.INSTANCE),
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldNames(),
                                getPredictionCol(),
                                getRawPredictionCol(),
                                getProbabilityCol()));
        DataStream<Row> predictionResult =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(inputStream),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        inputList -> {
                            //noinspection unchecked
                            DataStream<Row> inputData = (DataStream<Row>) inputList.get(0);
                            return inputData.map(
                                    new PredictLabelFunction(broadcastModelKey, getFeaturesCols()),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(predictionResult)};
    }

    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {

        private static final Sigmoid sigmoid = new Sigmoid();

        private final String broadcastModelKey;
        private final String[] featuresCols;
        private GBTModelData modelData;

        public PredictLabelFunction(String broadcastModelKey, String[] featuresCols) {
            this.broadcastModelKey = broadcastModelKey;
            this.featuresCols = featuresCols;
        }

        @Override
        public Row map(Row value) throws Exception {
            if (null == modelData) {
                modelData =
                        (GBTModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
            }
            IntDoubleHashMap features = modelData.rowToFeatures(value, featuresCols);
            double logits = modelData.predictRaw(features);
            double prob = sigmoid.value(logits);
            return Row.join(
                    value,
                    Row.of(
                            logits >= 0. ? 1. : 0.,
                            Vectors.dense(-logits, logits),
                            Vectors.dense(1 - prob, prob)));
        }
    }
}
