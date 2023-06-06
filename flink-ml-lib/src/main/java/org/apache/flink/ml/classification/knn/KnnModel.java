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

package org.apache.flink.ml.classification.knn;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

/** A Model which classifies data using the model data computed by {@link Knn}. */
public class KnnModel implements Model<KnnModel>, KnnModelParams<KnnModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public KnnModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public KnnModel setModelData(Table... inputs) {
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    @Override
    @SuppressWarnings("unchecked")
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> data = tEnv.toDataStream(inputs[0]);
        DataStream<KnnModelData> knnModel = KnnModelData.getModelDataStream(modelDataTable);
        final String broadcastModelKey = "broadcastModelKey";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), BasicTypeInfo.DOUBLE_TYPE_INFO),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), getPredictionCol()));
        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(data),
                        Collections.singletonMap(broadcastModelKey, knnModel),
                        inputList -> {
                            DataStream input = inputList.get(0);
                            return input.map(
                                    new PredictLabelFunction(
                                            broadcastModelKey, getK(), getFeaturesCol()),
                                    outputTypeInfo);
                        });
        return new Table[] {tEnv.fromDataStream(output)};
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                KnnModelData.getModelDataStream(modelDataTable),
                path,
                new KnnModelData.ModelDataEncoder());
    }

    /**
     * Loads model data from path.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path Model path.
     * @return Knn model.
     */
    public static KnnModel load(StreamTableEnvironment tEnv, String path) throws IOException {
        KnnModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable =
                ReadWriteUtils.loadModelData(tEnv, path, new KnnModelData.ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    /** This operator loads model data and predicts result. */
    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {
        private final String featureCol;
        private KnnModelData knnModelData;
        private final int k;
        private final String broadcastKey;
        private DenseIntDoubleVector distanceVector;

        public PredictLabelFunction(String broadcastKey, int k, String featureCol) {
            this.k = k;
            this.broadcastKey = broadcastKey;
            this.featureCol = featureCol;
        }

        @Override
        public Row map(Row row) {
            if (knnModelData == null) {
                knnModelData =
                        (KnnModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastKey).get(0);
                distanceVector = new DenseIntDoubleVector(knnModelData.labels.size());
            }
            DenseIntDoubleVector feature = ((IntDoubleVector) row.getField(featureCol)).toDense();
            double prediction = predictLabel(feature);
            return Row.join(row, Row.of(prediction));
        }

        private double predictLabel(DenseIntDoubleVector feature) {
            double normSquare = Math.pow(BLAS.norm2(feature), 2);
            BLAS.gemv(-2.0, knnModelData.packedFeatures, true, feature, 0.0, distanceVector);
            for (int i = 0; i < distanceVector.size(); i++) {
                distanceVector.values[i] =
                        Math.sqrt(
                                Math.abs(
                                        distanceVector.values[i]
                                                + normSquare
                                                + knnModelData.featureNormSquares.values[i]));
            }
            PriorityQueue<Tuple2<Double, Double>> nearestKNeighbors =
                    new PriorityQueue<>(
                            Comparator.comparingDouble(distanceAndLabel -> -distanceAndLabel.f0));
            double[] labelValues = knnModelData.labels.values;
            for (int i = 0; i < labelValues.length; ++i) {
                if (nearestKNeighbors.size() < k) {
                    nearestKNeighbors.add(Tuple2.of(distanceVector.get(i), labelValues[i]));
                } else {
                    Tuple2<Double, Double> currentFarthestNeighbor = nearestKNeighbors.peek();
                    if (currentFarthestNeighbor.f0 > distanceVector.get(i)) {
                        nearestKNeighbors.poll();
                        nearestKNeighbors.add(Tuple2.of(distanceVector.get(i), labelValues[i]));
                    }
                }
            }
            Map<Double, Double> labelWeights = new HashMap<>(nearestKNeighbors.size());
            while (!nearestKNeighbors.isEmpty()) {
                Tuple2<Double, Double> distanceAndLabel = nearestKNeighbors.poll();
                labelWeights.merge(distanceAndLabel.f1, 1.0, Double::sum);
            }
            double maxWeight = 0.0;
            double predictedLabel = -1.0;
            for (Map.Entry<Double, Double> entry : labelWeights.entrySet()) {
                if (entry.getValue() > maxWeight) {
                    maxWeight = entry.getValue();
                    predictedLabel = entry.getKey();
                }
            }
            return predictedLabel;
        }
    }
}
