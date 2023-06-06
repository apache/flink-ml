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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.Model;
import org.apache.flink.ml.classification.naivebayes.NaiveBayesModelData.ModelDataDecoder;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/** A Model which classifies data using the model data computed by {@link NaiveBayes}. */
public class NaiveBayesModel
        implements Model<NaiveBayesModel>, NaiveBayesModelParams<NaiveBayesModel> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private Table modelDataTable;

    public NaiveBayesModel() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);

        final String predictionCol = getPredictionCol();
        final String featuresCol = getFeaturesCol();
        final String broadcastModelKey = "NaiveBayesModelStream";

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputs[0].getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(
                                inputTypeInfo.getFieldTypes(), TypeInformation.of(Double.class)),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), predictionCol));

        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) modelDataTable).getTableEnvironment();
        DataStream<NaiveBayesModelData> modelDataStream =
                NaiveBayesModelData.getModelDataStream(modelDataTable);
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        Function<List<DataStream<?>>, DataStream<Row>> function =
                dataStreams -> {
                    DataStream stream = dataStreams.get(0);
                    return stream.map(
                            new PredictLabelFunction(featuresCol, broadcastModelKey),
                            outputTypeInfo);
                };
        DataStream<Row> output =
                BroadcastUtils.withBroadcastStream(
                        Collections.singletonList(input),
                        Collections.singletonMap(broadcastModelKey, modelDataStream),
                        function);

        Table outputTable = tEnv.fromDataStream(output);

        return new Table[] {outputTable};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
        ReadWriteUtils.saveModelData(
                NaiveBayesModelData.getModelDataStream(modelDataTable),
                path,
                new NaiveBayesModelData.ModelDataEncoder());
    }

    public static NaiveBayesModel load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        NaiveBayesModel model = ReadWriteUtils.loadStageParam(path);
        Table modelDataTable = ReadWriteUtils.loadModelData(tEnv, path, new ModelDataDecoder());
        return model.setModelData(modelDataTable);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public NaiveBayesModel setModelData(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        modelDataTable = inputs[0];
        return this;
    }

    @Override
    public Table[] getModelData() {
        return new Table[] {modelDataTable};
    }

    private static class PredictLabelFunction extends RichMapFunction<Row, Row> {
        private final String featuresCol;
        private final String broadcastModelKey;
        private NaiveBayesModelData modelData = null;

        public PredictLabelFunction(String featuresCol, String broadcastModelKey) {
            this.featuresCol = featuresCol;
            this.broadcastModelKey = broadcastModelKey;
        }

        @Override
        public Row map(Row row) {
            if (modelData == null) {
                modelData =
                        (NaiveBayesModelData)
                                getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
            }
            IntDoubleVector vector = (IntDoubleVector) row.getField(featuresCol);
            double label = findMaxProbLabel(calculateProb(modelData, vector), modelData.labels);
            return Row.join(row, Row.of(label));
        }
    }

    private static double findMaxProbLabel(DenseIntDoubleVector prob, IntDoubleVector label) {
        double result = 0.;
        int probSize = prob.size();
        double maxVal = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < probSize; ++i) {
            if (maxVal < prob.values[i]) {
                maxVal = prob.values[i];
                result = label.get(i);
            }
        }
        Preconditions.checkArgument(maxVal > Double.NEGATIVE_INFINITY);
        return result;
    }

    /** Calculate probability of the input data. */
    private static DenseIntDoubleVector calculateProb(
            NaiveBayesModelData modelData, IntDoubleVector data) {
        int labelSize = modelData.labels.size();
        DenseIntDoubleVector probs = new DenseIntDoubleVector(new double[labelSize]);
        for (int i = 0; i < labelSize; i++) {
            Map<Double, Double>[] labelData = modelData.theta[i];
            for (int j = 0; j < data.size(); j++) {
                probs.values[i] += labelData[j].get(data.get(j));
            }
        }
        BLAS.axpy(1, modelData.piArray, probs);
        return probs;
    }
}
