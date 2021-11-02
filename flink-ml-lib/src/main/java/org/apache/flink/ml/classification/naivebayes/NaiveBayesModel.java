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

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.api.core.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.common.broadcast.operator.HasBroadcastVariable;
import org.apache.flink.ml.dataproc.stringindexer.MultiStringIndexerModel;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.expressions.Expression;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.apache.flink.table.api.Expressions.$;

/**
 * Naive Bayes Predictor.
 */
public class NaiveBayesModel implements Model<NaiveBayesModel>, NaiveBayesParams<NaiveBayesModel> {
    private static final long serialVersionUID = -4673084154965905629L;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private DataStream<Row> modelStream;
    private final MultiStringIndexerModel multiStringIndexerPredictBatchOp;
    private static final String broadcastModelKey = "NaiveBayesModelStream";

    public NaiveBayesModel(
            @Nullable MultiStringIndexerModel multiStringIndexerPredictBatchOp) {
        this.multiStringIndexerPredictBatchOp = multiStringIndexerPredictBatchOp;
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkNotNull(getPredictionCol());

        String[] reservedCols = getReservedCols();
        String predictionCol = getPredictionCol();
        int reservedColSize = reservedCols.length;
        int outputSchemaSize = reservedColSize + 1;

        if (multiStringIndexerPredictBatchOp != null) {
            inputs[0] = inputs[0].select(inputs[0]
                    .getResolvedSchema()
                    .getColumnNames()
                    .stream()
                    .map((Function<String, Expression>) s -> $(s).as("raw-" + s))
                    .toArray(Expression[]::new));
            inputs = multiStringIndexerPredictBatchOp.transform(inputs);
            for (int i = 0; i < reservedColSize; i++) {
                reservedCols[i] = "raw-" + reservedCols[i];
            }
            predictionCol = "raw-" + predictionCol;
        }
        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put(broadcastModelKey, this.modelStream);

        TypeInformation<?>[] typeInformations = new TypeInformation[outputSchemaSize];
        for (int i = 0; i < reservedColSize; i++) {
            int index = inputs[0].getResolvedSchema().getColumnNames().indexOf(reservedCols[i]);
            typeInformations[i] = TypeInformation.of(inputs[0].getResolvedSchema().getColumnDataTypes().get(index).getConversionClass());
        }
        typeInformations[reservedColSize] = TypeInformation.of(String.class);

        String[] columnNames = new String[outputSchemaSize];
        System.arraycopy(reservedCols, 0, columnNames, 0, reservedColSize);
        columnNames[reservedColSize] = predictionCol;

        Function<List<DataStream<?>>, DataStream<Row>> function = dataStreams -> {
            DataStream stream = dataStreams.get(0);
            return stream.transform(
                    this.getClass().getSimpleName(),
                    new RowTypeInfo(typeInformations, columnNames),
                    new PredictOperator(reservedCols, columnNames[reservedColSize])
            );
        };
        DataStream<Row> output = BroadcastUtils.withBroadcastStream(Collections.singletonList(input), broadcastMap, function);

        Table outputTable = tEnv.fromDataStream(output);

        if (multiStringIndexerPredictBatchOp != null) {
            outputTable = outputTable.select(outputTable
                    .getResolvedSchema()
                    .getColumnNames()
                    .stream()
                    .map((Function<String, Expression>) s -> $(s).as(s.substring("raw-".length())))
                    .toArray(Expression[]::new));
        }

        return new Table[]{outputTable};
    }

    @Override
    public void save(String path) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<Param<?>, Object> getUserDefinedParamMap() {
        return paramMap;
    }

    @Override
    public void setModelData(Table... inputs) {
        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        modelStream = tEnv.toDataStream(inputs[0]);
    }

    private static class PredictOperator extends AbstractStreamOperator<Row>
            implements OneInputStreamOperator<Row, Row>, HasBroadcastVariable {
        private NaiveBayesModelData modelData = null;
        private final String[] reservedCols;
        private final String predictCol;
        private static final double constant = 0.5 * Math.log(2 * Math.PI);
        /*
         * If the ratio of data variance between dimensions is too small, it
         * will cause numerical errors. To address this, we modify the probability
         * of those data.
         */
        private static final double maxValue = 0.;
        private static final double minValue = Math.log(1e-9);

        private PredictOperator(String[] reservedCols, String predictCol) {
            this.reservedCols = reservedCols;
            this.predictCol = predictCol;
        }

        @Override
        public void setBroadcastVariable(String name, List<?> broadcastVariable) {
            if (name.equals(broadcastModelKey)) {
                Map<Integer, String> map = new HashMap<>();
                for (Object obj: broadcastVariable) {
                    Row row = (Row) obj;
                    row = (Row) row.getField(0);
                    map.put((int) row.getField(0), (String) row.getField(1));
                }
                List<String> list = new ArrayList<>();
                for (int i = 0; i < map.size(); i++) {
                    list.add(map.get(i));
                }
                modelData = NaiveBayesUtils.deserialize(list, NaiveBayesModelData.class);
            }
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) {
            Row row = streamRecord.getValue();
            Row ret = Row.withNames();
            for (String reservedCol : reservedCols) {
                ret.setField(reservedCol, row.getField(reservedCol));
            }
            double[] prob = calculateProb(modelData, row);
            Object result = findMaxProbLabel(prob, modelData.label);
            ret.setField(predictCol, result);
            output.collect(new StreamRecord<>(ret));
        }

        private static Object findMaxProbLabel(double[] prob, Object[] label) {
            Object result = null;
            int probSize = prob.length;
            double maxVal = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < probSize; ++i) {
                if (maxVal < prob[i]) {
                    maxVal = prob[i];
                    result = label[i];
                }
            }
            return result;
        }

        /**
         * Calculate probability of the input data.
         */
        private static double[] calculateProb(NaiveBayesModelData modelData, Row rowData) {
            int labelSize = modelData.label.length;
            double[] probs = new double[labelSize];
            int featureSize = modelData.featureNames.length;
            boolean allZero = true;
            for (String featureName: modelData.featureNames) {
                if (rowData.getField(featureName) != null) {
                    allZero = false;
                    break;
                }
            }

            if (allZero) {
                double prob = 1. / labelSize;
                Arrays.fill(probs, prob);
                return probs;
            }
            for (int i = 0; i < labelSize; i++) {
                Number[][] labelData = modelData.theta[i];
                for (int j = 0; j < featureSize; j++) {
                    Object indexObj = rowData.getField(modelData.featureNames[j]);
                    if (modelData.isCate[j]) {
                        if (indexObj != null) {
                            int index = (int) indexObj;
                            if (index < labelData[j].length) {
                                probs[i] += (Double) labelData[j][index];
                            }
                        }
                    } else {
                        double miu = (double) labelData[j][0];
                        double sigma2 = (double) labelData[j][1];
                        if (indexObj == null) {
                            probs[i] -= (constant + 0.5 * Math.log(sigma2));
                        } else {
                            double data = ((Number) indexObj).doubleValue();
                            if (sigma2 == 0) {
                                if (Math.abs(data - miu) <= 1e-5) {
                                    probs[i] += maxValue;
                                } else {
                                    probs[i] += minValue;
                                }
                            } else {
                                double item1 = Math.pow(data - miu, 2) / (2 * sigma2);
                                probs[i] -= (item1 + constant + 0.5 * Math.log(sigma2));
                            }
                        }
                    }

                }
            }
            BLAS.axpy(1, modelData.piArray, probs);
            return probs;
        }
    }
}
