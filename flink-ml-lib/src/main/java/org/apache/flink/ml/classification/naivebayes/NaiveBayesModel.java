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
import org.apache.flink.ml.api.core.Model;
import org.apache.flink.ml.common.broadcast.BroadcastUtils;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Naive Bayes Predictor.
 */
public class NaiveBayesModel implements Model<NaiveBayesModel>, NaiveBayesParams<NaiveBayesModel> {
    private static final long serialVersionUID = -4673084154965905629L;
    private final Map<Param<?>, Object> paramMap = new HashMap<>();
    private DataStream<NaiveBayesModelData> modelStream;
    private static final String broadcastModelKey = "NaiveBayesModelStream";

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkNotNull(getPredictionCol());

        List<String> colNames = new ArrayList<>(inputs[0].getResolvedSchema().getColumnNames());
        colNames.add(getPredictionCol());

        List<TypeInformation<?>> colTypes = new ArrayList<>(inputs[0].getResolvedSchema().getColumnDataTypes())
                .stream()
                .map((Function<DataType, TypeInformation<?>>) dataType -> TypeInformation.of(dataType.getConversionClass()))
                .collect(Collectors.toList());
        colTypes.add(TypeInformation.of(Object.class));

        StreamTableEnvironment tEnv = (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<Row> input = tEnv.toDataStream(inputs[0]);

        Map<String, DataStream<?>> broadcastMap = new HashMap<>();
        broadcastMap.put(broadcastModelKey, this.modelStream);

        Function<List<DataStream<?>>, DataStream<Row>> function = dataStreams -> {
            DataStream stream = dataStreams.get(0);
            return stream.transform(
                    this.getClass().getSimpleName(),
                    new RowTypeInfo(colTypes.toArray(new TypeInformation[0]), colNames.toArray(new String[0])),
                    new NaiveBayesPredictOp(new NaiveBayesPredictFunc())
            );
        };
        DataStream<Row> output = BroadcastUtils.withBroadcastStream(Collections.singletonList(input), broadcastMap, function);

        Table outputTable = tEnv.fromDataStream(output);

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
        modelStream = tEnv.toDataStream(inputs[0], DataTypes.RAW(NaiveBayesModelData.class));
    }

    private static class NaiveBayesPredictOp
            extends AbstractUdfStreamOperator<Row, NaiveBayesPredictFunc>
            implements OneInputStreamOperator<Row, Row> {
        public NaiveBayesPredictOp(NaiveBayesPredictFunc userFunction) {
            super(userFunction);
        }

        @Override
        public void processElement(StreamRecord<Row> streamRecord) {
            output.collect(new StreamRecord<>(userFunction.map(streamRecord.getValue())));
        }
    }

    private static class NaiveBayesPredictFunc extends RichMapFunction<Row, Row> {
        @Override
        public Row map(Row row) {
            NaiveBayesModelData modelData = (NaiveBayesModelData) getRuntimeContext().getBroadcastVariable(broadcastModelKey).get(0);
            Object label = findMaxProbLabel(calculateProb(modelData, row), modelData.label);
            return Row.join(row, Row.of(label));
        }
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
            Map<Object, Double>[] labelData = modelData.theta[i];
            for (int j = 0; j < featureSize; j++) {
                Object indexObj = rowData.getField(modelData.featureNames[j]);
                probs[i] += labelData[j].getOrDefault(indexObj, 0.);
            }
        }
        BLAS.axpy(1, modelData.piArray, probs);
        return probs;
    }
}
