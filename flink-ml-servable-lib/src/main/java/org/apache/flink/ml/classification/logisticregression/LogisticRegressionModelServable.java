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

package org.apache.flink.ml.classification.logisticregression;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.memory.DataInputViewStreamWrapper;
import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorSerializer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.ModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ServableReadWriteUtils;
import org.apache.flink.util.Preconditions;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A Servable which can be used to classifies data in online inference. */
public class LogisticRegressionModelServable
        implements ModelServable<LogisticRegressionModelServable>,
                LogisticRegressionModelParams<LogisticRegressionModelServable> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    private DenseVector coefficient;

    public LogisticRegressionModelServable() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public DataFrame transform(DataFrame input) {
        List<Double> predictionResults = new ArrayList<>();
        List<DenseVector> rawPredictionResults = new ArrayList<>();

        int featuresColIndex = input.getIndex(getFeaturesCol());
        for (Row row : input.collect()) {
            Vector features = (Vector) row.get(featuresColIndex);
            Tuple2<Double, DenseVector> dataPoint = predictOneDataPoint(features, coefficient);
            predictionResults.add(dataPoint.f0);
            rawPredictionResults.add(dataPoint.f1);
        }

        input.addColumn(getPredictionCol(), DataTypes.DOUBLE, predictionResults);
        input.addColumn(
                getRawPredictionCol(), DataTypes.VECTOR(BasicType.DOUBLE), rawPredictionResults);

        return input;
    }

    public LogisticRegressionModelServable setModelData(InputStream... modelDataInputs)
            throws IOException {
        Preconditions.checkArgument(modelDataInputs.length == 1);

        DataInputViewStreamWrapper inputViewStreamWrapper =
                new DataInputViewStreamWrapper(modelDataInputs[0]);

        DenseVectorSerializer serializer = new DenseVectorSerializer();
        coefficient = serializer.deserialize(inputViewStreamWrapper);

        return this;
    }

    public static LogisticRegressionModelServable load(String path) throws IOException {
        LogisticRegressionModelServable servable =
                ServableReadWriteUtils.loadServableParam(
                        path, LogisticRegressionModelServable.class);

        try (InputStream fsDataInputStream = ServableReadWriteUtils.loadModelData(path)) {
            DataInputViewStreamWrapper dataInputViewStreamWrapper =
                    new DataInputViewStreamWrapper(fsDataInputStream);
            DenseVectorSerializer serializer = new DenseVectorSerializer();
            servable.coefficient = serializer.deserialize(dataInputViewStreamWrapper);
            return servable;
        }
    }

    /**
     * The main logic that predicts one input data point.
     *
     * @param feature The input feature.
     * @param coefficient The model parameters.
     * @return The prediction label and the raw probabilities.
     */
    public static Tuple2<Double, DenseVector> predictOneDataPoint(
            Vector feature, DenseVector coefficient) {
        double dotValue = BLAS.dot(feature, coefficient);
        double prob = 1 - 1.0 / (1.0 + Math.exp(dotValue));
        return Tuple2.of(dotValue >= 0 ? 1. : 0., Vectors.dense(1 - prob, prob));
    }

    /**
     * Serializes the model data into byte array which can be saved to external storage and then be
     * used to update the Servable by `TransformerServable::setModelData` method.
     *
     * @param modelData The model data to be serialized.
     * @return The serialized model data in byte array.
     */
    public static byte[] serialize(DenseVector modelData) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        DataOutputViewStreamWrapper outputViewStreamWrapper =
                new DataOutputViewStreamWrapper(outputStream);

        DenseVectorSerializer serializer = new DenseVectorSerializer();
        serializer.serialize(modelData, outputViewStreamWrapper);

        return outputStream.toByteArray();
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    public DenseVector getCoefficient() {
        return coefficient;
    }
}
