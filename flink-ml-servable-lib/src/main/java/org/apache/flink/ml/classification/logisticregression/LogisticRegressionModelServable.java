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

import org.apache.flink.annotation.VisibleForTesting;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.linalg.BLAS;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.ModelServable;
import org.apache.flink.ml.servable.api.Row;
import org.apache.flink.ml.servable.types.BasicType;
import org.apache.flink.ml.servable.types.DataTypes;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ServableReadWriteUtils;
import org.apache.flink.util.Preconditions;

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

    private LogisticRegressionModelData modelData;

    public LogisticRegressionModelServable() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    LogisticRegressionModelServable(LogisticRegressionModelData modelData) {
        this();
        this.modelData = modelData;
    }

    @Override
    public DataFrame transform(DataFrame input) {
        List<Double> predictionResults = new ArrayList<>();
        List<DenseVector> rawPredictionResults = new ArrayList<>();

        int featuresColIndex = input.getIndex(getFeaturesCol());
        for (Row row : input.collect()) {
            Vector features = (Vector) row.get(featuresColIndex);
            Tuple2<Double, DenseVector> dataPoint = transform(features);
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
        List<LogisticRegressionModelData> modelPieces = new ArrayList<>();
        while (true) {
            try {
                LogisticRegressionModelData piece =
                        LogisticRegressionModelData.decode(modelDataInputs[0]);
                modelPieces.add(piece);
            } catch (IOException e) {
                // Reached the end of model stream.
                break;
            }
        }

        modelData = mergePieces(modelPieces);
        return this;
    }

    @VisibleForTesting
    public static LogisticRegressionModelData mergePieces(
            List<LogisticRegressionModelData> pieces) {
        long dim = 0;
        for (LogisticRegressionModelData piece : pieces) {
            dim = Math.max(dim, piece.endIndex);
        }
        // TODO: Add distributed inference for very large models.
        Preconditions.checkState(
                dim < Integer.MAX_VALUE,
                "The dimension of logistic regression model is larger than INT.MAX. Please consider using distributed inference.");
        int intDim = (int) dim;
        DenseVector mergedCoefficient = new DenseVector(intDim);
        for (LogisticRegressionModelData piece : pieces) {
            int startIndex = (int) piece.startIndex;
            int endIndex = (int) piece.endIndex;
            System.arraycopy(
                    piece.coefficient.values,
                    0,
                    mergedCoefficient.values,
                    startIndex,
                    endIndex - startIndex);
        }
        return new LogisticRegressionModelData(
                mergedCoefficient, 0, mergedCoefficient.size(), pieces.get(0).modelVersion);
    }

    public static LogisticRegressionModelServable load(String path) throws IOException {
        LogisticRegressionModelServable servable =
                ServableReadWriteUtils.loadServableParam(
                        path, LogisticRegressionModelServable.class);

        try (InputStream modelData = ServableReadWriteUtils.loadModelData(path)) {
            servable.setModelData(modelData);
            return servable;
        }
    }

    /**
     * The main logic that predicts one input data point.
     *
     * @param feature The input feature.
     * @return The prediction label and the raw probabilities.
     */
    protected Tuple2<Double, DenseVector> transform(Vector feature) {
        double dotValue = BLAS.dot(feature, modelData.coefficient);
        double prob = 1 - 1.0 / (1.0 + Math.exp(dotValue));
        return Tuple2.of(dotValue >= 0 ? 1. : 0., Vectors.dense(1 - prob, prob));
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
