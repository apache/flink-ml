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

package org.apache.flink.ml.feature.robustscaler;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
import org.apache.flink.ml.common.util.QuantileSummary;
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

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An Estimator which scales features using statistics that are robust to outliers.
 *
 * <p>This Scaler removes the median and scales the data according to the quantile range (defaults
 * to IQR: Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and
 * the 3rd quartile (75th quantile) but can be configured.
 *
 * <p>Centering and scaling happen independently on each feature by computing the relevant
 * statistics on the samples in the training set. Median and quantile range are then stored to be
 * used on later data using the transform method.
 *
 * <p>Standardization of a dataset is a common requirement for many machine learning estimators.
 * Typically this is done by removing the mean and scaling to unit variance. However, outliers can
 * often influence the sample mean / variance in a negative way. In such cases, the median and the
 * interquartile range often give better results.
 *
 * <p>Note that NaN values are ignored in the computation of medians and ranges.
 */
public class RobustScaler
        implements Estimator<RobustScaler, RobustScalerModel>, RobustScalerParams<RobustScaler> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public RobustScaler() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public RobustScalerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        final String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<DenseIntDoubleVector> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, DenseIntDoubleVector>)
                                        value ->
                                                ((IntDoubleVector) value.getField(inputCol))
                                                        .toDense());
        DataStream<RobustScalerModelData> modelData =
                DataStreamUtils.aggregate(
                        inputData,
                        new QuantileAggregator(getRelativeError(), getLower(), getUpper()));
        RobustScalerModel model =
                new RobustScalerModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * Computes the medians and quantile ranges of input column and builds the {@link
     * RobustScalerModelData}.
     */
    private static class QuantileAggregator
            implements AggregateFunction<
                    DenseIntDoubleVector, QuantileSummary[], RobustScalerModelData> {

        private final double relativeError;
        private final double lower;
        private final double upper;

        public QuantileAggregator(double relativeError, double lower, double upper) {
            this.relativeError = relativeError;
            this.lower = lower;
            this.upper = upper;
        }

        @Override
        public QuantileSummary[] createAccumulator() {
            return new QuantileSummary[0];
        }

        @Override
        public QuantileSummary[] add(
                DenseIntDoubleVector denseVector, QuantileSummary[] quantileSummaries) {
            if (quantileSummaries.length == 0) {
                quantileSummaries = new QuantileSummary[denseVector.size()];
                for (int i = 0; i < denseVector.size(); i++) {
                    quantileSummaries[i] = new QuantileSummary(relativeError);
                }
            }
            Preconditions.checkState(
                    denseVector.size() == quantileSummaries.length,
                    "Number of features must be %s but got %s.",
                    quantileSummaries.length,
                    denseVector.size());

            for (int i = 0; i < quantileSummaries.length; i++) {
                double value = denseVector.get(i);
                if (!Double.isNaN(value)) {
                    quantileSummaries[i] = quantileSummaries[i].insert(value);
                }
            }
            return quantileSummaries;
        }

        @Override
        public RobustScalerModelData getResult(QuantileSummary[] quantileSummaries) {
            Preconditions.checkState(quantileSummaries.length != 0, "The training set is empty.");
            DenseIntDoubleVector medianVector = new DenseIntDoubleVector(quantileSummaries.length);
            DenseIntDoubleVector rangeVector = new DenseIntDoubleVector(quantileSummaries.length);

            for (int i = 0; i < quantileSummaries.length; i++) {
                QuantileSummary compressed = quantileSummaries[i].compress();

                double[] quantiles = compressed.query(new double[] {0.5, lower, upper});
                medianVector.values[i] = quantiles[0];
                rangeVector.values[i] = quantiles[2] - quantiles[1];
            }
            return new RobustScalerModelData(medianVector, rangeVector);
        }

        @Override
        public QuantileSummary[] merge(QuantileSummary[] summaries, QuantileSummary[] acc) {
            if (summaries.length == 0) {
                return Arrays.stream(acc)
                        .map(QuantileSummary::compress)
                        .collect(Collectors.toList())
                        .toArray(acc);
            }
            if (acc.length == 0) {
                return Arrays.stream(summaries)
                        .map(QuantileSummary::compress)
                        .collect(Collectors.toList())
                        .toArray(summaries);
            }
            Preconditions.checkState(summaries.length == acc.length);

            for (int i = 0; i < summaries.length; i++) {
                acc[i] = acc[i].compress().merge(summaries[i].compress());
            }
            return acc;
        }
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static RobustScaler load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }
}
