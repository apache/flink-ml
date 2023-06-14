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

package org.apache.flink.ml.benchmark.datagenerator.common;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.param.IntParam;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidators;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

/**
 * A DataGenerator which creates a table of features, label and weight.
 *
 * <p>Users need to specify three column names as {@link #COL_NAMES}'s value in the following order:
 *
 * <ul>
 *   <li>features column name
 *   <li>label column name
 *   <li>weight column name
 * </ul>
 */
public class LabeledPointWithWeightGenerator
        extends InputTableGenerator<LabeledPointWithWeightGenerator>
        implements HasVectorDim<LabeledPointWithWeightGenerator> {

    public static final Param<Integer> FEATURE_ARITY =
            new IntParam(
                    "featureArity",
                    "Arity of each feature. "
                            + "If set to positive value, each feature would be an integer in range [0, arity - 1]. "
                            + "If set to zero, each feature would be a continuous double in range [0, 1).",
                    2,
                    ParamValidators.gtEq(0));

    public static final Param<Integer> LABEL_ARITY =
            new IntParam(
                    "labelArity",
                    "Arity of label. "
                            + "If set to positive value, the label would be an integer in range [0, arity - 1]. "
                            + "If set to zero, the label would be a continuous double in range [0, 1).",
                    2,
                    ParamValidators.gtEq(0));

    public LabeledPointWithWeightGenerator() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    public int getFeatureArity() {
        return get(FEATURE_ARITY);
    }

    public LabeledPointWithWeightGenerator setFeatureArity(int value) {
        return set(FEATURE_ARITY, value);
    }

    public int getLabelArity() {
        return get(LABEL_ARITY);
    }

    public LabeledPointWithWeightGenerator setLabelArity(int value) {
        return set(LABEL_ARITY, value);
    }

    @Override
    protected RowGenerator[] getRowGenerators() {
        String[][] colNames = getColNames();
        Preconditions.checkState(colNames.length == 1);
        Preconditions.checkState(colNames[0].length == 3);
        int vectorDim = getVectorDim();
        int featureArity = getFeatureArity();
        int labelArity = getLabelArity();

        return new RowGenerator[] {
            new RowGenerator(getNumValues(), getSeed()) {
                @Override
                protected Row getRow() {
                    double[] features = new double[vectorDim];
                    for (int i = 0; i < vectorDim; i++) {
                        features[i] = getValue(featureArity);
                    }

                    double label = getValue(labelArity);

                    double weight = random.nextDouble();

                    return Row.of(Vectors.dense(features), label, weight);
                }

                @Override
                protected RowTypeInfo getRowTypeInfo() {
                    return new RowTypeInfo(
                            new TypeInformation[] {
                                DenseIntDoubleVectorTypeInfo.INSTANCE, Types.DOUBLE, Types.DOUBLE
                            },
                            colNames[0]);
                }

                private double getValue(int arity) {
                    if (arity > 0) {
                        return random.nextInt(arity);
                    }
                    return random.nextDouble();
                }
            }
        };
    }
}
