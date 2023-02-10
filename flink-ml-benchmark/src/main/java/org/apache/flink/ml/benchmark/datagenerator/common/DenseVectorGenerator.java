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
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

/** A DataGenerator which creates a table of DenseVector. */
public class DenseVectorGenerator extends InputTableGenerator<DenseVectorGenerator>
        implements HasVectorDim<DenseVectorGenerator> {

    @Override
    public RowGenerator[] getRowGenerators() {
        String[][] columnNames = getColNames();
        Preconditions.checkState(columnNames.length == 1);
        Preconditions.checkState(columnNames[0].length == 1);
        int vectorDim = getVectorDim();

        return new RowGenerator[] {
            new RowGenerator(getNumValues(), getSeed()) {

                @Override
                protected Row getRow() {
                    double[] values = new double[vectorDim];
                    for (int i = 0; i < values.length; i++) {
                        values[i] = random.nextDouble();
                    }
                    return Row.of(Vectors.dense(values));
                }

                @Override
                protected RowTypeInfo getRowTypeInfo() {
                    return new RowTypeInfo(
                            new TypeInformation[] {DenseVectorTypeInfo.INSTANCE}, columnNames[0]);
                }
            }
        };
    }
}
