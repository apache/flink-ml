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
import org.apache.flink.ml.benchmark.datagenerator.param.HasArraySize;
import org.apache.flink.ml.benchmark.datagenerator.param.HasVectorDim;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

/** A DataGenerator which creates a table of DenseVector array. */
public class DenseVectorArrayGenerator extends InputTableGenerator<DenseVectorArrayGenerator>
        implements HasArraySize<DenseVectorArrayGenerator>,
                HasVectorDim<DenseVectorArrayGenerator> {

    @Override
    protected RowGenerator[] getRowGenerators() {
        String[][] columnNames = getColNames();
        Preconditions.checkState(columnNames.length == 1);
        Preconditions.checkState(columnNames[0].length == 1);
        int arraySize = getArraySize();
        int vectorDim = getVectorDim();

        return new RowGenerator[] {
            new RowGenerator(getNumValues(), getSeed()) {
                @Override
                protected Row getRow() {
                    DenseVector[] result = new DenseVector[arraySize];
                    for (int i = 0; i < arraySize; i++) {
                        result[i] = new DenseVector(vectorDim);
                        for (int j = 0; j < vectorDim; j++) {
                            result[i].values[j] = random.nextDouble();
                        }
                    }
                    Row row = new Row(1);
                    row.setField(0, result);
                    return row;
                }

                @Override
                protected RowTypeInfo getRowTypeInfo() {
                    return new RowTypeInfo(
                            new TypeInformation[] {TypeInformation.of(DenseVector[].class)},
                            columnNames[0]);
                }
            }
        };
    }
}
