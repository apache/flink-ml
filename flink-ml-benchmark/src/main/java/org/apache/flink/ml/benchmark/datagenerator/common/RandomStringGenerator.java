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
import org.apache.flink.ml.benchmark.datagenerator.param.HasNumDistinctValues;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import java.util.Arrays;

/** A DataGenerator which creates a table of random strings. */
public class RandomStringGenerator extends InputTableGenerator<RandomStringGenerator>
        implements HasNumDistinctValues<RandomStringGenerator> {

    @Override
    protected RowGenerator[] getRowGenerators() {
        String[][] colNames = getColNames();
        Preconditions.checkState(colNames.length == 1);
        int numOutputCols = colNames[0].length;
        int numDistinctValues = getNumDistinctValues();

        return new RowGenerator[] {
            new RowGenerator(getNumValues(), getSeed()) {
                @Override
                public Row getRow() {
                    Row r = new Row(numOutputCols);
                    for (int i = 0; i < numOutputCols; i++) {
                        r.setField(i, Integer.toString(random.nextInt(numDistinctValues)));
                    }
                    return r;
                }

                @Override
                protected RowTypeInfo getRowTypeInfo() {
                    TypeInformation[] outputTypes = new TypeInformation[colNames[0].length];
                    Arrays.fill(outputTypes, Types.STRING);
                    return new RowTypeInfo(outputTypes, colNames[0]);
                }
            }
        };
    }
}
