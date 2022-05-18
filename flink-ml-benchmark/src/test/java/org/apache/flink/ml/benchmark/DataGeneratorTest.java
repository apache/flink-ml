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

package org.apache.flink.ml.benchmark;

import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorArrayGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.DenseVectorGenerator;
import org.apache.flink.ml.benchmark.datagenerator.common.LabeledPointWithWeightGenerator;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.types.Row;
import org.apache.flink.util.CloseableIterator;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/** Tests data generators. */
public class DataGeneratorTest {
    @Test
    public void testDenseVectorGenerator() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        DenseVectorGenerator generator =
                new DenseVectorGenerator()
                        .setColNames(new String[] {"denseVector"})
                        .setNumValues(100)
                        .setVectorDim(10);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            assertEquals(row.getArity(), 1);
            DenseVector vector = (DenseVector) row.getField(generator.getColNames()[0][0]);
            assertNotNull(vector);
            assertEquals(vector.size(), generator.getVectorDim());
            count++;
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testDenseVectorArrayGenerator() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        DenseVectorArrayGenerator generator =
                new DenseVectorArrayGenerator()
                        .setColNames(new String[] {"denseVectors"})
                        .setNumValues(100)
                        .setVectorDim(10)
                        .setArraySize(20);

        DataStream<DenseVector[]> stream =
                tEnv.toDataStream(generator.getData(tEnv)[0], DenseVector[].class);

        int count = 0;
        for (CloseableIterator<DenseVector[]> it = stream.executeAndCollect(); it.hasNext(); ) {
            DenseVector[] vectors = it.next();
            assertEquals(generator.getArraySize(), vectors.length);
            for (DenseVector vector : vectors) {
                assertEquals(vector.size(), generator.getVectorDim());
            }
            count++;
        }
        assertEquals(generator.getNumValues(), count);
    }

    @Test
    public void testLabeledPointWithWeightGenerator() {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);

        String featuresCol = "features";
        String labelCol = "label";
        String weightCol = "weight";

        LabeledPointWithWeightGenerator generator =
                new LabeledPointWithWeightGenerator()
                        .setFeatureArity(10)
                        .setLabelArity(10)
                        .setColNames(new String[] {featuresCol, labelCol, weightCol})
                        .setNumValues(100);

        int count = 0;
        for (CloseableIterator<Row> it = generator.getData(tEnv)[0].execute().collect();
                it.hasNext(); ) {
            Row row = it.next();
            count++;
            DenseVector features = (DenseVector) row.getField(featuresCol);
            assertNotNull(features);
            for (double value : features.values) {
                assertTrue(value >= 0);
                assertTrue(value <= generator.getFeatureArity() - 1);
            }

            double label = (double) row.getField(labelCol);
            assertTrue(label >= 0);
            assertTrue(label <= generator.getLabelArity() - 1);

            double weight = (double) row.getField(weightCol);
            assertTrue(weight >= 0);
            assertTrue(weight < 1);
        }
        assertEquals(generator.getNumValues(), count);
    }
}
