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

package org.apache.flink.ml.clustering;

import org.apache.flink.ml.clustering.kmeans.KMeans;
import org.apache.flink.ml.clustering.kmeans.KMeansModel;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;

import org.junit.Test;

/** Tests KMeans and KMeansModel. */
public class KMeansTest extends AbstractTestBase {

    @Test
    public void testKMeansDataStream() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tEnv = StreamTableEnvironment.create(env);
        env.setParallelism(4);

        DataStream<DenseVector> input = KMeansData.getDefaultData(env);

        Table points =
                tEnv.fromDataStream(
                        input,
                        Schema.newBuilder().column("f0", DataTypes.of(DenseVector.class)).build());

        KMeans kmeans = new KMeans().setMaxIter(1);

        KMeansModel model = kmeans.fit(points);

        //        Table pointsWithCentroidId =
        //                model.transform(points)[0];
        //
        //        DataStream<Tuple2> result = tEnv.toDataStream(pointsWithCentroidId, Tuple2.class);
        //
        //        List<Tuple2> values = IteratorUtils.toList(result.executeAndCollect());
        //        for (Tuple2<Integer, DenseVector> value : values) {
        //            System.out.println("Value " + value.f0 + " " + value.f1);
        //        }
    }

    //    @Test
    //    public void testKMeansDataset() throws Exception {
    //        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
    //        DataSet<DenseVector> points = KMeansData.getDefaultData(env);
    //
    //        KMeans kmeans = new KMeans().setMaxIter(1);
    //
    //        KMeansModel model = kmeans.fitDataSet(points);
    //
    //        DataSet<Tuple2<Integer, DenseVector>> pointsWithCentroidId =
    //                model.transformDataSet(points)[0];
    //
    //        List<Tuple2<Integer, DenseVector>> values = pointsWithCentroidId.collect();
    //        for (Tuple2<Integer, DenseVector> value : values) {
    //            System.out.println("Value " + value.f0 + " " + value.f1);
    //        }
    //    }
}
