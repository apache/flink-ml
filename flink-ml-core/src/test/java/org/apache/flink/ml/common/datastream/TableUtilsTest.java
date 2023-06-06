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

package org.apache.flink.ml.common.datastream;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.DenseMatrix;
import org.apache.flink.ml.linalg.SparseIntDoubleVector;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.DenseMatrixTypeInfo;
import org.apache.flink.ml.linalg.typeinfo.SparseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.types.AbstractDataType;
import org.apache.flink.table.types.DataType;
import org.apache.flink.types.Row;

import org.apache.commons.lang3.ArrayUtils;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Tests the {@link TableUtils}. */
public class TableUtilsTest {
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);
    }

    @Test
    public void testGetRowTypeInfo() {
        List<AbstractDataType<?>> preDefinedDataTypes = new ArrayList<>();
        List<Object> dataFields = new ArrayList<>();

        preDefinedDataTypes.add(DataTypes.CHAR(10));
        dataFields.add("char");
        preDefinedDataTypes.add(DataTypes.VARCHAR(100));
        dataFields.add("varchar");
        preDefinedDataTypes.add(DataTypes.STRING());
        dataFields.add("string");
        preDefinedDataTypes.add(DataTypes.BOOLEAN());
        dataFields.add(true);
        preDefinedDataTypes.add(DataTypes.BINARY(10));
        dataFields.add(new byte[] {'a', 'b', 'c'});
        preDefinedDataTypes.add(DataTypes.VARBINARY(100));
        dataFields.add(new byte[] {'a', 'b', 'c'});
        preDefinedDataTypes.add(DataTypes.BYTES());
        dataFields.add(new byte[] {'a', 'b', 'c'});
        preDefinedDataTypes.add(DataTypes.DECIMAL(11, 3));
        dataFields.add(new BigDecimal(100));
        preDefinedDataTypes.add(DataTypes.TINYINT());
        dataFields.add((byte) 'a');
        preDefinedDataTypes.add(DataTypes.SMALLINT());
        dataFields.add((short) 1);
        preDefinedDataTypes.add(DataTypes.INT());
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.BIGINT());
        dataFields.add(1L);
        preDefinedDataTypes.add(DataTypes.FLOAT());
        dataFields.add(0.1f);
        preDefinedDataTypes.add(DataTypes.DOUBLE());
        dataFields.add(0.1);
        preDefinedDataTypes.add(DataTypes.DATE());
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.TIME(4));
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.TIMESTAMP(9));
        dataFields.add(new Timestamp(1));
        preDefinedDataTypes.add(DataTypes.TIMESTAMP_LTZ());
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.TIMESTAMP_WITH_LOCAL_TIME_ZONE(9));
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.INTERVAL(DataTypes.SECOND()));
        dataFields.add(1);
        preDefinedDataTypes.add(DataTypes.ARRAY(DataTypes.TIME()));
        dataFields.add(new int[] {1, 2});
        preDefinedDataTypes.add(DataTypes.MAP(DataTypes.INT(), DataTypes.DOUBLE()));
        dataFields.add(Collections.singletonMap(1, 0.1));
        preDefinedDataTypes.add(DataTypes.MULTISET(DataTypes.DOUBLE()));
        dataFields.add(Collections.singletonMap(0.1, 1));
        preDefinedDataTypes.add(DataTypes.ROW(DataTypes.INT(), DataTypes.BIGINT()));
        dataFields.add(Row.of(1, 2L));
        preDefinedDataTypes.add(DataTypes.RAW(DenseIntDoubleVectorTypeInfo.INSTANCE));
        dataFields.add(new DenseIntDoubleVector(new double[] {0.1, 0.2}));
        preDefinedDataTypes.add(DataTypes.RAW(SparseIntDoubleVectorTypeInfo.INSTANCE));
        dataFields.add(new SparseIntDoubleVector(2, new int[] {0}, new double[] {0.1}));
        preDefinedDataTypes.add(DataTypes.RAW(DenseMatrixTypeInfo.INSTANCE));
        dataFields.add(new DenseMatrix(2, 2));
        preDefinedDataTypes.add(
                DataTypes.STRUCTURED(
                        Tuple2.class,
                        DataTypes.FIELD("f0", DataTypes.BIGINT()),
                        DataTypes.FIELD("f1", DataTypes.BIGINT())));
        dataFields.add(Tuple2.of(1L, 2L));

        Schema.Builder builder = Schema.newBuilder();
        for (int i = 0; i < preDefinedDataTypes.size(); i++) {
            builder.column("f" + i, preDefinedDataTypes.get(i));
        }
        Schema schema = builder.build();

        Table inputTable =
                tEnv.fromDataStream(env.fromElements(Row.of(dataFields.toArray())), schema);

        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(inputTable.getResolvedSchema());
        RowTypeInfo outputTypeInfo =
                new RowTypeInfo(
                        ArrayUtils.addAll(inputTypeInfo.getFieldTypes(), Types.INT),
                        ArrayUtils.addAll(inputTypeInfo.getFieldNames(), "outputCol"));

        DataStream<Row> mappedOutput =
                tEnv.toDataStream(inputTable)
                        .map(
                                (MapFunction<Row, Row>) row -> Row.of(row.getField(0), 1),
                                outputTypeInfo);

        List<DataType> inputDataTypes = inputTable.getResolvedSchema().getColumnDataTypes();
        List<DataType> outputDataTypes =
                tEnv.fromDataStream(mappedOutput).getResolvedSchema().getColumnDataTypes();
        Assert.assertEquals(inputDataTypes.size() + 1, outputDataTypes.size());
        for (int i = 0; i < inputDataTypes.size(); i++) {
            Assert.assertEquals(inputDataTypes.get(i), outputDataTypes.get(i));
        }
    }
}
