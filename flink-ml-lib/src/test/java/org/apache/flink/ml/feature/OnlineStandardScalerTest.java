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

package org.apache.flink.ml.feature;

import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.window.CountTumblingWindows;
import org.apache.flink.ml.common.window.EventTimeTumblingWindows;
import org.apache.flink.ml.common.window.GlobalWindows;
import org.apache.flink.ml.common.window.ProcessingTimeTumblingWindows;
import org.apache.flink.ml.feature.standardscaler.OnlineStandardScaler;
import org.apache.flink.ml.feature.standardscaler.OnlineStandardScalerModel;
import org.apache.flink.ml.feature.standardscaler.StandardScalerModelData;
import org.apache.flink.ml.linalg.DenseIntDoubleVector;
import org.apache.flink.ml.linalg.IntDoubleVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseIntDoubleVectorTypeInfo;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.TestUtils;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.DataTypes;
import org.apache.flink.table.api.Schema;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.test.util.AbstractTestBase;
import org.apache.flink.types.Row;

import org.apache.commons.collections.IteratorUtils;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/** Tests {@link OnlineStandardScaler} and {@link OnlineStandardScalerModel}. */
public class OnlineStandardScalerTest extends AbstractTestBase {
    @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
    private StreamExecutionEnvironment env;
    private StreamTableEnvironment tEnv;
    private final List<Row> inputData =
            Arrays.asList(
                    Row.of(0L, Vectors.dense(-2.5, 9, 1)),
                    Row.of(1000L, Vectors.dense(1.4, -5, 1)),
                    Row.of(2000L, Vectors.dense(2, -1, -2)),
                    Row.of(6000L, Vectors.dense(0.7, 3, 1)),
                    Row.of(7000L, Vectors.dense(0, 1, 1)),
                    Row.of(8000L, Vectors.dense(0.5, 0, -2)),
                    Row.of(9000L, Vectors.dense(0.4, 1, 1)),
                    Row.of(10000L, Vectors.dense(0.3, 2, 1)),
                    Row.of(11000L, Vectors.dense(0.5, 1, -2)));

    private final List<StandardScalerModelData> expectedModelData =
            Arrays.asList(
                    new StandardScalerModelData(
                            Vectors.dense(0.3, 1, 0),
                            Vectors.dense(2.4433583, 7.2111026, 1.7320508),
                            0L,
                            2999L),
                    new StandardScalerModelData(
                            Vectors.dense(0.35, 1.1666667, 0),
                            Vectors.dense(1.5630099, 4.6654760, 1.5491933),
                            1L,
                            8999L),
                    new StandardScalerModelData(
                            Vectors.dense(0.3666667, 1.2222222, 0),
                            Vectors.dense(1.2369316, 3.7006005, 1.5),
                            2L,
                            11999L));

    private static final double TOLERANCE = 1e-7;

    private Table inputTable;

    private Table inputTableWithProcessingTime;

    private Table inputTableWithEventTime;

    @Before
    public void before() {
        env = TestUtils.getExecutionEnvironment();
        tEnv = StreamTableEnvironment.create(env);

        DataStream<Row> inputStream = env.fromCollection(inputData);
        inputTable =
                tEnv.fromDataStream(
                                inputStream,
                                Schema.newBuilder()
                                        .column("f0", DataTypes.BIGINT())
                                        .column(
                                                "f1",
                                                DataTypes.RAW(
                                                        DenseIntDoubleVectorTypeInfo.INSTANCE))
                                        .build())
                        .as("id", "input");

        DataStream<Row> inputStreamWithProcessingTimeGap =
                inputStream
                        .map(
                                new MapFunction<Row, Row>() {
                                    private int count = 0;

                                    @Override
                                    public Row map(Row value) throws Exception {
                                        count++;
                                        if (count % 3 == 0) {
                                            Thread.sleep(1000);
                                        }
                                        return value;
                                    }
                                },
                                new RowTypeInfo(
                                        new TypeInformation[] {
                                            Types.LONG, DenseIntDoubleVectorTypeInfo.INSTANCE
                                        },
                                        new String[] {"id", "input"}))
                        .setParallelism(1);

        inputTableWithProcessingTime = tEnv.fromDataStream(inputStreamWithProcessingTimeGap);

        DataStream<Row> inputStreamWithEventTime =
                inputStream.assignTimestampsAndWatermarks(
                        WatermarkStrategy.<Row>forMonotonousTimestamps()
                                .withTimestampAssigner(
                                        (SerializableTimestampAssigner<Row>)
                                                (element, recordTimestamp) ->
                                                        element.getFieldAs(0)));
        inputTableWithEventTime =
                tEnv.fromDataStream(
                                inputStreamWithEventTime,
                                Schema.newBuilder()
                                        .column("f0", DataTypes.BIGINT())
                                        .column(
                                                "f1",
                                                DataTypes.RAW(
                                                        DenseIntDoubleVectorTypeInfo.INSTANCE))
                                        .columnByMetadata("rowtime", "TIMESTAMP_LTZ(3)")
                                        .watermark("rowtime", "SOURCE_WATERMARK()")
                                        .build())
                        .as("id", "input");
    }

    @Test
    public void testParam() {
        OnlineStandardScaler standardScaler = new OnlineStandardScaler();

        assertEquals("input", standardScaler.getInputCol());
        assertEquals(false, standardScaler.getWithMean());
        assertEquals(true, standardScaler.getWithStd());
        assertEquals("output", standardScaler.getOutputCol());
        assertEquals("version", standardScaler.getModelVersionCol());
        assertEquals(GlobalWindows.getInstance(), standardScaler.getWindows());
        assertEquals(0L, standardScaler.getMaxAllowedModelDelayMs());

        standardScaler
                .setInputCol("test_input")
                .setWithMean(true)
                .setWithStd(false)
                .setOutputCol("test_output")
                .setModelVersionCol("test_version")
                .setWindows(EventTimeTumblingWindows.of(Time.milliseconds(3000)))
                .setMaxAllowedModelDelayMs(3000L);

        assertEquals("test_input", standardScaler.getInputCol());
        assertEquals(true, standardScaler.getWithMean());
        assertEquals(false, standardScaler.getWithStd());
        assertEquals("test_output", standardScaler.getOutputCol());
        assertEquals("test_version", standardScaler.getModelVersionCol());
        assertEquals(
                EventTimeTumblingWindows.of(Time.milliseconds(3000)), standardScaler.getWindows());
        assertEquals(3000L, standardScaler.getMaxAllowedModelDelayMs());
    }

    @Test
    public void testOutputSchema() {
        Table renamedTable = inputTable.as("test_id", "test_input");

        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler().setInputCol("test_input").setOutputCol("test_output");
        Table output = standardScaler.fit(renamedTable).transform(renamedTable)[0];

        assertEquals(
                Arrays.asList("test_id", "test_input", "test_output", "version"),
                output.getResolvedSchema().getColumnNames());

        // Tests the case when modelVersionCol is null.
        standardScaler.setModelVersionCol(null);
        output = standardScaler.fit(renamedTable).transform(renamedTable)[0];

        assertEquals(
                Arrays.asList("test_id", "test_input", "test_output"),
                output.getResolvedSchema().getColumnNames());
    }

    @Test
    public void testFitAndPredictWithEventTimeWindow() throws Exception {
        OnlineStandardScaler standardScaler = new OnlineStandardScaler();
        Table output;
        int windowSizeMs = 3000;

        // Tests event time window with maxAllowedModelDelayMs as 0.
        standardScaler.setWindows(EventTimeTumblingWindows.of(Time.milliseconds(windowSizeMs)));
        output = standardScaler.fit(inputTableWithEventTime).transform(inputTableWithEventTime)[0];
        verifyUsedModelVersion(
                output,
                standardScaler.getModelVersionCol(),
                standardScaler.getMaxAllowedModelDelayMs());

        // Tests event time window with maxAllowedModelDelayMs as window size.
        standardScaler.setMaxAllowedModelDelayMs(windowSizeMs);
        output = standardScaler.fit(inputTableWithEventTime).transform(inputTableWithEventTime)[0];
        verifyUsedModelVersion(
                output,
                standardScaler.getModelVersionCol(),
                standardScaler.getMaxAllowedModelDelayMs());
    }

    @Test
    public void testFitAndPredictWithProcessingTimeWindow() throws Exception {
        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler()
                        .setWindows(ProcessingTimeTumblingWindows.of(Time.milliseconds(1000)));
        OnlineStandardScalerModel model = standardScaler.fit(inputTableWithProcessingTime);

        DataStream<StandardScalerModelData> modelData =
                StandardScalerModelData.getModelDataStream(model.getModelData()[0]);
        List<StandardScalerModelData> collectedModelData =
                IteratorUtils.toList(modelData.executeAndCollect());
        assertTrue(collectedModelData.size() >= 1);

        Table output = model.transform(inputTableWithProcessingTime)[0];
        List<Row> predictedRows =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        assertEquals(inputData.size(), predictedRows.size());
    }

    @Test
    public void testFitAndPredictWithCountWindow() throws Exception {
        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler().setWindows(CountTumblingWindows.of(3L));

        OnlineStandardScalerModel standardScalerModel = standardScaler.fit(inputTable);

        List<StandardScalerModelData> collectedModelData =
                IteratorUtils.toList(
                        StandardScalerModelData.getModelDataStream(
                                        standardScalerModel.getModelData()[0])
                                .executeAndCollect());

        assertEquals(expectedModelData.size(), collectedModelData.size());
        for (int i = 0; i < expectedModelData.size(); i++) {
            verifyModelData(expectedModelData.get(i), collectedModelData.get(i), false);
        }

        Table output = standardScalerModel.transform(inputTableWithEventTime)[0];
        List<Row> predictions = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        assertEquals(inputData.size(), predictions.size());
    }

    @Test
    public void testFitAndPredictWithGlobalWindow() throws Exception {
        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler().setWindows(GlobalWindows.getInstance());
        Table output;

        Table input =
                tEnv.fromDataStream(
                                env.fromCollection(
                                        Arrays.asList(
                                                Row.of(Vectors.dense(-2.5, 9, 1)),
                                                Row.of(Vectors.dense(1.4, -5, 1)),
                                                Row.of(Vectors.dense(2, -1, -2)))))
                        .as("input");

        // Tests withMean option.
        List<DenseIntDoubleVector> expectedResWithMean =
                Arrays.asList(
                        Vectors.dense(-2.8, 8, 1),
                        Vectors.dense(1.1, -6, 1),
                        Vectors.dense(1.7, -2, -2));
        output = standardScaler.setWithMean(true).setWithStd(false).fit(input).transform(input)[0];
        verifyPredictionResult(expectedResWithMean, output, standardScaler.getOutputCol());

        // Tests withStd option.
        List<DenseIntDoubleVector> expectedResWithStd =
                Arrays.asList(
                        Vectors.dense(-1.0231819, 1.2480754, 0.5773502),
                        Vectors.dense(0.5729819, -0.6933752, 0.5773503),
                        Vectors.dense(0.8185455, -0.1386750, -1.1547005));
        output = standardScaler.setWithMean(false).setWithStd(true).fit(input).transform(input)[0];
        verifyPredictionResult(expectedResWithStd, output, standardScaler.getOutputCol());

        // Tests withMean, withStd Option.
        List<DenseIntDoubleVector> expectedResWithMeanAndStd =
                Arrays.asList(
                        Vectors.dense(-1.1459637, 1.1094004, 0.5773503),
                        Vectors.dense(0.45020003, -0.8320503, 0.5773503),
                        Vectors.dense(0.69576368, -0.2773501, -1.1547005));
        output = standardScaler.setWithMean(true).setWithStd(true).fit(input).transform(input)[0];
        verifyPredictionResult(expectedResWithMeanAndStd, output, standardScaler.getOutputCol());
    }

    @Test
    @SuppressWarnings("unchecked")
    public void testGetModelData() throws Exception {
        OnlineStandardScaler standardScaler = new OnlineStandardScaler();
        OnlineStandardScalerModel model =
                standardScaler
                        .setWindows(EventTimeTumblingWindows.of(Time.milliseconds(3000)))
                        .fit(inputTableWithEventTime);
        Table modelDataTable = model.getModelData()[0];

        assertEquals(
                Arrays.asList("mean", "std", "version", "timestamp"),
                modelDataTable.getResolvedSchema().getColumnNames());

        List<StandardScalerModelData> collectedModelData =
                (List<StandardScalerModelData>)
                        IteratorUtils.toList(
                                StandardScalerModelData.getModelDataStream(modelDataTable)
                                        .executeAndCollect());

        assertEquals(expectedModelData.size(), collectedModelData.size());
        for (int i = 0; i < expectedModelData.size(); i++) {
            verifyModelData(expectedModelData.get(i), collectedModelData.get(i), true);
        }
    }

    @Test
    public void testSetModelData() throws Exception {
        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler()
                        .setWindows(EventTimeTumblingWindows.of(Time.milliseconds(3000L)));
        OnlineStandardScalerModel model = standardScaler.fit(inputTableWithEventTime);

        OnlineStandardScalerModel newModel = new OnlineStandardScalerModel();
        ParamUtils.updateExistingParams(newModel, model.getParamMap());
        newModel.setModelData(model.getModelData());
        Table output = newModel.transform(inputTableWithEventTime)[0];

        verifyUsedModelVersion(
                output,
                standardScaler.getModelVersionCol(),
                standardScaler.getMaxAllowedModelDelayMs());
    }

    @Test
    public void testSaveLoadPredict() throws Exception {
        OnlineStandardScaler standardScaler =
                new OnlineStandardScaler()
                        .setWindows(EventTimeTumblingWindows.of(Time.milliseconds(3000L)));

        standardScaler =
                TestUtils.saveAndReload(
                        tEnv,
                        standardScaler,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        OnlineStandardScaler::load);
        OnlineStandardScalerModel model = standardScaler.fit(inputTableWithEventTime);
        Table[] modelData = model.getModelData();

        model =
                TestUtils.saveAndReload(
                        tEnv,
                        model,
                        TEMPORARY_FOLDER.newFolder().getAbsolutePath(),
                        OnlineStandardScalerModel::load);
        model.setModelData(modelData);

        Table output = model.transform(inputTableWithEventTime)[0];

        verifyUsedModelVersion(
                output,
                standardScaler.getModelVersionCol(),
                standardScaler.getMaxAllowedModelDelayMs());
    }

    private static void verifyModelData(
            StandardScalerModelData expected,
            StandardScalerModelData actual,
            boolean checkTimeStamp) {
        assertArrayEquals(expected.mean.values, actual.mean.values, TOLERANCE);
        assertArrayEquals(expected.std.values, actual.std.values, TOLERANCE);
        assertEquals(expected.version, actual.version);
        if (checkTimeStamp) {
            assertEquals(expected.timestamp, actual.timestamp);
        }
    }

    private void verifyUsedModelVersion(
            Table output, String modelVersionCol, long maxAllowedModelDelayMs) throws Exception {
        List<Row> predictions = IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());

        for (Row row : predictions) {
            long dataTimestamp = row.getFieldAs(0);
            long modelVersion = row.getFieldAs(modelVersionCol);
            long modelTimeStamp = expectedModelData.get((int) modelVersion).timestamp;
            assertTrue(dataTimestamp - modelTimeStamp <= maxAllowedModelDelayMs);
        }
    }

    @SuppressWarnings("unchecked")
    private void verifyPredictionResult(
            List<DenseIntDoubleVector> expectedOutput, Table output, String predictionCol)
            throws Exception {
        List<Row> collectedResult =
                IteratorUtils.toList(tEnv.toDataStream(output).executeAndCollect());
        List<DenseIntDoubleVector> predictions = new ArrayList<>(collectedResult.size());

        for (Row r : collectedResult) {
            IntDoubleVector vec = (IntDoubleVector) r.getField(predictionCol);
            predictions.add(vec.toDense());
        }

        assertEquals(expectedOutput.size(), predictions.size());

        predictions.sort(TestUtils::compare);

        for (int i = 0; i < predictions.size(); i++) {
            assertArrayEquals(expectedOutput.get(i).values, predictions.get(i).values, TOLERANCE);
        }
    }
}
