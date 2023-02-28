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

package org.apache.flink.ml.common.gbt;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.typeutils.RowTypeInfo;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.common.gbt.defs.FeatureMeta;
import org.apache.flink.ml.common.gbt.defs.GbtParams;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizer;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModel;
import org.apache.flink.ml.feature.kbinsdiscretizer.KBinsDiscretizerModelData;
import org.apache.flink.ml.feature.stringindexer.StringIndexer;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModel;
import org.apache.flink.ml.feature.stringindexer.StringIndexerModelData;
import org.apache.flink.ml.linalg.DenseVector;
import org.apache.flink.ml.linalg.Vectors;
import org.apache.flink.ml.linalg.typeinfo.DenseVectorTypeInfo;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.ApiExpression;
import org.apache.flink.table.api.Expressions;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.api.internal.TableImpl;
import org.apache.flink.types.Row;
import org.apache.flink.util.Preconditions;

import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static org.apache.flink.table.api.Expressions.$;

/**
 * Preprocesses input data table for gradient boosting trees algorithms.
 *
 * <p>Multiple non-vector columns or a single vector column can be specified for preprocessing.
 * Values of these column(s) are mapped to integers inplace through discretizer or string indexer,
 * and the meta information of column(s) are obtained.
 */
class Preprocess {

    /**
     * Maps continuous and categorical columns to integers inplace using quantile discretizer and
     * string indexer respectively, and obtains meta information for all columns.
     */
    static Tuple2<Table, DataStream<FeatureMeta>> preprocessCols(Table dataTable, GbtParams p) {

        final String[] relatedCols = ArrayUtils.add(p.featuresCols, p.labelCol);
        dataTable =
                dataTable.select(
                        Arrays.stream(relatedCols)
                                .map(Expressions::$)
                                .toArray(ApiExpression[]::new));

        // Maps continuous columns to integers, and obtain corresponding discretizer model.
        String[] continuousCols = ArrayUtils.removeElements(p.featuresCols, p.categoricalCols);
        Tuple2<Table, DataStream<KBinsDiscretizerModelData>> continuousMappedDataAndModelData =
                discretizeContinuousCols(dataTable, continuousCols, p.maxBins);
        dataTable = continuousMappedDataAndModelData.f0;
        DataStream<FeatureMeta> continuousFeatureMeta =
                buildContinuousFeatureMeta(continuousMappedDataAndModelData.f1, continuousCols);

        // Maps categorical columns to integers, and obtain string indexer model.
        DataStream<FeatureMeta> categoricalFeatureMeta;
        if (p.categoricalCols.length > 0) {
            String[] mappedCategoricalCols =
                    Arrays.stream(p.categoricalCols).map(d -> d + "_output").toArray(String[]::new);
            StringIndexer stringIndexer =
                    new StringIndexer()
                            .setInputCols(p.categoricalCols)
                            .setOutputCols(mappedCategoricalCols)
                            .setHandleInvalid("keep");
            StringIndexerModel stringIndexerModel = stringIndexer.fit(dataTable);
            dataTable = stringIndexerModel.transform(dataTable)[0];

            categoricalFeatureMeta =
                    buildCategoricalFeatureMeta(
                            StringIndexerModelData.getModelDataStream(
                                    stringIndexerModel.getModelData()[0]),
                            p.categoricalCols);
        } else {
            categoricalFeatureMeta =
                    continuousFeatureMeta
                            .<FeatureMeta>flatMap((value, out) -> {})
                            .returns(TypeInformation.of(FeatureMeta.class));
        }

        // Rename results columns.
        ApiExpression[] dropColumnExprs =
                Arrays.stream(p.categoricalCols).map(Expressions::$).toArray(ApiExpression[]::new);
        ApiExpression[] renameColumnExprs =
                Arrays.stream(p.categoricalCols)
                        .map(d -> $(d + "_output").as(d))
                        .toArray(ApiExpression[]::new);
        dataTable = dataTable.dropColumns(dropColumnExprs).renameColumns(renameColumnExprs);

        return Tuple2.of(dataTable, continuousFeatureMeta.union(categoricalFeatureMeta));
    }

    /**
     * Maps features values in vectors to integers using quantile discretizer, and obtains meta
     * information for all features.
     */
    static Tuple2<Table, DataStream<FeatureMeta>> preprocessVecCol(Table dataTable, GbtParams p) {
        dataTable = dataTable.select($(p.featuresCols[0]), $(p.labelCol));
        Tuple2<Table, DataStream<KBinsDiscretizerModelData>> mappedDataAndModelData =
                discretizeVectorCol(dataTable, p.featuresCols[0], p.maxBins);
        dataTable = mappedDataAndModelData.f0;
        DataStream<FeatureMeta> featureMeta =
                buildContinuousFeatureMeta(mappedDataAndModelData.f1, null);
        return Tuple2.of(dataTable, featureMeta);
    }

    /** Builds {@link FeatureMeta} from {@link StringIndexerModelData}. */
    private static DataStream<FeatureMeta> buildCategoricalFeatureMeta(
            DataStream<StringIndexerModelData> stringIndexerModelData, String[] cols) {
        return stringIndexerModelData
                .<FeatureMeta>flatMap(
                        (d, out) -> {
                            Preconditions.checkArgument(d.stringArrays.length == cols.length);
                            for (int i = 0; i < cols.length; i += 1) {
                                out.collect(
                                        FeatureMeta.categorical(
                                                cols[i],
                                                d.stringArrays[i].length,
                                                d.stringArrays[i]));
                            }
                        })
                .returns(TypeInformation.of(FeatureMeta.class));
    }

    /** Builds {@link FeatureMeta} from {@link KBinsDiscretizerModelData}. */
    private static DataStream<FeatureMeta> buildContinuousFeatureMeta(
            DataStream<KBinsDiscretizerModelData> discretizerModelData, String[] cols) {
        return discretizerModelData
                .<FeatureMeta>flatMap(
                        (d, out) -> {
                            double[][] binEdges = d.binEdges;
                            for (int i = 0; i < binEdges.length; i += 1) {
                                String name = (null != cols) ? cols[i] : "_vec_f" + i;
                                out.collect(
                                        FeatureMeta.continuous(
                                                name, binEdges[i].length - 1, binEdges[i]));
                            }
                        })
                .returns(TypeInformation.of(FeatureMeta.class));
    }

    /** Discretizes continuous columns inplace, and obtains quantile discretizer model data. */
    @SuppressWarnings("checkstyle:RegexpSingleline")
    private static Tuple2<Table, DataStream<KBinsDiscretizerModelData>> discretizeContinuousCols(
            Table dataTable, String[] continuousCols, int numBins) {
        final StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) dataTable).getTableEnvironment();
        final int nCols = continuousCols.length;

        // Merges all continuous columns into a vector columns.
        final String vectorCol = "_vec";
        RowTypeInfo inputTypeInfo = TableUtils.getRowTypeInfo(dataTable.getResolvedSchema());
        DataStream<Row> data = tEnv.toDataStream(dataTable, Row.class);
        DataStream<Row> dataWithVectors =
                data.map(
                        (row) -> {
                            double[] values = new double[nCols];
                            for (int i = 0; i < nCols; i += 1) {
                                Number number = row.getFieldAs(continuousCols[i]);
                                // Null values are represented using `Double.NaN` in `DenseVector`.
                                values[i] = (null == number) ? Double.NaN : number.doubleValue();
                            }
                            return Row.join(row, Row.of(Vectors.dense(values)));
                        },
                        new RowTypeInfo(
                                ArrayUtils.add(
                                        inputTypeInfo.getFieldTypes(),
                                        DenseVectorTypeInfo.INSTANCE),
                                ArrayUtils.add(inputTypeInfo.getFieldNames(), vectorCol)));

        Tuple2<Table, DataStream<KBinsDiscretizerModelData>> mappedDataAndModelData =
                discretizeVectorCol(tEnv.fromDataStream(dataWithVectors), vectorCol, numBins);
        DataStream<Row> discretized = tEnv.toDataStream(mappedDataAndModelData.f0);

        // Maps the result vector back to multiple continuous columns.
        final String[] otherCols =
                ArrayUtils.removeElements(inputTypeInfo.getFieldNames(), continuousCols);
        final TypeInformation<?>[] otherColTypes =
                Arrays.stream(otherCols)
                        .map(inputTypeInfo::getTypeAt)
                        .toArray(TypeInformation[]::new);
        final TypeInformation<?>[] mappedColTypes =
                Arrays.stream(continuousCols).map(d -> Types.INT).toArray(TypeInformation[]::new);

        DataStream<Row> mapped =
                discretized.map(
                        (row) -> {
                            DenseVector vec = row.getFieldAs(vectorCol);
                            Integer[] ints =
                                    Arrays.stream(vec.values)
                                            .mapToObj(d -> (Integer) ((int) d))
                                            .toArray(Integer[]::new);
                            Row result = Row.project(row, otherCols);
                            for (int i = 0; i < ints.length; i += 1) {
                                result.setField(continuousCols[i], ints[i]);
                            }
                            return result;
                        },
                        new RowTypeInfo(
                                ArrayUtils.addAll(otherColTypes, mappedColTypes),
                                ArrayUtils.addAll(otherCols, continuousCols)));

        return Tuple2.of(tEnv.fromDataStream(mapped), mappedDataAndModelData.f1);
    }

    /**
     * Discretize the vector column inplace using quantile discretizer, and obtains quantile
     * discretizer model data..
     */
    private static Tuple2<Table, DataStream<KBinsDiscretizerModelData>> discretizeVectorCol(
            Table dataTable, String vectorCol, int numBins) {
        final String outputCol = "_output_col";
        KBinsDiscretizer kBinsDiscretizer =
                new KBinsDiscretizer()
                        .setInputCol(vectorCol)
                        .setOutputCol(outputCol)
                        .setStrategy("quantile")
                        .setNumBins(numBins);
        KBinsDiscretizerModel model = kBinsDiscretizer.fit(dataTable);
        Table discretizedDataTable = model.transform(dataTable)[0];
        return Tuple2.of(
                discretizedDataTable
                        .dropColumns($(vectorCol))
                        .renameColumns($(outputCol).as(vectorCol)),
                KBinsDiscretizerModelData.getModelDataStream(model.getModelData()[0]));
    }
}
