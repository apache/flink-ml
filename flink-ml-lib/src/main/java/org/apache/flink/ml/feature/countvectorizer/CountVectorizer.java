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

package org.apache.flink.ml.feature.countvectorizer;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.Estimator;
import org.apache.flink.ml.common.datastream.DataStreamUtils;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * An Estimator which converts a collection of text documents to vectors of token counts. When an
 * a-priori dictionary is not available, CountVectorizer can be used as an estimator to extract the
 * vocabulary, and generates a {@link CountVectorizerModel}. The model produces sparse
 * representations for the documents over the vocabulary, which can then be passed to other
 * algorithms like LDA.
 */
public class CountVectorizer
        implements Estimator<CountVectorizer, CountVectorizerModel>,
                CountVectorizerParams<CountVectorizer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public CountVectorizer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public CountVectorizerModel fit(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        double minDF = getMinDF();
        double maxDF = getMaxDF();
        if (minDF >= 1.0 && maxDF >= 1.0 || minDF < 1.0 && maxDF < 1.0) {
            Preconditions.checkArgument(maxDF >= minDF, "maxDF must be >= minDF.");
        }

        String inputCol = getInputCol();
        StreamTableEnvironment tEnv =
                (StreamTableEnvironment) ((TableImpl) inputs[0]).getTableEnvironment();
        DataStream<String[]> inputData =
                tEnv.toDataStream(inputs[0])
                        .map(
                                (MapFunction<Row, String[]>)
                                        value -> ((String[]) value.getField(inputCol)));

        DataStream<CountVectorizerModelData> modelData =
                DataStreamUtils.aggregate(
                        inputData,
                        new VocabularyAggregator(getMinDF(), getMaxDF(), getVocabularySize()),
                        Types.TUPLE(
                                Types.LONG,
                                Types.MAP(Types.STRING, Types.TUPLE(Types.LONG, Types.LONG))),
                        TypeInformation.of(CountVectorizerModelData.class));

        CountVectorizerModel model =
                new CountVectorizerModel().setModelData(tEnv.fromDataStream(modelData));
        ParamUtils.updateExistingParams(model, getParamMap());
        return model;
    }

    /**
     * Extracts a vocabulary from input document collections and builds the {@link
     * CountVectorizerModelData}.
     */
    private static class VocabularyAggregator
            implements AggregateFunction<
                    String[],
                    Tuple2<Long, Map<String, Tuple2<Long, Long>>>,
                    CountVectorizerModelData> {
        private final double minDF;
        private final double maxDF;
        private final int vocabularySize;

        public VocabularyAggregator(double minDF, double maxDF, int vocabularySize) {
            this.minDF = minDF;
            this.maxDF = maxDF;
            this.vocabularySize = vocabularySize;
        }

        @Override
        public Tuple2<Long, Map<String, Tuple2<Long, Long>>> createAccumulator() {
            return Tuple2.of(0L, new HashMap<>());
        }

        @Override
        public Tuple2<Long, Map<String, Tuple2<Long, Long>>> add(
                String[] terms, Tuple2<Long, Map<String, Tuple2<Long, Long>>> vocabAccumulator) {
            Map<String, Long> wc = new HashMap<>();
            Arrays.stream(terms)
                    .forEach(
                            x -> {
                                if (wc.containsKey(x)) {
                                    wc.put(x, wc.get(x) + 1);
                                } else {
                                    wc.put(x, 1L);
                                }
                            });

            Map<String, Tuple2<Long, Long>> counts = vocabAccumulator.f1;
            wc.forEach(
                    (w, c) -> {
                        if (counts.containsKey(w)) {
                            counts.get(w).f0 += c;
                            counts.get(w).f1 += 1;
                        } else {
                            counts.put(w, Tuple2.of(c, 1L));
                        }
                    });
            vocabAccumulator.f0 += 1;

            return vocabAccumulator;
        }

        @Override
        public CountVectorizerModelData getResult(
                Tuple2<Long, Map<String, Tuple2<Long, Long>>> vocabAccumulator) {
            Preconditions.checkState(vocabAccumulator.f0 > 0, "The training set is empty.");

            boolean filteringRequired =
                    !MIN_DF.defaultValue.equals(minDF) || !MAX_DF.defaultValue.equals(maxDF);
            if (filteringRequired) {
                long rowNum = vocabAccumulator.f0;
                double actualMinDF = minDF >= 1.0 ? minDF : minDF * rowNum;
                double actualMaxDF = maxDF >= 1.0 ? maxDF : maxDF * rowNum;
                Preconditions.checkState(actualMaxDF >= actualMinDF, "maxDF must be >= minDF.");

                vocabAccumulator.f1 =
                        vocabAccumulator.f1.entrySet().stream()
                                .filter(
                                        x ->
                                                x.getValue().f1 >= actualMinDF
                                                        && x.getValue().f1 <= actualMaxDF)
                                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            }

            List<Map.Entry<String, Tuple2<Long, Long>>> list =
                    new ArrayList<>(vocabAccumulator.f1.entrySet());
            list.sort((o1, o2) -> (o2.getValue().f1.compareTo(o1.getValue().f1)));
            List<String> vocabulary =
                    list.stream().map(Map.Entry::getKey).collect(Collectors.toList());
            String[] topTerms =
                    vocabulary
                            .subList(0, Math.min(vocabulary.size(), vocabularySize))
                            .toArray(new String[0]);
            return new CountVectorizerModelData(topTerms);
        }

        @Override
        public Tuple2<Long, Map<String, Tuple2<Long, Long>>> merge(
                Tuple2<Long, Map<String, Tuple2<Long, Long>>> acc1,
                Tuple2<Long, Map<String, Tuple2<Long, Long>>> acc2) {
            if (acc1.f0 == 0) {
                return acc2;
            }

            if (acc2.f0 == 0) {
                return acc1;
            }
            acc2.f0 += acc1.f0;
            acc1.f1.forEach(
                    (term, counts) -> {
                        if (acc2.f1.containsKey(term)) {
                            acc2.f1.put(
                                    term,
                                    Tuple2.of(
                                            counts.f0 + acc2.f1.get(term).f0,
                                            counts.f1 + acc2.f1.get(term).f1));
                        } else {
                            acc2.f1.put(term, counts);
                        }
                    });
            return acc2;
        }
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static CountVectorizer load(StreamTableEnvironment tEnv, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }
}
