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

package org.apache.flink.ml.feature.ngram;

import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Expressions;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.expressions.Expression;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.flink.table.api.Expressions.$;

/**
 * A Transformer that converts the input string array into an array of n-grams, where each n-gram is
 * represented by a space-separated string of words. If the length of the input array is less than
 * `n`, no n-grams are returned.
 *
 * <p>See https://en.wikipedia.org/wiki/N-gram.
 */
public class NGram implements Transformer<NGram>, NGramParams<NGram> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public NGram() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Expression nGramUdf =
                Expressions.call(NGramUdf.class, $(getInputCol()), getN()).as(getOutputCol());
        Table output = inputs[0].addColumns(nGramUdf);
        return new Table[] {output};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    public static NGram load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /**
     * The main logic of {@link NGram}, which converts the input string array to an array of
     * n-grams.
     */
    public static class NGramUdf extends ScalarFunction {

        public String[] eval(String[] items, int n) {
            int numItems = items.length;
            if (n > numItems) {
                return new String[0];
            } else {
                String[] output = new String[numItems - n + 1];
                for (int i = 0; i < numItems - n + 1; i++) {
                    StringBuilder stringBuilder = new StringBuilder();
                    for (int j = 0; j < n; j++) {
                        stringBuilder.append(items[i + j]);
                        stringBuilder.append(" ");
                    }

                    output[i] = stringBuilder.deleteCharAt(stringBuilder.length() - 1).toString();
                }

                return output;
            }
        }
    }
}
