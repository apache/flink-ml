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

package org.apache.flink.ml.feature.stopwordsremover;

import org.apache.flink.ml.api.Transformer;
import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.util.ParamUtils;
import org.apache.flink.ml.util.ReadWriteUtils;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.table.expressions.Expression;
import org.apache.flink.table.functions.FunctionContext;
import org.apache.flink.table.functions.ScalarFunction;
import org.apache.flink.util.Preconditions;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static org.apache.flink.table.api.Expressions.$;
import static org.apache.flink.table.api.Expressions.call;

/**
 * A feature transformer that filters out stop words from input.
 *
 * <p>Note: null values from input array are preserved unless adding null to stopWords explicitly.
 *
 * @see <a href="http://en.wikipedia.org/wiki/Stop_words">Stop words (Wikipedia)</a>
 */
public class StopWordsRemover
        implements Transformer<StopWordsRemover>, StopWordsRemoverParams<StopWordsRemover> {

    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public StopWordsRemover() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Preconditions.checkArgument(getInputCols().length == getOutputCols().length);

        String[] inputCols = getInputCols();
        String[] outputCols = getOutputCols();

        ScalarFunction function =
                new RemoveStopWordsFunction(
                        new HashSet<>(Arrays.asList(getStopWords())),
                        new Locale(getLocale()),
                        getCaseSensitive());

        Expression[] expressions = new Expression[inputCols.length + 1];
        expressions[0] = $("*");
        for (int i = 0; i < inputCols.length; i++) {
            expressions[i + 1] = call(function, $(inputCols[i])).as(outputCols[i]);
        }

        Table output = inputs[0].select(expressions);

        return new Table[] {output};
    }

    @Override
    public void save(String path) throws IOException {
        ReadWriteUtils.saveMetadata(this, path);
    }

    public static StopWordsRemover load(StreamTableEnvironment env, String path)
            throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    @Override
    public Map<Param<?>, Object> getParamMap() {
        return paramMap;
    }

    /**
     * Loads the default stop words for the given language.
     *
     * <p>Supported languages: danish, dutch, english, finnish, french, german, hungarian, italian,
     * norwegian, portuguese, russian, spanish, swedish, turkish
     *
     * @see <a
     *     href="http://anoncvs.postgresql.org/cvsweb.cgi/pgsql/src/backend/snowball/stopwords/">here</a>
     */
    public static String[] loadDefaultStopWords(String language) {
        return StopWordsRemoverUtils.loadDefaultStopWords(language);
    }

    /**
     * Returns system default locale, or {@link Locale#US} if the default locale is not in available
     * locales in JVM. The locale is returned as a String.
     */
    public static String getDefaultOrUS() {
        return StopWordsRemoverUtils.getDefaultOrUS();
    }

    /**
     * Returns a set of all installed locales. It must contain at least a {@link Locale} instance
     * equal to {@link Locale#US}. The locales are returned as Strings.
     *
     * @return A set of installed locales.
     */
    public static Set<String> getAvailableLocales() {
        return StopWordsRemoverUtils.getAvailableLocales();
    }

    /** A Scalar Function that removes stop words from input string array. */
    public static class RemoveStopWordsFunction extends ScalarFunction {
        private final Set<String> stopWords;
        private final Locale locale;
        private final boolean caseSensitive;
        private transient Predicate<String> predicate;

        public RemoveStopWordsFunction(
                Set<String> stopWords, Locale locale, boolean caseSensitive) {
            this.locale = locale;
            this.caseSensitive = caseSensitive;
            if (caseSensitive) {
                this.stopWords = stopWords;
            } else {
                this.stopWords =
                        stopWords.stream()
                                .map(x -> x == null ? null : x.toLowerCase(locale))
                                .collect(Collectors.toSet());
            }
        }

        @Override
        public void open(FunctionContext context) throws Exception {
            super.open(context);
            if (caseSensitive) {
                predicate = x -> !stopWords.contains(x);
            } else {
                predicate =
                        x -> {
                            if (x != null) {
                                x = x.toLowerCase(locale);
                            }
                            return !stopWords.contains(x);
                        };
            }
        }

        public String[] eval(String[] input) {
            return Arrays.stream(input).filter(predicate).toArray(String[]::new);
        }
    }
}
