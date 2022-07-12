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

package org.apache.flink.ml.feature.regextokenizer;

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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.flink.table.api.Expressions.$;

/**
 * A Transformer which converts the input string to lowercase and then splits it by white spaces
 * based on regex. It provides two options to extract tokens:
 *
 * <ul>
 *   <li>if "gaps" is true: uses the provided pattern to split the input string.
 *   <li>else: repeatedly matches the regex (the provided pattern) with the input string.
 * </ul>
 *
 * <p>Moreover, it provides parameters to filter tokens with a minimal length and converts input to
 * lowercase. The output of each input string is an array of strings that can be empty.
 */
public class RegexTokenizer
        implements Transformer<RegexTokenizer>, RegexTokenizerParams<RegexTokenizer> {
    private final Map<Param<?>, Object> paramMap = new HashMap<>();

    public RegexTokenizer() {
        ParamUtils.initializeMapWithDefaultValues(paramMap, this);
    }

    @Override
    public Table[] transform(Table... inputs) {
        Preconditions.checkArgument(inputs.length == 1);
        Expression tokenizerUdf =
                Expressions.call(
                                RegexTokenizerUdf.class,
                                $(getInputCol()),
                                getPattern(),
                                getGaps(),
                                getToLowercase(),
                                getMinTokenLength())
                        .as(getOutputCol());
        Table output = inputs[0].addColumns(tokenizerUdf);
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

    public static RegexTokenizer load(StreamTableEnvironment tEnv, String path) throws IOException {
        return ReadWriteUtils.loadStageParam(path);
    }

    /**
     * The main logic of ${@link RegexTokenizer}, which converts the input string to an array of
     * tokens.
     */
    public static class RegexTokenizerUdf extends ScalarFunction {

        public String[] eval(
                String input,
                String pattern,
                Boolean gaps,
                boolean toLowercase,
                int minTokenLength) {
            Pattern regPattern = Pattern.compile(pattern);
            input = toLowercase ? input.toLowerCase() : input;

            List<String> tokens = new ArrayList<>();
            if (gaps) {
                String[] tokenArray = regPattern.split(input);
                for (String token : tokenArray) {
                    if (token.length() >= minTokenLength) {
                        tokens.add(token);
                    }
                }
            } else {
                Matcher matcher = regPattern.matcher(input);
                while (matcher.find()) {
                    String token = matcher.group();
                    if (token.length() >= minTokenLength) {
                        tokens.add(token);
                    }
                }
            }

            return tokens.toArray(new String[0]);
        }
    }
}
