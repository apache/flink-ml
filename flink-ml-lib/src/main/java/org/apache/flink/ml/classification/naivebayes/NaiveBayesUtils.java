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

package org.apache.flink.ml.classification.naivebayes;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonProcessingException;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/** Utility class to support NaiveBayes. */
public class NaiveBayesUtils {
    public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final int batchSize = 1024;

    public static class Serializer implements FlatMapFunction<NaiveBayesModelData, Row> {
        @Override
        public void flatMap(NaiveBayesModelData naiveBayesModelData, Collector<Row> collector) {
            int index = 0;
            for (String str: serialize(naiveBayesModelData)) {
                collector.collect(Row.of(index, str));
                index++;
            }
        }
    }

    public static List<String> serialize(Object obj) {
        try {
            return splitString(OBJECT_MAPPER.writeValueAsString(obj));
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public static <T> T deserialize(List<String> list, Class<T> clazz) {
        try {
            return OBJECT_MAPPER.readValue(mergeString(list), clazz);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public static List<String> splitString(String str) {
        List<String> list = new ArrayList<>();
        for (int i = 0; i < str.length(); i += batchSize) {
            list.add(str.substring(i, Math.min(i + batchSize, str.length())));
        }
        return list;
    }

    public static String mergeString(List<String> list) {
        StringBuilder builder = new StringBuilder();
        for (String str: list) {
            builder.append(str);
        }
        return builder.toString();
    }
}
