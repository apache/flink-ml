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

package org.apache.flink.ml.servable;

import org.apache.flink.ml.servable.api.DataFrame;
import org.apache.flink.ml.servable.api.Row;

import org.junit.Assert;

import java.util.List;

/** Utility methods for tests. */
public class TestUtils {

    /** Asserts that the two dataframes are equivalent. */
    public static void assertDataFrameEquals(DataFrame first, DataFrame second) {

        List<String> firstColNames = first.getColumnNames();
        Assert.assertEquals(first.getColumnNames(), second.getColumnNames());

        List<Row> firstRows = first.collect();
        List<Row> secondRows = second.collect();

        for (int i = 0; i < firstColNames.size(); i++) {
            String colName = firstColNames.get(i);
            Assert.assertEquals(first.getDataType(colName), second.getDataType(colName));

            for (int j = 0; j < firstRows.size(); j++) {
                Assert.assertEquals(firstRows.get(j).get(i), secondRows.get(j).get(i));
            }
        }
    }
}
