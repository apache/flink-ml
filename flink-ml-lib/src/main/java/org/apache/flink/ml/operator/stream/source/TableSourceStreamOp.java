/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.flink.ml.operator.stream.source;

import org.apache.flink.ml.operator.stream.StreamOperator;
import org.apache.flink.table.api.Table;
import org.apache.flink.util.Preconditions;

import java.io.IOException;

/** Utility class to wrap a table as {@link StreamOperator}. */
public final class TableSourceStreamOp extends StreamOperator<TableSourceStreamOp> {

    public TableSourceStreamOp(Table table) {
        super(null);
        Preconditions.checkArgument(table != null, "The source table cannot be null.");
        this.setOutput(table);
    }

    @Override
    public Table[] transform(Table... inputs) {
        throw new UnsupportedOperationException(
                "Table source operator should not have any upstream to link from.");
    }

    @Override
    public void save(String path) throws IOException {
        // TODO: add save table
    }

    public static TableSourceStreamOp load(String path) throws IOException {
        // TODO: add save table
        return null;
    }
}
