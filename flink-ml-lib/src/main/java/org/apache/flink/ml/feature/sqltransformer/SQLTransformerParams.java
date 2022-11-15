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

package org.apache.flink.ml.feature.sqltransformer;

import org.apache.flink.ml.param.Param;
import org.apache.flink.ml.param.ParamValidator;
import org.apache.flink.ml.param.StringParam;
import org.apache.flink.ml.param.WithParams;

import static org.apache.flink.ml.feature.sqltransformer.SQLTransformer.TABLE_IDENTIFIER;

/**
 * Params for {@link SQLTransformer}.
 *
 * @param <T> The class type of this instance.
 */
public interface SQLTransformerParams<T> extends WithParams<T> {
    Param<String> STATEMENT =
            new StringParam("statement", "SQL statement.", null, new SQLStatementValidator());

    default String getStatement() {
        return get(STATEMENT);
    }

    default T setStatement(String value) {
        return set(STATEMENT, value);
    }

    /** Param validator for SQL statements. */
    class SQLStatementValidator implements ParamValidator<String> {
        @Override
        public boolean validate(String value) {
            return value != null && value.contains(TABLE_IDENTIFIER);
        }
    }
}
