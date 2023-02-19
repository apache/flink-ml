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

package org.apache.flink.ml.common.metrics;

import org.apache.flink.annotation.Experimental;

/**
 * A collection class for handling metrics in Flink ML.
 *
 * <p>All metrics of Flink ML are registered under group "ml", which is a child group of {@link
 * org.apache.flink.metrics.groups.OperatorMetricGroup}. Metrics related to model data will be
 * registered in the group "ml.model".
 *
 * <p>For example, the timestamp of the current model data will be reported in metric:
 * "{some_parent_groups}.operator.ml.model.timestamp". And the version of the current model data
 * will be reported in metric: "{some_parent_groups}.operator.ml.model.version".
 */
@Experimental
public class MLMetrics {
    public static final String ML_GROUP = "ml";
    public static final String ML_MODEL_GROUP = "model";
    public static final String TIMESTAMP = "timestamp";
    public static final String VERSION = "version";
}
