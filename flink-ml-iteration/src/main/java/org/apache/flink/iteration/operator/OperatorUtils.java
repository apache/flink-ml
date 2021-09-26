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

package org.apache.flink.iteration.operator;

import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.utils.ReflectionUtils;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackConsumer;
import org.apache.flink.statefun.flink.core.feedback.FeedbackKey;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;

import java.util.Arrays;
import java.util.concurrent.Executor;
import java.util.function.Consumer;

/** Utility class for operators. */
public class OperatorUtils {

    /** Returns the unique id for the specified operator. */
    public static String getUniqueSenderId(OperatorID operatorId, int subtaskIndex) {
        return operatorId.toHexString() + "-" + subtaskIndex;
    }

    /** Creates {@link FeedbackKey} from the {@code iterationId} and {@code feedbackIndex}. */
    public static <V> FeedbackKey<V> createFeedbackKey(IterationID iterationId, int feedbackIndex) {
        return new FeedbackKey<>(iterationId.toHexString(), feedbackIndex);
    }

    /** Registers the specified {@code feedbackConsumer} to the {@code feedbackChannel}. */
    public static <V> void registerFeedbackConsumer(
            FeedbackChannel<V> feedbackChannel,
            FeedbackConsumer<V> feedbackConsumer,
            Executor executor) {
        ReflectionUtils.callMethod(
                feedbackChannel,
                FeedbackChannel.class,
                "registerConsumer",
                Arrays.asList(FeedbackConsumer.class, Executor.class),
                Arrays.asList(feedbackConsumer, executor));
    }

    public static <T> void processOperatorOrUdfIfSatisfy(
            StreamOperator<?> operator, Class<T> targetInterface, Consumer<T> action) {
        if (targetInterface.isAssignableFrom(operator.getClass())) {
            action.accept((T) operator);
        } else if (operator instanceof AbstractUdfStreamOperator<?, ?>) {
            Object udf = ((AbstractUdfStreamOperator<?, ?>) operator).getUserFunction();
            if (targetInterface.isAssignableFrom(udf.getClass())) {
                action.accept((T) udf);
            }
        }
    }
}
