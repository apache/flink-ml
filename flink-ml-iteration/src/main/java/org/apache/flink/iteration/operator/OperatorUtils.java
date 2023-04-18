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

import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.Path;
import org.apache.flink.iteration.IterationID;
import org.apache.flink.iteration.config.IterationOptions;
import org.apache.flink.iteration.proxy.ProxyKeySelector;
import org.apache.flink.iteration.typeinfo.IterationRecordSerializer;
import org.apache.flink.iteration.typeinfo.IterationRecordTypeInfo;
import org.apache.flink.iteration.utils.ReflectionUtils;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.statefun.flink.core.feedback.FeedbackChannel;
import org.apache.flink.statefun.flink.core.feedback.FeedbackConsumer;
import org.apache.flink.statefun.flink.core.feedback.FeedbackKey;
import org.apache.flink.streaming.api.graph.StreamConfig;
import org.apache.flink.streaming.api.graph.StreamConfig.NetworkInputConfig;
import org.apache.flink.streaming.api.operators.AbstractUdfStreamOperator;
import org.apache.flink.streaming.api.operators.StreamOperator;
import org.apache.flink.util.ExceptionUtils;
import org.apache.flink.util.OutputTag;
import org.apache.flink.util.function.SupplierWithException;
import org.apache.flink.util.function.ThrowingConsumer;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.Executor;
import java.util.stream.Stream;

import static org.apache.flink.util.Preconditions.checkState;

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
            StreamOperator<?> operator,
            Class<T> targetInterface,
            ThrowingConsumer<T, Exception> action) {
        try {
            if (targetInterface.isAssignableFrom(operator.getClass())) {
                action.accept((T) operator);
            } else if (operator instanceof AbstractUdfStreamOperator<?, ?>) {
                Object udf = ((AbstractUdfStreamOperator<?, ?>) operator).getUserFunction();
                if (targetInterface.isAssignableFrom(udf.getClass())) {
                    action.accept((T) udf);
                }
            }
        } catch (Exception e) {
            ExceptionUtils.rethrow(e);
        }
    }

    public static StreamConfig createWrappedOperatorConfig(StreamConfig config, ClassLoader cl) {
        StreamConfig wrappedConfig = new StreamConfig(config.getConfiguration().clone());
        for (int i = 0; i < wrappedConfig.getNumberOfNetworkInputs(); ++i) {
            KeySelector keySelector = config.getStatePartitioner(i, cl);
            if (keySelector != null) {
                checkState(
                        keySelector instanceof ProxyKeySelector,
                        "The state partitioner for the wrapper operator should always be ProxyKeySelector, but it is "
                                + keySelector);
                wrappedConfig.setStatePartitioner(
                        i, ((ProxyKeySelector) keySelector).getWrappedKeySelector());
            }
        }

        StreamConfig.InputConfig[] inputs = config.getInputs(cl);
        for (int i = 0; i < inputs.length; ++i) {
            if (inputs[i] instanceof NetworkInputConfig) {
                TypeSerializer<?> typeSerializerIn =
                        ((NetworkInputConfig) inputs[i]).getTypeSerializer();
                checkState(
                        typeSerializerIn instanceof IterationRecordSerializer,
                        "The serializer of input[%s] should be IterationRecordSerializer but it is %s.",
                        i,
                        typeSerializerIn);
                inputs[i] =
                        new NetworkInputConfig(
                                ((IterationRecordSerializer<?>) typeSerializerIn)
                                        .getInnerSerializer(),
                                i);
            }
        }
        wrappedConfig.setInputs(inputs);

        TypeSerializer<?> typeSerializerOut = config.getTypeSerializerOut(cl);
        checkState(
                typeSerializerOut instanceof IterationRecordSerializer,
                "The serializer of output should be IterationRecordSerializer but it is %s.",
                typeSerializerOut);
        wrappedConfig.setTypeSerializerOut(
                ((IterationRecordSerializer<?>) typeSerializerOut).getInnerSerializer());

        Stream.concat(
                        config.getChainedOutputs(cl).stream(),
                        config.getNonChainedOutputs(cl).stream())
                .forEach(
                        edge -> {
                            OutputTag<?> outputTag = edge.getOutputTag();
                            if (outputTag != null) {
                                TypeSerializer<?> typeSerializerSideOut =
                                        config.getTypeSerializerSideOut(outputTag, cl);
                                checkState(
                                        typeSerializerSideOut instanceof IterationRecordSerializer,
                                        "The serializer of side output with tag[%s] should be IterationRecordSerializer but it is %s.",
                                        outputTag,
                                        typeSerializerSideOut);
                                wrappedConfig.setTypeSerializerSideOut(
                                        new OutputTag<>(
                                                outputTag.getId(),
                                                ((IterationRecordTypeInfo<?>)
                                                                outputTag.getTypeInfo())
                                                        .getInnerTypeInfo()),
                                        ((IterationRecordSerializer) typeSerializerSideOut)
                                                .getInnerSerializer());
                            }
                        });

        return wrappedConfig;
    }

    public static Path getDataCachePath(Configuration configuration, String[] localSpillPaths) {
        String pathStr = configuration.get(IterationOptions.DATA_CACHE_PATH);
        if (pathStr == null) {
            Random random = new Random();
            final String localSpillPath = localSpillPaths[random.nextInt(localSpillPaths.length)];
            pathStr = Paths.get(localSpillPath).toUri().toString();
        }
        return new Path(pathStr);
    }

    public static SupplierWithException<Path, IOException> createDataCacheFileGenerator(
            Path basePath, String fileTypeName, OperatorID operatorId) {
        return () ->
                new Path(
                        String.format(
                                "%s/%s-%s-%s",
                                basePath.toString(),
                                fileTypeName,
                                operatorId,
                                UUID.randomUUID().toString()));
    }
}
