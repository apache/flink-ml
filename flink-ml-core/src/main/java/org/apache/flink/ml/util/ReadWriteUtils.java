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

package org.apache.flink.ml.util;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.Encoder;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.connector.source.Source;
import org.apache.flink.connector.file.sink.FileSink;
import org.apache.flink.connector.file.src.FileSource;
import org.apache.flink.connector.file.src.reader.SimpleStreamFormat;
import org.apache.flink.core.fs.Path;
import org.apache.flink.ml.api.Stage;
import org.apache.flink.ml.builder.Graph;
import org.apache.flink.ml.builder.GraphData;
import org.apache.flink.ml.builder.GraphModel;
import org.apache.flink.ml.builder.GraphNode;
import org.apache.flink.ml.common.datastream.TableUtils;
import org.apache.flink.ml.param.Param;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.filesystem.bucketassigners.BasePathBucketAssigner;
import org.apache.flink.streaming.api.functions.sink.filesystem.rollingpolicies.OnCheckpointRollingPolicy;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;
import org.apache.flink.util.Preconditions;

import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.core.JsonParser;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Utility methods for reading and writing stages. */
public class ReadWriteUtils {
    public static final ObjectMapper OBJECT_MAPPER =
            new ObjectMapper().enable(JsonParser.Feature.ALLOW_COMMENTS);

    // A helper method that calls encodes the given parameter value to a json string. We can not
    // call param.jsonEncode(value) directly because Param::jsonEncode(...) needs the actual type
    // of the value.
    private static <T> Object jsonEncodeHelper(Param<T> param, Object value) throws IOException {
        return param.jsonEncode((T) value);
    }

    // Converts Map<Param<?>, Object> to Map<String, Object> which maps the parameter name to the
    // json-supported parameter value.
    private static Map<String, Object> jsonEncode(Map<Param<?>, Object> paramMap)
            throws IOException {
        Map<String, Object> result = new HashMap<>(paramMap.size());
        for (Map.Entry<Param<?>, Object> entry : paramMap.entrySet()) {
            Object json = jsonEncodeHelper(entry.getKey(), entry.getValue());
            result.put(entry.getKey().name, json);
        }
        return result;
    }

    /**
     * Saves the metadata of the given stage and the extra metadata to a file named `metadata` under
     * the given path. The metadata of a stage includes the stage class name, parameter values etc.
     *
     * <p>Required: the metadata file under the given path should not exist.
     *
     * @param stage The stage instance.
     * @param path The parent directory to save the stage metadata.
     * @param extraMetadata The extra metadata to be saved.
     */
    public static void saveMetadata(Stage<?> stage, String path, Map<String, ?> extraMetadata)
            throws IOException {
        Map<String, Object> metadata = new HashMap<>(extraMetadata);
        metadata.put("className", stage.getClass().getName());
        metadata.put("timestamp", System.currentTimeMillis());
        metadata.put("paramMap", jsonEncode(stage.getParamMap()));
        // TODO: add version in the metadata.
        String metadataStr = OBJECT_MAPPER.writeValueAsString(metadata);

        FileUtils.saveToFile(new Path(path, "metadata").toUri().toString(), metadataStr, false);
    }

    /**
     * Saves the metadata of the given stage to a file named `metadata` under the given path. The
     * metadata of a stage includes the stage class name, parameter values etc.
     *
     * <p>Required: the metadata file under the given path should not exist.
     *
     * @param stage The stage instance.
     * @param path The parent directory to save the stage metadata.
     */
    public static void saveMetadata(Stage<?> stage, String path) throws IOException {
        saveMetadata(stage, path, new HashMap<>());
    }

    /**
     * Saves a Pipeline or PipelineModel with the given list of stages to the given path.
     *
     * @param pipeline A Pipeline or PipelineModel instance.
     * @param stages A list of stages of the given pipeline.
     * @param path The parent directory to save the pipeline metadata and its stages.
     */
    public static void savePipeline(Stage<?> pipeline, List<Stage<?>> stages, String path)
            throws IOException {
        // Creates parent directories if not already created.
        FileUtils.mkdirs(new Path(path));

        Map<String, Object> extraMetadata = new HashMap<>();
        extraMetadata.put("numStages", stages.size());
        saveMetadata(pipeline, path, extraMetadata);

        int numStages = stages.size();
        for (int i = 0; i < numStages; i++) {
            String stagePath = FileUtils.getPathForPipelineStage(i, numStages, path);
            stages.get(i).save(stagePath);
        }
    }

    /**
     * Loads the stages of a Pipeline or PipelineModel from the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved Pipeline or PipelineModel.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory to load the pipeline metadata and its stages.
     * @param expectedClassName The expected class name of the pipeline.
     * @return A list of stages.
     */
    public static List<Stage<?>> loadPipeline(
            StreamTableEnvironment tEnv, String path, String expectedClassName) throws IOException {
        Map<String, ?> metadata = FileUtils.loadMetadata(path, expectedClassName);
        int numStages = (Integer) metadata.get("numStages");
        List<Stage<?>> stages = new ArrayList<>(numStages);

        for (int i = 0; i < numStages; i++) {
            String stagePath = FileUtils.getPathForPipelineStage(i, numStages, path);
            stages.add(loadStage(tEnv, stagePath));
        }
        return stages;
    }

    /**
     * Saves a Graph or GraphModel with the given GraphData to the given path.
     *
     * @param graph A Graph or GraphModel instance.
     * @param graphData A GraphData instance.
     * @param path The parent directory to save the graph metadata and its stages.
     */
    public static void saveGraph(Stage<?> graph, GraphData graphData, String path)
            throws IOException {
        // Creates parent directories if not already created.
        FileUtils.mkdirs(new Path(path));

        Map<String, Object> extraMetadata = new HashMap<>();
        extraMetadata.put("graphData", graphData.toMap());
        saveMetadata(graph, path, extraMetadata);
        int maxNodeId =
                graphData.nodes.stream()
                        .map(node -> node.nodeId)
                        .max(Comparator.naturalOrder())
                        .orElse(-1);

        for (GraphNode node : graphData.nodes) {
            String stagePath = FileUtils.getPathForPipelineStage(node.nodeId, maxNodeId + 1, path);
            node.stage.save(stagePath);
        }
    }

    /**
     * Loads a Graph or GraphModel from the given path.
     *
     * <p>The method throws RuntimeException if the expectedClassName is not empty AND it does not
     * match the className of the previously saved Pipeline or PipelineModel.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory to load the pipeline metadata and its stages.
     * @param expectedClassName The expected class name of the pipeline.
     * @return A Graph or GraphModel instance.
     */
    @SuppressWarnings("unchecked")
    public static Stage<?> loadGraph(
            StreamTableEnvironment tEnv, String path, String expectedClassName) throws IOException {
        Map<String, ?> metadata = FileUtils.loadMetadata(path, expectedClassName);
        GraphData graphData = GraphData.fromMap((Map<String, Object>) metadata.get("graphData"));

        int maxNodeId =
                graphData.nodes.stream()
                        .map(node -> node.nodeId)
                        .max(Comparator.naturalOrder())
                        .orElse(-1);

        for (GraphNode node : graphData.nodes) {
            String stagePath = FileUtils.getPathForPipelineStage(node.nodeId, maxNodeId + 1, path);
            node.stage = loadStage(tEnv, stagePath);
        }

        if (expectedClassName.equals(GraphModel.class.getName())) {
            return new GraphModel(
                    graphData.nodes,
                    graphData.modelInputIds,
                    graphData.outputIds,
                    graphData.inputModelDataIds,
                    graphData.outputModelDataIds);
        }
        Preconditions.checkState(expectedClassName.equals(Graph.class.getName()));
        return new Graph(
                graphData.nodes,
                graphData.estimatorInputIds,
                graphData.modelInputIds,
                graphData.outputIds,
                graphData.inputModelDataIds,
                graphData.outputModelDataIds);
    }

    /**
     * Loads the stage with the saved parameters from the given path. This method reads the metadata
     * file under the given path, instantiates the stage using its no-argument constructor, and
     * loads the stage with the paramMap from the metadata file.
     *
     * <p>Note: This method does not attempt to read model data from the given path. Caller needs to
     * read model data from the given path if the stage has model data.
     *
     * <p>Required: the class with type T must have a no-argument constructor.
     *
     * @param path The parent directory of the stage metadata file.
     * @param <T> The class type of the Stage subclass.
     * @return An instance of class type T.
     */
    public static <T extends Stage<T>> T loadStageParam(String path) throws IOException {
        try {
            return ParamUtils.instantiateWithParams(FileUtils.loadMetadata(path, ""));
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Failed to load stage.", e);
        }
    }

    /**
     * Loads the stage from the given path by invoking the static load() method of the stage. The
     * stage class name is read from the metadata file under the given path. The load() method is
     * expected to construct the stage instance with the saved parameters, model data and other
     * metadata if exists.
     *
     * <p>Required: the stage class must have a static load() method.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory of the stage metadata file.
     * @return An instance of Stage.
     */
    public static Stage<?> loadStage(StreamTableEnvironment tEnv, String path) throws IOException {
        Map<String, ?> metadata = FileUtils.loadMetadata(path, "");
        String className = (String) metadata.get("className");

        try {
            Class<?> clazz = Class.forName(className);
            Method method = clazz.getMethod("load", StreamTableEnvironment.class, String.class);
            method.setAccessible(true);
            return (Stage<?>) method.invoke(null, tEnv, path);
        } catch (NoSuchMethodException e) {
            String methodName =
                    String.format("%s::load(StreamTableEnvironment, String)", className);
            throw new RuntimeException(
                    "Failed to load stage because the static method "
                            + methodName
                            + " is not implemented.",
                    e);
        } catch (ClassNotFoundException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException("Failed to load stage.", e);
        }
    }

    /**
     * Saves the model data stream to the given path using the model encoder.
     *
     * @param model The model data stream.
     * @param path The parent directory of the model data file.
     * @param modelEncoder The encoder to encode the model data.
     * @param <T> The class type of the model data.
     */
    public static <T> void saveModelData(
            DataStream<T> model, String path, Encoder<T> modelEncoder) {
        FileSink<T> sink =
                FileSink.forRowFormat(FileUtils.getDataPath(path), modelEncoder)
                        .withRollingPolicy(OnCheckpointRollingPolicy.build())
                        .withBucketAssigner(new BasePathBucketAssigner<>())
                        .build();
        model.sinkTo(sink);
    }

    /**
     * Loads the model data from the given path using the model decoder.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory of the model data file.
     * @param modelDecoder The decoder used to decode the model data.
     * @param <T> The class type of the model data.
     * @return The loaded model data.
     */
    public static <T> Table loadModelData(
            StreamTableEnvironment tEnv, String path, SimpleStreamFormat<T> modelDecoder) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        Source<T, ?, ?> source =
                FileSource.forRecordStreamFormat(modelDecoder, FileUtils.getDataPath(path)).build();
        DataStream<T> modelDataStream =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "modelData");
        return tEnv.fromDataStream(modelDataStream);
    }

    /**
     * Loads the model data from the given path using the model decoder. This overloaded version
     * returns a table with only 1 column whose type is the class of the model data.
     *
     * @param tEnv A StreamTableEnvironment instance.
     * @param path The parent directory of the model data file.
     * @param modelDecoder The decoder used to decode the model data.
     * @param typeInfo The type information of model data.
     * @param <T> The class type of the model data.
     * @return The loaded model data.
     */
    public static <T> Table loadModelData(
            StreamTableEnvironment tEnv,
            String path,
            SimpleStreamFormat<T> modelDecoder,
            TypeInformation<T> typeInfo) {
        StreamExecutionEnvironment env = TableUtils.getExecutionEnvironment(tEnv);
        Source<T, ?, ?> source =
                FileSource.forRecordStreamFormat(modelDecoder, FileUtils.getDataPath(path)).build();
        DataStream<T> modelDataStream =
                env.fromSource(source, WatermarkStrategy.noWatermarks(), "modelData", typeInfo);
        return tEnv.fromDataStream(modelDataStream);
    }
}
