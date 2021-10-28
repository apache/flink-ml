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

package org.apache.flink.iteration.datacache.nonkeyed;

import org.apache.flink.api.common.typeutils.base.IntSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.runtime.fs.hdfs.HadoopFileSystem;
import org.apache.flink.util.OperatingSystem;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.junit.AfterClass;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;

/** Tests the behavior of the {@link DataCacheSnapshot}. */
@RunWith(Parameterized.class)
public class DataCacheSnapshotTest {

    @ClassRule public static final TemporaryFolder CLASS_TEMPORARY_FOLDER = new TemporaryFolder();

    private static MiniDFSCluster hdfsCluster;

    private final FileSystem fileSystem;

    private final Path basePath;

    @BeforeClass
    public static void createHDFS() throws Exception {
        Assume.assumeTrue(!OperatingSystem.isWindows());

        Configuration hdfsConfig = new Configuration();
        hdfsConfig.set(
                MiniDFSCluster.HDFS_MINIDFS_BASEDIR,
                CLASS_TEMPORARY_FOLDER.newFolder().getAbsolutePath());
        hdfsCluster = new MiniDFSCluster.Builder(hdfsConfig).build();
    }

    @AfterClass
    public static void destroyHDFS() {
        if (hdfsCluster != null) {
            hdfsCluster.shutdown();
        }

        hdfsCluster = null;
    }

    @Parameterized.Parameters(name = "{0}")
    public static Object[][] testData() throws IOException {
        return new Object[][] {new Object[] {"local"}, new Object[] {"hdfs"}};
    }

    public DataCacheSnapshotTest(String fileSystemType) throws IOException {
        if (fileSystemType.equals("local")) {
            fileSystem = FileSystem.getLocalFileSystem();
            basePath = new Path("file://" + CLASS_TEMPORARY_FOLDER.newFolder().getAbsolutePath());
        } else if (fileSystemType.equals("hdfs")) {
            fileSystem = new HadoopFileSystem(hdfsCluster.getNewFileSystemInstance(0));
            basePath =
                    new Path(hdfsCluster.getURI().toString() + "/" + UUID.randomUUID().toString());
        } else {
            throw new UnsupportedEncodingException("Unsupported fs type: " + fileSystemType);
        }
    }

    @Test
    public void testWithoutReaderPosition() throws Exception {
        int[] numRecordsPerSegment = {100, 200, 300};
        DataCacheWriter<Integer> writer = createWriterAndAddRecords(numRecordsPerSegment);
        DataCacheSnapshot dataCacheSnapshot =
                new DataCacheSnapshot(fileSystem, null, writer.getFinishSegments());
        checkWriteAndRecoverAndReplay(dataCacheSnapshot, numRecordsPerSegment);
    }

    @Test
    public void testWithReadPosition() throws Exception {
        int[] numRecordsPerSegment = {100, 200, 300};
        DataCacheWriter<Integer> writer = createWriterAndAddRecords(numRecordsPerSegment);
        DataCacheSnapshot dataCacheSnapshot =
                new DataCacheSnapshot(fileSystem, new Tuple2<>(0, 50), writer.getFinishSegments());
        checkWriteAndRecoverAndReplay(dataCacheSnapshot, numRecordsPerSegment);
    }

    private DataCacheWriter<Integer> createWriterAndAddRecords(int[] numRecordsPerSegment)
            throws IOException {
        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "writer." + UUID.randomUUID().toString()));
        int nextNumber = 0;
        for (int numRecord : numRecordsPerSegment) {
            for (int i = 0; i < numRecord; ++i) {
                writer.addRecord(nextNumber++);
            }
            writer.finishCurrentSegment();
            writer.startNewSegment();
        }
        writer.finishAddingRecords();
        return writer;
    }

    private void checkWriteAndRecoverAndReplay(
            DataCacheSnapshot dataCacheSnapshot, int[] numRecordsPerSegment) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        dataCacheSnapshot.writeTo(bos);
        byte[] data = bos.toByteArray();

        checkRecover(dataCacheSnapshot, data);
        checkReplay(dataCacheSnapshot, data, numRecordsPerSegment);
    }

    private void checkRecover(DataCacheSnapshot dataCacheSnapshot, byte[] serializedData)
            throws IOException {
        ByteArrayInputStream bis = new ByteArrayInputStream(serializedData);
        DataCacheSnapshot copied =
                DataCacheSnapshot.recover(
                        bis,
                        dataCacheSnapshot.getFileSystem(),
                        () -> new Path(basePath, "writer." + UUID.randomUUID().toString()));
        if (dataCacheSnapshot.getFileSystem().isDistributedFS()) {
            assertEquals(dataCacheSnapshot.getSegments(), copied.getSegments());
        } else {
            assertEquals(readElements(dataCacheSnapshot), readElements(copied));
        }

        assertEquals(dataCacheSnapshot.getReaderPosition(), copied.getReaderPosition());
    }

    private void checkReplay(
            DataCacheSnapshot dataCacheSnapshot, byte[] serializedData, int[] numRecordsPerSegment)
            throws Exception {
        ByteArrayInputStream bis = new ByteArrayInputStream(serializedData);
        List<Integer> elements = new ArrayList<>();
        DataCacheSnapshot.replay(bis, IntSerializer.INSTANCE, fileSystem, elements::add);

        int totalRecords = IntStream.of(numRecordsPerSegment).sum();
        assertEquals(
                IntStream.range(0, totalRecords).boxed().collect(Collectors.toList()), elements);
    }

    private List<Integer> readElements(DataCacheSnapshot dataCacheSnapshot) throws IOException {
        DataCacheReader<Integer> reader =
                new DataCacheReader<>(
                        IntSerializer.INSTANCE,
                        dataCacheSnapshot.getFileSystem(),
                        dataCacheSnapshot.getSegments());
        List<Integer> result = new ArrayList<>();
        while (reader.hasNext()) {
            result.add(reader.next());
        }

        return result;
    }
}
