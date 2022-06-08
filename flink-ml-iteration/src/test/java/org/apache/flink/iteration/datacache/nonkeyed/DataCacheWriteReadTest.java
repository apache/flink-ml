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
import org.apache.flink.core.fs.FileSystem;
import org.apache.flink.core.fs.Path;
import org.apache.flink.runtime.fs.hdfs.HadoopFileSystem;
import org.apache.flink.runtime.memory.MemoryManager;
import org.apache.flink.table.runtime.util.LazyMemorySegmentPool;
import org.apache.flink.table.runtime.util.MemorySegmentPool;
import org.apache.flink.util.OperatingSystem;
import org.apache.flink.util.TestLogger;

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

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Tests the behavior of {@link DataCacheWriter}. */
@RunWith(Parameterized.class)
public class DataCacheWriteReadTest extends TestLogger {

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
    public static Object[][] testData() {
        return new Object[][] {new Object[] {"local"}, new Object[] {"hdfs"}};
    }

    public DataCacheWriteReadTest(String fileSystemType) throws IOException {
        if (fileSystemType.equals("local")) {
            fileSystem = FileSystem.getLocalFileSystem();
            basePath = new Path("file://" + CLASS_TEMPORARY_FOLDER.newFolder().getAbsolutePath());
        } else if (fileSystemType.equals("hdfs")) {
            fileSystem = new HadoopFileSystem(hdfsCluster.getNewFileSystemInstance(0));
            basePath = new Path(hdfsCluster.getURI().toString() + "/" + UUID.randomUUID());
        } else {
            throw new UnsupportedEncodingException("Unsupported fs type: " + fileSystemType);
        }
    }

    @Test
    public void testWriteAndReadEmptyCache() throws IOException {
        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "test." + UUID.randomUUID()));
        List<Segment> segments = writer.finish();

        assertEquals(0, segments.size());

        DataCacheReader<Integer> reader = new DataCacheReader<>(IntSerializer.INSTANCE, segments);
        assertFalse(reader.hasNext());
    }

    @Test
    public void testWriteAndReadSingleSegment() throws IOException {
        final int numRecords = 10240;

        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "test_single." + UUID.randomUUID()));
        for (int i = 0; i < numRecords; ++i) {
            writer.addRecord(i);
        }

        List<Segment> segments = writer.finish();

        assertEquals(1, segments.size());
        verifySegment(numRecords, segments.get(0));

        DataCacheReader<Integer> reader = new DataCacheReader<>(IntSerializer.INSTANCE, segments);
        List<Integer> read = new ArrayList<>();
        while (reader.hasNext()) {
            read.add(reader.next());
        }

        assertEquals(IntStream.range(0, numRecords).boxed().collect(Collectors.toList()), read);
    }

    @Test
    public void testCacheInMemory() throws IOException {
        int numRecords = 10240;
        int pageSize = 4096;
        int pageNum = 64;

        MemoryManager memoryManager = MemoryManager.create(pageSize * pageNum, pageSize);
        MemorySegmentPool segmentPool =
                new LazyMemorySegmentPool(
                        this, memoryManager, memoryManager.computeNumberOfPages(1.0));

        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "test_cache." + UUID.randomUUID()),
                        segmentPool);
        for (int i = 0; i < numRecords; ++i) {
            writer.addRecord(i);
        }

        List<Segment> segments = writer.finish();
        assertTrue(segmentPool.freePages() < pageNum);
        int freePages = segmentPool.freePages();

        for (int i = 0; i < 2; i++) {
            DataCacheReader<Integer> reader =
                    new DataCacheReader<>(IntSerializer.INSTANCE, segments);
            List<Integer> read = new ArrayList<>();
            while (reader.hasNext()) {
                read.add(reader.next());
            }

            assertEquals(IntStream.range(0, numRecords).boxed().collect(Collectors.toList()), read);
            assertEquals(freePages, segmentPool.freePages());
        }
    }

    @Test
    public void testInsufficientMemoryForWriter() throws IOException {
        int numRecords = 10240;
        int pageSize = 4096;
        int pageNum = 4;

        MemoryManager memoryManager = MemoryManager.create(pageSize * pageNum, pageSize);
        MemorySegmentPool segmentPool =
                new LazyMemorySegmentPool(
                        this, memoryManager, memoryManager.computeNumberOfPages(1.0));

        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "test_not_cache_writer." + UUID.randomUUID()),
                        segmentPool);
        for (int i = 0; i < numRecords; ++i) {
            writer.addRecord(i);
        }

        List<Segment> segments = writer.finish();
        assertTrue(memoryManager.availableMemory() < memoryManager.getMemorySize());
        long availableMemory = memoryManager.availableMemory();

        for (int i = 0; i < 2; i++) {
            DataCacheReader<Integer> reader =
                    new DataCacheReader<>(IntSerializer.INSTANCE, segments);
            List<Integer> read = new ArrayList<>();
            while (reader.hasNext()) {
                read.add(reader.next());
            }

            assertEquals(IntStream.range(0, numRecords).boxed().collect(Collectors.toList()), read);
            assertEquals(availableMemory, memoryManager.availableMemory());
        }
    }

    @Test
    public void testInsufficientMemoryForReader() throws IOException {
        int numRecords = 10240;
        int pageSize = 4096;
        int pageNum = 4;

        MemoryManager memoryManager = MemoryManager.create(pageSize * pageNum, pageSize);
        MemorySegmentPool segmentPool =
                new LazyMemorySegmentPool(
                        this, memoryManager, memoryManager.computeNumberOfPages(1.0));

        DataCacheWriter<Integer> writer =
                new DataCacheWriter<>(
                        IntSerializer.INSTANCE,
                        fileSystem,
                        () -> new Path(basePath, "test_not_cache_reader." + UUID.randomUUID()));
        for (int i = 0; i < numRecords; ++i) {
            writer.addRecord(i);
        }

        List<Segment> segments = writer.finish();
        assertEquals(segmentPool.freePages(), pageNum);

        for (int i = 0; i < 3; i++) {
            DataCacheReader<Integer> reader =
                    new DataCacheReader<>(IntSerializer.INSTANCE, segments);
            List<Integer> read = new ArrayList<>();
            while (reader.hasNext()) {
                read.add(reader.next());
            }

            assertEquals(IntStream.range(0, numRecords).boxed().collect(Collectors.toList()), read);
            assertEquals(pageNum, segmentPool.freePages());
        }
    }

    private void verifySegment(int expectedCount, Segment segment) throws IOException {
        assertEquals(expectedCount, segment.getCount());
        assertEquals(fileSystem.getFileStatus(segment.getPath()).getLen(), segment.getFsSize());
    }
}
