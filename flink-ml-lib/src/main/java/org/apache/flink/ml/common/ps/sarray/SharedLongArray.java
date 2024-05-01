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

package org.apache.flink.ml.common.ps.sarray;

import it.unimi.dsi.fastutil.longs.LongArrayList;

import java.io.Serializable;

/** A resizable long array that can be shared among different iterations for memory efficiency. */
public class SharedLongArray implements Serializable {

    /** The underlying LongArrayList that holds the elements. */
    private final LongArrayList longs;

    /**
     * Constructs a new SLArray from the given long array.
     *
     * @param array the long array to wrap
     */
    public SharedLongArray(long[] array) {
        longs = LongArrayList.wrap(array);
    }

    /**
     * Constructs a new SLArray with the given initial capacity.
     *
     * @param capacity the initial capacity
     */
    public SharedLongArray(int capacity) {
        longs = new LongArrayList(capacity);
    }

    /** Constructs a new empty SLArray. */
    public SharedLongArray() {
        longs = new LongArrayList();
    }

    /**
     * Returns the element at the specified index.
     *
     * @param index the index of the element to return
     * @return the element at the specified index
     */
    public long get(int index) {
        return longs.getLong(index);
    }

    /**
     * Appends the specified element to the end of this array.
     *
     * @param v the element to add
     */
    public void add(long v) {
        longs.add(v);
    }

    /**
     * Appends all the elements from the specified long array to the end of this array.
     *
     * @param src the long array to append
     */
    public void addAll(long[] src) {
        int sizeBefore = size();
        longs.size(sizeBefore + src.length);
        System.arraycopy(src, 0, elements(), sizeBefore, src.length);
    }

    /**
     * Returns the number of valid elements in this array.
     *
     * @return the number of valid elements in this array
     */
    public int size() {
        return longs.size();
    }

    /**
     * Resizes this array to the specified size. Sets the size of the array to the provided size. If
     * the new size is larger than the current size, the new allocated memory are filled with zero.
     *
     * @param size the new size of the array
     */
    public void size(int size) {
        longs.size(size);
    }

    /** Clears the elements in this array. Note that the memory is not recycled. */
    public void clear() {
        longs.clear();
    }

    /**
     * Returns a long array containing the valid elements in this array. Only the first {@link
     * SharedLongArray#size()} elements are valid.
     *
     * @return a long array containing the valid elements in this array
     */
    public long[] elements() {
        return longs.elements();
    }
}
