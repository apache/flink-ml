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

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;

import java.io.Serializable;

/** A resizable double array that can be shared among different iterations for memory efficiency. */
public class SharedDoubleArray implements Serializable {

    /** The underlying DoubleArrayList that holds the elements. */
    private final DoubleArrayList doubles;

    /**
     * Constructs a new SDArray from the given double array.
     *
     * @param array the double array to wrap
     */
    public SharedDoubleArray(double[] array) {
        doubles = DoubleArrayList.wrap(array);
    }

    /**
     * Constructs a new SDArray with the given initial capacity.
     *
     * @param capacity the initial capacity
     */
    public SharedDoubleArray(int capacity) {
        doubles = new DoubleArrayList(capacity);
    }

    /** Constructs a new empty SDArray. */
    public SharedDoubleArray() {
        doubles = new DoubleArrayList();
    }

    /**
     * Returns the element at the specified index.
     *
     * @param index the index of the element to return
     * @return the element at the specified index
     */
    public double get(int index) {
        return doubles.getDouble(index);
    }

    /**
     * Appends the specified element to the end of this array.
     *
     * @param v the element to add
     */
    public void add(double v) {
        doubles.add(v);
    }

    /**
     * Appends all the elements from the specified double array to the end of this array.
     *
     * @param src the double array to append
     */
    public void addAll(double[] src) {
        int sizeBefore = size();
        doubles.size(sizeBefore + src.length);
        System.arraycopy(src, 0, elements(), sizeBefore, src.length);
    }

    /**
     * Returns the number of valid elements in this array.
     *
     * @return the number of valid elements in this array
     */
    public int size() {
        return doubles.size();
    }

    /**
     * Sets the size of the array to the provided size. If the new size is larger than the current
     * size, the new allocated memory are filled with zero.
     *
     * @param size the new size of the array
     */
    public void size(int size) {
        doubles.size(size);
    }

    /** Clears the elements in this array. Note that the memory is not recycled. */
    public void clear() {
        doubles.clear();
    }

    /**
     * Returns a double array containing all the elements in this array. Only the first {@link
     * SharedDoubleArray#size()} elements are valid.
     *
     * @return a double array containing the all the elements in this array
     */
    public double[] elements() {
        return doubles.elements();
    }
}
