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

package org.apache.flink.ml.common.fpgrowth;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.streaming.api.operators.Output;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Implementation of local FPGrowth algorithm. Reference: Christian Borgelt, An Implementation of
 * the FP-growth Algorithm.
 */
public class FPTree implements Serializable {

    private static final Logger LOG = LoggerFactory.getLogger(FPTree.class);

    /** The tree node. Notice that no reference to children are kept. */
    private static class Node implements Serializable {
        private static final long serialVersionUID = -3963529487030357584L;
        int itemId;
        int support;
        Node parent;
        Node successor;
        Node auxPtr;

        public Node(int itemId, int support, Node parent) {
            this.itemId = itemId;
            this.support = support;
            this.parent = parent;
            this.successor = null;
            this.auxPtr = null;
        }
    }

    /** Summary of an item in the Fp-tree. */
    private static class Summary implements Serializable {
        private static final long serialVersionUID = 7641916158660339302L;
        /** Number of nodes in the tree. */
        int count;

        /** The head of the linked list of all nodes of an item. */
        Node head;

        public Summary(Node head) {
            this.head = head;
        }

        public void countAll() {
            Node p = head;
            count = 0;
            while (p != null) {
                count += p.support;
                p = p.successor;
            }
        }

        @Override
        public String toString() {
            StringBuilder sbd = new StringBuilder();
            Node p = head;
            while (p != null) {
                sbd.append("->")
                        .append(
                                String.format(
                                        "(%d,%d,%d)",
                                        p.itemId,
                                        p.support,
                                        p.parent == null ? -1 : p.parent.itemId));
                p = p.successor;
            }
            return sbd.toString();
        }
    }

    private Map<Integer, Summary> summaries; // item -> summary of the item

    // transient data for building trees.
    private Map<Integer, Node> roots; // item -> root node of the item
    private Map<Integer, List<Node>> itemNodes; // item -> list of nodes of the item

    public FPTree() {}

    private FPTree(Map<Integer, Summary> summaries) {
        this.summaries = summaries;
        this.summaries.forEach(
                (itemId, summary) -> {
                    summary.countAll();
                });
    }

    public void createTree() {
        this.summaries = new HashMap<>();
        this.roots = new HashMap<>();
        this.itemNodes = new HashMap<>();
    }

    public void destroyTree() {
        if (summaries != null) {
            this.summaries.clear();
        }
        if (roots != null) {
            this.roots.clear();
        }
        if (itemNodes != null) {
            this.itemNodes.clear();
        }
    }

    public void addTransaction(int[] transaction) {
        if (transaction.length == 0) {
            return;
        }
        int firstItem = transaction[0];
        Node curr;
        if (roots.containsKey(firstItem)) {
            curr = roots.get(firstItem);
            curr.support += 1;
        } else {
            curr = new Node(firstItem, 1, null);
            List<Node> list = new ArrayList<>();
            list.add(curr);
            itemNodes.merge(
                    firstItem,
                    list,
                    (old, delta) -> {
                        old.addAll(delta);
                        return old;
                    });
            roots.put(firstItem, curr);
        }

        for (int i = 1; i < transaction.length; i++) {
            int item = transaction[i];
            Node p = curr.auxPtr; // use auxPtr as head of siblings
            while (p != null && p.itemId != item) {
                p = p.successor;
            }
            if (p != null) { // found
                p.support += 1;
                curr = p;
            } else { // not found
                Node newNode = new Node(item, 1, curr);
                // insert newNode at the beginning of siblings.
                newNode.successor = curr.auxPtr;
                curr.auxPtr = newNode;
                curr = newNode;
                List<Node> list = new ArrayList<>();
                list.add(newNode);
                itemNodes.merge(
                        item,
                        list,
                        (old, delta) -> {
                            old.addAll(delta);
                            return old;
                        });
            }
        }
    }

    public void initialize() {
        this.itemNodes.forEach(
                (item, nodesList) -> {
                    int n = nodesList.size();
                    for (int i = 0; i < n; i++) {
                        Node curr = nodesList.get(i);
                        curr.auxPtr = null;
                        curr.successor = (i + 1) >= n ? null : nodesList.get(i + 1);
                    }
                    this.summaries.put(item, new Summary(nodesList.get(0)));
                });

        // clear data buffer
        this.roots.clear();
        this.itemNodes.clear();

        this.summaries.forEach((item, summary) -> summary.countAll());
    }

    /** Project the tree on the given item. */
    private FPTree project(int itemId, int minSupportCnt) {
        if (!this.summaries.containsKey(itemId)) {
            throw new RuntimeException("not contain item " + itemId);
        }
        Summary summary = this.summaries.get(itemId);
        Map<Integer, Summary> projectedSummaries = new HashMap<>();

        Node p = summary.head;
        while (p != null) {
            // trace upward
            // auxiliary pointer is copied and linked from its original ancestor f.
            Node lastShadow = null;
            Node f = p.parent;
            while (f != null) {
                if (f.auxPtr == null) {
                    Node shadow = new Node(f.itemId, p.support, null);
                    if (projectedSummaries.containsKey(shadow.itemId)) {
                        Summary summary0 = projectedSummaries.get(shadow.itemId);
                        shadow.successor = summary0.head;
                        summary0.head = shadow;
                    } else {
                        Summary summary0 = new Summary(shadow);
                        projectedSummaries.put(shadow.itemId, summary0);
                    }
                    f.auxPtr = shadow;
                } else { // aux ptr already created by another branch
                    f.auxPtr.support += p.support;
                }
                if (lastShadow != null) {
                    // to set parent ptr of auxPtr
                    lastShadow.parent = f.auxPtr;
                }
                lastShadow = f.auxPtr;
                f = f.parent;
            }
            p = p.successor;
        }

        // prune
        Set<Integer> toPrune = new HashSet<>();
        projectedSummaries.forEach(
                (item, s) -> {
                    s.countAll();
                    if (s.count < minSupportCnt) {
                        toPrune.add(item);
                    }
                });
        toPrune.forEach(projectedSummaries::remove);

        p = summary.head;
        while (p != null) {
            Node f = p.parent;
            if (f != null) {
                Node leaf = f.auxPtr;
                while (leaf != null && toPrune.contains(leaf.itemId)) {
                    leaf = leaf.parent;
                }
                while (leaf != null) {
                    Node leafParent = leaf.parent;
                    while (leafParent != null && toPrune.contains(leafParent.itemId)) {
                        leafParent = leafParent.parent;
                    }
                    leaf.parent = leafParent;
                    leaf = leafParent;
                }
            }
            p = p.successor;
        }

        // clear auxPtr
        p = summary.head;
        while (p != null) {
            Node f = p.parent;
            while (f != null) {
                f.auxPtr = null;
                f = f.parent;
            }
            p = p.successor;
        }

        return new FPTree(projectedSummaries);
    }

    private void extractImpl(
            int minSupportCnt,
            int item,
            int maxLength,
            int[] suffix,
            Output<StreamRecord<Tuple2<Integer, int[]>>> collector) {
        if (maxLength < 1) {
            return;
        }
        Summary summary = summaries.get(item);
        if (summary.count < minSupportCnt) {
            return;
        }
        int[] newSuffix = new int[suffix.length + 1];
        newSuffix[0] = item;
        System.arraycopy(suffix, 0, newSuffix, 1, suffix.length);
        Arrays.sort(newSuffix);
        collector.collect(new StreamRecord<>(Tuple2.of(summary.count, newSuffix.clone())));
        if (maxLength == 1) {
            return;
        }
        FPTree projectedTree = this.project(item, minSupportCnt);
        projectedTree.summaries.forEach(
                (pItem, pSummary) -> {
                    projectedTree.extractImpl(
                            minSupportCnt, pItem, maxLength - 1, newSuffix, collector);
                });
    }

    public void extractAll(
            int[] suffices,
            int minSupport,
            int maxPatternLength,
            Output<StreamRecord<Tuple2<Integer, int[]>>> collector) {
        for (int item : suffices) {
            extractImpl(minSupport, item, maxPatternLength, new int[0], collector);
        }
    }

    /** Print the tree profile for debugging purpose. */
    public void printProfile() {
        // tuple:
        // 1) num distinct items in the tree,
        // 2) sum of support of each items,
        // 3) num tree nodes in the tree
        Tuple3<Integer, Integer, Integer> counts = Tuple3.of(0, 0, 0);
        summaries.forEach(
                (item, summary) -> {
                    counts.f0 += 1;
                    counts.f1 += summary.count;
                    Node p = summary.head;
                    while (p != null) {
                        counts.f2 += 1;
                        p = p.successor;
                    }
                });
        LOG.info("fptree_profile {}", counts);
    }
}
