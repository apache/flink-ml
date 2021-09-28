package org.apache.flink.ml.common.broadcast.operator;

import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.common.broadcast.BroadcastContext;
import org.apache.flink.runtime.jobgraph.OperatorID;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.runtime.tasks.MultipleInputStreamTask;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarness;
import org.apache.flink.streaming.runtime.tasks.StreamTaskMailboxTestHarnessBuilder;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class CacheStreamOperatorTest {
	final String[] broadcastNames = new String[] {"source1", "source2"};
	final TypeInformation[] typeInformations = new TypeInformation[] {BasicTypeInfo.INT_TYPE_INFO,
		BasicTypeInfo.INT_TYPE_INFO};

	@Test
	public void testCacheStreamOperator() throws Exception {
		OperatorID operatorId = new OperatorID();

		try (StreamTaskMailboxTestHarness <Integer> harness =
				 new StreamTaskMailboxTestHarnessBuilder <>(
					 MultipleInputStreamTask::new, BasicTypeInfo.INT_TYPE_INFO)
					 .addInput(BasicTypeInfo.INT_TYPE_INFO)
					 .addInput(BasicTypeInfo.INT_TYPE_INFO)
					 .setupOutputForSingletonOperatorChain(
						 new CacheStreamOperatorFactory <>(broadcastNames, typeInformations), operatorId)
					 .build()) {
			harness.processElement(new StreamRecord <>(1, 2), 0);
			harness.processElement(new StreamRecord <>(2, 3), 0);
			harness.processElement(new StreamRecord <>(3, 2), 1);
			harness.processElement(new StreamRecord <>(4, 2), 1);
			harness.processElement(new StreamRecord <>(5, 3), 1);
			harness.waitForTaskCompletion();
			List <Integer> cache1 = BroadcastContext.getBroadcastVariable(broadcastNames[0]);
			List <Integer> cache2 = BroadcastContext.getBroadcastVariable(broadcastNames[1]);
			// check broadcast inputs
			assertEquals(Arrays.asList(1, 2), cache1);
			assertEquals(Arrays.asList(3, 4, 5), cache2);
		}
	}

}