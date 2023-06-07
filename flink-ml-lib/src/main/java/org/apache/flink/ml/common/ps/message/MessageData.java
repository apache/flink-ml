// package org.apache.flink.ml.common.ps.message;
//
// import org.apache.flink.api.common.typeutils.TypeSerializer;
// import org.apache.flink.api.java.tuple.Tuple2;
// import org.apache.flink.core.memory.DataInputViewStreamWrapper;
// import org.apache.flink.core.memory.DataOutputViewStreamWrapper;
// import org.apache.flink.ml.util.Bits;
//
// import java.io.ByteArrayInputStream;
// import java.io.ByteArrayOutputStream;
// import java.io.IOException;
// import java.nio.ByteBuffer;
// import java.util.ArrayList;
// import java.util.List;
//
/// **
// * Message body.
// */
// public class MessageData {
//	byte[] bytes;
//	ByteBuffer byteBuffer;
//	int offset = 0;
//
//	public MessageData(Meta meta, int messageSize) {
//		byteBuffer = ByteBuffer.allocate(messageSize);
//	}
//
//	/**
//	 * Adds data for generics.
//	 */
//	public <V> void addData(long[] keys, V[] values, TypeSerializer<V> serializer) throws IOException
// {
//		ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
//		    DataOutputViewStreamWrapper dataOutputViewStreamWrapper = new
// DataOutputViewStreamWrapper(byteArrayOutputStream);
//		    for (int i = 0; i < values.length; i ++) {
//				serializer.serialize(values[i], dataOutputViewStreamWrapper);
//		    }
//		    byte[] serializedValues = byteArrayOutputStream.toByteArray();
//
//	}
//
//	public void addData(long[] keys, double[] values) {
//		offset = MessageUtils.putLongDoubleArray(Tuple2.of(keys, values), bytes, offset);
//	}
//
//	/** Gets a long-double array from the byte array starting from the given offset. */
//	public static <V> V[] getGenericArray(byte[] bytes, int offset, TypeSerializer <V>
// typeSerializer)
//	    throws IOException {
//	    int n = Bits.getInt(bytes, offset);
//	    Object[] result = new Object[n];
//	    ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(bytes, offset,
// bytes.length - offset);
//	    DataInputViewStreamWrapper dataInputViewStreamWrapper = new
// DataInputViewStreamWrapper(byteArrayInputStream);
//	    for (int i = 0; i < n; i ++) {
//	        result[i] = typeSerializer.deserialize(dataInputViewStreamWrapper);
//	    }
//
//	    return (V[]) result;
//	}
//
//	public static <V> byte[] getSerializedBytes(V[] values, TypeSerializer<V> typeSerializer) throws
// IOException {
//	    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
//	    DataOutputViewStreamWrapper dataOutputViewStreamWrapper = new
// DataOutputViewStreamWrapper(byteArrayOutputStream);
//	    for (int i = 0; i < values.length; i ++) {
//	        typeSerializer.serialize(values[i], dataOutputViewStreamWrapper);
//	    }
//	    byte[] serializedValues = byteArrayOutputStream.toByteArray();
//	    return serializedValues;
//	}
// }
