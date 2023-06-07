// package org.apache.flink.ml.common.ps.message;
//
// import org.apache.flink.ml.util.Bits;
//
/// **
// * Meta information of a message.
// */
// public class Meta {
//	/**
//	 * Index of the sender of this message.
//	 */
//	public int sender;
//	/**
//	 * Index of the receiver of this message.
//	 */
//	public int receiver;
//	/**
//	 * Whether this is a push message.
//	 */
//	public boolean push;
//	/**
//	 * Whether this is a pull message.
//	 */
//	public boolean pull;
//	/**
//	 * The size of data in bytes.
//	 */
//	public int dataSize;
//
//	public Meta(int sender, int receiver, boolean push, boolean pull, int dataSize) {
//		this.sender = sender;
//		this.receiver = receiver;
//		this.push = push;
//		this.pull = pull;
//		this.dataSize = dataSize;
//	}
//
//	/**
//	 * Empty constructor to make it as a pojo.
//	 */
//	public Meta() {}
//
//	/**
//	 * Restores meta instance from a given byte array starting from the given offset.
//	 */
//	public Meta fromBytes(byte[] bytes, int offset) {
//		Meta meta = new Meta();
//		meta.sender = Bits.getInt(bytes, offset);
//		offset += Integer.BYTES;
//		meta.receiver = Bits.getInt(bytes, offset);
//		offset += Integer.BYTES;
//		meta.push = Bits.getChar(bytes, offset) == (char) 1;
//		offset += Character.BYTES;
//		meta.pull = Bits.getChar(bytes, offset) == (char) 1;
//		offset += Character.BYTES;
//		meta.dataSize = Bits.getInt(bytes, offset);
//		return meta;
//	}
//
//	/**
//	 * Writes a meta instance to a given byte array starting from the given offset.
//	 */
//	public int toBytes(byte[] bytes, int offset) {
//		Bits.putInt(bytes, offset, sender);
//		offset += Integer.BYTES;
//		Bits.putInt(bytes, offset, receiver);
//		offset += Integer.BYTES;
//
//		Bits.putChar(bytes, offset, push? (char) 1: (char) 0);
//		offset += Character.BYTES;
//		Bits.putChar(bytes, offset, pull? (char) 1: (char) 0);
//		offset += Character.BYTES;
//
//		Bits.putInt(bytes, offset, dataSize);
//		offset += Integer.BYTES;
//		return offset;
//	}
//
//	public static int getSizeInBytes() {
//		return Integer.BYTES + Integer.BYTES + Character.BYTES + Character.BYTES + Character.BYTES +
// Integer.BYTES;
//	}
// }
