package net.jhorstmann.vectorapitest;

import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SumKernelTest {
    @Test
    public void testSum() {
        double[] data = IntStream.rangeClosed(1, 17).mapToDouble(i -> i).toArray();
        double sum = SumKernel.sum(data);

        assertEquals(1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17, sum);
    }

    @Test
    public void testSumNoRemainder() {
        double[] data = IntStream.rangeClosed(1, 16).mapToDouble(i -> i).toArray();
        double sum = SumKernel.sum(data);

        assertEquals(1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16, sum);
    }

    @Test
    public void testSumMasked() {
        double[] data = IntStream.rangeClosed(1, 17).mapToDouble(i -> i).toArray();
        byte[] valid = new byte[]{0b01010101, (byte) 0b10101010, 0b1};
        double sum = SumKernel.sum(data, valid);

        assertEquals(1+3+5+7+10+12+14+16+17, sum);
    }

    @Test
    public void testSumMaskedNoRemainder() {
        double[] data = IntStream.rangeClosed(1, 16).mapToDouble(i -> i).toArray();
        byte[] valid = new byte[]{0b01010101, (byte) 0b10101010};
        double sum = SumKernel.sum(data, valid);

        assertEquals(1+3+5+7+10+12+14+16, sum);
    }

}
