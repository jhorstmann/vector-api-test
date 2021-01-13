package net.jhorstmann.vectorapitest;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddKernelTest {
    @Test
    void testSum() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {10, 20, 30, 40, 50};
        double[] c = new double[a.length];

        AddKernel.add(a, b, c);

        double[] expected = {11, 22, 33, 44, 55};

        assertArrayEquals(expected, c);
    }

    @Test
    void testSumMasked() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {10, 20, 30, 40, 50};
        double[] c = new double[a.length];

        AddKernel.addMasked(a, b, c);

        double[] expected = {11, 22, 33, 44, 55};

        assertArrayEquals(expected, c);
    }

}
