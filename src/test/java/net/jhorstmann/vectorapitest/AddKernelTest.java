package net.jhorstmann.vectorapitest;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AddKernelTest {
    @Test
    void testAdd() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {10, 20, 30, 40, 50};
        double[] result = new double[a.length];

        AddKernel.add(a, b, result);

        double[] expected = {11, 22, 33, 44, 55};

        assertArrayEquals(expected, result);
    }

    @Test
    void testAddMasked() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {10, 20, 30, 40, 50};
        double[] result = new double[a.length];

        AddKernel.addMasked(a, b, result);

        double[] expected = {11, 22, 33, 44, 55};

        assertArrayEquals(expected, result);
    }

    @Test
    void testAddIfSmaller() {
        double[] a = {1, 2, 3, 4, 5};
        double[] b = {10, 20, 30, 40, 50};
        double[] result = new double[a.length];

        AddKernel.addIfSmaller(a, b, 4, result);

        double[] expected = {11, 22, 33, 4, 5};

        assertArrayEquals(expected, result);
    }

    @Test
    void testAddShuffled() {
        double[] a = {1, 2, 3, 4};
        double[] b = {40, 30, 20, 10};
        double[] result = new double[a.length];

        AddKernel.addShuffled(a, b, result);

        double[] expected = {11, 22, 33, 44};

        assertArrayEquals(expected, result);
    }

}
