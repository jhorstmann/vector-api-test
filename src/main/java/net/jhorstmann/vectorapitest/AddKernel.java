package net.jhorstmann.vectorapitest;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class AddKernel {
    private static final VectorSpecies<Double> F64X4 = DoubleVector.SPECIES_256;

    public static void add(double[] a, double[] b, double[] result) {
        assert (a.length == b.length);
        assert (a.length == result.length);
        int len = a.length;
        int i = 0;

        // vectorized loop
        for (; i < F64X4.loopBound(len); i += F64X4.length()) {
            var v1 = DoubleVector.fromArray(F64X4, a, i);
            var v2 = DoubleVector.fromArray(F64X4, b, i);
            v1.add(v2).intoArray(result, i);
        }

        // scalar loop for the remaining elements
        for (; i < len; i++) {
            result[i] = a[i] + b[i];
        }
    }

    public static void addMasked(double[] a, double[] b, double[] result) {
        assert (a.length == b.length);
        assert (a.length == result.length);
        int len = a.length;
        for (int i = 0; i < len; i += F64X4.length()) {
            // builds a VectorMask that will ignore the out of bounds elements when loading/storing
            var mask = F64X4.indexInRange(i, len);
            var v1 = DoubleVector.fromArray(F64X4, a, i, mask);
            var v2 = DoubleVector.fromArray(F64X4, b, i, mask);

            v1.add(v2).intoArray(result, i, mask);
        }
    }

}
