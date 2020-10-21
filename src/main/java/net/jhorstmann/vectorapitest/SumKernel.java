package net.jhorstmann.vectorapitest;

import jdk.incubator.vector.*;

public class SumKernel {
    private static final VectorSpecies<Double> F64X4 = DoubleVector.SPECIES_256;
    private static final VectorSpecies<Long> I64X4 = LongVector.SPECIES_256;
    private static final VectorSpecies<Integer> I32X8 = IntVector.SPECIES_256;
    private static final LongVector IDX_I64X4 = LongVector.fromArray(I64X4, new long[]{8, 4, 2, 1}, 0);
    private static final DoubleVector IDX_F64X4 = IDX_I64X4.reinterpretAsDoubles();

    private static final IntVector IDX_I32X8 = IntVector.fromArray(I32X8, new int[]{128, 64, 32, 16, 8, 4, 2, 1}, 0);
    private static final FloatVector IDX_F32X8 = IDX_I32X8.reinterpretAsFloats();

    private static VectorMask<Long> maskFromBits(long mask) {
        var broadcast = LongVector.broadcast(I64X4, mask & 0x0F);
        return IDX_I64X4.and(broadcast).eq(IDX_I64X4);
    }

    private static DoubleVector maskedSum(DoubleVector accumulator, double[] data, int i, long mask) {
        var vmask = maskFromBits(mask);

        var v1 = DoubleVector.fromArray(F64X4, data, i);

        // blend as longs in order to use VectorMask<Long>, which should be slightly faster to compute
        var blended = v1.reinterpretAsLongs()
                .blend(Double.doubleToLongBits(0.0), vmask)
                .reinterpretAsDoubles();

        return accumulator.add(blended);
    }

    public static double sum(double[] data, byte[] valid) {

        int size = data.length;

        if (size == 0) {
            return 0.0;
        }

        int vsize = size & ~7;

        if (valid.length < (size + 7) >>> 3) {
            throw new IllegalArgumentException("invalid mask len");
        }

        var vsum1 = DoubleVector.broadcast(F64X4, 0.0);
        var vsum2 = DoubleVector.broadcast(F64X4, 0.0);
        for (int i = 0; i < vsize; i += 8) {
            long mask = valid[i >>> 3] & 0xFF;

            vsum1 = maskedSum(vsum1, data, i, mask);
            vsum2 = maskedSum(vsum2, data, i + 4, mask >>> 4);
        }

        double total = vsum1.reduceLanes(VectorOperators.ADD);
        total += vsum2.reduceLanes(VectorOperators.ADD);

        for (int i = vsize; i < size; i++) {
            long mask = valid[i / 8] & 0xFF;
            if ((mask & (1 << (i % 8))) != 0) {
                total += data[i];
            }
        }

        return total;
    }

    public static double sum(double[] data) {
        int size = data.length;

        if (size == 0) {
            return 0.0;
        }

        int vsize = size & ~7;

        var vsum1 = DoubleVector.broadcast(F64X4, 0.0);
        var vsum2 = DoubleVector.broadcast(F64X4, 0.0);
        for (int i = 0; i < vsize; i += 8) {
            DoubleVector v1 = DoubleVector.fromArray(F64X4, data, i);
            DoubleVector v2 = DoubleVector.fromArray(F64X4, data, i + 4);
            vsum1 = vsum1.add(v1);
            vsum2 = vsum2.add(v2);
        }

        double total = vsum1.reduceLanes(VectorOperators.ADD);
        total += vsum2.reduceLanes(VectorOperators.ADD);

        for (int i = vsize; i < size; i++) {
            total += data[i];
        }

        return total;
    }
}
