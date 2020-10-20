package net.jhorstmann.vectorapitest;

import jdk.incubator.vector.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.profile.LinuxPerfAsmProfiler;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.ThreadLocalRandom;

@BenchmarkMode(Mode.AverageTime)
public class SumBenchmark {

    public static final VectorSpecies<Double> F64X4 = DoubleVector.SPECIES_256;
    public static final VectorSpecies<Double> F64X8 = DoubleVector.SPECIES_512;
    public static final VectorSpecies<Long> I64X4 = LongVector.SPECIES_256;
    public static final VectorSpecies<Long> I64X8 = LongVector.SPECIES_512;
    public static final DoubleVector IDX_F64X4 = DoubleVector.fromArray(F64X4, new double[]{8, 4, 2, 1}, 0);
    public static final LongVector IDX_I64X4 = LongVector.fromArray(I64X4, new long[]{8, 4, 2, 1}, 0);

    @State(Scope.Benchmark)
    public static class Input {
        @Param({"100000000"})
        int size;
        double[] data;
        byte[] valid;

        @Setup
        public void setup() {
            var random = ThreadLocalRandom.current();

            data = new double[size];

            for (int i = 0; i < size; i++) {
                data[i] = random.nextInt(10);
            }

            valid = new byte[(size + 7) / 8];
            for (int i = 0; i < size; i++) {
                if (i % 2 == 0 || i % 3 == 0) {
                    valid[i / 8] |= 1 << (i % 8);
                }
            }
        }
    }

    @Benchmark
    public double sum(Input input) {
        int size = input.size;
        int vsize = size & ~7;
        int rsize = size & 7;

        double[] data = input.data;
        var vsum = DoubleVector.broadcast(F64X4, 0.0);
        int i = 0;
        for (; i < vsize; i += F64X4.length()) {
            DoubleVector v1 = DoubleVector.fromArray(F64X4, data, i);
            vsum = vsum.add(v1);
        }

        double total = vsum.reduceLanes(VectorOperators.ADD);
        for (; i < rsize; i++) {
            total += data[i];
        }

        return total;
    }

    private static VectorMask<Double> maskFromBits(long mask) {
        var vecidx = IDX_I64X4;
        var broadcast = LongVector.broadcast(I64X4, mask & 0x0F);
        return vecidx.and(broadcast).reinterpretAsDoubles().eq(IDX_F64X4);
    }

    private static DoubleVector maskedSum(DoubleVector accumulator, double[] data, int i, long mask) {
        var vmask = maskFromBits(mask);

        var v1 = DoubleVector.fromArray(F64X4, data, i, vmask);
        return accumulator.add(v1);
    }

    @Benchmark
    public double maskedSum(Input input) {
        int size = input.size;
        int vsize = size & ~7;
        int rsize = size & 7;

        double[] data = input.data;
        byte[] valid = input.valid;

        if (data.length != size) {
            throw new IllegalArgumentException("invalid data len");
        }

        if (valid.length * 8 < size) {
            throw new IllegalArgumentException("invalid mask len");
        }

        var vsum1 = DoubleVector.broadcast(F64X4, 0.0);
        var vsum2 = DoubleVector.broadcast(F64X4, 0.0);
        int i = 0;
        for (; i < vsize; i += 2 * F64X4.length()) {
            long mask = valid[i / 8] & 0xFF;

            vsum1 = maskedSum(vsum1, data, i, mask);
            vsum2 = maskedSum(vsum2, data, i + F64X8.length(), mask >> 4);
        }

        double total = vsum1.reduceLanes(VectorOperators.ADD) + vsum2.reduceLanes(VectorOperators.ADD);
        if (rsize > 0) {
            int rvalid = valid[i / 8] & 0xFF;
            for (; i < rsize; i++) {
                if ((rvalid & (1 << (i % 8))) != 0) {
                    total += data[i];
                }
            }
        }

        return total;
    }


    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .jvmArgs("--add-modules=jdk.incubator.vector", "-XX:MaxInlineLevel=32", "-XX:+UnlockExperimentalVMOptions", "-XX:+TrustFinalNonStaticFields")
                .include(SumBenchmark.class.getSimpleName())
                .addProfiler(LinuxPerfAsmProfiler.class)
                .threads(1)
                .forks(1)
                .build();

        new Runner(opt).run();
    }
}
