package net.jhorstmann.vectorapitest;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.profile.LinuxPerfAsmProfiler;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.ThreadLocalRandom;

@BenchmarkMode(Mode.AverageTime)
public class SumBenchmark {


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
        return SumKernel.sum(input.data);
    }

    @Benchmark
    public double maskedSum(Input input) {
        return SumKernel.sum(input.data, input.valid);
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
