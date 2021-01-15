package net.jhorstmann.vectorapitest;

import java.util.concurrent.ThreadLocalRandom;

import net.jhorstmann.vectorapitest.SumBenchmark.Input;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.profile.LinuxPerfAsmProfiler;
import org.openjdk.jmh.profile.LinuxPerfNormProfiler;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

@BenchmarkMode(Mode.Throughput)

public class AddKernelBenchmark {

    @State(Scope.Benchmark)
    public static class Input {

        @Param({ "1000000" })
        int size;
        double[] left;
        double[] right;
        double[] result;

        @Setup
        public void setup() {
            var random = ThreadLocalRandom.current();

            left = new double[size];
            right = new double[size];
            result = new double[size];

            for (int i = 0; i < size; i++) {
                left[i] = random.nextInt(10);
                right[i] = random.nextInt(10);
                result[i] = 0.0;
            }
        }
    }

    //@Benchmark
    public void benchmarkAddScalar(Input input, Blackhole blackhole) {
        AddKernel.addScalar(input.left, input.right, input.result);
        blackhole.consume(input.result);
    }

    @Benchmark
    public void benchmarkAddWithScalarRemainder(Input input, Blackhole blackhole) {
        AddKernel.add(input.left, input.right, input.result);
        blackhole.consume(input.result);
    }

    //@Benchmark
    public void benchmarkAddWithRangeMask(Input input, Blackhole blackhole) {
        AddKernel.addMasked(input.left, input.right, input.result);
        blackhole.consume(input.result);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .jvmArgs("--add-modules=jdk.incubator.vector",
                        "-XX:MaxInlineLevel=32",
                        "-XX:+UnlockExperimentalVMOptions",
                        "-XX:+UnlockDiagnosticVMOptions",
                        "-XX:+TrustFinalNonStaticFields",
                        //"-XX:+PrintAssembly",
                        //"-XX:PrintAssemblyOptions=intel",
                        //"-XX:CompileCommand=print,net.jhorstmann.vectorapitest.AddKernel.*",
                        // disable autovectorization to compare against pure scalar code
                        "-XX:-UseSuperWord")
                .include(AddKernelBenchmark.class.getSimpleName())
                //.addProfiler(LinuxPerfAsmProfiler.class)
                //.addProfiler(LinuxPerfNormProfiler.class)
                .threads(1)
                .forks(1)
                .build();

        new Runner(opt).run();
    }

}
