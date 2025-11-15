add_executable(karl-cute-gemm-perf-enhance-test cute-gemm-perf-enhance-test.cu)
target_link_libraries(karl-cute-gemm-perf-enhance-test PUBLIC -lcublas -lcudart -lcublasLt)