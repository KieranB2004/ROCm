double *deviceA1;
double *deviceB1;
double *deviceA2;
double *deviceB2;
double *A1;
double *A2;
double *B1;
double *B2;

hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

HIP_ASSERT(hipMemcpy(&deviceA1, &A1, arraySize, hipMemcpyHostToDevice));
HIP_ASSERT(hipMemcpy(&deviceA2, &A2, arraySize, hipMemcpyHostToDevice));


