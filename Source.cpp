// Muhammad Usman Shahid
//      20I-1797
//        CY-M

// setting the OpenCL version to 1.2
// #define CL_TARGET_OPENCL_VERSION 120

// necessary header files
#include <iostream>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#include <chrono>

using namespace std;

// for printing the matrix
void printMatrix(int* array, int rows, int cols, string name);

// for running code on gpu or CPU in parallel fashion and would return the milliseconds
// if choice is 0 then parallely on CPU else on GPU
int* parallel(int* array_1, int* array_2, int* resultant, const int row, const int col, int choice);

// the serial code
int* serial(int* array_1, int* array_2, int* resultant, const int row, const int col);

int main()
{
    const int row = 512; // size of the array
    const int col = 512;

    srand(time(NULL));

    // the arrays
    // then would map in 1d array
    int* array_1 = new int[row * col];
    int* array_2 = new int[row * col];

    // random initalization of the array
    for (int i = 0; i < row * col; i++)
    {
        array_1[i] = (rand() % 90) + 10;
        array_2[i] = (rand() % 90) + 10;
    }

    // the arrays
    //printMatrix(array_1, row, col, "A");
    //printMatrix(array_2, row, col, "B");

    int* resultant = new int[row * col];

    // ------------------------------------------------------------
    // for Serial
    // -------------------------------------------------------------
    cout << "\n :::::::::: Running as Serial :::::::::\n"
        << endl;
    int* serialTimes = NULL;
    serialTimes = serial(array_1, array_2, resultant, row, col);
    cout << "\n :::::::::::::::::::::::::::::::::::\n"
        << endl;

    // ------------------------------------------------------------
    // for cpu
    // -------------------------------------------------------------
    cout << "\n :::::::::: Running On CPU :::::::::\n"
        << endl;

    int* cpuTimes = NULL;
    cpuTimes = parallel(array_1, array_2, resultant, row, col, 0);

    cout << "\n :::::::::::::::::::::::::::::::::::\n"
        << endl;

    // ------------------------------------------------------------
    // for gpu
    // -------------------------------------------------------------
    cout << "\n :::::::::: Running On Gpu :::::::::\n"
        << endl;
    int* gpuTimes = NULL;
    gpuTimes = parallel(array_1, array_2, resultant, row, col, 1);
    cout << "\n :::::::::::::::::::::::::::::::::::\n"
        << endl;

    
    delete serialTimes;
    delete cpuTimes;
    delete gpuTimes;

    return 0;
}

// for running code on gpu or CPU in parallel fashion and would return the milliseconds
// if choice is 0 then parallely on CPU else on GPU
int* parallel(int* array_1, int* array_2, int* resultant, const int row, const int col, int choice)
{
    // declaring the basic OpenCL variables
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // the memory buffers for processing
    cl_mem A, B, C = NULL;

    // first will count and then will get the ids accordingly
    cl_uint numPlatforms;
    clGetPlatformIDs(0, NULL, &numPlatforms);

    if (numPlatforms == 0) {
        cout<< "No OpenCL platforms found." << endl;
    }

    // allocating memory and getting all platforms ids
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    clGetPlatformIDs(numPlatforms, platforms, NULL); // getting the ids

    // Iterate through each platform
    //for (cl_uint i = 0; i < numPlatforms; ++i) {
    //    cout << "\n--- Platform " << i + 1 << " ---" << endl;

    //    // Get platform name
    //    size_t nameSize;
    //    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &nameSize);
    //    char* platformName = new char[nameSize];
    //    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameSize, platformName, NULL);
    //    cout << "Platform Name: " << platformName << endl;
    //    delete[] platformName;

    //    // Add more information as needed (e.g., CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, etc.)
    //}

    //-- - Platform 1 -- -
    //    Platform Name : Intel(R) OpenCL HD Graphics

    //    -- - Platform 2 -- -
    //    Platform Name : NVIDIA CUDA

    //    -- - Platform 3 -- -
    //    Platform Name : Intel(R) OpenCL

    //    -- - Platform 4 -- -
    //    Platform Name : Intel(R) FPGA Emulation Platform for OpenCL(TM)

    //    -- - Platform 5 -- -
    //    Platform Name : Intel(R) FPGA SDK for OpenCL(TM)

    //    Device Name : Intel(R) Core(TM) i7 - 1065G7 CPU @ 1.30GHz

    //    -- - Doing Multiplication on Intel(R) Core(TM) i7 - 1065G7 CPU @ 1.30GHz-- -

    //    Execution Time : 1056 milliseconds
    //    Execution Time : 1 seconds

       
    // the GPU Code
    // getting platform and device
   /* err = clGetPlatformIDs(3, &platform_id, NULL);
    if (err != CL_SUCCESS)
    {
        cout << "\nError getting platform id\n"
            << endl;
        return 0;
    }*/

    // getting the device
    // 1 for nividia and 2 for cpu
    err = clGetDeviceIDs(choice ? platforms[0] : platforms[2], choice ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        cout << "\nError getting device id\n"
            << endl;
        return 0;
    }

    // printing the device name and info
    char* name = new char[100];
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 100, name, NULL);
    if (err != CL_SUCCESS)
    {
        cout << "\nError getting device info\n"
            << endl;
        return 0;
    }
    cout << "\nDevice Name: " << name << endl;

    cout << "\n --- Doing Multiplication on " << name << " ---\n"
        << endl;
    // creating the context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        cout << "\nError creating\n"
            << endl;
        return 0;
    }

    // creating the command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS)
    {
        cout << "\nError creating command queue\n"
            << endl;
        return 0;
    }

    // creating the program
    const char* source = ""
        " __kernel void mul(__global int* a, __global int* b, __global int* c, int s ) { "
        "int i = get_global_id(0);"
        "int j = get_global_id(1);"
        "int sum = 0;"
        "   for(int f = 0 ; f< s ; f++)"
        "   {"
        "        sum += (a[j * s + f] )* (b[f * s + i]); "
        "   }"
        "c[j * s + i] = sum;"
        "}";

    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating program.");
    }

    // Build the program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error building program.");
    }

    // Create the kernel
    kernel = clCreateKernel(program, "mul", &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating kernel.");
    }

    const int size = row * col;

    // Create memory buffers
    A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(int), array_1, &err);
    B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * sizeof(int), array_2, &err);

    // the resultant
    C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(int), NULL, &err);

    // the size (the row or like that)
    cl_mem s = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (int*)&row, &err);

    // the resultant row element
    // C = clCreateBuffer(context , CL_MEM)
    if (err != CL_SUCCESS)
    {
        printf("Error creating memory buffers.");
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
    err = clSetKernelArg(kernel, 3, sizeof(int), &row);
    if (err != CL_SUCCESS)
    {
        printf("Error setting kernel arguments.");
    }

    const size_t global[2] = { row, col };
    /*const size_t local[2] = { 1, 1 };*/

    // Execute the kernel
    size_t global_size = size;

    // getting the current time
    auto start_time = chrono::high_resolution_clock::now();


    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error executing kernel.");
    }

    // now retriving the results that kernal had written all after processing
    // Read the memory buffer C on the device to the local variable C
    err = clEnqueueReadBuffer(queue, C, CL_TRUE, 0, size * sizeof(int), resultant, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error reading memory buffer.");
        return 0;
    }

    // printing the results
    // printMatrix(resultant, row, col, "Resultant");

    // Clean up
    if (A)
        clReleaseMemObject(A);
    if (B)
        clReleaseMemObject(B);
    if (C)
        clReleaseMemObject(C);
    if (s)
        clReleaseMemObject(s);
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);

    delete[] platforms;

    // getting ending point
    auto end_time = chrono::high_resolution_clock::now();

    // getting time in seconds and milli seconds
    auto milliSecond_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    auto second_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

    // saving the times and would run for making graphs
    int* times = new int[2];
    times[0] = milliSecond_duration.count();
    times[1] = second_duration.count();

    // Print the execution time
    cout << "Execution Time: " << times[0] << " milliseconds" << endl;
    cout << "Execution Time: " << times[1] << " seconds" << endl;

    // returning the time for graphs
    return times;
}

// the serial code
int* serial(int* array_1, int* array_2, int* resultant, const int row, const int col)
{
    // getting the current time
    auto start_time = chrono::high_resolution_clock::now();

    int sum = 0;

    // doing multiplication in serial
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            sum = 0;
            for (int k = 0; k < col; k++)
            {
                sum += array_1[i * col + k ] * array_2[k * row + j];
            }

            // the resultant
            resultant[j * row + i] = sum;
        }

        
    }

    // getting ending point
    auto end_time = chrono::high_resolution_clock::now();

    // getting time in seconds and milli seconds
    auto milliSecond_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    auto second_duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

    // saving the times and would run for making graphs
    int* times = new int[2];
    times[0] = milliSecond_duration.count();
    times[1] = second_duration.count();

    // Print the execution time
    cout << "Execution Time: " << times[0] << " milliseconds" << endl;
    cout << "Execution Time: " << times[1] << " seconds" << endl;

    // returning the time for graphs
    return times;
}

// for printing the matrix
void printMatrix(int* array, int rows, int cols, string name)
{
    cout << "\n:::::::::::::::::::::::::" << endl;
    cout << "\nMatrix " << name << " is : \n " << endl;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            cout << *(array + i * cols + j) << " ";
        }

        cout << endl;
    }
    cout << "\n::::::::::::::::::::::::: \n"
        << endl;
}