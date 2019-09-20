#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <test/test.h>
#include <readers/input.h>
#include <readers/error.h>

#define STR_LEN 2048


int check_file_readable(void);
int check_file_writeable(void);
int check_write(const char *filename);
int check_init(const char *filename);
int check_header(const char *filename);
int test_parse_arg(void);
int test_parse(void);
int test_broadcast(int rank, int size, const char *filename);

int main(int argc, char **argv)
{
        char testfile[STR_LEN];

        int err = 0;
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (argc == 2) {
                assert(strlen(argv[1]) < STR_LEN);
                sprintf(testfile, "%s", argv[1]);
        }
        else {
                sprintf(testfile, "../tests/fixtures/input1.txt");
        }

        if (rank == 0) {
                test_divider();
        }
        print_master("Testing input.c\n");
        print_master("Test file: %s\n", testfile);
        print_master("\n");
        print_master("Running tests:\n");

        if (rank == 0) {
                err |= test_parse_arg();
                err |= test_parse();
                err = check_init(testfile);
                error_print(err);
                err |= check_header(testfile);
                err |= check_file_readable();
                err |= check_file_writeable();
                err |= check_write(testfile);
        }

        err |= test_broadcast(rank, size, testfile);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();
        return err;
}

int check_header(const char *filename)
{
        int err = 0;
        test_t test = test_init(" * input_check_header", 0, 0);

        input_t input;
        err |= input_init(&input, filename);
        err |= input_check_header(&input);
        input.steps = 101;
        err |= s_assert(input_check_header(&input) ==
                        ERR_CONFIG_PARSE_NOT_DIVISIBLE);

        test_finalize(&test, err);

        return err;
}

int check_file_readable(void)
{
        int err = 0;
        test_t test = test_init(" * input_file_readable", 0, 0);

        FILE *fp;
        // file exists
        fp = fopen("tmp", "w");
        err |= s_assert(input_file_readable("tmp") == 0);
        remove("tmp");
        fclose(fp);
        error_print(err);

        //file does not exist
        err |= s_assert(input_file_readable("tmp") == ERR_FILE_READ);
        assert(err == 0 && "Test failed because the file 'tmp' exists. \
                            Delete/rename this file and try again");
        test_finalize(&test, err);


        return err;
}

int check_file_writeable(void)
{
        FILE *fp;
        int err = 0;
        test_t test = test_init(" * input_file_writeable", 0, 0);
        // file is writeable
        fp = fopen("tmp", "w");
        err |= s_assert(input_file_writeable("tmp") == 0);
        error_print(err);
        fclose(fp);
        remove("tmp");

        // file is not writeable
        fp = fopen("tmp/tmp", "w");
        err |= s_assert(input_file_writeable("tmp/tmp") == ERR_FILE_WRITE);
        assert(err == 0 && "Test failed because the directory 'tmp' exists. \
                            Delete/rename this directory and try again.");
        
        test_finalize(&test, err);

        return err;
}

int check_init(const char *filename)
{
        int err = 0;
        test_t test = test_init(" * input_init", 0, 0);
        input_t out;
        err |= input_init(&out, filename);
        error_print(err);
        assert(err == 0);
        error_print(err);
        assert(err == 0);

        input_finalize(&out);
        test_finalize(&test, err);

        return err;
}

int check_write(const char *testfile)
{
        int err = 0;
        test_t test = test_init(" * input_write", 0, 0);
        input_t out;
        err |= input_init(&out, testfile);
        error_print(err);
        assert(err == 0);

        err = s_assert(input_write(&out, "input.txt") == 0);  

        //Read it back in
        input_t in;
        err = input_init(&in, "input.txt");
        error_print(err);
        remove("input.txt");
        err |= s_assert(input_equals(&in, &out));

        input_finalize(&in);
        input_finalize(&out);

        error_print(err);

        test_finalize(&test, err);
        return err;
}

int test_parse_arg(void)
{
        int err = 0;
        test_t test = test_init(" * input_parse_arg", 0, 0);

        char variable[2048];
        char value[2048];

        // Wrong delimiter
        err |= s_assert(input_parse_arg(variable, value, "a:b") ==
                        ERR_CONFIG_PARSE_ARG);
        // Missing variable
        err |= s_assert(input_parse_arg(variable, value, "=b") ==
                        ERR_CONFIG_PARSE_ARG);
        // Missing value
        err |= s_assert(input_parse_arg(variable, value, "a=") ==
                        ERR_CONFIG_PARSE_ARG);
        err |= input_parse_arg(variable, value, "test=a");
        err |= s_assert(strcmp("test", variable) == 0);
        err |= s_assert(strcmp("a", value) == 0);

        err = test_finalize(&test, err);
        return test_last_error();
}

int test_parse(void)
{
        int err = 0;
        test_t test = test_init(" * input_parse", 0, 0);
        
        input_t out;

        err |= input_parse(&out, "file=test");
        err |= s_assert(strcmp(out.file, "test") == 0);
        err |= input_parse(&out, "length=2");
        err |= s_assert(out.length == 2);
        err |= input_parse(&out, "stride=10");
        err |= s_assert(out.stride == 10);
        err |= input_parse(&out, "degree=3");
        err |= s_assert(out.degree == 3);
        err |= input_parse(&out, "steps=1e3");
        err |= s_assert(out.steps == 1000);
        err |= input_parse(&out, "num_writes=10");
        err |= s_assert(out.num_writes == 10);
        err |= input_parse(&out, "gpu_buffer_size=10");
        err |= s_assert(out.gpu_buffer_size == 10);
        err |= input_parse(&out, "cpu_buffer_size=100");
        err |= s_assert(out.cpu_buffer_size == 100);

        err = test_finalize(&test, err);
        return test_last_error();
}

int test_broadcast(int rank, int size, const char *filename)
{
        int err = 0;
        test_t test = test_init(" * input_broadcast", rank, size);
        input_t out;
        input_t ans;
        err |= input_init(&ans, filename);
        err |= mpi_assert(err == 0, rank);
        err |= input_init_default(&out);
        err |= mpi_assert(err == 0, rank);

        if (rank == 0) {
                err |= input_init(&out, filename);
                err |= mpi_assert(err == 0, rank);
        } 

        err |= input_broadcast(&out, rank, 0, MPI_COMM_WORLD);
        err |= mpi_assert(err == 0, rank);
        err |= mpi_assert(input_equals(&ans, &out) == 1, rank);

        input_finalize(&out);
        input_finalize(&ans);

        err = test_finalize(&test, err);
        return err;
}

