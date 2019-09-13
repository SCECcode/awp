#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <test/test.h>
#include <topography/input/version.h>
#include <awp/error.h>


int check_version(void);
int check_equals(void);
int check_greater_than(void);
int check_compatible(void);
int check_to_string(void);
int check_broadcast(int rank, int size);

int main(int argc, char **argv)
{
        int err = 0;
        int rank, size;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
                test_divider();
                test_divider();
                printf("Testing version.c\n");
                printf("Running tests:\n");
                err |= check_version();
                err |= check_equals();
                err |= check_greater_than();
                err |= check_compatible();
                err |= check_to_string();
        }
        err |= check_broadcast(rank, size);

        if (rank == 0) {
                printf("Testing completed.\n");
                test_divider();
        }

        MPI_Finalize();
        return err;
}


int check_version(void)
{
        int err = 0;
        test_t test = test_init(" * version_init", 0, 0);
        {
        version_t ver;
        char version[80] = "1.0.0";
        version_init(&ver, version);
        err |= s_assert(ver.major == 1);
        err |= s_assert(ver.minor == 0);
        err |= s_assert(ver.patch == 0);
        }

        {
        version_t ver;
        char version[80] = "1.10.20";
        version_init(&ver, version);
        err |= s_assert(ver.major == 1);
        err |= s_assert(ver.minor == 10);
        err |= s_assert(ver.patch == 20);
        }

        err = test_finalize(&test, err);
        return err;
}

int check_equals(void)
{
        int err = 0;
        test_t test = test_init(" * version_equals", 0, 0);
        {
                version_t a = {.major = 1, .minor = 0, .patch = 0};
                version_t b = {.major = 1, .minor = 0, .patch = 0};
                err |= s_assert(version_equals(&a, &b));
        }
        
        err = test_finalize(&test, err);
        return err;
}

int check_greater_than(void)
{
        int err = 0;
        test_t test = test_init(" * version_greater_than", 0, 0);
        {
                version_t a = {.major = 1, .minor = 0, .patch = 0};
                version_t b = {.major = 0, .minor = 0, .patch = 0};
                err |= s_assert(version_greater_than(&a, &b));
        }

        {
                version_t a = {.major = 1, .minor = 1, .patch = 0};
                version_t b = {.major = 1, .minor = 0, .patch = 0};
                err |= s_assert(version_greater_than(&a, &b));
        }

        {
                version_t a = {.major = 1, .minor = 1, .patch = 1};
                version_t b = {.major = 1, .minor = 1, .patch = 0};
                err |= s_assert(version_greater_than(&a, &b));
        }

        // Should fail a == b
        {
                version_t a = {.major = 1, .minor = 1, .patch = 1};
                version_t b = {.major = 1, .minor = 1, .patch = 1};
                err |= s_no_except(version_greater_than(&a, &b) == 0);
        }

        err = test_finalize(&test, err);
        return err;
}

int check_compatible(void)
{
        int err = 0;
        test_t test = test_init(" * version_compatible", 0, 0);
        {
                version_t a = {.major = 1, .minor = 0, .patch = 0};
                version_t b = {.major = 1, .minor = 0, .patch = 0};
                err |= s_assert(version_compatible(&a, &b));
        }

        // Should fail, major versions are not the same (not backwards
        // compatible)
        {
                version_t a = {.major = 2, .minor = 0, .patch = 0};
                version_t b = {.major = 1, .minor = 0, .patch = 0};
                err |= s_no_except(version_compatible(&a, &b) == 0);
        }

        err = test_finalize(&test, err);
        return err;
}

int check_to_string(void)
{
        int err = 0;
        test_t test = test_init(" * version_to_string", 0, 0);

        version_t a = {.major = 1, .minor = 2, .patch = 3};
        char str_a[80];
        int num_written = version_to_string(&a, str_a);
        err |= s_assert(num_written > 0);
        err |= s_assert(strcmp(str_a, "1.2.3") == 0);

        err = test_finalize(&test, err);

        return err;
}

int check_broadcast(int rank, int size)
{
        int err = 0;
        test_t test = test_init(" * version_broadcast", rank, size);
        version_t a;
        if (rank == 0) {
                a.major = 1;
                a.minor = 2;
                a.patch = 3;
        }
        err = version_broadcast(&a, 0, MPI_COMM_WORLD);
        err |= s_assert(a.major == 1);
        err |= s_assert(a.minor == 2);
        err |= s_assert(a.patch == 3);
        err = test_finalize(&test, err);

        return err;
}

