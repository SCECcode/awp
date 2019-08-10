#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

#include <test/test.h>
#include <awp/error.h>


int LASTERR = 0;

void repeat_char(char *label, const int n, const char *cr);
void format_label(const char *label, char *new_label);

void print_status(int x) 
{
        if (x == 0) {
                printf("%sPASS%s", KGRN, KNRM);
        } 
        else {
                printf("%sFAIL%s", KRED, KNRM);
        }
}

test_t _test_init(const char *label, int line, int rank, int size) 
{
        test_t out = {.label = label, .rank = rank, .size = size, 
                      .line = line,  
                      .start = clock()};
        return out;
}

int test_finalize(test_t *test, int x) 
{
        if (test->size != 0) {
                x = test_all(x, test->rank);
        }
        clock_t t = clock() - test->start;
        test->elapsed = 1e3*((double)t)/CLOCKS_PER_SEC;
        double total = 0.0;
        if (test->size != 0) {
                MPI_Reduce(&test->elapsed, &total, 1, MPI_DOUBLE, MPI_SUM, 0,
                              MPI_COMM_WORLD);
        } 
        else {
                total = test->elapsed;
        }

        if (test->rank != 0) {
                return x;
        }

        double average = total;
        if (test->size != 0) {
                average = total/test->size;
        }
        char formatted_label[LABEL_MAX_LEN];
        char space[DISP_MAX_LEN];
        format_label(test->label, formatted_label);
        repeat_char(space, DISP_MAX_LEN - strlen(formatted_label), " ");

        if (ADDLINENUM) {
                printf(LINEFORMAT ": ", test->line);
        }
        printf("%s", formatted_label);
        printf("%s ", space);
        printf("%.2g ms \t", average);
        print_status(x);
        printf("\n");
        LASTERR = x;
        return x;
}

int test_last_error(void)
{
        return LASTERR;
}


int test_all(int x, int rank) 
{
        int sum;
        MPI_Allreduce(&x, &sum, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (sum == 0) {
                return SUCCESS;
        } 
        else {
                return ERR_TEST_FAILED;
        }
}

void test_divider(void)
{
        char div[DIVIDER_LEN+1];
        repeat_char(div, DIVIDER_LEN, "=");
        printf("%s\n", div);
}

void format_label(const char *label, char *new_label)
{
        if (strlen(label) >= LABEL_MAX_LEN) {
                int n = LABEL_MAX_LEN;
                for (int i = 0; i < n - 3; ++i) {
                        new_label[i] = label[i];
                }
                new_label[n - 3] = '.';
                new_label[n - 2] = '.';
                new_label[n - 1] = '.';
                new_label[n] = '\0';
        } 
        else {
                sprintf(new_label, "%s", label);
        }
}

void repeat_char(char *label, const int n, const char *cr)
{
        for (int i = 0; i < n; ++i) {
                label[i] = cr[0];
        }
        label[n] = '\0';
}

