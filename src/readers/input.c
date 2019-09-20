#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

#include <readers/input.h>
#include <awp/error.h>
#include <test/test.h>

#define DATA_ELEMENT_LENGTH 1024

int _input_read(input_t *out, const char *filename);
int _read_header(input_t *out, FILE *fp);
int _malloc_data(input_t *out);
int _read_data(input_t *out, FILE *fp);

int input_init_default(input_t *out)
{
        out->gpu_buffer_size = 0;
        out->cpu_buffer_size = 0;
        out->degree = 0;
        out->stride = 1;
        out->steps = 1;
        out->num_writes = 1;
        sprintf(out->file, "default");
        out->x = NULL;
        out->y = NULL;
        out->z = NULL;
        out->type = NULL;

        return SUCCESS;
}

int input_init(input_t *out, const char *filename)
{
        int err = 0;
        err |= input_init_default(out);

        err = _input_read(out, filename);

        return err;
}

int input_write(input_t *out, const char *filename)
{
        int err = input_file_writeable(filename);
        if (err > 0) {
                return err;
        }
        FILE *fp = fopen(filename, "w");


        char str_version[VERSION_STRING_LENGTH];
        int num_chars = version_to_string(&out->version, str_version);
        if (num_chars == 0) {
                return ERR_WRONG_VERSION;
        }

        // Header
        fprintf(fp,"%s\n", str_version);
        fprintf(fp,"file=%s\n", out->file);
        fprintf(fp,"length=%lu\n", out->length);
        fprintf(fp, "gpu_buffer_size=%lu\n", out->gpu_buffer_size);
        fprintf(fp, "cpu_buffer_size=%lu\n", out->cpu_buffer_size);
        fprintf(fp, "steps=%lu\n", out->steps);
        fprintf(fp, "num_writes=%lu\n", out->num_writes);
        fprintf(fp, "stride=%d\n", out->stride);
        fprintf(fp, "degree=%d\n", out->degree);
        fprintf(fp, "coordinates\n");

        // Data
        for (size_t i = 0; i < out->length; ++i) {
                fprintf(fp, "%d %g %g %g\n", out->type[i], out->x[i], out->y[i],
                        out->z[i]);
        }

        fclose(fp);
        return SUCCESS;
}

// Configure interpolation
void input_buffer(void);

// Clear input and write to disk
void input_flush(void);

void input_finalize(input_t *out)
{
        if(out->type != NULL) free(out->type);
        if(out->x != NULL) free(out->x);
        if(out->y != NULL) free(out->y);
        if(out->z != NULL) free(out->z);
}

int input_file_readable(const char *filename)
{
       // Check if the file exists
       FILE *fh = fopen(filename, "r");
       if (fh == NULL) {
               return ERR_FILE_READ;
       } 
       fclose(fh);
       return SUCCESS;
}

int input_file_writeable(const char *filename)
{
       // Check if the file exists, and if it does conclude that it is possible
       // to write to this file.
       int err = input_file_readable(filename);
       if (err == 0) {
               return SUCCESS;
       }

       // Try creating the file, and remove it on success.
       FILE *fh = fopen(filename, "w");
       if (fh == NULL) {
               return ERR_FILE_WRITE;
       } else {
               remove(filename);
               fclose(fh);
       }
       return SUCCESS;
}

int input_parse(input_t *out, const char *line)
{

        char variable[INPUT_DATA_STRING_LENGTH];
        char value[INPUT_DATA_STRING_LENGTH];
        int err = input_parse_arg(variable, value, line);
        if (err != SUCCESS) {
                fprintf(stderr, "Failed to parse: %s \n", line);
                return err;
        }
        if (strcmp(variable, "file") == 0) {
                strcpy(out->file, value);
        }
        // use atof so that use can write e.g., 1e3
        else if (strcmp(variable, "length") == 0) {
                out->length = (size_t)atof(value);    
        }
        else if (strcmp(variable, "stride") == 0) {
                out->stride = (size_t)atof(value);
        }
        else if (strcmp(variable, "steps") == 0) {
                out->steps = (size_t)atof(value);
        }
        else if (strcmp(variable, "num_writes") == 0) {
                out->num_writes = (size_t)atof(value);
        }
        else if (strcmp(variable, "degree") == 0) {
                out->degree = atoi(value);
        }
        else if (strcmp(variable, "gpu_buffer_size") == 0) {
                out->gpu_buffer_size = atoi(value);
        }
        else if (strcmp(variable, "cpu_buffer_size") == 0) {
                out->cpu_buffer_size = atoi(value);
        }
        else {
                fprintf(stderr, "Unknown argument: %s \n", variable);
                return ERR_CONFIG_PARSE_UNKNOWN_ARG;
        }
        return SUCCESS;
}

int input_parse_arg(char *variable, char *value, const char *line)
{
        char delim[] = "=";
        char *str = malloc(sizeof str * strlen(line));
        strcpy(str, line);

        char *ptr = strtok(str, delim);

        if (ptr == NULL) { 
                free(str);
                return ERR_CONFIG_PARSE_ARG;
        }

        strcpy(variable, ptr);
        ptr = strtok(NULL, delim);

        if (ptr == NULL) {
                free(str);
                return ERR_CONFIG_PARSE_ARG;
        }
        strcpy(value, ptr);

        free(str);

        return SUCCESS;
}

int input_equals(input_t *a, input_t *b)
{
        int equals = 1;
        equals &= version_equals(&a->version, &b->version);
        equals &= strcmp(a->file, b->file) == 0;
        equals &= a->length == b->length;
        equals &= a->gpu_buffer_size == b->gpu_buffer_size;
        equals &= a->cpu_buffer_size == b->cpu_buffer_size;
        equals &= a->steps == b->steps;
        equals &= a->num_writes == b->num_writes;
        equals &= a->stride == b->stride;
        equals &= a->degree == b->degree;
        if (!equals)
                return equals;
        return equals;
}

int input_broadcast(input_t *out, int rank, int root, MPI_Comm communicator)
{
        int err = 0;

        if (rank != root) {
                input_init_default(out);
        }

        // Header section
        err = version_broadcast(&out->version, root, communicator);
        int datafile_len = strlen(out->file);
        err |= MPI_Bcast(&datafile_len, 1, MPI_INT, root, communicator);
        err |= MPI_Bcast(&out->file, datafile_len + 1, MPI_BYTE, root,
                         communicator);
        err |= MPI_Bcast(&out->length, 1, MPI_AINT, root, communicator);
        err |= MPI_Bcast(&out->gpu_buffer_size, 1, MPI_AINT, root, communicator);
        err |= MPI_Bcast(&out->cpu_buffer_size, 1, MPI_AINT, root, communicator);
        err |= MPI_Bcast(&out->steps, 1, MPI_AINT, root, communicator);
        err |= MPI_Bcast(&out->num_writes, 1, MPI_AINT, root, communicator);
        err |= MPI_Bcast(&out->stride, 1, MPI_INT, root, communicator);
        err |= MPI_Bcast(&out->degree, 1, MPI_INT, root, communicator);

        if (err > 0) {
                return ERR_CONFIG_BROADCAST;
        }
        
        if (rank != root) {
                err = _malloc_data(out);
        }

        // Data section
        err |= MPI_Bcast(out->type, out->length, MPI_PREC, root, communicator);
        err |= MPI_Bcast(out->x, out->length, MPI_PREC, root, communicator);
        err |= MPI_Bcast(out->y, out->length, MPI_PREC, root, communicator);
        err |= MPI_Bcast(out->z, out->length, MPI_PREC, root, communicator);

        if (err > 0) {
                return ERR_CONFIG_BROADCAST;
        }
        return SUCCESS;
}

int _input_read(input_t *out, const char *filename)
{
        int err = 0;
        FILE *fp;
        fp = fopen(filename, "r");
        if (!fp) {
                return ERR_FILE_READ;
        }
        
        err = _read_header(out, fp);
        if (err > 0) {
                fclose(fp);
                return err;
        }

        err = input_check_header(out);
        if (err > 0) {
                fclose(fp);
                return err;
        }

        err = _malloc_data(out);
        if (err > 0) {
                fclose(fp);
                return err;
        }

        err = _read_data(out, fp);
        if (err > 0) {
                fclose(fp);
                return err;
        }
        
        fclose(fp);

        return err;
}

int _read_header(input_t *out, FILE *fp)
{
        int err = version_from_file(&out->version, fp);
        if (err > 0) {
                fclose(fp);
                return err;
        }

        char current_string[VERSION_STRING_LENGTH];

        sprintf(current_string, "%d.%d.%d", INPUT_MAJOR, INPUT_MINOR,
                                            INPUT_PATCH);
        err = version_check_compatibility((const version_t *)&out->version,
                                          current_string);
        if (err > 0) {
                fclose(fp);
                return err;
        }

        int parse_arg = SUCCESS;
        char line[INPUT_DATA_STRING_LENGTH];
        int match = 1;
        while (parse_arg == SUCCESS && match == 1) {
                match = fscanf(fp, "%s\n", line);
                if (strcmp(line, "coordinates") == 0) {
                        break;
                }
                parse_arg = input_parse(out, line);
        }

        err = input_file_writeable(out->file);
        if (err > 0) {
                fclose(fp);
                return err;
        }


        return parse_arg;
}

int input_check_header(const input_t *out)
{
        // Buffer size must be divisible by number of steps in file
        // it would be nice if this was not a requirement. The reader that reads
        // the binary data would then need to adjust the amount of data it reads
        // at the final read.
        if (out->steps > 1 &&
            out->steps % (out->gpu_buffer_size * out->cpu_buffer_size) != 0) {
                return ERR_CONFIG_PARSE_NOT_DIVISIBLE;
        }
        return SUCCESS;
}

int _malloc_data(input_t *out)
{

        assert(!out->type && !out->x && !out->y && !out->z);

        out->type = malloc(sizeof(out->type) * out->length);
        out->x = malloc(sizeof(out->x) * out->length);
        out->y = malloc(sizeof(out->y) * out->length);
        out->z = malloc(sizeof(out->z) * out->length);
        if (!out->x || !out->y || !out->z) {
                return ERR_CONFIG_DATA_MALLOC;
        }
        return SUCCESS;
}

int _read_data(input_t *out, FILE *fp)
{
        //FIXME: Rewrite this function using strtok instead of scanf
        // The problem with scanf is that it does not respect line numbers,
        // and hence it can lead to some unexpected behavior if the input is
        // incorrectly formatted
        for (size_t id = 0; id < out->length; ++id) {
             prec x, y, z;
             int type;
             int match = fscanf(fp, "%d %g %g %g\n", &type, &x, &y, &z);
             if (match != 4) {
                     return ERR_CONFIG_DATA_READ_ELEMENT;
             }
             out->x[id] = x;
             out->y[id] = y;
             out->z[id] = z;
             out->type[id] = type;
        }

        return SUCCESS;
}

