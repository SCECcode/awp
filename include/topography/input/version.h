#ifndef VERSION_H
#define VERSION_H
#define VERSION_STRING_LENGTH 1024

#include <mpi.h>

typedef struct
{
        int major;
        int minor;
        int patch;
} version_t;

int version_init(version_t *out, const char *version);
int version_equals(const version_t *a, const version_t *b);
int version_greater_than(const version_t *recent, const version_t *old);
int version_compatible(const version_t *recent, const version_t *old);
int version_to_string(const version_t *version, char *string);
int version_from_file(version_t *version, FILE *fp);
int version_check_compatibility(const version_t *version, 
                                const char *str_current_version);
int version_broadcast(version_t *version, int root, MPI_Comm communicator);

#endif

