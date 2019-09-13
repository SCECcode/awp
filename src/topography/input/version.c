#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <topography/input/version.h>
#include <awp/error.h>

int set_version_number(char  *token, int *number);

int version_init(version_t *out, const char *version)
{
   const char s[2] = ".";

   int err = 0;
   err &= set_version_number(strtok((char*)version, s), &out->major);
   err &= set_version_number(strtok(NULL, s), &out->minor);
   err &= set_version_number(strtok(NULL, s), &out->patch);

   return err;
}

int version_equals(const version_t *a, const version_t *b)
{
        int equals = 1;
        equals &= a->major == b->major;
        equals &= a->minor == b->minor;
        equals &= a->patch == b->patch;
        return equals;
}

int version_greater_than(const version_t *recent, const version_t *old)
{
        if (recent->major == old->major) {
                if (recent->minor == old->minor) {
                        return recent->patch > old->patch;
                }
                return recent->minor > old->minor;
        }
        return recent->major > old->major;
}

// Only require major version number to match for compatibility
int version_compatible(const version_t *recent, const version_t *old)
{
        if (recent->major == old->major) {
                return 1;
        } 
        else {
                return 0;
        }
        
}

int version_to_string(const version_t *version, char *string)
{
        return sprintf(string, "%d.%d.%d", version->major, version->minor,
                       version->patch);
}

/*
 * Initialize version from file handle. 
 */ 
int version_from_file(version_t *version, FILE *fp)
{
        char version_string[VERSION_STRING_LENGTH];
      
        int version_err = 0;
        if (fgets(version_string, VERSION_STRING_LENGTH, fp) != NULL) 
        {
                version_err = version_init(version, version_string);
        } else {
                return ERR_GET_VERSION;
        }

        return version_err;
}

int version_check_compatibility(const version_t *version, 
                                const char *str_current_version)
{
        version_t current = {0, 0, 0};
        version_init(&current, str_current_version);
        if (!version_equals(version, &current) &&
            !version_compatible(version, &current)
           ) {
                return ERR_WRONG_VERSION;
        }
        return SUCCESS;
}

int version_broadcast(version_t *version, int root, MPI_Comm communicator)
{
        int err;
        err = MPI_Bcast(&version->major, 1, MPI_INT, root, communicator);
        err |= MPI_Bcast(&version->minor, 1, MPI_INT, root, communicator);
        err |= MPI_Bcast(&version->patch, 1, MPI_INT, root, communicator);

        if (err > 0) {
                return ERR_BROADCAST_VERSION;
        }
        
        return SUCCESS;
}

int set_version_number(char  *token, int *number)
{
   if (token != NULL) {
        *number = atoi(token);
   } else {
           return 1;
   }
   return 0;
}


