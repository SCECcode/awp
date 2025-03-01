#ifndef ERROR_H
#define ERROR_H

enum error_codes {SUCCESS, 
                  ERR_FILE_OPEN = 100, 
                  ERR_FILE_READ = 101, 
                  ERR_FILE_WRITE = 102, 
                  ERR_GET_VERSION = 200, 
                  ERR_WRONG_VERSION = 201, 
                  ERR_BROADCAST_VERSION = 202, 
                  ERR_CONFIG_PARSE_SIZES = 300,
                  ERR_CONFIG_SIZE_OVERFLOW = 301,
                  ERR_CONFIG_DATA_FILENAME = 302,
                  ERR_CONFIG_DATA_WRITEABLE = 303,
                  ERR_CONFIG_DATA_MALLOC = 304,
                  ERR_CONFIG_DATA_READ_ELEMENT = 305,
                  ERR_CONFIG_BROADCAST = 306,
                  ERR_CONFIG_DATA_SIZE = 307,
                  ERR_CONFIG_PARSE_ARG = 308,
                  ERR_CONFIG_PARSE_UNKNOWN_ARG = 309,
                  ERR_CONFIG_PARSE_WRONG_DIMENSION = 310,
                  ERR_CONFIG_PARSE_NOT_DIVISIBLE = 311,
                  ERR_GRID_OUT_OF_BOUNDS = 400,
                  ERR_TEST_FAILED = 500,
                  ERR_LAGRANGE_LOWER_EXTRAPOLATION = 600,
                  ERR_LAGRANGE_UPPER_EXTRAPOLATION = 601,
                  ERR_OUT_OF_BOUNDS_LOWER = 700,
                  ERR_OUT_OF_BOUNDS_UPPER = 701,
                  ERR_NON_POSITIVE = 800,
                  ERR_INTERP_MALLOC = 900,
                  ERR_INTERP_WRITE_END_OF_FILE = 901,
                  ERR_INCONSISTENT_SIZE = 902,
                  ERR_INCOMPATIBLE_SOURCE_TYPE = 1001
};
// Display the error message associated with an error code.
const char* error_message(const int err);
// Call this function whenever an error occurs. Use 'error_last_message' to
// obtain the error message for the most recent error that occurred.
int error_set(const int error);

const char* error_last_message(void);

// Print error message to stderr
void error_print(const int err);

#define AWPCHK(err) {                                                  \
  if( (err) != 0) {                                                    \
  fprintf(stderr, "AWP error: %s:%s():%i %s.\n",                       \
          __FILE__, __func__, __LINE__, error_message( err) );         \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

#endif

