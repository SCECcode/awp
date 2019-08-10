#ifndef READERS_ERROR_H
#define READERS_ERROR_H

int _last_error;

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
};
// Display the error message associated with an error code.
const char* error_message(const int err);
void error_print(const int err);

#ifdef __cplusplus
}
#endif
#endif
