#include <stdlib.h>
#include <stdio.h>

#include <awp/error.h>

const char* error_message(const int err)
{
        switch (err) {
                case SUCCESS:
                        return "";
                        break;
                case ERR_FILE_READ:
                        return "Failed to read file.";
                        break;
                case ERR_FILE_WRITE:
                        return "Failed to write file.";
                        break;
                case ERR_GET_VERSION:
                        return "Failed to get version number.";
                        break;
                case ERR_WRONG_VERSION:
                        return "Incompatible version number.";
                        break;
                case ERR_BROADCAST_VERSION:
                        return "Failed to broadcast version.";
                        break;
                case ERR_CONFIG_PARSE_SIZES:
                        return "Failed to parse number of elements, etc .";
                        break;
                case ERR_CONFIG_SIZE_OVERFLOW:
                        return "Too many elements. Integer overflow occured.";
                        break;
                case ERR_CONFIG_DATA_FILENAME:
                        return "Failed to parse output data filename.";
                        break;
                case ERR_CONFIG_DATA_WRITEABLE:
                        return "Failed to write output data file.";
                        break;
                case ERR_CONFIG_DATA_MALLOC:
                        return "Failed to allocate output data.";
                        break;
                case ERR_CONFIG_DATA_READ_ELEMENT:
                        return "Failed to read data element description.";
                        break;
                case ERR_CONFIG_BROADCAST:
                        return "Failed to broadcast data description.";
                        break;
                case ERR_CONFIG_DATA_SIZE:
                        return "Data size must be positive.";
                        break;
                case ERR_CONFIG_PARSE_ARG:
                        return "Failed to parse argument in input file.";
                        break;
                case ERR_CONFIG_PARSE_UNKNOWN_ARG:
                        return "Unknown argument found in input file.";
                        break;
                case ERR_CONFIG_PARSE_WRONG_DIMENSION:
                        return "Invalid dimension in input file.";
                        break;
                case ERR_CONFIG_PARSE_NOT_DIVISIBLE:
                        return "Number of steps is not divisible by buffer "
                               "size.";
                        break;
                case ERR_GRID_OUT_OF_BOUNDS:
                        return "Grid is out of bounds.";
                        break;
                case ERR_TEST_FAILED:
                        return "Test failed.";
                        break;
                case ERR_NON_POSITIVE:
                        return "Integer must be greater than zero.";
                        break;
                case ERR_INTERP_MALLOC:
                        return "Memory allocation failed in cuinterpolation";
                        break;
                case ERR_INTERP_WRITE_END_OF_FILE:
                        return "Tried to write past end of file.";
                        break;
                case ERR_INCONSISTENT_SIZE:
                        return "Inconsistent size.";
                        break;
                default:
                        return "Unknown error code.";
                        break;
        }
        return "";
}

void error_print(const int err)
{
        if (err > 0) {
                fprintf(stderr, "%s\n", error_message(err));
        }
}
