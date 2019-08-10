#include <stdio.h>
#include <grid/error.h>

const char* error_message(const int err)
{
        switch (err) {
                case SUCCESS:
                        return "";
                        break;
                case ERR_GRID_OUT_OF_BOUNDS:
                        return "Grid is out of bounds.";
                        break;
                case ERR_OUT_OF_BOUNDS_LOWER:
                        return "Out of bounds (lower)";
                        break;
                case ERR_OUT_OF_BOUNDS_UPPER:
                        return "Out of bounds (upper)";
                        break;
                case ERR_NON_POSITIVE:
                        return "Integer must be greater than zero.";
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

