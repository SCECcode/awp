#include <stdio.h>
#include <interpolation/error.h>

const char* error_message(const int err)
{
        switch (err) {
                case SUCCESS:
                        return "";
                        break;
                case ERR_LAGRANGE_LOWER_EXTRAPOLATION:
                        return "Evaluation point is out of bounds (lower) "\
                               " for Lagrange interpolation.";
                        break;
                case ERR_LAGRANGE_UPPER_EXTRAPOLATION:
                        return "Evaluation point is out of bounds (upper) "\
                               " for Lagrange interpolation.";
                        break;
                case ERR_OUT_OF_BOUNDS_LOWER:
                        return "Out of bounds (lower)";
                        break;
                case ERR_OUT_OF_BOUNDS_UPPER:
                        return "Out of bounds (upper)";
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
