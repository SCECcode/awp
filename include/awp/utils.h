#ifndef UTILS_H
#define UTILS_H

double gethrtime();
void error_check(int ierr, char *message);
int copyfile(const char *output, const char *input);

#endif

