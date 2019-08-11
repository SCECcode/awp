#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <utils/copy.h>

int copyfile(const char *output, const char *input)
{
        FILE *fin = fopen(input, "r"); 
        FILE *fout = fopen(output, "w"); 
        int count = -1;

        if (fin == NULL) {
                fprintf(stderr, "Cannot open file %s. \n", input);
                return count;
        }

        if (fout == NULL) {
                fprintf(stderr, "Cannot write to file %s. \n", output);
                return count;
        }
  
        char ch;
        while ((ch = fgetc(fin)) != EOF)
                fputc(ch, fout);
        fclose(fin);
        fclose(fout);
        return count;
}

