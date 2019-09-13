#include <stdlib.h>
#include <stdio.h>
#include "md5.h"

int main(int argc, char **argv)
{
        float *data;
        int n = (int)1e7;
        data = malloc(sizeof(data) * n);
        for (int i = 0; i < n; ++i) {
                data[i] = i*1.0;
        }
        unsigned char hash[64];
        MD5_CTX ctx;
        MD5_Init(&ctx);
        MD5_Update(&ctx, data, n*sizeof(float));
        MD5_Final(hash, &ctx);
        char checksum[33];
        for(int i = 0; i < 16; ++i)
        sprintf(&checksum[i*2], "%02x", (unsigned int)hash[i]);
        printf("checksum: %s \n", checksum);

}

