#ifndef CHECKSUM_H
#define CHECKSUM_H

/*
Write MD5 checksums to disk.

Usage: call checksum_init to initialize and open file to write to. If the file
does not exist, it will be created. Call checksum_update for each object to
include in the checksum. If you have multiple objects that you want to compute
a checksum for, you can call checksum_update once for each object. To compute
the checksum and write it to disk, call checksum_write. You can then call
checksum_update to start constructing a new checksum. When done, call
checksum_finalize to perform cleanup.
*/

#include <stdio.h>

#include "md5/md5.h"

typedef struct {
        const char *filename;
        FILE *fp;
        MD5_CTX ctx;
} checksum_t;

int checksum_init(checksum_t *chk, const char *filename);
void checksum_update(checksum_t *chk, const void *data, unsigned long size);
void checksum_write(checksum_t *chk, const char *msg);
void checksum_finalize(checksum_t *chk);

#endif

