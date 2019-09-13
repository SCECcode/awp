#include <checksum.h>
#include <md5/md5.h>

int checksum_init(checksum_t *chk, const char *filename)
{
        FILE *fp;
        fp = fopen(filename, "w");
        if (!fp) {
                return 1;
        }

        MD5_CTX ctx;
        MD5_Init(&ctx);

        chk->fp = fp;
        chk->filename = filename;
        chk->ctx = ctx;

        return 0;
}

void checksum_update(checksum_t *chk, const void *data, unsigned long num_bytes)
{
        MD5_Update(&chk->ctx, data, num_bytes);
}

void checksum_write(checksum_t *chk, const char *msg)
{
        unsigned char hash[64];
        MD5_Final(hash, &chk->ctx);
        char checksum[33];
        for(int i = 0; i < 16; ++i)
        sprintf(&checksum[i*2], "%02x", (unsigned int)hash[i]);
        fprintf(chk->fp, "%s%s\n", msg, checksum);
        MD5_Init(&chk->ctx);
}

void checksum_finalize(checksum_t *chk)
{
        unsigned char hash[64];
        MD5_Final(hash, &chk->ctx);
        fclose(chk->fp);
}

