
void seism_init(MPI_Comm *tcomm, int *rank, int *tcoords, int *tmaxdim, 
    int *tnx, int *tny, int *tnz, 
    int *tnxt, int *tnyt, int *tnzt, 
    int *tghostx, int *tghosty, int *tghostz,
    int *tpx, int *tpy, int *tpz,
    char *t_seism_method, int *err);

void seism_createRegularGrid(int *nbgx, int *nedx, int *nskpx, 
    int *nbgy, int *nedy, int *nskpy, int *nbgz, int *nedz, int *nskpz, 
    int *gridID, int *err);

void seism_file_open(char *fname, char *fmode, int *write_step, char *fdata,
    int *psID, int *fpID, int *err);

void seism_write(int *seism_f, void *var, int *err);

void seism_file_close(int *seism_f, int *err);

