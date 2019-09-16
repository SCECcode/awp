set(HEADERS 
   ${AWP_MINI_SOURCE_DIR}/include/awp/definitions.h 
   ${AWP_MINI_SOURCE_DIR}/include/awp/pmcl3d_cons.h 
   )

add_library(awp
        kernel.cu 
        ${HEADERS})

target_include_directories(awp
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )


add_library(error
        error.c
        )

target_include_directories(error
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_library(lpmcl3d
        calc.c
        cerjan.c
        command.c
        dump.c
        grid.c
        init.c
        io.c
        kernel.cu 
        mesh.c
        source.c
        swap.c
        utils.c
        ${HEADERS})

target_link_libraries(lpmcl3d opt_topography_attenuation buffers mpi checksum)
#
target_include_directories(lpmcl3d
        PUBLIC
        ${AWP_MINI_SOURCE_DIR}/include/
        )

add_executable(pmcl3d pmcl3d.c)
target_link_libraries(pmcl3d lpmcl3d)
