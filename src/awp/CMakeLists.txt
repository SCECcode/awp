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

