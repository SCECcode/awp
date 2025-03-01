set(HEADERS
    ${AWP_SOURCE_DIR}/include/awp/definitions.h
    ${AWP_SOURCE_DIR}/include/vtk/vtk.h
    )

add_library(vtk
        vtk.c
        )

target_include_directories(vtk
        PUBLIC
        ${AWP_SOURCE_DIR}/include/
        )
