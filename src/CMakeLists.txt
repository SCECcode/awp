#Specify files to be compiled
set(HEADERS ${AWP_MINI_SOURCE_DIR}/include/hello/cuhello.cuh) 


#Combine files to create an executable
add_library(hellolib 
        cuhello.cu 
        ${HEADERS})
include_directories(../include)

target_include_directories(hellolib
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hellolib ${MPI_C_LIBRARIES})

