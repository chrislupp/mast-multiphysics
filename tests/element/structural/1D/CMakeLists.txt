target_sources(mast_catch_tests
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/mast_structural_element_1d.cpp)

# 1D Structural Element Test Basic Tests
add_test(NAME Element_1D_Structural_Basic_Tests
    COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "structural_element_1d_base_tests")
set_tests_properties(Element_1D_Structural_Basic_Tests
    PROPERTIES
        LABELS "SEQ"
        FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     Element_1D_Structural_Basic_Tests)

add_test(NAME Element_1D_Structural_Basic_Tests_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} -np 2  $<TARGET_FILE:mast_catch_tests> -w NoTests "structural_element_1d_base_tests")
set_tests_properties(Element_1D_Structural_Basic_Tests_mpi
    PROPERTIES
        LABELS "MPI"
        FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural_mpi;libMesh_Mesh_Generation_1d_mpi"
        FIXTURES_SETUP     Element_1D_Structural_Basic_Tests_mpi)

                     
add_subdirectory(edge2)
