target_sources(mast
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_discipline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_discipline.h
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_element_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_element_base.h
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_system_initialization.cpp
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_system_initialization.h
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_transient_assembly.cpp
        ${CMAKE_CURRENT_LIST_DIR}/conservative_fluid_transient_assembly.h
        ${CMAKE_CURRENT_LIST_DIR}/flight_condition.h
        ${CMAKE_CURRENT_LIST_DIR}/fluid_elem_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/fluid_elem_base.h
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_linearized_complex_assembly.cpp
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_linearized_complex_assembly.h
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_linearized_conservative_fluid_elem.cpp
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_linearized_conservative_fluid_elem.h
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_pressure_function.cpp
        ${CMAKE_CURRENT_LIST_DIR}/frequency_domain_pressure_function.h
        ${CMAKE_CURRENT_LIST_DIR}/gas_property.h
        ${CMAKE_CURRENT_LIST_DIR}/integrated_force_output.cpp
        ${CMAKE_CURRENT_LIST_DIR}/integrated_force_output.h
        ${CMAKE_CURRENT_LIST_DIR}/pressure_function.cpp
        ${CMAKE_CURRENT_LIST_DIR}/pressure_function.h
        ${CMAKE_CURRENT_LIST_DIR}/primitive_fluid_solution.cpp
        ${CMAKE_CURRENT_LIST_DIR}/primitive_fluid_solution.h
        ${CMAKE_CURRENT_LIST_DIR}/small_disturbance_primitive_fluid_solution.cpp
        ${CMAKE_CURRENT_LIST_DIR}/small_disturbance_primitive_fluid_solution.h
        ${CMAKE_CURRENT_LIST_DIR}/surface_integrated_pressure_output.cpp
        ${CMAKE_CURRENT_LIST_DIR}/surface_integrated_pressure_output.h)

# Install MAST headers for this directory.
install(DIRECTORY ./ DESTINATION include/fluid
        FILES_MATCHING PATTERN "*.h")