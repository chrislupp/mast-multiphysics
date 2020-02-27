// C++ Stanard Includes
#include <math.h>

// Catch2 includes
#include "catch.hpp"

// libMesh includes
#include "libmesh/libmesh.h"
#include "libmesh/replicated_mesh.h"
#include "libmesh/point.h"
#include "libmesh/elem.h"
#include "libmesh/edge_edge2.h"
#include "libmesh/equation_systems.h"
#include "libmesh/dof_map.h"

// MAST includes
#include "base/parameter.h"
#include "base/constant_field_function.h"
#include "property_cards/isotropic_material_property_card.h"
#include "property_cards/solid_1d_section_element_property_card.h"
#include "elasticity/structural_element_1d.h"
#include "elasticity/structural_system_initialization.h"
#include "base/physics_discipline_base.h"
#include "base/nonlinear_implicit_assembly.h"
#include "elasticity/structural_nonlinear_assembly.h"
#include "base/nonlinear_system.h"
#include "elasticity/structural_element_base.h"
#include "mesh/geom_elem.h"

// Custom includes
#include "test_helpers.h"
#include "element/structural/1D/mast_structural_element_1d.h"

#define pi 3.14159265358979323846

extern libMesh::LibMeshInit* p_global_init;

TEST_CASE("structural_element_1d_base_tests",
          "[1D],[structural],[base]")
{
    RealMatrixX coords = RealMatrixX::Zero(3, 2);
    coords << -1.0, 1.0, 0.0,
               0.0, 0.0, 0.0;
    TEST::TestStructuralSingleElement1D test_struct_elem(libMesh::EDGE2, coords);

    SECTION("number_strain_components")
    {
        REQUIRE(test_struct_elem.elem->n_direct_strain_components() == 2);
        REQUIRE(test_struct_elem.elem->n_von_karman_strain_components() == 2);
    }
    
    SECTION("no_incompatible_modes")
    {
        REQUIRE_FALSE(test_struct_elem.elem->if_incompatible_modes() );
    }

    SECTION("return_section_property")
    {
        const MAST::ElementPropertyCardBase& elem_section = test_struct_elem.elem->elem_property();
        CHECK( elem_section.if_isotropic() );
    }

    SECTION("set_get_local_solution")
    {
        const libMesh::DofMap& dof_map = test_struct_elem.assembly.system().get_dof_map();
        std::vector<libMesh::dof_id_type> dof_indices;
        dof_map.dof_indices (test_struct_elem.reference_elem, dof_indices);
        uint n_dofs = uint(dof_indices.size());

        RealVectorX elem_solution = 5.3*RealVectorX::Ones(n_dofs);
        test_struct_elem.elem->set_solution(elem_solution);

        const RealVectorX& local_solution = test_struct_elem.elem->local_solution();

        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> test =  eigen_matrix_to_std_vector(elem_solution);
        std::vector<double> truth = eigen_matrix_to_std_vector(local_solution);

        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( test, Catch::Approx<double>(truth) );
    }

    SECTION("set_get_local_solution_sensitivity")
    {
        const libMesh::DofMap& dof_map = test_struct_elem.assembly.system().get_dof_map();
        std::vector<libMesh::dof_id_type> dof_indices;
        dof_map.dof_indices (test_struct_elem.reference_elem, dof_indices);
        uint n_dofs = uint(dof_indices.size());

        RealVectorX elem_solution_sens = 3.1*RealVectorX::Ones(n_dofs);
        test_struct_elem.elem->set_solution(elem_solution_sens, true);

        const RealVectorX& local_solution_sens = test_struct_elem.elem->local_solution(true);

        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> test =  eigen_matrix_to_std_vector(elem_solution_sens);
        std::vector<double> truth = eigen_matrix_to_std_vector(local_solution_sens);

        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( test, Catch::Approx<double>(truth) );
    }
    
    SECTION("element shape can be transformed")
    {
        const Real V0 = test_struct_elem.reference_elem->volume();
        
        // Stretch in x-direction
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 3.1, 1.0, 0.0, 0.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == 6.2);
        
        // Rotation about z-axis
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 60.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
        
        // Rotation about y-axis
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 30.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
        
        // Rotation about x-axis
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 1.0, 1.0, 20.0, 0.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
        
        // Shifted in x-direction
        transform_element(test_struct_elem.mesh, coords, 10.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
        
        // Shifted in y-direction
        transform_element(test_struct_elem.mesh, coords, 0.0, 7.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
        
        // Shifted in z-direction
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 4.2, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE(test_struct_elem.reference_elem->volume() == V0);
    }
}
