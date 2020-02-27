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


TEST_CASE("edge2_linear_structural_inertial_consistent",
          "[1D],[dynamic],[edge],[edge2],[element]")
{
    RealMatrixX coords = RealMatrixX::Zero(3, 2);
    coords << -1.0, 1.0, 0.0,
               0.0, 0.0, 0.0;
    TEST::TestStructuralSingleElement1D test_struct_elem(libMesh::EDGE2, coords);
    
    const Real V0 = test_struct_elem.reference_elem->volume();
    
    // Calculate residual and jacobian
    RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
    RealMatrixX jac0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    RealMatrixX jac_xddot0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    RealMatrixX jac_xdot0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    test_struct_elem.elem->inertial_residual(true, residual, jac_xddot0, jac_xdot0, jac0);
            
    double val_margin = (jac_xddot0.array().abs()).mean() * 1.490116119384766e-08;
    
    //libMesh::out << "Jac_xddot0:\n" << jac_xddot0 << std::endl;
    
    SECTION("inertial_jacobian_finite_difference_check")                   
    {
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_inertial_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_accel, jacobian_fd);
        
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        
        std::vector<double> test =  eigen_matrix_to_std_vector(jac_xddot0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian_fd);
        
        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    
    SECTION("inertial_jacobian_symmetry_check")
    {
        // Element inertial jacobian should be symmetric
        std::vector<double> test =  eigen_matrix_to_std_vector(jac_xddot0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jac_xddot0.transpose());
        REQUIRE_THAT( test, Catch::Approx<double>(truth) );
    }
    
    
    SECTION("inertial_jacobian_determinant_check")
    {
        // Determinant of inertial jacobian should be positive
        REQUIRE( jac_xddot0.determinant() > 0.0 );
    }
    
    
    SECTION("inertial_jacobian_eigenvalue_check")
    {
        /**
         * A lumped mass matrix should have all positive eigenvalues since it
         * is a diagonal matrix and masses should not be zero or negative.
         */
        SelfAdjointEigenSolver<RealMatrixX> eigensolver(jac_xddot0, false);
        RealVectorX eigenvalues = eigensolver.eigenvalues();
        //libMesh::out << "Eigenvalues are:\n" << eigenvalues << std::endl;
        REQUIRE(eigenvalues.minCoeff()>0.0);
    }
}


TEST_CASE("edge2_linear_structural_inertial_lumped",
          "[1D],[dynamic],[edge],[edge2]")
{
    RealMatrixX coords = RealMatrixX::Zero(3, 2);
    coords << -1.0, 1.0, 0.0,
            0.0, 0.0, 0.0;
    TEST::TestStructuralSingleElement1D test_struct_elem(libMesh::EDGE2, coords);

    const Real V0 = test_struct_elem.reference_elem->volume();
    
    // Calculate residual and jacobian
    RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
    RealMatrixX jac0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    RealMatrixX jac_xddot0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    RealMatrixX jac_xdot0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    test_struct_elem.elem->inertial_residual(true, residual, jac_xddot0, jac_xdot0, jac0);
            
    double val_margin = (jac_xddot0.array().abs()).mean() * 1.490116119384766e-08;
    
    //libMesh::out << "Jac_xddot0:\n" << jac_xddot0 << std::endl;
    
    SECTION("inertial_jacobian_finite_difference_check")                   
    {
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_inertial_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_accel, jacobian_fd);
        
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        
        std::vector<double> test =  eigen_matrix_to_std_vector(jac_xddot0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian_fd);
        
        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    
    SECTION("inertial_jacobian_symmetry_check")
    {
        // Element interial jacobian should be symmetric
        std::vector<double> test =  eigen_matrix_to_std_vector(jac_xddot0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jac_xddot0.transpose());
        REQUIRE_THAT( test, Catch::Approx<double>(truth) );
    }
    
    
    SECTION("inertial_jacobian_determinant_check")
    {
        // Determinant of inertial jacobian should be positive
        REQUIRE( jac_xddot0.determinant() > 0.0 );
    }
    
    
    SECTION("inertial_jacobian_eigenvalue_check")
    {
        /**
         * A lumped mass matrix should have all positive eigenvalues since it
         * is a diagonal matrix and masses should not be zero or negative.
         */
        SelfAdjointEigenSolver<RealMatrixX> eigensolver(jac_xddot0, false);
        RealVectorX eigenvalues = eigensolver.eigenvalues();
        //libMesh::out << "Eigenvalues are:\n" << eigenvalues << std::endl;
        REQUIRE(eigenvalues.minCoeff()>0.0);
    }
}
