// We need access to the protected thermal_residual method to test it
// NOTE: Be careful with this, it could cause unexpected problems
#define protected public

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
#include "base/boundary_condition_base.h"


// Custom includes
#include "test_helpers.h"
#include "element/structural/1D/mast_structural_element_1d.h"

#define pi 3.14159265358979323846

extern libMesh::LibMeshInit* p_global_init;


TEST_CASE("edge2_nonlinear_structural_thermal_jacobian",
          "[1D],[thermoelastic],[edge],[edge2],[nonlinear],[protected]")
{
    RealMatrixX coords = RealMatrixX::Zero(3, 2);
    coords << -1.0, 1.0, 0.0,
               0.0, 0.0, 0.0;
    TEST::TestStructuralSingleElement1D test_struct_elem(libMesh::EDGE2, coords);

    // Define the Uniform Temperature and Uniform Reference Temperature
    MAST::Parameter temperature("T", 400.0);
    MAST::Parameter ref_temperature("T0", 0.0);
    MAST::ConstantFieldFunction temperature_f("temperature", temperature);
    MAST::ConstantFieldFunction ref_temperature_f("ref_temperature", ref_temperature);

    // Setup the temperature change boundary condition
    MAST::BoundaryConditionBase temperature_load(MAST::TEMPERATURE);
    temperature_load.add(temperature_f);
    temperature_load.add(ref_temperature_f);
    test_struct_elem.discipline.add_volume_load(0, temperature_load);

    // Set the strain type to linear for the section
    test_struct_elem.section.set_strain(MAST::NONLINEAR_STRAIN);
    // Set the bending operator to Euler-Bernoulli
    test_struct_elem.section.set_bending_model(MAST::TIMOSHENKO);

    const Real V0 = test_struct_elem.reference_elem->volume();
    
    // Calculate residual and jacobian
    RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
    RealMatrixX jacobian0 = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
    test_struct_elem.elem->thermal_residual(true, residual, jacobian0, temperature_load);
            
    double val_margin = (jacobian0.array().abs()).mean() * 1.490116119384766e-08;
    
    //libMesh::out << "R=\n" << residual << std::endl;
    //libMesh::out << "J =\n" << jacobian0 << std::endl;
    

    SECTION("thermal_jacobian_finite_difference_check")                   
    {
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian_fd);
        
        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    
    SECTION("thermal_jacobian_symmetry_check")
    {
        // Element stiffness matrix should be symmetric
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian0);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0.transpose());
        REQUIRE_THAT( test, Catch::Approx<double>(truth) );
    }
    
    
    SECTION("thermal_jacobian_determinant_check")
    {
        // Determinant of linear thermoelastic jacobian should be zero
        REQUIRE( jacobian0.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    
//     SECTION("thermal_jacobian_eigenvalue_check")
//     {
//         /**
//          * Linear thermoelastic Jacobian should be independent of the 
//          * displacements and thus should be a zero matrix.
//          */
//         SelfAdjointEigenSolver<RealMatrixX> eigensolver(jacobian0, false);
//         RealVectorX eigenvalues = eigensolver.eigenvalues();
//         libMesh::out << "Eigenvalues are:\n" << eigenvalues << std::endl;
//         uint nz = 0;
//         for (uint i=0; i<eigenvalues.size(); i++)
//         {
//             if (std::abs(eigenvalues(i))<0.0001220703125)
//             {
//                 nz++;
//             }
//         }
//         REQUIRE( nz == 12);
//     }
    
    
    SECTION("thermal_jacobian_orientation_invariant")
    {
        test_struct_elem.section.clear();
        RealVectorX orientation = RealVectorX::Zero(3);
        orientation(2) = 1.0;
        test_struct_elem.section.y_vector() = orientation;
        test_struct_elem.section.init();
        
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
                
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0);

        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    
    SECTION("thermal_jacobian_displacement_invariant")
    {
        // Calculate residual and jacobian at arbitrary displacement
        RealVectorX elem_sol = RealVectorX::Zero(test_struct_elem.n_dofs);
        elem_sol << 0.5727841,  0.8896581,  0.9541619, -0.3774913,  
                    0.7510557, -0.7122266, -0.0979117, -0.8300009, 
                    -0.3453369, -0.5487761, -0.1407677, -0.9268421;
        test_struct_elem.elem->set_solution(elem_sol);
        
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
                
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0);

        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    SECTION("thermal_jacobian_shifted_x_invariant")
    {
        // Shifted in x-direction
        transform_element(test_struct_elem.mesh, coords, 5.2, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
                
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0);

        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    SECTION("thermal_jacobian_shifted_y_invariant")
    {
        // Shifted in y-direction
        transform_element(test_struct_elem.mesh, coords, 0.0, -11.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0);
        
        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    SECTION("thermal_jacobian_shifted_z_invariant")
    {
        // Shifted in y-direction
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 7.6, 1.0, 1.0, 0.0, 0.0, 0.0);
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        std::vector<double> test =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> truth = eigen_matrix_to_std_vector(jacobian0);
        
        REQUIRE_THAT( test, Catch::Approx<double>(truth).margin(val_margin) );
    }
    
    
    SECTION("thermal_jacobian_aligned_y")
    {
        /*
         * NOTE: We could try to use the transform_element method here, but the
         * issue is that if the sin and cos calculations are not exact, then we
         * may not be perfectly aligned along the y axis like we want.
         */
        RealMatrixX X= RealMatrixX::Zero(3,test_struct_elem.n_nodes);
        X << 0.0, 0.0, -1.0, 1.0, 0.0, 0.0;
        for (int i=0; i<test_struct_elem.n_nodes; i++)
        {
            (*test_struct_elem.mesh.node_ptr(i)) = libMesh::Point(X(0,i), X(1,i), X(2,i));
        }
        
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_aligned_z")
    {
        /*
         * NOTE: We could try to use the transform_element method here, but the
         * issue is that if the sin and cos calculations are not exact, then we
         * may not be perfectly aligned along the z axis like we want.
         */
        RealMatrixX X= RealMatrixX::Zero(3,test_struct_elem.n_nodes);
        X << 0.0, 0.0, 0.0, 0.0, -1.0, 1.0;
        for (int i=0; i<test_struct_elem.n_nodes; i++)
        {
            (*test_struct_elem.mesh.node_ptr(i)) = libMesh::Point(X(0,i), X(1,i), X(2,i));
        }
        
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_rotated_about_z")
    {
        // Rotated 63.4 about z-axis at element's centroid
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 63.4);
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_rotated_about_y")
    {
        // Rotated 35.8 about y-axis at element's centroid
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 35.8, 0.0);
        REQUIRE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_scaled_x")
    {
        // Rotated 63.4 about z-axis at element's centroid
        transform_element(test_struct_elem.mesh, coords, 0.0, 0.0, 0.0, 3.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        REQUIRE_FALSE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_arbitrary_transformation")
    {
        // Arbitrary transformations applied to the element
        transform_element(test_struct_elem.mesh, coords, -5.0, 7.8, -13.1, 2.7, 6.4, 20.0, 47.8, -70.1);
        REQUIRE_FALSE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
    
    SECTION("thermal_jacobian_arbitrary_with_displacements")
    {
        // Arbitrary transformations applied to the element
        transform_element(test_struct_elem.mesh, coords, 4.1, -6.3, 7.5, 4.2, 1.5, -18.0, -24.8, 
                          30.1);
        
        // Calculate residual and jacobian at arbitrary displacement
        RealVectorX elem_sol = RealVectorX::Zero(test_struct_elem.n_dofs);
        elem_sol << 0.8158724,  0.7991906, -0.0719128,  0.2025461, 
                    -0.4602193, 0.5280159,  0.3700081,  0.4636344,  
                    0.5559377,  0.6448206, 0.8919238, -0.3079122;
        test_struct_elem.elem->set_solution(elem_sol);
        
        REQUIRE_FALSE( test_struct_elem.reference_elem->volume() == Approx(V0) );
        
        // Calculate residual and jacobian
        RealVectorX residual = RealVectorX::Zero(test_struct_elem.n_dofs);
        RealMatrixX jacobian = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        test_struct_elem.elem->thermal_residual(true, residual, jacobian, temperature_load);
        
        // Approximate Jacobian with Finite Difference
        RealMatrixX jacobian_fd = RealMatrixX::Zero(test_struct_elem.n_dofs, test_struct_elem.n_dofs);
        approximate_thermal_jacobian_with_finite_difference(*test_struct_elem.elem, test_struct_elem.elem_solution, jacobian_fd, temperature_load);
        
        // This is necessary because MAST manually (hard-coded) adds a small 
        // value to the diagonal to prevent singularities at inactive DOFs
        //double val_margin = (jacobian_fd.array().abs()).maxCoeff() * 1.490116119384766e-08;
        val_margin = (jacobian_fd.array().abs()).mean() * 1.490116119384766e-08;
        //std::cout << "val_margin = " << val_margin << std::endl;
        
        // Convert the test and truth Eigen::Matrix objects to std::vector
        // since Catch2 has built in methods to compare vectors
        std::vector<double> J =  eigen_matrix_to_std_vector(jacobian);
        std::vector<double> J_fd = eigen_matrix_to_std_vector(jacobian_fd);
        
        // Floating point approximations are diffcult to compare since the
        // values typically aren't exactly equal due to numerical error.
        // Therefore, we use the Approx comparison instead of Equals
        REQUIRE_THAT( J, Catch::Approx<double>(J_fd).margin(val_margin) );
        
        // Symmetry check
        std::vector<double> Jt = eigen_matrix_to_std_vector(jacobian.transpose());
        REQUIRE_THAT( Jt, Catch::Approx<double>(J) );
        
        // Determinant check
        REQUIRE( jacobian.determinant() == Approx(0.0).margin(1e-06) );
    }
}
