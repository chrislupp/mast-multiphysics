/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2020  Manav Bhatia and MAST authors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */


#ifndef _mast__level_set_volume_output_h__
#define _mast__level_set_volume_output_h__

// MAST includes
#include "base/output_assembly_elem_operations.h"


namespace MAST {
    
    class LevelSetVolume:
    public MAST::OutputAssemblyElemOperations {
      
    public:
        
        LevelSetVolume();

        virtual ~LevelSetVolume();
        
        /*!
         *   virtual function, nothing to be done for level set
         */
        virtual void
        set_elem_data(unsigned int dim,
                      const libMesh::Elem& ref_elem,
                      MAST::GeomElem& elem) const { }

        /*!
         *   initializes the object for calculation of element quantities for
         *   the specified \p elem.
         */
        virtual void
        init(const MAST::GeomElem& elem);

        /*!
         *   zeroes the output quantity values stored inside this object
         *   so that assembly process can begin. This will zero out data
         *   so that it is ready for a new evaluation. Before sensitivity
         *   analysis, call the other method, since some nonlinear
         *   functionals need the forward quantities for sensitivity analysis,
         *   eg., stress output.
         */
        virtual void zero_for_analysis();
        
        
        /*!
         *   zeroes the output quantity values stored inside this object
         *   so that assembly process can begin. This will only zero the
         *   data to compute new sensitivity analysis.
         */
        virtual void zero_for_sensitivity();

        /*!
         *   @returns the output quantity value contribution for the
         *   present element for which this object has been initialized.
         *   This does not make sense for some output quantities that are
         *   defined as averaged over a domain, example the p-norm of
         *   the stress (see MAST::StressStrainOutputBase). For such cases,
         *   the output_total must be called.
         */
        virtual Real output_for_elem();
        
        /*!
         *   @returns the output quantity value accumulated over all elements
         */
        virtual Real output_total();

        /*!
         *   @returns the output quantity sensitivity for parameter.
         *   This method calculates the partial derivative of quantity
         *    \f[ \frac{\partial q(X, p)}{\partial p} \f]  with
         *    respect to parameter \f$ p \f$. This returns the quantity
         *    evaluated for on element for which this object is initialized.
         */
        virtual Real output_sensitivity_for_elem(const MAST::FunctionBase& p);
        
        /*!
         *   @returns the output quantity sensitivity for parameter.
         *   This method calculates the partial derivative of quantity
         *    \f[ \frac{\partial q(X, p)}{\partial p} \f]  with
         *    respect to parameter \f$ p \f$. This returns the quantity
         *   accumulated over all elements.
         */
        virtual Real output_sensitivity_total(const MAST::FunctionBase& p);

        /*!
         *   returns the output quantity derivative with respect to
         *   state vector in \p dq_dX.
         *   This method calculates the quantity
         *    \f[ \frac{\partial q(X, p)}{\partial X} \f] for this
         *    output function. This is returned for the element for which
         *   this
         */
        virtual void output_derivative_for_elem(RealVectorX& dq_dX){
            // does not make sense for this output. // should not get called.
            libmesh_error();
        }
        
        
        /*!
         *    this is the abstract interface to be implemented by derived
         *    classes. This method calculates the quantity \f$ q(X, p) \f$.
         *    This is provided as an interface since some quantities
         *    requires evaluation of all inputs before the final quantity
         *    can be calculated. So, the user can call this routine on each
         *    element before requesting the output values using either
         *    \p output_for_elem() or \p output_total().
         */
        virtual void evaluate();

        /*!
         *    this evaluates all relevant sensitivity components on
         *    the element.
         *    This is only done on the current element for which this
         *    object has been initialized.
         */
        virtual void evaluate_sensitivity(const MAST::FunctionBase& f);

        /*!
         *    this evaluates all relevant shape sensitivity components on
         *    the element.
         *    This is only done on the current element for which this
         *    object has been initialized.
         */
        virtual void evaluate_shape_sensitivity(const MAST::FunctionBase& f) {
            
            libmesh_assert(false); // to be implemented
        }
        
        /*!
         *    this evaluates all relevant topological sensitivity components on
         *    the element.
         *    This is only done on the current element for which this
         *    object has been initialized.
         */
        virtual void
        evaluate_topology_sensitivity(const MAST::FunctionBase& f);

        /*!
         *    This evaluates the contribution to the topology sensitivity on the
         *    boundary \f$ \int_\Gamma V_n~d\Gamma \f$
         */
        virtual void evaluate_topology_sensitivity(const MAST::FunctionBase& f,
                                                   const MAST::FieldFunction<RealVectorX>& vel);

    protected:

        Real                                _vol;
        Real                                _dvol_dp;
    };
    
}




#endif // _mast__level_set_volume_output_h__
